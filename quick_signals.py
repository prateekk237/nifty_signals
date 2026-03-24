"""
╔══════════════════════════════════════════════════════════════╗
║  QUICK SIGNALS — Volatile Market Fast-Trigger Engine        ║
║                                                              ║
║  SEPARATE from main confluence system. This fires on just   ║
║  2 out of 3 fast indicators agreeing:                        ║
║                                                              ║
║  1. Supertrend (5, 1.5) — fastest trend direction           ║
║  2. VWAP position       — institutional flow bias           ║
║  3. RSI (7)             — momentum confirmation             ║
║                                                              ║
║  WHY THIS EXISTS:                                            ║
║  In high-VIX markets (>20), the main system waits for 11    ║
║  indicators to agree — which rarely happens when VIX is     ║
║  spiking. This module needs only 2/3 agreement and gives    ║
║  5-8 signals per day instead of 0-1.                        ║
║                                                              ║
║  BEST FOR: Scalping in volatile markets, ₹5K-50K capital    ║
║  TIMEFRAME: 5-minute candles                                 ║
║  HOLD TIME: 10-30 minutes per trade                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pytz

from config import (
    NIFTY_LOT_SIZE, BANKNIFTY_LOT_SIZE,
    STRIKE_STEP_NIFTY, STRIKE_STEP_BANKNIFTY,
)
from data_fetcher import get_atm_strike

IST = pytz.timezone("Asia/Kolkata")


def generate_quick_signal(
    df: pd.DataFrame,
    symbol: str = "NIFTY50",
    capital: float = 10000,
) -> Dict:
    """
    Generate fast scalping signal using only 3 indicators.

    Fires when 2 out of 3 agree on direction:
      - Supertrend (5, 1.5): trend direction
      - Price vs VWAP: institutional bias
      - RSI (7): momentum

    Returns complete trade recommendation or NO SIGNAL.
    """
    if df is None or df.empty or len(df) < 10:
        return _no_quick_signal("Not enough data")

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])

    # ═══════════════════ 3 FAST INDICATORS ═══════════════════

    # 1. SUPERTREND (5, 1.5) — direction
    st_col = "STd_5_1.5"
    st_signal = 0
    st_fresh_cross = False
    st_detail = "N/A"
    if st_col in df.columns:
        curr_dir = int(last[st_col])
        prev_dir = int(prev[st_col])
        st_signal = curr_dir  # +1 = bullish, -1 = bearish
        st_fresh_cross = (curr_dir != prev_dir)
        st_detail = f"{'BULLISH' if curr_dir > 0 else 'BEARISH'}" + (" ★FRESH" if st_fresh_cross else "")

    # 2. VWAP — institutional bias
    vwap_signal = 0
    vwap_detail = "N/A"
    if "VWAP" in df.columns:
        vwap = float(last["VWAP"])
        pct_from_vwap = ((price - vwap) / vwap) * 100
        if price > vwap:
            vwap_signal = 1
            vwap_detail = f"ABOVE VWAP ({pct_from_vwap:+.2f}%)"
        else:
            vwap_signal = -1
            vwap_detail = f"BELOW VWAP ({pct_from_vwap:+.2f}%)"
    else:
        # Fallback: use EMA 9 vs 21 if VWAP not available
        if "EMA_9" in df.columns and "EMA_21" in df.columns:
            e9 = float(last["EMA_9"])
            e21 = float(last["EMA_21"])
            vwap_signal = 1 if e9 > e21 else -1
            vwap_detail = f"EMA9 {'>' if e9 > e21 else '<'} EMA21"

    # 3. RSI (7) — momentum
    rsi_signal = 0
    rsi_val = 50
    rsi_detail = "N/A"
    if "RSI_7" in df.columns and not pd.isna(last["RSI_7"]):
        rsi_val = float(last["RSI_7"])
        if rsi_val > 55:
            rsi_signal = 1
            rsi_detail = f"BULLISH ({rsi_val:.1f})"
        elif rsi_val < 45:
            rsi_signal = -1
            rsi_detail = f"BEARISH ({rsi_val:.1f})"
        else:
            rsi_signal = 0
            rsi_detail = f"NEUTRAL ({rsi_val:.1f})"

    # ═══════════════════ SIGNAL LOGIC ════════════════════════
    # Count agreement: +1 for bullish, -1 for bearish
    signals = [st_signal, vwap_signal, rsi_signal]
    bullish_count = sum(1 for s in signals if s > 0)
    bearish_count = sum(1 for s in signals if s < 0)

    # Need at least 2 out of 3 to agree
    if bullish_count >= 2:
        direction = "BULLISH"
        action = "BUY CE"
        score = bullish_count / 3.0
    elif bearish_count >= 2:
        direction = "BEARISH"
        action = "BUY PE"
        score = bearish_count / 3.0
    else:
        return _no_quick_signal(
            f"Indicators split — ST:{st_detail}, VWAP:{vwap_detail}, RSI:{rsi_detail}"
        )

    # Boost confidence if all 3 agree or if Supertrend just crossed
    confidence = 55 + (score * 20)
    if bullish_count == 3 or bearish_count == 3:
        confidence = 85
    if st_fresh_cross:
        confidence = min(confidence + 15, 95)

    # ═══════════════════ TRADE DETAILS ═══════════════════════
    if "BANK" in symbol.upper():
        lot_size = BANKNIFTY_LOT_SIZE
        strike_step = STRIKE_STEP_BANKNIFTY
    else:
        lot_size = NIFTY_LOT_SIZE
        strike_step = STRIKE_STEP_NIFTY

    atm_strike = get_atm_strike(price, strike_step)
    atr = float(last["ATR_7"]) if "ATR_7" in df.columns and not pd.isna(last["ATR_7"]) else price * 0.005

    # Quick scalp targets — smaller than main system
    if direction == "BULLISH":
        strike = atm_strike
        sl_underlying = round(price - atr * 1.0, 2)
        target1 = round(price + atr * 0.8, 2)
        target2 = round(price + atr * 1.5, 2)
    else:
        strike = atm_strike
        sl_underlying = round(price + atr * 1.0, 2)
        target1 = round(price - atr * 0.8, 2)
        target2 = round(price - atr * 1.5, 2)

    # Estimate premium — scalping uses cheaper OTM options too
    est_premium = max(atr * 0.5 * 2.0, 30)  # Cheaper options for scalp
    sl_premium = round(est_premium * 0.70, 1)     # 30% SL
    t1_premium = round(est_premium * 1.40, 1)     # 40% profit
    t2_premium = round(est_premium * 1.80, 1)     # 80% profit

    # Position sizing for small capital
    cost_per_lot = est_premium * lot_size
    max_lots = max(1, int((capital * 0.80) / cost_per_lot)) if cost_per_lot > 0 else 1
    risk_per_lot = (est_premium - sl_premium) * lot_size

    # ═══════════════════ RSI EXTREME WARNINGS ════════════════
    warnings = []
    if rsi_val > 75:
        warnings.append("⚠️ RSI > 75 — overbought, reversal risk HIGH. Quick exit!")
    elif rsi_val < 25:
        warnings.append("⚠️ RSI < 25 — oversold, bounce likely. Quick exit!")
    if st_fresh_cross:
        warnings.append("★ FRESH Supertrend crossover — strongest signal!")

    return {
        "has_signal": True,
        "action": action,
        "direction": direction,
        "strike": strike,
        "confidence": round(confidence, 0),
        "score": round(score, 2),

        # Indicators
        "supertrend": {"signal": st_signal, "detail": st_detail, "fresh": st_fresh_cross},
        "vwap": {"signal": vwap_signal, "detail": vwap_detail},
        "rsi": {"signal": rsi_signal, "detail": rsi_detail, "value": round(rsi_val, 1)},
        "agreement": f"{max(bullish_count, bearish_count)}/3",

        # Trade details
        "entry_premium": round(est_premium, 1),
        "sl_premium": sl_premium,
        "target1_premium": t1_premium,
        "target2_premium": t2_premium,
        "sl_underlying": sl_underlying,
        "target1_underlying": target1,
        "target2_underlying": target2,
        "lot_size": lot_size,
        "max_lots": max_lots,
        "risk_per_lot": round(risk_per_lot, 0),
        "total_investment": round(est_premium * lot_size * max_lots, 0),

        "current_price": round(price, 2),
        "atr": round(atr, 2),
        "warnings": warnings,
        "hold_time": "10-30 minutes",
        "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
    }


def _no_quick_signal(reason: str) -> Dict:
    return {
        "has_signal": False,
        "action": "NO SIGNAL",
        "direction": "NEUTRAL",
        "reason": reason,
        "strike": 0, "confidence": 0, "score": 0,
        "supertrend": {"signal": 0, "detail": "N/A", "fresh": False},
        "vwap": {"signal": 0, "detail": "N/A"},
        "rsi": {"signal": 0, "detail": "N/A", "value": 50},
        "agreement": "0/3",
        "entry_premium": 0, "sl_premium": 0,
        "target1_premium": 0, "target2_premium": 0,
        "sl_underlying": 0, "target1_underlying": 0, "target2_underlying": 0,
        "lot_size": 0, "max_lots": 0, "risk_per_lot": 0, "total_investment": 0,
        "current_price": 0, "atr": 0, "warnings": [],
        "hold_time": "", "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
    }


def get_quick_signal_history(df: pd.DataFrame, symbol: str = "NIFTY50", lookback: int = 20) -> List[Dict]:
    """
    Scan last N candles for quick signals — shows recent signal history.
    Useful to see where signals WOULD have triggered.
    """
    if df is None or df.empty or len(df) < lookback + 10:
        return []

    history = []
    for i in range(max(10, len(df) - lookback), len(df)):
        sub_df = df.iloc[:i+1]
        sig = generate_quick_signal(sub_df, symbol)
        if sig["has_signal"]:
            sig["candle_time"] = str(sub_df.index[-1])
            history.append(sig)

    return history
