"""
╔══════════════════════════════════════════════════════════════╗
║  QUICK SIGNALS v2 — Fixed: RSI blocking, real prices,       ║
║  sideways detection, proper BUY PE                          ║
║                                                              ║
║  FIXES FROM v1:                                              ║
║  1. RSI > 75 now BLOCKS BUY CE (not just warns)            ║
║  2. RSI < 25 now BLOCKS BUY PE                              ║
║  3. Uses real option chain LTP for entry price              ║
║  4. ADX < 18 = SIDEWAYS → NO TRADE                         ║
║  5. Proper BUY PE when indicators bearish                   ║
║  6. VWAP NaN handled with EMA fallback                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
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
    oc_df: pd.DataFrame = None,
    oc_expiry: str = None,
) -> Dict:
    """
    Generate fast scalping signal with real price validation.

    NEW: Pass oc_df (option chain) for real LTP prices.
    NEW: RSI extremes BLOCK signals, not just warn.
    NEW: ADX-based sideways detection.
    """
    if df is None or df.empty or len(df) < 10:
        return _no_quick_signal("Not enough data")

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])

    # ═══════════════════ SIDEWAYS CHECK (ADX) ════════════════
    adx_val = 0
    if "ADX" in df.columns and not pd.isna(last["ADX"]):
        adx_val = float(last["ADX"])

    is_sideways = adx_val < 18 and adx_val > 0
    if is_sideways:
        return _no_quick_signal(
            f"SIDEWAYS MARKET — ADX at {adx_val:.1f} (below 18). "
            f"No clear trend. Avoid directional trades. "
            f"Consider option selling strategies instead."
        )

    # ═══════════════════ 3 FAST INDICATORS ═══════════════════

    # 1. SUPERTREND (5, 1.5)
    st_col = "STd_5_1.5"
    st_signal = 0
    st_fresh_cross = False
    st_detail = "N/A"
    if st_col in df.columns:
        curr_dir = int(last[st_col])
        prev_dir = int(prev[st_col])
        st_signal = curr_dir
        st_fresh_cross = (curr_dir != prev_dir)
        st_detail = f"{'BULLISH' if curr_dir > 0 else 'BEARISH'}" + (" ★FRESH" if st_fresh_cross else "")

    # 2. VWAP — with NaN protection + EMA fallback
    vwap_signal = 0
    vwap_detail = "N/A"
    vwap_used = False

    if "VWAP" in df.columns:
        vwap_raw = last["VWAP"]
        if not pd.isna(vwap_raw) and float(vwap_raw) > 0:
            vwap = float(vwap_raw)
            pct = ((price - vwap) / vwap) * 100
            if not np.isnan(pct) and not np.isinf(pct):
                vwap_signal = 1 if price > vwap else -1
                vwap_detail = f"{'ABOVE' if price > vwap else 'BELOW'} VWAP ({pct:+.2f}%)"
                vwap_used = True

    # EMA fallback if VWAP failed
    if not vwap_used and "EMA_9" in df.columns and "EMA_21" in df.columns:
        e9 = float(last["EMA_9"]) if not pd.isna(last["EMA_9"]) else 0
        e21 = float(last["EMA_21"]) if not pd.isna(last["EMA_21"]) else 0
        if e9 > 0 and e21 > 0:
            vwap_signal = 1 if e9 > e21 else -1
            vwap_detail = f"EMA9{'>' if e9 > e21 else '<'}EMA21 (VWAP N/A)"

    # 3. RSI (7) — with EXTREME BLOCKING
    rsi_signal = 0
    rsi_val = 50
    rsi_detail = "N/A"
    rsi_extreme = None  # "OVERBOUGHT" or "OVERSOLD" or None

    if "RSI_7" in df.columns and not pd.isna(last["RSI_7"]):
        rsi_val = float(last["RSI_7"])

        if rsi_val > 75:
            # OVERBOUGHT — DO NOT BUY CE, favor BUY PE
            rsi_signal = -1  # Treat as bearish (reversal likely)
            rsi_detail = f"OVERBOUGHT ({rsi_val:.1f}) ⚠️"
            rsi_extreme = "OVERBOUGHT"
        elif rsi_val < 25:
            # OVERSOLD — DO NOT BUY PE, favor BUY CE
            rsi_signal = 1  # Treat as bullish (bounce likely)
            rsi_detail = f"OVERSOLD ({rsi_val:.1f}) ⚠️"
            rsi_extreme = "OVERSOLD"
        elif rsi_val > 55:
            rsi_signal = 1
            rsi_detail = f"BULLISH ({rsi_val:.1f})"
        elif rsi_val < 45:
            rsi_signal = -1
            rsi_detail = f"BEARISH ({rsi_val:.1f})"
        else:
            rsi_signal = 0
            rsi_detail = f"NEUTRAL ({rsi_val:.1f})"

    # ═══════════════════ SIGNAL LOGIC ════════════════════════
    signals = [st_signal, vwap_signal, rsi_signal]
    bullish_count = sum(1 for s in signals if s > 0)
    bearish_count = sum(1 for s in signals if s < 0)

    # Need at least 2 out of 3
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

    # ═══════════════════ RSI EXTREME BLOCKING ════════════════
    # RSI > 75 but signal says BUY CE → BLOCK (reversal risk)
    # RSI < 25 but signal says BUY PE → BLOCK (bounce risk)
    warnings = []
    if rsi_extreme == "OVERBOUGHT" and action == "BUY CE":
        return _no_quick_signal(
            f"BLOCKED: RSI at {rsi_val:.1f} is OVERBOUGHT. "
            f"BUY CE blocked — reversal risk too high. "
            f"Wait for RSI to drop below 65 or look for BUY PE."
        )
    if rsi_extreme == "OVERSOLD" and action == "BUY PE":
        return _no_quick_signal(
            f"BLOCKED: RSI at {rsi_val:.1f} is OVERSOLD. "
            f"BUY PE blocked — bounce risk too high. "
            f"Wait for RSI to rise above 35 or look for BUY CE."
        )

    # RSI warnings (not blocking but caution)
    if rsi_val > 70:
        warnings.append(f"⚠️ RSI at {rsi_val:.1f} — nearing overbought. Book profits quickly!")
    elif rsi_val < 30:
        warnings.append(f"⚠️ RSI at {rsi_val:.1f} — nearing oversold. Book profits quickly!")
    if st_fresh_cross:
        warnings.append("★ FRESH Supertrend crossover — strongest signal!")

    # Confidence
    confidence = 55 + (score * 20)
    if bullish_count == 3 or bearish_count == 3:
        confidence = 85
    if st_fresh_cross:
        confidence = min(confidence + 15, 95)
    if adx_val > 25:
        confidence = min(confidence + 5, 95)  # Strong trend boost

    # ═══════════════════ TRADE DETAILS ═══════════════════════
    if "BANK" in symbol.upper():
        lot_size = BANKNIFTY_LOT_SIZE
        strike_step = STRIKE_STEP_BANKNIFTY
    else:
        lot_size = NIFTY_LOT_SIZE
        strike_step = STRIKE_STEP_NIFTY

    atm_strike = get_atm_strike(price, strike_step)
    atr = float(last["ATR_7"]) if "ATR_7" in df.columns and not pd.isna(last["ATR_7"]) else price * 0.005

    # Strike selection
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

    # ═══════════════════ REAL PREMIUM FROM OPTION CHAIN ══════
    real_premium = None
    if oc_df is not None and not oc_df.empty:
        # Filter by expiry if available
        chain = oc_df.copy()
        if oc_expiry:
            chain = chain[chain["expiry"] == oc_expiry]

        strike_data = chain[chain["strike"] == strike]
        if not strike_data.empty:
            row = strike_data.iloc[0]
            if direction == "BULLISH" and row.get("ce_ltp", 0) > 0:
                real_premium = float(row["ce_ltp"])
            elif direction == "BEARISH" and row.get("pe_ltp", 0) > 0:
                real_premium = float(row["pe_ltp"])

    # Use real premium or better estimate
    if real_premium and real_premium > 5:
        entry = round(real_premium, 1)
        price_source = "LIVE"
    else:
        # Better estimate: delta × ATR × time_value
        # ATM delta ≈ 0.5, typical ATM premium ≈ 0.4-0.6% of spot
        entry = round(max(price * 0.004, 50), 1)  # ~0.4% of spot, min ₹50
        price_source = "EST"

    sl_premium = round(entry * 0.70, 1)      # 30% SL
    t1_premium = round(entry * 1.40, 1)      # 40% profit
    t2_premium = round(entry * 1.80, 1)      # 80% profit

    # Position sizing
    cost_per_lot = entry * lot_size
    max_lots = max(1, int((capital * 0.80) / cost_per_lot)) if cost_per_lot > 0 else 1
    risk_per_lot = (entry - sl_premium) * lot_size

    return {
        "has_signal": True,
        "action": action,
        "direction": direction,
        "strike": strike,
        "confidence": round(confidence, 0),
        "score": round(score, 2),

        "supertrend": {"signal": st_signal, "detail": st_detail, "fresh": st_fresh_cross},
        "vwap": {"signal": vwap_signal, "detail": vwap_detail},
        "rsi": {"signal": rsi_signal, "detail": rsi_detail, "value": round(rsi_val, 1)},
        "adx": round(adx_val, 1),
        "agreement": f"{max(bullish_count, bearish_count)}/3",

        "entry_premium": entry,
        "sl_premium": sl_premium,
        "target1_premium": t1_premium,
        "target2_premium": t2_premium,
        "sl_underlying": sl_underlying,
        "target1_underlying": target1,
        "target2_underlying": target2,
        "lot_size": lot_size,
        "max_lots": max_lots,
        "risk_per_lot": round(risk_per_lot, 0),
        "total_investment": round(entry * lot_size * max_lots, 0),
        "price_source": price_source,

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
        "adx": 0, "agreement": "0/3",
        "entry_premium": 0, "sl_premium": 0,
        "target1_premium": 0, "target2_premium": 0,
        "sl_underlying": 0, "target1_underlying": 0, "target2_underlying": 0,
        "lot_size": 0, "max_lots": 0, "risk_per_lot": 0, "total_investment": 0,
        "price_source": "", "current_price": 0, "atr": 0, "warnings": [],
        "hold_time": "", "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
    }
