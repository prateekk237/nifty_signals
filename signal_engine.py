"""
╔══════════════════════════════════════════════════════════════╗
║  SIGNAL ENGINE v2.0 — Enhanced with Global + VIX scoring    ║
║                                                              ║
║  ══════════════════════════════════════════════════           ║
║   HOW THE OVERALL SIGNAL IS CALCULATED (EXPLAINED)           ║
║  ══════════════════════════════════════════════════           ║
║                                                              ║
║  The system uses CONFLUENCE SCORING — multiple independent   ║
║  signals are combined with weights to produce one score.     ║
║                                                              ║
║  STEP 1: Each indicator produces a signal from -1 to +1     ║
║    Supertrend(5,1.5) + RSI(7) combo → e.g., +0.8           ║
║    EMA 9/21 crossover              → e.g., +0.5            ║
║    MACD histogram                  → e.g., +0.4            ║
║    VWAP position                   → e.g., +0.3            ║
║    Bollinger Bands %B              → e.g., -0.2            ║
║    ADX trend strength              → e.g., +0.5            ║
║    PCR (Put-Call Ratio)            → e.g., +0.3            ║
║    OI Buildup analysis             → e.g., +0.6            ║
║    News sentiment (VADER)          → e.g., +0.1            ║
║    Global markets score            → e.g., +0.5  ← NEW    ║
║    India VIX analysis              → e.g., +0.2  ← NEW    ║
║                                                              ║
║  STEP 2: Multiply each by its WEIGHT                         ║
║    Supertrend+RSI: +0.8 × 18% = +0.144                     ║
║    Global cues:    +0.5 × 14% = +0.070                     ║
║    VIX analysis:   +0.2 × 12% = +0.024                     ║
║    ... and so on for all 11 components                       ║
║                                                              ║
║  STEP 3: Sum all weighted scores                             ║
║    Total = 0.144 + 0.070 + 0.024 + ... = 0.523             ║
║                                                              ║
║  STEP 4: Apply VIX adjustment (high VIX reduces confidence) ║
║    If VIX > 25: score × 0.7                                ║
║                                                              ║
║  STEP 5: Map to signal label                                 ║
║    > +0.65  → STRONG BUY                                    ║
║    > +0.45  → BUY                                           ║
║    -0.30 to +0.45 → NEUTRAL (no trade)                      ║
║    < -0.30  → SELL                                          ║
║    < -0.65  → STRONG SELL                                   ║
║                                                              ║
║  STEP 6: Generate trade recommendation                       ║
║    BUY CE at strike X, entry ₹Y, SL ₹Z, target ₹T          ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pytz

from config import (
    STRATEGY_WEIGHTS, STRONG_BUY_THRESHOLD, BUY_THRESHOLD,
    NEUTRAL_LOW, STRONG_SELL_THRESHOLD,
    NIFTY_LOT_SIZE, BANKNIFTY_LOT_SIZE,
    STRIKE_STEP_NIFTY, STRIKE_STEP_BANKNIFTY,
    PCR_EXTREME_BULLISH, PCR_BULLISH, PCR_BEARISH, PCR_EXTREME_BEARISH,
    VIX_LOW, VIX_NORMAL_HIGH, VIX_HIGH,
    SL_OPTION_BUY_PCT, SL_ATR_MULTIPLIER, RISK_REWARD_MIN,
    MAX_RISK_PER_TRADE_PCT, DEFAULT_CAPITAL,
)
from indicators import get_indicator_signals, calc_cpr, calc_orb_levels
from data_fetcher import get_atm_strike, get_previous_day_ohlc, is_market_open, get_market_session

IST = pytz.timezone("Asia/Kolkata")


def calculate_confluence_score(
    indicator_signals: Dict[str, dict],
    pcr_data: Optional[dict] = None,
    oi_bias: str = "NEUTRAL",
    news_score: float = 0.0,
    vix_level: float = 0.0,
    global_score: float = 0.0,
    vix_signal_score: float = 0.0,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Calculate weighted confluence score from ALL signal sources.
    Now includes global markets and VIX analysis.

    Returns:
        (score, signal_label, component_scores)
    """
    components = {}
    total_weight = 0

    # ── Technical Indicators ─────────────────────────────────
    st_fast = indicator_signals.get("supertrend_fast", {}).get("signal", 0)
    rsi_7 = indicator_signals.get("rsi_7", {}).get("signal", 0)
    if st_fast != 0 and rsi_7 != 0:
        if (st_fast > 0 and rsi_7 > 0) or (st_fast < 0 and rsi_7 < 0):
            combo = (st_fast + rsi_7) / 2 * 1.3
        else:
            combo = (st_fast + rsi_7) / 2 * 0.5
        components["supertrend_rsi"] = np.clip(combo, -1, 1)
    elif st_fast != 0:
        components["supertrend_rsi"] = st_fast * 0.6

    ema_sig = indicator_signals.get("ema_cross", {}).get("signal", 0)
    if ema_sig: components["ema_crossover"] = ema_sig

    macd_sig = indicator_signals.get("macd", {}).get("signal", 0)
    if macd_sig: components["macd"] = macd_sig

    vwap_sig = indicator_signals.get("vwap", {}).get("signal", 0)
    if vwap_sig: components["vwap"] = vwap_sig

    bb_sig = indicator_signals.get("bollinger", {}).get("signal", 0)
    if bb_sig: components["bollinger"] = bb_sig

    adx_sig = indicator_signals.get("adx", {}).get("signal", 0)
    if adx_sig: components["adx_trend"] = adx_sig

    # ── PCR / OI ─────────────────────────────────────────────
    if pcr_data and pcr_data.get("pcr_oi", 0) > 0:
        pcr = pcr_data["pcr_oi"]
        if pcr >= PCR_EXTREME_BULLISH: pcr_sig = 0.7
        elif pcr >= PCR_BULLISH: pcr_sig = 0.3
        elif pcr <= PCR_EXTREME_BEARISH: pcr_sig = -0.7
        elif pcr <= PCR_BEARISH: pcr_sig = -0.3
        else: pcr_sig = 0.0
        components["pcr_sentiment"] = pcr_sig

    oi_map = {"BULLISH": 0.6, "BEARISH": -0.6, "NEUTRAL": 0.0}
    components["oi_analysis"] = oi_map.get(oi_bias, 0.0)

    # ── News Sentiment ───────────────────────────────────────
    components["news_sentiment"] = np.clip(news_score, -1, 1)

    # ── NEW: Global Market Score ─────────────────────────────
    components["global_cues"] = np.clip(global_score, -1, 1)

    # ── NEW: VIX Analysis Score ──────────────────────────────
    components["vix_analysis"] = np.clip(vix_signal_score, -1, 1)

    # ── Calculate weighted score ─────────────────────────────
    score = 0
    for key, weight in STRATEGY_WEIGHTS.items():
        comp_val = components.get(key, 0)
        score += comp_val * weight
        total_weight += weight

    if total_weight > 0:
        score = score / total_weight
    score = np.clip(score, -1, 1)

    # VIX panic adjustment
    if vix_level > VIX_HIGH:
        score *= 0.7
    elif vix_level > VIX_NORMAL_HIGH:
        score *= 0.85

    # Signal label
    if score >= STRONG_BUY_THRESHOLD: label = "STRONG BUY"
    elif score >= BUY_THRESHOLD: label = "BUY"
    elif score <= STRONG_SELL_THRESHOLD: label = "STRONG SELL"
    elif score <= NEUTRAL_LOW: label = "SELL"
    else: label = "NEUTRAL"

    return round(score, 3), label, components


def generate_trade_recommendation(
    symbol: str, current_price: float, confluence_score: float,
    signal_label: str, df: pd.DataFrame,
    oc_df: pd.DataFrame = None, vix_level: float = 15.0,
    capital: float = DEFAULT_CAPITAL, timeframe: str = "Intraday",
) -> Dict:
    """Generate complete actionable trade recommendation."""
    if signal_label == "NEUTRAL" or abs(confluence_score) < 0.15:
        return _no_trade("Insufficient confluence — signals are mixed")

    if "BANK" in symbol.upper():
        lot_size = BANKNIFTY_LOT_SIZE; strike_step = STRIKE_STEP_BANKNIFTY
    else:
        lot_size = NIFTY_LOT_SIZE; strike_step = STRIKE_STEP_NIFTY

    atm_strike = get_atm_strike(current_price, strike_step)
    is_bullish = confluence_score > 0
    confidence = min(abs(confluence_score) * 100 / 0.65 * 100, 98)

    strategy_type = "BUY"
    if vix_level > VIX_HIGH and abs(confluence_score) < STRONG_BUY_THRESHOLD:
        strategy_type = "SELL"

    atr = float(df["ATR_14"].iloc[-1]) if "ATR_14" in df.columns and not pd.isna(df["ATR_14"].iloc[-1]) else current_price * 0.01

    if is_bullish:
        action = f"{strategy_type} CE"
        strike = atm_strike
        otm_strike = atm_strike + strike_step
        sl_underlying = current_price - atr * SL_ATR_MULTIPLIER
        target1_underlying = current_price + atr * 1.5
        target2_underlying = current_price + atr * 2.5
    else:
        action = f"{strategy_type} PE"
        strike = atm_strike
        otm_strike = atm_strike - strike_step
        sl_underlying = current_price + atr * SL_ATR_MULTIPLIER
        target1_underlying = current_price - atr * 1.5
        target2_underlying = current_price - atr * 2.5

    est_delta = 0.50 if strike == atm_strike else 0.35
    est_premium = max(atr * est_delta * 2.5, 50)

    actual_premium = None
    if oc_df is not None and not oc_df.empty:
        nearest = oc_df[oc_df["strike"] == strike]
        if not nearest.empty:
            row = nearest.iloc[0]
            col = "ce_ltp" if is_bullish else "pe_ltp"
            if row[col] > 0: actual_premium = float(row[col])

    entry = actual_premium if actual_premium else round(est_premium, 1)

    if strategy_type == "BUY":
        sl_prem = round(entry * (1 - SL_OPTION_BUY_PCT / 100), 1)
        t1_prem = round(entry * 1.5, 1)
        t2_prem = round(entry * 2.0, 1)
        risk_per_lot = (entry - sl_prem) * lot_size
    else:
        sl_prem = round(entry * 1.5, 1)
        t1_prem = round(entry * 0.5, 1)
        t2_prem = round(entry * 0.25, 1)
        risk_per_lot = (sl_prem - entry) * lot_size

    max_risk = capital * (MAX_RISK_PER_TRADE_PCT / 100)
    max_lots = max(1, int(max_risk / risk_per_lot)) if risk_per_lot > 0 else 1

    # ── Capital affordability cap ────────────────────────────
    # Ensure total investment doesn't exceed 80% of capital
    cost_per_lot = entry * lot_size
    if cost_per_lot > 0:
        affordable_lots = max(1, int((capital * 0.80) / cost_per_lot))
        max_lots = min(max_lots, affordable_lots)

    # Warn if capital is too low for even 1 lot
    capital_warning = ""
    if cost_per_lot > capital:
        capital_warning = (f"⚠️ Capital ₹{capital:,} is below 1-lot cost ₹{cost_per_lot:,.0f}. "
                          f"Consider increasing capital or trading BankNifty (smaller lots).")

    reasons = []
    if abs(confluence_score) >= STRONG_BUY_THRESHOLD:
        reasons.append(f"Strong confluence ({confluence_score:+.2f})")
    else:
        reasons.append(f"Moderate confluence ({confluence_score:+.2f})")
    reasons.append(f"{'Bullish' if is_bullish else 'Bearish'} across multiple indicators")
    if vix_level < VIX_LOW:
        reasons.append(f"VIX low ({vix_level:.1f}) — premiums cheap")
    elif vix_level > VIX_HIGH:
        reasons.append(f"VIX elevated ({vix_level:.1f})")

    tf_advice = {
        "Scalping": "Quick scalp: Exit within 15-30 mins",
        "Intraday": "Must exit before 3:15 PM. Trail SL after T1",
        "Swing": "Hold 2-5 days. Daily close below SL = exit",
        "Positional": "Hold till weekly expiry. Weekly close for SL",
    }
    reasons.append(tf_advice.get(timeframe, ""))

    return {
        "action": action, "direction": "BULLISH" if is_bullish else "BEARISH",
        "strike": strike, "otm_strike": otm_strike,
        "entry_premium": entry, "sl_premium": sl_prem,
        "target1_premium": t1_prem, "target2_premium": t2_prem,
        "sl_underlying": round(sl_underlying, 2),
        "target1_underlying": round(target1_underlying, 2),
        "target2_underlying": round(target2_underlying, 2),
        "lot_size": lot_size, "max_lots": max_lots,
        "risk_per_lot": round(risk_per_lot, 0),
        "total_risk": round(risk_per_lot * max_lots, 0),
        "total_investment": round(entry * lot_size * max_lots, 0),
        "potential_profit_t1": round(abs(t1_prem - entry) * lot_size * max_lots, 0),
        "potential_profit_t2": round(abs(t2_prem - entry) * lot_size * max_lots, 0),
        "risk_reward": round(abs(t1_prem - entry) / max(abs(entry - sl_prem), 0.01), 2),
        "confidence": round(confidence, 0),
        "strategy_type": strategy_type, "reasoning": reasons,
        "atr": round(atr, 2), "timeframe": timeframe,
        "capital_warning": capital_warning,
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }


def _no_trade(reason: str) -> Dict:
    return {
        "action": "NO TRADE", "direction": "NEUTRAL",
        "strike": 0, "otm_strike": 0,
        "entry_premium": 0, "sl_premium": 0,
        "target1_premium": 0, "target2_premium": 0,
        "sl_underlying": 0, "target1_underlying": 0, "target2_underlying": 0,
        "lot_size": 0, "max_lots": 0, "risk_per_lot": 0,
        "total_risk": 0, "total_investment": 0,
        "potential_profit_t1": 0, "potential_profit_t2": 0,
        "risk_reward": 0, "confidence": 0, "strategy_type": "NONE",
        "reasoning": [reason], "atr": 0, "timeframe": "",
        "capital_warning": "",
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }


def select_best_strategy(vix, pcr, adx, timeframe, is_expiry_day=False):
    strategies = []
    if is_expiry_day:
        strategies.append({"name": "Expiry Credit Spread", "type": "SELL", "win_rate": "65-70%",
                           "detail": "Sell OTM spreads; theta accelerates after 12 PM"})
    if vix < VIX_LOW:
        strategies.append({"name": "Directional Option Buying", "type": "BUY", "win_rate": "~65%",
                           "detail": "Low VIX = cheap premiums. Buy ATM with trend"})
    elif vix > VIX_HIGH:
        strategies.append({"name": "Short Strangle (hedged)", "type": "SELL", "win_rate": "68%",
                           "detail": "High VIX = rich premiums. Sell OTM CE+PE"})
    if timeframe == "Scalping":
        strategies.append({"name": "VWAP Bounce Scalp", "type": "BUY", "win_rate": "~78%",
                           "detail": "Buy CE/PE on VWAP bounce with volume"})
    elif timeframe == "Intraday":
        strategies.append({"name": "ORB + Supertrend", "type": "BUY", "win_rate": "~71%",
                           "detail": "ORB breakout confirmed by Supertrend"})
    elif timeframe == "Swing":
        strategies.append({"name": "EMA 9/21 + MACD", "type": "BUY", "win_rate": "~73%",
                           "detail": "Enter on EMA cross, confirm with MACD"})
    elif timeframe == "Positional":
        strategies.append({"name": "Triple Supertrend + 200 EMA", "type": "BUY", "win_rate": "~60%",
                           "detail": "All 3 Supertrends align + above 200 EMA"})
    if not strategies:
        strategies.append({"name": "Wait for Setup", "type": "NONE", "win_rate": "N/A",
                           "detail": "No clear edge"})
    return strategies
