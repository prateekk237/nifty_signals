"""
╔══════════════════════════════════════════════════════════════╗
║  BTST PREDICTOR v2 — Fixed scoring (less dampening)         ║
║                                                              ║
║  v1 BUG: Each factor divided by 2.0 then multiplied by      ║
║  weight (0.10-0.25), so even strong signals produced tiny    ║
║  scores. 3 bullish factors → only +0.19 → FLAT OPENING.    ║
║                                                              ║
║  v2 FIX: Normalize by 1.0 (not 2.0), lower thresholds      ║
║  from 0.45 to 0.25. Now 3 bullish factors → ~0.35 → GAP UP ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import pytz

from config import BTST_WEIGHTS

IST = pytz.timezone("Asia/Kolkata")

# Lowered thresholds — v1 was 0.45/0.70, never triggered
GAP_STRONG = 0.35
GAP_MODERATE = 0.15


def predict_next_day_gap(
    us_futures_data: Dict = None,
    asian_data: Dict = None,
    fii_net_flow: float = 0.0,
    vix_current: float = 0.0,
    vix_prev_close: float = 0.0,
    df_today: pd.DataFrame = None,
    pcr_eod: float = 0.0,
    indicator_signals: Dict = None,
) -> Dict:
    """Predict next-day gap with fixed scoring."""
    factors = {}
    total_score = 0.0

    # ── 1. US FUTURES (25%) ──────────────────────────────────
    us_score = 0.0
    if us_futures_data:
        changes = [d.get("change_pct", 0) for k, d in us_futures_data.items()
                   if k in ["SP500_FUT", "DOW_FUT", "NASDAQ_FUT"]]
        if changes:
            avg = np.mean(changes)
            us_score = np.clip(avg / 1.0, -1, 1)  # FIX: was /2.0, now /1.0
            factors["us_futures"] = {
                "score": round(us_score, 3), "raw": round(avg, 2),
                "detail": f"US Futures avg: {avg:+.2f}%",
                "impact": "BULLISH" if us_score > 0 else "BEARISH",
            }
    total_score += us_score * BTST_WEIGHTS["us_futures"]

    # ── 2. ASIAN CLOSE (15%) ─────────────────────────────────
    asian_score = 0.0
    if asian_data:
        changes = [d.get("change_pct", 0) for k, d in asian_data.items()
                   if k in ["NIKKEI", "HANGSENG", "SHANGHAI", "STRAITS"]]
        if changes:
            avg = np.mean(changes)
            asian_score = np.clip(avg / 1.5, -1, 1)  # FIX: was /2.0
            factors["asian_close"] = {
                "score": round(asian_score, 3), "raw": round(avg, 2),
                "detail": f"Asian markets avg: {avg:+.2f}%",
                "impact": "BULLISH" if asian_score > 0 else "BEARISH",
            }
    total_score += asian_score * BTST_WEIGHTS["asian_close"]

    # ── 3. FII/DII FLOW (15%) ───────────────────────────────
    fii_score = 0.0
    if fii_net_flow != 0:
        fii_score = np.clip(fii_net_flow / 1500, -1, 1)  # FIX: was /2000
        factors["fii_dii_flow"] = {
            "score": round(fii_score, 3), "raw": round(fii_net_flow, 0),
            "detail": f"FII net: ₹{fii_net_flow:+,.0f} Cr",
            "impact": "BULLISH" if fii_score > 0 else "BEARISH",
        }
    total_score += fii_score * BTST_WEIGHTS["fii_dii_flow"]

    # ── 4. VIX TREND (10%) ──────────────────────────────────
    vix_score = 0.0
    if vix_current > 0 and vix_prev_close > 0:
        vix_change = ((vix_current - vix_prev_close) / vix_prev_close) * 100
        vix_score = np.clip(-vix_change / 3.0, -1, 1)  # FIX: was /5.0
        factors["vix_trend"] = {
            "score": round(vix_score, 3), "raw": round(vix_change, 2),
            "detail": f"India VIX: {vix_current:.2f} ({vix_change:+.2f}%)",
            "impact": "BULLISH" if vix_score > 0 else "BEARISH",
        }
    total_score += vix_score * BTST_WEIGHTS["vix_trend"]

    # ── 5. TECHNICAL TREND (15%) ─────────────────────────────
    tech_score = 0.0
    if indicator_signals:
        st_sig = indicator_signals.get("supertrend_fast", {}).get("signal", 0)
        ema_sig = indicator_signals.get("ema_cross", {}).get("signal", 0)
        macd_sig = indicator_signals.get("macd", {}).get("signal", 0)
        tech_score = np.clip((st_sig + ema_sig + macd_sig) / 2.0, -1, 1)  # FIX: was /3.0
        factors["technical_trend"] = {
            "score": round(tech_score, 3), "raw": round(tech_score, 3),
            "detail": f"ST: {st_sig:+.2f} | EMA: {ema_sig:+.2f} | MACD: {macd_sig:+.2f}",
            "impact": "BULLISH" if tech_score > 0 else "BEARISH",
        }
    total_score += tech_score * BTST_WEIGHTS["technical_trend"]

    # ── 6. CLOSING PATTERN (10%) ─────────────────────────────
    close_score = 0.0
    if df_today is not None and not df_today.empty:
        tail = df_today.tail(6)
        if len(tail) >= 2:
            lc = float(tail["Close"].iloc[-1])
            lh = float(tail["High"].max())
            ll = float(tail["Low"].min())
            rng = lh - ll
            if rng > 0:
                pos = (lc - ll) / rng
                close_score = np.clip((pos - 0.5) * 2.5, -1, 1)  # FIX: was *2.0
                if pos > 0.7: detail = "STRONG CLOSE near high"
                elif pos < 0.3: detail = "WEAK CLOSE near low"
                else: detail = "NEUTRAL CLOSE mid-range"
                factors["closing_pattern"] = {
                    "score": round(close_score, 3), "raw": round(pos, 2),
                    "detail": detail, "impact": "BULLISH" if close_score > 0 else "BEARISH",
                }
    total_score += close_score * BTST_WEIGHTS["closing_pattern"]

    # ── 7. PCR END-OF-DAY (10%) ──────────────────────────────
    pcr_score = 0.0
    if pcr_eod > 0:
        if pcr_eod > 1.2: pcr_score = 0.8    # FIX: was 0.7
        elif pcr_eod > 0.9: pcr_score = 0.4   # FIX: was 0.3
        elif pcr_eod < 0.5: pcr_score = -0.8
        elif pcr_eod < 0.7: pcr_score = -0.4
        factors["oi_pcr_eod"] = {
            "score": round(pcr_score, 3), "raw": round(pcr_eod, 3),
            "detail": f"EOD PCR: {pcr_eod:.3f}",
            "impact": "BULLISH" if pcr_score > 0 else "BEARISH",
        }
    total_score += pcr_score * BTST_WEIGHTS["oi_pcr_eod"]

    # ── MAJORITY DIRECTION BONUS ─────────────────────────────
    # When 3+ factors agree on direction, boost the score
    # This prevents a single strong bearish factor (e.g., US futures)
    # from overriding 3-4 bullish factors
    all_scores = [f["score"] for f in factors.values()]
    bullish = sum(1 for s in all_scores if s > 0.05)
    bearish = sum(1 for s in all_scores if s < -0.05)
    total_factors = len(all_scores)

    if total_factors >= 3:
        if bullish >= 3 and bullish > bearish:
            majority_bonus = 0.10 * (bullish / total_factors)
            total_score += majority_bonus
        elif bearish >= 3 and bearish > bullish:
            majority_bonus = -0.10 * (bearish / total_factors)
            total_score += majority_bonus

    # ── FINAL PREDICTION ─────────────────────────────────────
    final_score = np.clip(total_score, -1, 1)
    confidence = min(abs(final_score) / 0.35 * 100, 95)  # Scale to 0.35 not 0.5

    if final_score > GAP_STRONG:
        prediction = "STRONG GAP UP"; emoji = "🟢🟢"
    elif final_score > GAP_MODERATE:
        prediction = "GAP UP"; emoji = "🟢"
    elif final_score < -GAP_STRONG:
        prediction = "STRONG GAP DOWN"; emoji = "🔴🔴"
    elif final_score < -GAP_MODERATE:
        prediction = "GAP DOWN"; emoji = "🔴"
    else:
        prediction = "FLAT OPENING"; emoji = "⚪"

    # BTST Trade
    btst_trade = None
    if confidence >= 40 and prediction != "FLAT OPENING":
        is_bull = final_score > 0
        btst_trade = {
            "action": "BUY CE (BTST)" if is_bull else "BUY PE (BTST)",
            "entry_time": "3:15 - 3:25 PM today",
            "exit_time": "9:20 - 9:45 AM tomorrow",
            "strike": "ATM in tomorrow's direction",
            "sl": "30% of premium",
            "target": "50-100% profit on gap",
            "confidence": round(confidence, 0),
            "detail": (
                f"{'Buy ATM CE' if is_bull else 'Buy ATM PE'} at 3:20 PM. "
                f"Expected {'gap-up' if is_bull else 'gap-down'} tomorrow. "
                f"Sell in first 15-30 min. SL: 30% of premium."
            ),
        }

    bullish = sum(1 for f in factors.values() if f["score"] > 0.05)
    bearish = sum(1 for f in factors.values() if f["score"] < -0.05)

    return {
        "prediction": prediction, "emoji": emoji,
        "score": round(final_score, 3), "confidence": round(confidence, 1),
        "factors": factors,
        "bullish_count": bullish, "bearish_count": bearish,
        "btst_trade": btst_trade,
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "best_check_time": "3:00 PM - 3:30 PM IST",
    }
