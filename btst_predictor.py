"""
╔══════════════════════════════════════════════════════════════╗
║  BTST PREDICTOR — Gap Up / Gap Down Prediction Engine       ║
║                                                              ║
║  HOW IT WORKS (step by step):                                ║
║  ─────────────────────────────                               ║
║  This module predicts whether TOMORROW's market will open    ║
║  GAP UP or GAP DOWN based on 7 factors analyzed before       ║
║  today's market close (best checked 3:00-3:30 PM):           ║
║                                                              ║
║  1. US FUTURES (weight: 25%)                                 ║
║     → If S&P/Dow/Nasdaq futures are positive at 3 PM IST,   ║
║       US market likely to close positive tonight, and Nifty  ║
║       opens gap-up tomorrow with ~70% probability.           ║
║                                                              ║
║  2. ASIAN MARKET CLOSE (weight: 15%)                         ║
║     → Nikkei, Hang Seng closing direction correlates with    ║
║       next-day Indian market sentiment.                      ║
║                                                              ║
║  3. FII/DII FLOW (weight: 15%)                               ║
║     → Net FII buying today = gap-up probability increases.   ║
║       Net FII selling = gap-down probability increases.      ║
║                                                              ║
║  4. VIX TREND (weight: 10%)                                  ║
║     → VIX falling through the day = market confident = gap-up║
║       VIX rising through the day = fear building = gap-down  ║
║                                                              ║
║  5. TECHNICAL TREND AT CLOSE (weight: 15%)                   ║
║     → Supertrend + EMA direction at 3:30 PM closing candle   ║
║       determines the trend continuation probability.         ║
║                                                              ║
║  6. CLOSING PATTERN (weight: 10%)                            ║
║     → Last 30 minutes behavior:                              ║
║       Strong close (close near high) = gap-up likely         ║
║       Weak close (close near low) = gap-down likely          ║
║                                                              ║
║  7. OI/PCR END-OF-DAY (weight: 10%)                         ║
║     → PCR > 1.0 at close = Put writers confident = bullish   ║
║       PCR < 0.7 at close = Call writers confident = bearish  ║
║                                                              ║
║  FINAL FORMULA:                                              ║
║  btst_score = Σ(factor_score × factor_weight)                ║
║  Normalized to [-1, +1]                                      ║
║  > +0.45 = GAP UP prediction                                 ║
║  < -0.45 = GAP DOWN prediction                               ║
║  Between = FLAT OPENING prediction                           ║
║                                                              ║
║  BTST TRADE: If gap-up predicted with >70% confidence,       ║
║  BUY CE at 3:20 PM today → SELL at 9:20-9:45 AM tomorrow    ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import pytz

from config import BTST_WEIGHTS, GAP_STRONG_CONFIDENCE, GAP_MODERATE_CONFIDENCE

IST = pytz.timezone("Asia/Kolkata")


def predict_next_day_gap(
    us_futures_data: Dict[str, dict] = None,
    asian_data: Dict[str, dict] = None,
    fii_net_flow: float = 0.0,
    vix_current: float = 0.0,
    vix_prev_close: float = 0.0,
    df_today: pd.DataFrame = None,
    pcr_eod: float = 0.0,
    indicator_signals: Dict = None,
) -> Dict:
    """
    Predict next-day gap-up/gap-down based on multiple factors.

    Call this between 3:00 PM - 3:30 PM IST for best accuracy.

    Returns dict with:
        prediction: "GAP UP" / "GAP DOWN" / "FLAT"
        confidence: 0-100%
        score: -1 to +1
        factor_breakdown: individual factor scores
        btst_recommendation: trade details if confident enough
    """
    factors = {}
    total_score = 0.0

    # ═══════════════════════════════════════════════════════════
    #  FACTOR 1: US FUTURES DIRECTION (Weight: 25%)
    # ═══════════════════════════════════════════════════════════
    us_score = 0.0
    if us_futures_data:
        us_changes = []
        for ticker, data in us_futures_data.items():
            if ticker in ["SP500_FUT", "DOW_FUT", "NASDAQ_FUT"]:
                us_changes.append(data.get("change_pct", 0))

        if us_changes:
            avg_us = np.mean(us_changes)
            # Normalize: +1% US futures ≈ +0.5 score
            us_score = np.clip(avg_us / 2.0, -1, 1)
            factors["us_futures"] = {
                "score": round(us_score, 3),
                "raw": round(avg_us, 2),
                "detail": f"US Futures avg: {avg_us:+.2f}%",
                "impact": "BULLISH" if us_score > 0 else "BEARISH",
            }

    weighted_us = us_score * BTST_WEIGHTS["us_futures"]
    total_score += weighted_us

    # ═══════════════════════════════════════════════════════════
    #  FACTOR 2: ASIAN MARKET CLOSE (Weight: 15%)
    # ═══════════════════════════════════════════════════════════
    asian_score = 0.0
    if asian_data:
        asian_changes = []
        for ticker, data in asian_data.items():
            if ticker in ["NIKKEI", "HANGSENG", "SHANGHAI", "STRAITS"]:
                asian_changes.append(data.get("change_pct", 0))

        if asian_changes:
            avg_asian = np.mean(asian_changes)
            asian_score = np.clip(avg_asian / 2.0, -1, 1)
            factors["asian_close"] = {
                "score": round(asian_score, 3),
                "raw": round(avg_asian, 2),
                "detail": f"Asian markets avg: {avg_asian:+.2f}%",
                "impact": "BULLISH" if asian_score > 0 else "BEARISH",
            }

    total_score += asian_score * BTST_WEIGHTS["asian_close"]

    # ═══════════════════════════════════════════════════════════
    #  FACTOR 3: FII/DII NET FLOW (Weight: 15%)
    # ═══════════════════════════════════════════════════════════
    fii_score = 0.0
    if fii_net_flow != 0:
        # FII net flow in crores; ₹500cr+ is significant
        fii_score = np.clip(fii_net_flow / 2000, -1, 1)
        factors["fii_dii_flow"] = {
            "score": round(fii_score, 3),
            "raw": round(fii_net_flow, 0),
            "detail": f"FII net flow: ₹{fii_net_flow:+,.0f} Cr",
            "impact": "BULLISH" if fii_score > 0 else "BEARISH",
        }

    total_score += fii_score * BTST_WEIGHTS["fii_dii_flow"]

    # ═══════════════════════════════════════════════════════════
    #  FACTOR 4: VIX TREND (Weight: 10%)
    # ═══════════════════════════════════════════════════════════
    vix_score = 0.0
    if vix_current > 0 and vix_prev_close > 0:
        vix_change = ((vix_current - vix_prev_close) / vix_prev_close) * 100
        # VIX falling = bullish for gap-up; VIX rising = bearish
        vix_score = np.clip(-vix_change / 5.0, -1, 1)  # 5% VIX change = max signal
        factors["vix_trend"] = {
            "score": round(vix_score, 3),
            "raw": round(vix_change, 2),
            "detail": f"India VIX: {vix_current:.2f} ({vix_change:+.2f}%)",
            "impact": "BULLISH" if vix_score > 0 else "BEARISH",
        }

    total_score += vix_score * BTST_WEIGHTS["vix_trend"]

    # ═══════════════════════════════════════════════════════════
    #  FACTOR 5: TECHNICAL TREND AT CLOSE (Weight: 15%)
    # ═══════════════════════════════════════════════════════════
    tech_score = 0.0
    if indicator_signals:
        # Use Supertrend + EMA direction
        st_sig = indicator_signals.get("supertrend_fast", {}).get("signal", 0)
        ema_sig = indicator_signals.get("ema_cross", {}).get("signal", 0)
        macd_sig = indicator_signals.get("macd", {}).get("signal", 0)

        tech_score = np.clip((st_sig + ema_sig + macd_sig) / 3, -1, 1)
        factors["technical_trend"] = {
            "score": round(tech_score, 3),
            "raw": round(tech_score, 3),
            "detail": f"ST: {st_sig:+.2f} | EMA: {ema_sig:+.2f} | MACD: {macd_sig:+.2f}",
            "impact": "BULLISH" if tech_score > 0 else "BEARISH",
        }

    total_score += tech_score * BTST_WEIGHTS["technical_trend"]

    # ═══════════════════════════════════════════════════════════
    #  FACTOR 6: CLOSING PATTERN (Weight: 10%)
    # ═══════════════════════════════════════════════════════════
    close_score = 0.0
    if df_today is not None and not df_today.empty:
        # Last 6 candles (30 min in 5-min TF or last few candles)
        tail = df_today.tail(6)
        if len(tail) >= 2:
            last_close = float(tail["Close"].iloc[-1])
            last_high = float(tail["High"].max())
            last_low = float(tail["Low"].min())
            last_range = last_high - last_low

            if last_range > 0:
                # Close position within the range: 1 = closed at high, 0 = at low
                close_position = (last_close - last_low) / last_range
                close_score = np.clip((close_position - 0.5) * 2, -1, 1)

                if close_position > 0.75:
                    detail = "STRONG CLOSE — closed near high of last 30 min"
                elif close_position < 0.25:
                    detail = "WEAK CLOSE — closed near low of last 30 min"
                else:
                    detail = "NEUTRAL CLOSE — closed in middle of range"

                factors["closing_pattern"] = {
                    "score": round(close_score, 3),
                    "raw": round(close_position, 2),
                    "detail": detail,
                    "impact": "BULLISH" if close_score > 0 else "BEARISH",
                }

    total_score += close_score * BTST_WEIGHTS["closing_pattern"]

    # ═══════════════════════════════════════════════════════════
    #  FACTOR 7: OI/PCR END-OF-DAY (Weight: 10%)
    # ═══════════════════════════════════════════════════════════
    pcr_score = 0.0
    if pcr_eod > 0:
        if pcr_eod > 1.2:
            pcr_score = 0.7  # Strong put writing = bullish
        elif pcr_eod > 0.9:
            pcr_score = 0.3
        elif pcr_eod < 0.5:
            pcr_score = -0.7  # Strong call writing = bearish
        elif pcr_eod < 0.7:
            pcr_score = -0.3
        else:
            pcr_score = 0.0

        factors["oi_pcr_eod"] = {
            "score": round(pcr_score, 3),
            "raw": round(pcr_eod, 3),
            "detail": f"EOD PCR: {pcr_eod:.3f}",
            "impact": "BULLISH" if pcr_score > 0 else "BEARISH",
        }

    total_score += pcr_score * BTST_WEIGHTS["oi_pcr_eod"]

    # ═══════════════════════════════════════════════════════════
    #  FINAL PREDICTION
    # ═══════════════════════════════════════════════════════════
    final_score = np.clip(total_score, -1, 1)
    abs_score = abs(final_score)

    # Confidence = how strong the signal is
    confidence = min(abs_score * 100 / 0.7, 95)

    if final_score > GAP_STRONG_CONFIDENCE:
        prediction = "STRONG GAP UP"
        emoji = "🟢🟢"
    elif final_score > GAP_MODERATE_CONFIDENCE:
        prediction = "GAP UP"
        emoji = "🟢"
    elif final_score < -GAP_STRONG_CONFIDENCE:
        prediction = "STRONG GAP DOWN"
        emoji = "🔴🔴"
    elif final_score < -GAP_MODERATE_CONFIDENCE:
        prediction = "GAP DOWN"
        emoji = "🔴"
    else:
        prediction = "FLAT OPENING"
        emoji = "⚪"

    # ═══════════════════════════════════════════════════════════
    #  BTST TRADE RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    btst_trade = None
    if confidence >= 55 and prediction != "FLAT OPENING":
        is_bullish = final_score > 0
        btst_trade = {
            "action": "BUY CE (BTST)" if is_bullish else "BUY PE (BTST)",
            "entry_time": "3:15 - 3:25 PM today",
            "exit_time": "9:20 - 9:45 AM tomorrow",
            "strike": "ATM or 1-OTM in tomorrow's direction",
            "sl": "30% of premium",
            "target": "50-100% profit on gap",
            "confidence": round(confidence, 0),
            "detail": (
                f"{'Buy ATM CE' if is_bullish else 'Buy ATM PE'} at 3:20 PM. "
                f"Tomorrow expected to open {'gap-up' if is_bullish else 'gap-down'}. "
                f"Sell in first 15-30 minutes of market open. "
                f"STRICT SL: Exit if premium drops 30% from entry."
            ),
        }

    # Count bullish vs bearish factors
    bullish = sum(1 for f in factors.values() if f["score"] > 0.1)
    bearish = sum(1 for f in factors.values() if f["score"] < -0.1)

    return {
        "prediction": prediction,
        "emoji": emoji,
        "score": round(final_score, 3),
        "confidence": round(confidence, 1),
        "factors": factors,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "btst_trade": btst_trade,
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "best_check_time": "3:00 PM - 3:30 PM IST for most accurate prediction",
    }
