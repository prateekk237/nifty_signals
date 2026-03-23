"""
╔══════════════════════════════════════════════════════════════╗
║  GLOBAL ANALYSIS ENGINE — Indices, VIX, Correlation         ║
║                                                              ║
║  HOW IT WORKS:                                               ║
║  1. Fetches ALL global indices (US, Europe, Asia)           ║
║  2. Calculates % change for each market                      ║
║  3. Applies correlation weights (e.g., Crude ↑ = India ↓)  ║
║  4. Combines into a single GLOBAL SCORE (-1 to +1)          ║
║  5. VIX analysis adds strategy recommendation               ║
║                                                              ║
║  FORMULA:                                                    ║
║  global_score = Σ (market_change% × correlation × weight)   ║
║  Normalized to [-1, +1] range                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pytz
import logging

from config import (
    TICKERS, GLOBAL_SIGNAL_MARKETS, CORRELATION_DIRECTION,
    VIX_ZONES, VIX_LOW, VIX_NORMAL_HIGH, VIX_HIGH, VIX_EXTREME,
    VIX_SPIKE_PCT, VIX_INTRADAY_SPIKE,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════════
#  FETCH ALL GLOBAL MARKET DATA
# ═══════════════════════════════════════════════════════════════
def fetch_all_global_data() -> Dict[str, dict]:
    """
    Fetch price + change data for ALL tracked global markets.
    Returns dict of {ticker_name: {price, change, change_pct, status}}.
    """
    results = {}

    # Collect all unique tickers needed
    all_tickers = set()
    for group in GLOBAL_SIGNAL_MARKETS.values():
        all_tickers.update(group["tickers"])

    for name in all_tickers:
        yahoo_code = TICKERS.get(name, name)
        try:
            data = yf.download(yahoo_code, period="5d", interval="1d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and len(data) >= 2:
                last_close = float(data["Close"].iloc[-1])
                prev_close = float(data["Close"].iloc[-2])
                change = last_close - prev_close
                change_pct = (change / prev_close) * 100

                results[name] = {
                    "price": round(last_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "prev_close": round(prev_close, 2),
                    "status": "🟢" if change_pct > 0.1 else ("🔴" if change_pct < -0.1 else "⚪"),
                }
        except Exception as e:
            logger.warning(f"Failed to fetch {name}: {e}")
            continue

    return results


def fetch_global_data_fast(tickers_list: List[str]) -> Dict[str, dict]:
    """Fetch specific tickers quickly using batch download."""
    yahoo_codes = [TICKERS.get(t, t) for t in tickers_list]
    results = {}

    try:
        data = yf.download(yahoo_codes, period="5d", interval="1d",
                           progress=False, group_by="ticker")
        for name, yahoo_code in zip(tickers_list, yahoo_codes):
            try:
                if len(yahoo_codes) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[yahoo_code] if yahoo_code in data.columns.get_level_values(0) else None

                if ticker_data is not None and len(ticker_data) >= 2:
                    last_close = float(ticker_data["Close"].iloc[-1])
                    prev_close = float(ticker_data["Close"].iloc[-2])
                    change_pct = ((last_close - prev_close) / prev_close) * 100
                    results[name] = {
                        "price": round(last_close, 2),
                        "change_pct": round(change_pct, 2),
                        "status": "🟢" if change_pct > 0.1 else ("🔴" if change_pct < -0.1 else "⚪"),
                    }
            except Exception:
                continue
    except Exception:
        # Fallback to individual fetching
        return fetch_all_global_data()

    return results


# ═══════════════════════════════════════════════════════════════
#  GLOBAL MARKET SCORE CALCULATION
# ═══════════════════════════════════════════════════════════════
def calculate_global_score(global_data: Dict[str, dict]) -> Tuple[float, str, Dict[str, dict]]:
    """
    Calculate weighted global market score for Nifty direction.

    HOW THE SCORE IS CALCULATED:
    ─────────────────────────────
    For each market group (US Futures, Asian, European, etc.):
      1. Get the % change of each market in the group
      2. Multiply by CORRELATION_DIRECTION:
         - Positive markets (S&P, Nikkei): change × +1
         - Inverse markets (Crude, DXY, VIX): change × -1
      3. Average the correlated changes within the group
      4. Multiply by group WEIGHT (US Futures = 30%, Asian = 20%, etc.)
      5. Sum all weighted group scores
      6. Normalize to [-1, +1]

    EXAMPLE:
      S&P futures +0.5% → 0.5 × +1 = +0.5 (bullish for Nifty)
      Crude +2.0%       → 2.0 × -1 = -2.0 (bearish for Nifty, India imports oil)
      India VIX +5%     → 5.0 × -1 = -5.0 (fear rising, bearish)

    Returns:
        (score, label, group_details)
        score: -1.0 to +1.0
        label: STRONG BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG BEARISH
    """
    if not global_data:
        return 0.0, "NO DATA", {}

    group_details = {}
    total_score = 0.0
    total_weight = 0.0

    for group_name, group_config in GLOBAL_SIGNAL_MARKETS.items():
        group_tickers = group_config["tickers"]
        group_weight = group_config["weight"]

        group_scores = []
        ticker_details = []

        for ticker in group_tickers:
            if ticker in global_data:
                mkt = global_data[ticker]
                change_pct = mkt["change_pct"]
                corr = CORRELATION_DIRECTION.get(ticker, 1.0)

                # Correlated score: how this market affects Nifty
                # Positive = bullish for Nifty, Negative = bearish
                correlated_score = change_pct * corr

                group_scores.append(correlated_score)
                ticker_details.append({
                    "ticker": ticker,
                    "change_pct": change_pct,
                    "correlation": corr,
                    "nifty_impact": round(correlated_score, 3),
                    "status": mkt["status"],
                    "price": mkt.get("price", 0),
                })

        if group_scores:
            # Average score for the group
            avg_score = np.mean(group_scores)
            # Normalize to roughly [-1, 1] range (assuming max 3% move is extreme)
            normalized = np.clip(avg_score / 3.0, -1, 1)
            weighted = normalized * group_weight

            total_score += weighted
            total_weight += group_weight

            group_details[group_name] = {
                "score": round(normalized, 3),
                "weighted_score": round(weighted, 4),
                "weight": group_weight,
                "tickers": ticker_details,
                "direction": "BULLISH" if normalized > 0.1 else ("BEARISH" if normalized < -0.1 else "NEUTRAL"),
            }

    # Final score
    if total_weight > 0:
        final_score = total_score / total_weight
    else:
        final_score = 0.0

    final_score = np.clip(final_score, -1, 1)

    # Label
    if final_score > 0.4:
        label = "STRONG BULLISH"
    elif final_score > 0.15:
        label = "BULLISH"
    elif final_score < -0.4:
        label = "STRONG BEARISH"
    elif final_score < -0.15:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return round(final_score, 3), label, group_details


# ═══════════════════════════════════════════════════════════════
#  INDIA VIX DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════
def analyze_india_vix(vix_current: float = 0, vix_history: pd.DataFrame = None) -> Dict:
    """
    Deep analysis of India VIX for strategy selection.

    HOW VIX ANALYSIS WORKS:
    ────────────────────────
    1. ZONE DETECTION: Which VIX zone are we in (Low/Normal/High/Extreme)?
       → Determines whether to BUY or SELL options
    2. VIX TREND: Is VIX rising or falling over last 5 days?
       → Rising VIX = increasing fear = more bearish
    3. VIX SPIKE: Did VIX jump suddenly today?
       → Sudden spike = panic → expect big move, tighten stops
    4. VIX vs US VIX: Is India VIX higher or lower than US VIX?
       → Divergence signals local vs global risk perception
    5. VIX PERCENTILE: Where is current VIX vs. 52-week range?
       → Below 20th percentile = extreme low = complacency warning

    Returns dict with zone, trend, signals, and strategy recommendation.
    """
    result = {
        "current": vix_current,
        "zone": "UNKNOWN",
        "zone_color": "#ffc107",
        "action": "WAIT",
        "trend": "NEUTRAL",
        "signal_score": 0.0,  # -1 to +1 for Nifty direction
        "spike_alert": False,
        "strategy_advice": "",
        "details": [],
    }

    if vix_current <= 0:
        return result

    # ── 1. Zone Detection ────────────────────────────────────
    for zone_name, zone_cfg in VIX_ZONES.items():
        lo, hi = zone_cfg["range"]
        if lo <= vix_current < hi:
            result["zone"] = zone_name
            result["zone_color"] = zone_cfg["color"]
            result["action"] = zone_cfg["action"]
            break

    # ── 2. VIX Trend (from history) ──────────────────────────
    if vix_history is not None and len(vix_history) >= 3:
        if isinstance(vix_history.columns, pd.MultiIndex):
            vix_history.columns = vix_history.columns.get_level_values(0)

        closes = vix_history["Close"].tail(5)
        if len(closes) >= 2:
            vix_5d_change = ((float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0])) * 100
            vix_1d_change = ((float(closes.iloc[-1]) - float(closes.iloc[-2])) / float(closes.iloc[-2])) * 100

            result["change_1d"] = round(vix_1d_change, 2)
            result["change_5d"] = round(vix_5d_change, 2)

            # Trend determination
            if vix_5d_change > 5:
                result["trend"] = "RISING FAST"
                result["details"].append(f"VIX rose {vix_5d_change:.1f}% in 5 days — fear increasing")
            elif vix_5d_change > 0:
                result["trend"] = "RISING"
                result["details"].append(f"VIX trending up {vix_5d_change:.1f}% — caution")
            elif vix_5d_change < -5:
                result["trend"] = "FALLING FAST"
                result["details"].append(f"VIX fell {vix_5d_change:.1f}% in 5 days — complacency")
            else:
                result["trend"] = "FALLING"
                result["details"].append(f"VIX easing {vix_5d_change:.1f}% — positive for market")

            # ── 3. Spike Detection ───────────────────────────
            if vix_1d_change >= VIX_INTRADAY_SPIKE:
                result["spike_alert"] = True
                result["details"].append(
                    f"⚠️ VIX SPIKED {vix_1d_change:.1f}% today — expect sharp move! Tighten SL!"
                )
            elif vix_1d_change >= VIX_SPIKE_PCT:
                result["spike_alert"] = True
                result["details"].append(
                    f"🚨 MAJOR VIX SPIKE {vix_1d_change:.1f}% — consider exiting positions!"
                )

            # ── 4. Percentile (52-week context) ──────────────
            if len(vix_history) >= 20:
                all_closes = vix_history["Close"].values
                percentile = (np.sum(all_closes < vix_current) / len(all_closes)) * 100
                result["percentile"] = round(percentile, 1)

                if percentile < 10:
                    result["details"].append(
                        f"VIX at {percentile:.0f}th percentile — EXTREME LOW, correction risk building"
                    )
                elif percentile > 90:
                    result["details"].append(
                        f"VIX at {percentile:.0f}th percentile — EXTREME HIGH, bottom may be near"
                    )

    # ── 5. Signal Score for Nifty ────────────────────────────
    # Low VIX + Falling = mildly bullish (+0.3)
    # High VIX + Rising = strongly bearish (-0.8)
    # VIX spike = bearish (-0.6)
    if vix_current < VIX_LOW:
        base_score = 0.2  # Low fear = mildly positive
    elif vix_current < VIX_NORMAL_HIGH:
        base_score = 0.0  # Neutral
    elif vix_current < VIX_HIGH:
        base_score = -0.3  # Elevated fear
    else:
        base_score = -0.6  # High fear

    # Adjust by trend
    if result["trend"] == "FALLING FAST":
        base_score += 0.2
    elif result["trend"] == "FALLING":
        base_score += 0.1
    elif result["trend"] == "RISING FAST":
        base_score -= 0.3
    elif result["trend"] == "RISING":
        base_score -= 0.15

    if result["spike_alert"]:
        base_score -= 0.3

    result["signal_score"] = round(np.clip(base_score, -1, 1), 3)

    # ── 6. Strategy Advice ───────────────────────────────────
    if vix_current < 12:
        result["strategy_advice"] = (
            "VIX extremely low — Options are CHEAP. "
            "BUY ATM straddles/strangles. Great for directional buying. "
            "BUT: complacency risk — buy protective puts for swing trades."
        )
    elif vix_current < 16:
        result["strategy_advice"] = (
            "VIX low-normal — Good for directional option BUYING. "
            "Premiums are fair. Use Supertrend+RSI signal for direction."
        )
    elif vix_current < 20:
        result["strategy_advice"] = (
            "VIX normal — Use technical signals. "
            "Both buying and selling strategies can work. "
            "Prefer selling on expiry days, buying otherwise."
        )
    elif vix_current < 25:
        result["strategy_advice"] = (
            "VIX elevated — Premiums are RICH. Favor SELLING strategies: "
            "credit spreads, short strangles with hedges. "
            "Reduce position size by 30-50%."
        )
    else:
        result["strategy_advice"] = (
            "VIX very high — DANGER ZONE. Reduce all positions. "
            "Only sell far-OTM options with strict hedges. "
            "Or go to cash and wait for VIX to drop below 20."
        )

    return result


# ═══════════════════════════════════════════════════════════════
#  INDIAN INDEX COMPARISON
# ═══════════════════════════════════════════════════════════════
def analyze_indian_indices() -> Dict[str, dict]:
    """
    Fetch and compare Indian sectoral indices.
    Shows which sectors are strong/weak for signal confirmation.
    """
    indices = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Nifty IT": "^CNXIT",
        "Nifty Fin Services": "^CNXFIN",
    }

    results = {}
    for name, ticker in indices.items():
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and len(data) >= 2:
                last = float(data["Close"].iloc[-1])
                prev = float(data["Close"].iloc[-2])
                chg = ((last - prev) / prev) * 100
                results[name] = {
                    "price": round(last, 2),
                    "change_pct": round(chg, 2),
                    "status": "🟢" if chg > 0 else "🔴",
                }
        except Exception:
            continue

    return results
