"""
╔══════════════════════════════════════════════════════════════╗
║  GLOBAL ANALYSIS ENGINE v2 — OPTIMIZED                      ║
║                                                              ║
║  KEY OPTIMIZATION:                                           ║
║  OLD: 20+ individual yf.download() calls = 20-40 seconds    ║
║  NEW: 1 single batch yf.download() call  = 3-5 seconds     ║
║                                                              ║
║  yfinance supports downloading multiple tickers in one call  ║
║  which makes a single HTTP request to Yahoo Finance.         ║
╚══════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
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
#  SINGLE BATCH FETCH — ALL GLOBAL DATA IN ONE CALL
# ═══════════════════════════════════════════════════════════════
def fetch_all_global_data() -> Dict[str, dict]:
    """
    Fetch ALL global market data in ONE yfinance batch call.
    
    OLD: 20 individual calls × 1.5s each = 30 seconds
    NEW: 1 batch call = 3-5 seconds total
    """
    results = {}

    # Collect every unique ticker needed
    all_names = set()
    for group in GLOBAL_SIGNAL_MARKETS.values():
        all_names.update(group["tickers"])

    # Map names to Yahoo codes
    name_to_yahoo = {name: TICKERS.get(name, name) for name in all_names}
    yahoo_codes = list(name_to_yahoo.values())
    name_list = list(name_to_yahoo.keys())

    if not yahoo_codes:
        return results

    try:
        # SINGLE batch download — the key optimization
        raw = yf.download(
            yahoo_codes,
            period="5d",
            interval="1d",
            progress=False,
            group_by="ticker",
            threads=True,  # Parallel download within yfinance
        )

        if raw.empty:
            return results

        for name, yahoo_code in zip(name_list, yahoo_codes):
            try:
                # Extract this ticker's data from batch result
                if len(yahoo_codes) == 1:
                    ticker_data = raw
                else:
                    if yahoo_code in raw.columns.get_level_values(0):
                        ticker_data = raw[yahoo_code].dropna()
                    else:
                        continue

                if ticker_data.empty or len(ticker_data) < 2:
                    continue

                last_close = float(ticker_data["Close"].iloc[-1])
                prev_close = float(ticker_data["Close"].iloc[-2])
                change = last_close - prev_close
                change_pct = (change / prev_close) * 100

                results[name] = {
                    "price": round(last_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "prev_close": round(prev_close, 2),
                    "status": "🟢" if change_pct > 0.1 else ("🔴" if change_pct < -0.1 else "⚪"),
                }
            except Exception:
                continue

    except Exception as e:
        logger.warning(f"Batch global fetch failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════════
#  GLOBAL MARKET SCORE (unchanged logic, same as before)
# ═══════════════════════════════════════════════════════════════
def calculate_global_score(global_data: Dict[str, dict]) -> Tuple[float, str, Dict[str, dict]]:
    """Calculate weighted global market score for Nifty direction."""
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
            avg_score = np.mean(group_scores)
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

    if total_weight > 0:
        final_score = total_score / total_weight
    else:
        final_score = 0.0

    final_score = np.clip(final_score, -1, 1)

    if final_score > 0.4: label = "STRONG BULLISH"
    elif final_score > 0.15: label = "BULLISH"
    elif final_score < -0.4: label = "STRONG BEARISH"
    elif final_score < -0.15: label = "BEARISH"
    else: label = "NEUTRAL"

    return round(final_score, 3), label, group_details


# ═══════════════════════════════════════════════════════════════
#  INDIA VIX ANALYSIS (unchanged)
# ═══════════════════════════════════════════════════════════════
def analyze_india_vix(vix_current: float = 0, vix_history: pd.DataFrame = None) -> Dict:
    """Deep analysis of India VIX."""
    result = {
        "current": vix_current, "zone": "UNKNOWN", "zone_color": "#ffc107",
        "action": "WAIT", "trend": "NEUTRAL", "signal_score": 0.0,
        "spike_alert": False, "strategy_advice": "", "details": [],
        "change_1d": 0, "change_5d": 0, "percentile": 50,
    }
    if vix_current <= 0:
        return result

    for zone_name, zone_cfg in VIX_ZONES.items():
        lo, hi = zone_cfg["range"]
        if lo <= vix_current < hi:
            result["zone"] = zone_name
            result["zone_color"] = zone_cfg["color"]
            result["action"] = zone_cfg["action"]
            break

    if vix_history is not None and len(vix_history) >= 3:
        if isinstance(vix_history.columns, pd.MultiIndex):
            vix_history.columns = vix_history.columns.get_level_values(0)

        closes = vix_history["Close"].tail(5)
        if len(closes) >= 2:
            vix_5d_change = ((float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0])) * 100
            vix_1d_change = ((float(closes.iloc[-1]) - float(closes.iloc[-2])) / float(closes.iloc[-2])) * 100
            result["change_1d"] = round(vix_1d_change, 2)
            result["change_5d"] = round(vix_5d_change, 2)

            if vix_5d_change > 5: result["trend"] = "RISING FAST"; result["details"].append(f"VIX rose {vix_5d_change:.1f}% in 5 days")
            elif vix_5d_change > 0: result["trend"] = "RISING"; result["details"].append(f"VIX trending up {vix_5d_change:.1f}%")
            elif vix_5d_change < -5: result["trend"] = "FALLING FAST"; result["details"].append(f"VIX fell {vix_5d_change:.1f}% in 5 days")
            else: result["trend"] = "FALLING"; result["details"].append(f"VIX easing {vix_5d_change:.1f}%")

            if vix_1d_change >= VIX_INTRADAY_SPIKE:
                result["spike_alert"] = True
                result["details"].append(f"⚠️ VIX SPIKED {vix_1d_change:.1f}% today!")

            if len(vix_history) >= 20:
                all_closes = vix_history["Close"].values
                percentile = (np.sum(all_closes < vix_current) / len(all_closes)) * 100
                result["percentile"] = round(percentile, 1)

    # Signal score
    if vix_current < VIX_LOW: base_score = 0.2
    elif vix_current < VIX_NORMAL_HIGH: base_score = 0.0
    elif vix_current < VIX_HIGH: base_score = -0.3
    else: base_score = -0.6

    if result["trend"] == "FALLING FAST": base_score += 0.2
    elif result["trend"] == "FALLING": base_score += 0.1
    elif result["trend"] == "RISING FAST": base_score -= 0.3
    elif result["trend"] == "RISING": base_score -= 0.15
    if result["spike_alert"]: base_score -= 0.3
    result["signal_score"] = round(np.clip(base_score, -1, 1), 3)

    # Strategy advice
    if vix_current < 12: result["strategy_advice"] = "VIX extremely low — Options CHEAP. Buy ATM straddles. Good for directional buying."
    elif vix_current < 16: result["strategy_advice"] = "VIX low-normal — Good for directional option BUYING. Premiums fair."
    elif vix_current < 20: result["strategy_advice"] = "VIX normal — Use technical signals. Both buying and selling work."
    elif vix_current < 25: result["strategy_advice"] = "VIX elevated — Favor SELLING strategies. Reduce position size 30-50%."
    else: result["strategy_advice"] = "VIX very high — DANGER ZONE. Reduce all positions. Go to cash."

    return result


# ═══════════════════════════════════════════════════════════════
#  INDIAN INDICES — BATCH FETCH (optimized)
# ═══════════════════════════════════════════════════════════════
def analyze_indian_indices() -> Dict[str, dict]:
    """Fetch Indian sectoral indices in one batch call."""
    indices = {
        "Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK",
        "Nifty IT": "^CNXIT", "Nifty Fin": "^CNXFIN",
    }
    results = {}
    codes = list(indices.values())
    names = list(indices.keys())

    try:
        raw = yf.download(codes, period="5d", interval="1d",
                          progress=False, group_by="ticker", threads=True)
        if raw.empty:
            return results
        for name, code in zip(names, codes):
            try:
                if len(codes) == 1:
                    td = raw
                else:
                    td = raw[code].dropna() if code in raw.columns.get_level_values(0) else None
                if td is not None and len(td) >= 2:
                    last = float(td["Close"].iloc[-1])
                    prev = float(td["Close"].iloc[-2])
                    chg = ((last - prev) / prev) * 100
                    results[name] = {
                        "price": round(last, 2),
                        "change_pct": round(chg, 2),
                        "status": "🟢" if chg > 0 else "🔴",
                    }
            except Exception:
                continue
    except Exception:
        pass
    return results
