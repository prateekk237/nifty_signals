"""
╔══════════════════════════════════════════════════════════════╗
║  DATA FETCHER v2.0 — yfinance + NSE scraping                ║
║  Enhanced: VIX history, global batch fetch                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests, json, time, logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import pytz
from config import (
    TICKERS, NSE_BASE, NSE_OPTION_CHAIN_URL, NSE_HEADERS,
    NSE_FII_DII_URL, STRIKE_STEP_NIFTY, STRIKE_STEP_BANKNIFTY, TIMEFRAMES
)
logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

class NSESession:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._last_request_time = 0
        self._min_interval = 3
        self._cookies_valid = False
    def _init_cookies(self):
        try:
            resp = self.session.get(NSE_BASE, timeout=10)
            if resp.status_code == 200: self._cookies_valid = True; return True
        except: pass
        return False
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval: time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    def get(self, url, retries=3):
        if not self._cookies_valid: self._init_cookies()
        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200: return resp.json()
                elif resp.status_code == 401: self._cookies_valid = False; self._init_cookies()
            except Exception as e:
                logger.warning(f"NSE attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        return None

_nse_session = NSESession()

def fetch_ohlcv(symbol, interval="15m", period="10d"):
    ticker_code = TICKERS.get(symbol, symbol)
    try:
        data = yf.download(ticker_code, interval=interval, period=period, progress=False, auto_adjust=True)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [c.title() if c.islower() else c for c in data.columns]
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC").tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        return data.dropna()
    except Exception as e:
        logger.error(f"yfinance failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_vix_history(period="3mo"):
    """Fetch India VIX history for analysis."""
    try:
        data = yf.download("^INDIAVIX", period=period, interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna()
    except: return pd.DataFrame()

def get_vix_all() -> dict:
    """
    Get VIX current, previous close, and history in ONE download.
    OLD: 3 separate yf.download calls = 6-9 seconds
    NEW: 1 call reused for all 3 values = 2-3 seconds
    """
    try:
        data = yf.download("^INDIAVIX", period="3mo", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        if data.empty:
            return {"current": 0.0, "prev_close": 0.0, "history": pd.DataFrame()}
        current = round(float(data["Close"].iloc[-1]), 2)
        prev_close = round(float(data["Close"].iloc[-2]), 2) if len(data) >= 2 else current
        return {"current": current, "prev_close": prev_close, "history": data}
    except:
        return {"current": 0.0, "prev_close": 0.0, "history": pd.DataFrame()}

# Keep old functions for backward compat but they use cached data
def get_india_vix():
    return get_vix_all()["current"]

def get_vix_prev_close():
    return get_vix_all()["prev_close"]

def fetch_option_chain(symbol="NIFTY"):
    url = NSE_OPTION_CHAIN_URL.format(symbol=symbol)
    return _nse_session.get(url)

def parse_option_chain(raw_data):
    if not raw_data or "records" not in raw_data: return pd.DataFrame(), {}
    records = raw_data["records"]
    rows = []
    for item in records.get("data", []):
        ce = item.get("CE", {}); pe = item.get("PE", {})
        rows.append({
            "strike": item.get("strikePrice", 0), "expiry": item.get("expiryDate", ""),
            "ce_oi": ce.get("openInterest", 0), "ce_chg_oi": ce.get("changeinOpenInterest", 0),
            "ce_volume": ce.get("totalTradedVolume", 0), "ce_iv": ce.get("impliedVolatility", 0),
            "ce_ltp": ce.get("lastPrice", 0), "ce_bid": ce.get("bidprice", 0), "ce_ask": ce.get("askPrice", 0),
            "pe_oi": pe.get("openInterest", 0), "pe_chg_oi": pe.get("changeinOpenInterest", 0),
            "pe_volume": pe.get("totalTradedVolume", 0), "pe_iv": pe.get("impliedVolatility", 0),
            "pe_ltp": pe.get("lastPrice", 0), "pe_bid": pe.get("bidprice", 0), "pe_ask": pe.get("askPrice", 0),
        })
    metadata = {
        "underlying_value": records.get("underlyingValue", 0),
        "expiry_dates": records.get("expiryDates", []),
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }
    return pd.DataFrame(rows), metadata

def calculate_pcr(oc_df, expiry=None):
    if oc_df.empty: return {"pcr_oi": 0, "pcr_volume": 0, "pcr_chg_oi": 0}
    df = oc_df[oc_df["expiry"] == expiry].copy() if expiry else oc_df.copy()
    tce = df["ce_oi"].sum(); tpe = df["pe_oi"].sum()
    return {
        "pcr_oi": round(tpe / tce, 3) if tce > 0 else 0,
        "pcr_volume": round(df["pe_volume"].sum() / max(df["ce_volume"].sum(), 1), 3),
        "pcr_chg_oi": round(df["pe_chg_oi"].sum() / max(df["ce_chg_oi"].sum(), 1), 3),
        "total_ce_oi": tce, "total_pe_oi": tpe,
    }

def calculate_max_pain(oc_df, expiry=None):
    if oc_df.empty: return 0.0
    df = oc_df[oc_df["expiry"] == expiry].copy() if expiry else oc_df.copy()
    strikes = df["strike"].unique(); min_pain = float("inf"); mp = 0
    for ts in strikes:
        ce_p = df.apply(lambda r: max(0, ts - r["strike"]) * r["ce_oi"], axis=1).sum()
        pe_p = df.apply(lambda r: max(0, r["strike"] - ts) * r["pe_oi"], axis=1).sum()
        total = ce_p + pe_p
        if total < min_pain: min_pain = total; mp = ts
    return mp

def get_oi_support_resistance(oc_df, underlying, expiry=None, n=3):
    if oc_df.empty: return {"support": [], "resistance": []}
    df = oc_df[oc_df["expiry"] == expiry].copy() if expiry else oc_df.copy()
    above = df[df["strike"] >= underlying].nlargest(n, "ce_oi")
    below = df[df["strike"] <= underlying].nlargest(n, "pe_oi")
    return {"support": sorted(below["strike"].tolist(), reverse=True),
            "resistance": sorted(above["strike"].tolist())}

def analyze_oi_buildup(oc_df, underlying, expiry=None):
    if oc_df.empty: return "NEUTRAL"
    df = oc_df[oc_df["expiry"] == expiry].copy() if expiry else oc_df.copy()
    step = STRIKE_STEP_NIFTY
    atm = df[(df["strike"] >= underlying - 5*step) & (df["strike"] <= underlying + 5*step)]
    ce_chg = atm["ce_chg_oi"].sum(); pe_chg = atm["pe_chg_oi"].sum()
    if pe_chg > ce_chg * 1.3: return "BULLISH"
    elif ce_chg > pe_chg * 1.3: return "BEARISH"
    return "NEUTRAL"

def is_market_open():
    now = datetime.now(IST)
    if now.weekday() >= 5: return False
    mo = now.replace(hour=9, minute=15, second=0, microsecond=0)
    mc = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return mo <= now <= mc

def get_market_session():
    now = datetime.now(IST)
    h, m = now.hour, now.minute
    if now.weekday() >= 5: return "WEEKEND"
    if h < 9 or (h == 9 and m < 15): return "PRE-MARKET"
    if h == 9 and m < 30: return "OPENING (9:15-9:30)"
    if h < 11: return "MORNING SESSION"
    if h < 13: return "MID-DAY SESSION"
    if h < 14: return "AFTERNOON SESSION"
    if h < 15 or (h == 15 and m <= 30): return "CLOSING SESSION"
    return "POST-MARKET"

def get_atm_strike(price, step): return int(round(price / step) * step)

def get_previous_day_ohlc(symbol):
    df = fetch_ohlcv(symbol, "1d", "5d")
    if len(df) >= 2:
        p = df.iloc[-2]
        return {"high": float(p["High"]), "low": float(p["Low"]),
                "close": float(p["Close"]), "open": float(p["Open"])}
    return {"high": 0, "low": 0, "close": 0, "open": 0}


# ═══════════════════════════════════════════════════════════════
#  NSE LIVE INDEX QUOTE (faster than yfinance for current price)
# ═══════════════════════════════════════════════════════════════
def fetch_nse_live_indices() -> dict:
    """
    Fetch live index prices from NSE allIndices API.
    Returns dict of {index_name: {last, change, pctChange, open, high, low}}.
    Much faster than yfinance for current prices.
    """
    from config import NSE_INDEX_URL
    data = _nse_session.get(NSE_INDEX_URL)
    if not data or "data" not in data:
        return {}

    results = {}
    for item in data["data"]:
        name = item.get("index", "")
        results[name] = {
            "last": item.get("last", 0),
            "change": item.get("variation", 0),
            "pctChange": item.get("percentChange", 0),
            "open": item.get("open", 0),
            "high": item.get("high", 0),
            "low": item.get("low", 0),
            "prev_close": item.get("previousClose", 0),
        }
    return results


def get_nse_live_price(symbol: str) -> dict:
    """Get live price for a specific index from NSE."""
    index_map = {
        "NIFTY50": "NIFTY 50",
        "BANKNIFTY": "NIFTY BANK",
        "NIFTY_IT": "NIFTY IT",
        "NIFTY_FIN": "NIFTY FINANCIAL SERVICES",
        "INDIAVIX": "INDIA VIX",
    }
    nse_name = index_map.get(symbol, symbol)

    all_indices = fetch_nse_live_indices()
    if nse_name in all_indices:
        data = all_indices[nse_name]
        return {
            "price": float(data["last"]),
            "change": float(data["change"]),
            "change_pct": float(data["pctChange"]),
            "open": float(data["open"]),
            "high": float(data["high"]),
            "low": float(data["low"]),
            "prev_close": float(data["prev_close"]),
            "source": "NSE_LIVE",
        }
    return {"price": 0, "change": 0, "change_pct": 0, "source": "UNAVAILABLE"}


def fetch_fast_5min(symbol: str) -> pd.DataFrame:
    """
    Fetch latest 5-min data optimized for quick signals.
    Uses shorter period (2 days) for faster download.
    """
    return fetch_ohlcv(symbol, "5m", "2d")
