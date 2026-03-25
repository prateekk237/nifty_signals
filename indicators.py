"""
╔══════════════════════════════════════════════════════════════╗
║  INDICATORS ENGINE — Technical analysis calculations        ║
║  Supertrend, RSI, MACD, EMA, VWAP, CPR, BB, ADX, etc.     ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from config import (
    ST_FAST, ST_MEDIUM, ST_SLOW,
    RSI_FAST, RSI_STANDARD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_INTRADAY, EMA_SWING, EMA_POSITIONAL, EMA_SCALP,
    BB_LENGTH, BB_STD, ADX_LENGTH,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  MANUAL INDICATOR CALCULATIONS (fallback if pandas-ta missing)
# ═══════════════════════════════════════════════════════════════
def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════
#  SUPERTREND CALCULATION
# ═══════════════════════════════════════════════════════════════
def calc_supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate Supertrend indicator.
    Adds columns: ST_{length}_{mult}, STd_{length}_{mult} (direction: 1=up, -1=down)
    """
    result = df.copy()
    hl2 = (result["High"] + result["Low"]) / 2
    atr = _atr(result["High"], result["Low"], result["Close"], length)

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=result.index, dtype=float)
    direction = pd.Series(index=result.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(result)):
        curr_close = result["Close"].iloc[i]
        prev_close = result["Close"].iloc[i - 1]

        # Adjust bands
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or prev_close < lower_band.iloc[i - 1]:
            pass
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        if upper_band.iloc[i] < upper_band.iloc[i - 1] or prev_close > upper_band.iloc[i - 1]:
            pass
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Direction
        prev_st = supertrend.iloc[i - 1]
        if prev_st == upper_band.iloc[i - 1]:
            if curr_close > upper_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1  # Bullish
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1  # Bearish
        else:
            if curr_close < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1

    suffix = f"{length}_{multiplier}"
    result[f"ST_{suffix}"] = supertrend
    result[f"STd_{suffix}"] = direction

    return result


# ═══════════════════════════════════════════════════════════════
#  FULL INDICATOR SUITE
# ═══════════════════════════════════════════════════════════════
def add_all_indicators(df: pd.DataFrame, timeframe: str = "Intraday") -> pd.DataFrame:
    """
    Add all technical indicators to OHLCV DataFrame.
    Adapts indicator parameters based on timeframe.
    """
    if df.empty or len(df) < 30:
        return df

    result = df.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"] if "Volume" in result.columns else pd.Series(0, index=result.index)

    # ── RSI ──────────────────────────────────────────────────
    result["RSI_7"] = _rsi(close, RSI_FAST)
    result["RSI_14"] = _rsi(close, RSI_STANDARD)

    # ── MACD ─────────────────────────────────────────────────
    ema_fast = _ema(close, MACD_FAST)
    ema_slow = _ema(close, MACD_SLOW)
    result["MACD"] = ema_fast - ema_slow
    result["MACD_Signal"] = _ema(result["MACD"], MACD_SIGNAL)
    result["MACD_Hist"] = result["MACD"] - result["MACD_Signal"]

    # ── EMAs ─────────────────────────────────────────────────
    for period in [5, 8, 9, 13, 20, 21, 50, 200]:
        result[f"EMA_{period}"] = _ema(close, period)

    # ── Supertrend (3 timeframes) ────────────────────────────
    result = calc_supertrend(result, **ST_FAST)
    result = calc_supertrend(result, **ST_MEDIUM)
    result = calc_supertrend(result, **ST_SLOW)

    # ── Bollinger Bands ──────────────────────────────────────
    result["BB_Mid"] = _sma(close, BB_LENGTH)
    bb_std = close.rolling(window=BB_LENGTH).std()
    result["BB_Upper"] = result["BB_Mid"] + BB_STD * bb_std
    result["BB_Lower"] = result["BB_Mid"] - BB_STD * bb_std
    result["BB_Width"] = (result["BB_Upper"] - result["BB_Lower"]) / result["BB_Mid"]
    result["BB_Pct"] = (close - result["BB_Lower"]) / (result["BB_Upper"] - result["BB_Lower"])

    # ── ATR ───────────────────────────────────────────────────
    result["ATR_14"] = _atr(high, low, close, 14)
    result["ATR_7"] = _atr(high, low, close, 7)

    # ── ADX ───────────────────────────────────────────────────
    result = calc_adx(result, ADX_LENGTH)

    # ── VWAP (intraday only) ─────────────────────────────────
    if timeframe in ["Scalping", "Intraday"]:
        result = calc_vwap(result)

    # ── Stochastic RSI ────────────────────────────────────────
    result = calc_stoch_rsi(result)

    # ── Heikin Ashi ───────────────────────────────────────────
    result = calc_heikin_ashi(result)

    return result


# ═══════════════════════════════════════════════════════════════
#  ADX (Average Directional Index)
# ═══════════════════════════════════════════════════════════════
def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ADX, +DI, -DI."""
    result = df.copy()
    high = result["High"]
    low = result["Low"]
    close = result["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Where +DM > -DM, -DM = 0 and vice versa
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0

    atr = _atr(high, low, close, period)
    plus_di = 100 * _ema(plus_dm, period) / atr
    minus_di = 100 * _ema(minus_dm, period) / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = _ema(dx, period)

    result["ADX"] = adx
    result["Plus_DI"] = plus_di
    result["Minus_DI"] = minus_di

    return result


# ═══════════════════════════════════════════════════════════════
#  VWAP (Volume Weighted Average Price)
# ═══════════════════════════════════════════════════════════════
def calc_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate VWAP — resets daily. NaN-safe for zero volume."""
    result = df.copy()
    if "Volume" not in result.columns or result["Volume"].sum() == 0:
        # No volume data — use typical price as proxy
        result["VWAP"] = (result["High"] + result["Low"] + result["Close"]) / 3
        return result

    typical_price = (result["High"] + result["Low"] + result["Close"]) / 3
    tp_vol = typical_price * result["Volume"]

    result["_date"] = result.index.date
    cum_vol = result["Volume"].groupby(result["_date"]).cumsum()
    cum_tp_vol = tp_vol.groupby(result["_date"]).cumsum()

    # Prevent division by zero — replace 0 volume with NaN, then forward-fill
    result["VWAP"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    result["VWAP"] = result["VWAP"].ffill()  # Forward-fill NaN gaps
    result.drop("_date", axis=1, inplace=True)

    return result


# ═══════════════════════════════════════════════════════════════
#  CPR (Central Pivot Range)
# ═══════════════════════════════════════════════════════════════
def calc_cpr(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    """
    Calculate Central Pivot Range from previous day's OHLC.
    Returns dict with pivot, bc (bottom), tc (top), r1, r2, s1, s2.
    """
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2  # Bottom Central Pivot
    tc = 2 * pivot - bc              # Top Central Pivot

    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)

    cpr_width = abs(tc - bc)
    is_narrow = cpr_width < (prev_high - prev_low) * 0.1  # < 10% of range

    return {
        "pivot": round(pivot, 2),
        "bc": round(min(bc, tc), 2),
        "tc": round(max(bc, tc), 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
        "cpr_width": round(cpr_width, 2),
        "is_narrow_cpr": is_narrow,
    }


# ═══════════════════════════════════════════════════════════════
#  STOCHASTIC RSI
# ═══════════════════════════════════════════════════════════════
def calc_stoch_rsi(df: pd.DataFrame, period: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    """Calculate Stochastic RSI."""
    result = df.copy()
    rsi = _rsi(result["Close"], period)
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    result["StochRSI_K"] = stoch_rsi.rolling(window=k).mean()
    result["StochRSI_D"] = result["StochRSI_K"].rolling(window=d).mean()
    return result


# ═══════════════════════════════════════════════════════════════
#  HEIKIN ASHI
# ═══════════════════════════════════════════════════════════════
def calc_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Heikin Ashi candles for trend confirmation."""
    result = df.copy()
    ha_close = (result["Open"] + result["High"] + result["Low"] + result["Close"]) / 4
    ha_open = pd.Series(index=result.index, dtype=float)
    ha_open.iloc[0] = (result["Open"].iloc[0] + result["Close"].iloc[0]) / 2

    for i in range(1, len(result)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    result["HA_Close"] = ha_close
    result["HA_Open"] = ha_open
    result["HA_High"] = pd.concat(
        [result["High"], ha_open, ha_close], axis=1
    ).max(axis=1)
    result["HA_Low"] = pd.concat(
        [result["Low"], ha_open, ha_close], axis=1
    ).min(axis=1)
    result["HA_Bullish"] = ha_close > ha_open

    return result


# ═══════════════════════════════════════════════════════════════
#  OPENING RANGE BREAKOUT LEVELS
# ═══════════════════════════════════════════════════════════════
def calc_orb_levels(df: pd.DataFrame, orb_minutes: int = 30) -> Dict[str, float]:
    """
    Calculate Opening Range Breakout levels.
    Uses first N minutes of data (default: 30 min = 9:15 to 9:45).
    """
    if df.empty:
        return {"orb_high": 0, "orb_low": 0, "orb_mid": 0}

    today = df.index[-1].date()
    today_data = df[df.index.date == today]

    if today_data.empty:
        return {"orb_high": 0, "orb_low": 0, "orb_mid": 0}

    # First N minutes
    market_open = today_data.index[0]
    orb_end = market_open + pd.Timedelta(minutes=orb_minutes)
    orb_data = today_data[today_data.index <= orb_end]

    if orb_data.empty:
        return {"orb_high": 0, "orb_low": 0, "orb_mid": 0}

    orb_high = float(orb_data["High"].max())
    orb_low = float(orb_data["Low"].min())
    orb_mid = (orb_high + orb_low) / 2

    return {
        "orb_high": round(orb_high, 2),
        "orb_low": round(orb_low, 2),
        "orb_mid": round(orb_mid, 2),
        "orb_range": round(orb_high - orb_low, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  INDICATOR SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════════════════
def get_indicator_signals(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Extract current signal from each indicator.
    Returns dict with signal (-1 to +1), description, and raw values.
    """
    if df.empty or len(df) < 5:
        return {}

    signals = {}
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── Supertrend Fast (5, 1.5) ─────────────────────────────
    st_col = "STd_5_1.5"
    if st_col in df.columns:
        direction = int(last[st_col])
        # Check for fresh crossover
        prev_dir = int(prev[st_col])
        crossover = direction != prev_dir
        strength = 1.0 if crossover else 0.7
        signals["supertrend_fast"] = {
            "signal": direction * strength,
            "label": "BUY" if direction > 0 else "SELL",
            "detail": f"ST(5,1.5): {'Bullish' if direction > 0 else 'Bearish'}"
                      + (" ★ FRESH CROSS" if crossover else ""),
            "value": round(float(last[f"ST_5_1.5"]), 2),
        }

    # ── Supertrend Medium (10, 3) ────────────────────────────
    st_col2 = "STd_10_3.0"
    if st_col2 in df.columns:
        direction = int(last[st_col2])
        signals["supertrend_med"] = {
            "signal": direction * 0.8,
            "label": "BUY" if direction > 0 else "SELL",
            "detail": f"ST(10,3): {'Bullish' if direction > 0 else 'Bearish'}",
            "value": round(float(last[f"ST_10_3.0"]), 2),
        }

    # ── RSI(7) — for Supertrend combo ────────────────────────
    if "RSI_7" in df.columns:
        rsi7 = float(last["RSI_7"])
        if rsi7 > 70:
            sig = -0.6  # Overbought
            label = "OVERBOUGHT"
        elif rsi7 < 30:
            sig = 0.6   # Oversold = potential buy
            label = "OVERSOLD"
        elif rsi7 > 55:
            sig = 0.3
            label = "BULLISH"
        elif rsi7 < 45:
            sig = -0.3
            label = "BEARISH"
        else:
            sig = 0.0
            label = "NEUTRAL"
        signals["rsi_7"] = {
            "signal": sig,
            "label": label,
            "detail": f"RSI(7): {rsi7:.1f}",
            "value": round(rsi7, 1),
        }

    # ── RSI(14) ──────────────────────────────────────────────
    if "RSI_14" in df.columns:
        rsi14 = float(last["RSI_14"])
        if rsi14 > RSI_OVERBOUGHT:
            sig = -0.5
            label = "OVERBOUGHT"
        elif rsi14 < RSI_OVERSOLD:
            sig = 0.5
            label = "OVERSOLD"
        elif rsi14 > 55:
            sig = 0.2
            label = "BULLISH"
        elif rsi14 < 45:
            sig = -0.2
            label = "BEARISH"
        else:
            sig = 0.0
            label = "NEUTRAL"
        signals["rsi_14"] = {
            "signal": sig,
            "label": label,
            "detail": f"RSI(14): {rsi14:.1f}",
            "value": round(rsi14, 1),
        }

    # ── MACD ─────────────────────────────────────────────────
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd = float(last["MACD"])
        macd_sig = float(last["MACD_Signal"])
        macd_hist = float(last["MACD_Hist"])
        prev_hist = float(prev["MACD_Hist"])

        if macd > macd_sig and prev["MACD"] <= prev["MACD_Signal"]:
            sig = 0.8  # Fresh bullish crossover
            label = "BUY CROSS"
        elif macd < macd_sig and prev["MACD"] >= prev["MACD_Signal"]:
            sig = -0.8  # Fresh bearish crossover
            label = "SELL CROSS"
        elif macd > macd_sig:
            sig = 0.4
            label = "BULLISH"
        elif macd < macd_sig:
            sig = -0.4
            label = "BEARISH"
        else:
            sig = 0.0
            label = "NEUTRAL"

        # Histogram divergence
        if macd_hist > 0 and macd_hist > prev_hist:
            sig = min(sig + 0.2, 1.0)
        elif macd_hist < 0 and macd_hist < prev_hist:
            sig = max(sig - 0.2, -1.0)

        signals["macd"] = {
            "signal": sig,
            "label": label,
            "detail": f"MACD: {macd:.2f} | Signal: {macd_sig:.2f} | Hist: {macd_hist:.2f}",
            "value": round(macd_hist, 2),
        }

    # ── EMA Crossover (9/21) ─────────────────────────────────
    if "EMA_9" in df.columns and "EMA_21" in df.columns:
        ema9 = float(last["EMA_9"])
        ema21 = float(last["EMA_21"])
        prev_ema9 = float(prev["EMA_9"])
        prev_ema21 = float(prev["EMA_21"])

        if ema9 > ema21 and prev_ema9 <= prev_ema21:
            sig = 0.9
            label = "GOLDEN CROSS ★"
        elif ema9 < ema21 and prev_ema9 >= prev_ema21:
            sig = -0.9
            label = "DEATH CROSS ★"
        elif ema9 > ema21:
            gap_pct = ((ema9 - ema21) / ema21) * 100
            sig = min(0.3 + gap_pct * 0.5, 0.7)
            label = "BULLISH"
        elif ema9 < ema21:
            gap_pct = ((ema21 - ema9) / ema21) * 100
            sig = max(-0.3 - gap_pct * 0.5, -0.7)
            label = "BEARISH"
        else:
            sig = 0.0
            label = "NEUTRAL"

        signals["ema_cross"] = {
            "signal": sig,
            "label": label,
            "detail": f"EMA 9: {ema9:.1f} | EMA 21: {ema21:.1f}",
            "value": round(ema9 - ema21, 2),
        }

    # ── VWAP ─────────────────────────────────────────────────
    if "VWAP" in df.columns:
        vwap = float(last["VWAP"])
        price = float(last["Close"])
        pct_from_vwap = ((price - vwap) / vwap) * 100

        if price > vwap:
            sig = min(0.3 + abs(pct_from_vwap) * 0.3, 0.7)
            label = "ABOVE VWAP"
        else:
            sig = max(-0.3 - abs(pct_from_vwap) * 0.3, -0.7)
            label = "BELOW VWAP"

        signals["vwap"] = {
            "signal": sig,
            "label": label,
            "detail": f"VWAP: {vwap:.1f} | Price: {price:.1f} ({pct_from_vwap:+.2f}%)",
            "value": round(vwap, 2),
        }

    # ── Bollinger Bands ──────────────────────────────────────
    if "BB_Upper" in df.columns:
        bb_pct = float(last["BB_Pct"]) if not np.isnan(last["BB_Pct"]) else 0.5
        bb_width = float(last["BB_Width"]) if not np.isnan(last["BB_Width"]) else 0

        if bb_pct > 1.0:
            sig = -0.6  # Above upper band = overbought
            label = "ABOVE UPPER BAND"
        elif bb_pct < 0.0:
            sig = 0.6   # Below lower band = oversold
            label = "BELOW LOWER BAND"
        elif bb_pct > 0.8:
            sig = -0.3
            label = "NEAR UPPER"
        elif bb_pct < 0.2:
            sig = 0.3
            label = "NEAR LOWER"
        else:
            sig = 0.0
            label = "MID-RANGE"

        signals["bollinger"] = {
            "signal": sig,
            "label": label,
            "detail": f"BB%: {bb_pct:.2f} | Width: {bb_width:.4f}",
            "value": round(bb_pct, 2),
        }

    # ── ADX (Trend Strength) ─────────────────────────────────
    if "ADX" in df.columns:
        adx = float(last["ADX"]) if not np.isnan(last["ADX"]) else 0
        plus_di = float(last["Plus_DI"]) if not np.isnan(last["Plus_DI"]) else 0
        minus_di = float(last["Minus_DI"]) if not np.isnan(last["Minus_DI"]) else 0

        if adx > 25:
            if plus_di > minus_di:
                sig = 0.5
                label = "STRONG UPTREND"
            else:
                sig = -0.5
                label = "STRONG DOWNTREND"
        elif adx > 20:
            if plus_di > minus_di:
                sig = 0.2
                label = "MILD UPTREND"
            else:
                sig = -0.2
                label = "MILD DOWNTREND"
        else:
            sig = 0.0
            label = "NO TREND (SIDEWAYS)"

        signals["adx"] = {
            "signal": sig,
            "label": label,
            "detail": f"ADX: {adx:.1f} | +DI: {plus_di:.1f} | -DI: {minus_di:.1f}",
            "value": round(adx, 1),
        }

    # ── Heikin Ashi ──────────────────────────────────────────
    if "HA_Bullish" in df.columns:
        # Count consecutive HA candles
        ha_bull = df["HA_Bullish"].tail(5)
        consec_bull = 0
        consec_bear = 0
        for v in ha_bull.iloc[::-1]:
            if v:
                consec_bull += 1
            else:
                break
        for v in ha_bull.iloc[::-1]:
            if not v:
                consec_bear += 1
            else:
                break

        if consec_bull >= 3:
            sig = 0.5
            label = f"BULLISH ({consec_bull} candles)"
        elif consec_bear >= 3:
            sig = -0.5
            label = f"BEARISH ({consec_bear} candles)"
        elif bool(last["HA_Bullish"]):
            sig = 0.2
            label = "BULLISH"
        else:
            sig = -0.2
            label = "BEARISH"

        signals["heikin_ashi"] = {
            "signal": sig,
            "label": label,
            "detail": f"HA: {'Bullish' if last['HA_Bullish'] else 'Bearish'}",
            "value": 1 if last["HA_Bullish"] else -1,
        }

    return signals
