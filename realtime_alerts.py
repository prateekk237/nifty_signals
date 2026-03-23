"""
╔══════════════════════════════════════════════════════════════╗
║  REAL-TIME ALERTS — Trend Reversal & Exit Signal Engine     ║
║                                                              ║
║  This module monitors your trades in real-time and tells you ║
║  WHEN TO EXIT — whether for profit booking or loss cutting.  ║
║                                                              ║
║  ALERT TYPES:                                                ║
║  ─────────────                                               ║
║  🔴 CRITICAL: EXIT IMMEDIATELY                               ║
║     → Supertrend flipped against your position               ║
║     → Key support/resistance broken                          ║
║     → VIX spike > 5% intraday                                ║
║     → Breaking negative news detected                        ║
║                                                              ║
║  🟡 HIGH: TIGHTEN STOP-LOSS                                  ║
║     → EMA crossover against position                         ║
║     → VWAP broken against position                           ║
║     → ATR expanded 2x (volatility surge)                     ║
║     → 30 minutes before market close                         ║
║                                                              ║
║  🔵 MEDIUM: WATCH CLOSELY                                    ║
║     → RSI hit extreme (75+ or 25-)                           ║
║     → PCR hit extreme range                                  ║
║     → Heikin Ashi color changed                              ║
║                                                              ║
║  HOW TO USE:                                                 ║
║  1. Pass your current position (BUY CE or BUY PE)           ║
║  2. Pass current indicator data                              ║
║  3. System checks ALL conditions and returns alerts          ║
║  4. Follow CRITICAL alerts immediately                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import pytz

from config import (
    ALERT_CONFIG, BREAKING_NEWS_KEYWORDS,
    VIX_INTRADAY_SPIKE, RSI_OVERBOUGHT, RSI_OVERSOLD,
    PCR_EXTREME_BULLISH, PCR_EXTREME_BEARISH,
)

IST = pytz.timezone("Asia/Kolkata")


def generate_realtime_alerts(
    current_position: str = "NONE",  # "BUY CE", "BUY PE", or "NONE"
    df: pd.DataFrame = None,
    prev_df: pd.DataFrame = None,
    vix_current: float = 0,
    vix_prev: float = 0,
    pcr_current: float = 0,
    news_headlines: List[dict] = None,
    cpr_levels: dict = None,
    oi_support: List[float] = None,
    oi_resistance: List[float] = None,
) -> List[Dict]:
    """
    Generate real-time alerts based on current market conditions.

    Returns list of alert dicts sorted by severity:
        [{severity, type, message, action, timestamp}, ...]
    """
    alerts = []
    now = datetime.now(IST)

    if df is None or df.empty or len(df) < 3:
        return alerts

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])
    is_long = "CE" in current_position.upper()
    is_short = "PE" in current_position.upper()
    has_position = is_long or is_short

    # ═══════════════════════════════════════════════════════════
    #  🔴 CRITICAL ALERTS — EXIT IMMEDIATELY
    # ═══════════════════════════════════════════════════════════

    # 1. SUPERTREND FLIP
    st_col = "STd_5_1.5"
    if st_col in df.columns:
        curr_dir = int(last[st_col])
        prev_dir = int(prev[st_col])

        if curr_dir != prev_dir:
            if has_position:
                if (is_long and curr_dir < 0) or (is_short and curr_dir > 0):
                    alerts.append({
                        "severity": "CRITICAL",
                        "type": "SUPERTREND FLIP",
                        "emoji": "🔴",
                        "message": (
                            f"Supertrend FLIPPED {'BEARISH' if curr_dir < 0 else 'BULLISH'} — "
                            f"This is AGAINST your {current_position} position!"
                        ),
                        "action": f"EXIT {current_position} NOW. Book {'loss' if True else 'profit'} immediately.",
                        "timestamp": now.strftime("%H:%M:%S"),
                    })
                else:
                    alerts.append({
                        "severity": "HIGH",
                        "type": "SUPERTREND CONFIRM",
                        "emoji": "🟢",
                        "message": f"Supertrend confirmed {'BULLISH' if curr_dir > 0 else 'BEARISH'} — supports your position",
                        "action": "HOLD. Trail your stop-loss tighter.",
                        "timestamp": now.strftime("%H:%M:%S"),
                    })
            else:
                alerts.append({
                    "severity": "HIGH",
                    "type": "SUPERTREND FLIP",
                    "emoji": "⚡",
                    "message": f"Supertrend just flipped {'BULLISH' if curr_dir > 0 else 'BEARISH'}!",
                    "action": f"New trend starting. Consider {'BUY CE' if curr_dir > 0 else 'BUY PE'}.",
                    "timestamp": now.strftime("%H:%M:%S"),
                })

    # 2. VIX SPIKE
    if vix_current > 0 and vix_prev > 0:
        vix_change_pct = ((vix_current - vix_prev) / vix_prev) * 100
        if vix_change_pct >= VIX_INTRADAY_SPIKE:
            alerts.append({
                "severity": "CRITICAL",
                "type": "VIX SPIKE",
                "emoji": "🚨",
                "message": f"India VIX SPIKED {vix_change_pct:+.1f}% to {vix_current:.2f} — Major fear event!",
                "action": (
                    "TIGHTEN ALL STOP-LOSSES to 50% of original. "
                    "If holding option BUYS, consider booking 50% profit NOW. "
                    "BIG move incoming — could be either direction."
                ),
                "timestamp": now.strftime("%H:%M:%S"),
            })

    # 3. BREAKING NEWS
    if news_headlines:
        for headline in news_headlines[:5]:
            title_lower = headline.get("title", "").lower()
            sentiment = headline.get("sentiment", 0)

            # Check for breaking/crisis keywords
            has_breaking = any(kw in title_lower for kw in BREAKING_NEWS_KEYWORDS)

            if has_breaking or sentiment < -0.6:
                alerts.append({
                    "severity": "CRITICAL",
                    "type": "BREAKING NEWS",
                    "emoji": "📰🔴",
                    "message": f"NEGATIVE NEWS: {headline.get('title', '')[:100]}",
                    "action": (
                        "Review this news immediately. "
                        "If it affects your sector, EXIT and go to cash. "
                        "Wait for clarity before re-entering."
                    ),
                    "timestamp": now.strftime("%H:%M:%S"),
                })
                break  # Only one breaking news alert

    # 4. KEY SUPPORT/RESISTANCE BREAK
    if oi_support and is_long:
        nearest_support = max(s for s in oi_support if s < price) if any(s < price for s in oi_support) else None
        if nearest_support and price < nearest_support:
            alerts.append({
                "severity": "CRITICAL",
                "type": "SUPPORT BROKEN",
                "emoji": "🔴",
                "message": f"Price ({price:.0f}) broke below OI support {nearest_support:.0f}!",
                "action": "EXIT LONG (CE) positions. Support broken = further downside likely.",
                "timestamp": now.strftime("%H:%M:%S"),
            })

    if oi_resistance and is_short:
        nearest_resistance = min(r for r in oi_resistance if r > price) if any(r > price for r in oi_resistance) else None
        if nearest_resistance and price > nearest_resistance:
            alerts.append({
                "severity": "CRITICAL",
                "type": "RESISTANCE BROKEN",
                "emoji": "🔴",
                "message": f"Price ({price:.0f}) broke above OI resistance {nearest_resistance:.0f}!",
                "action": "EXIT SHORT (PE) positions. Resistance broken = further upside likely.",
                "timestamp": now.strftime("%H:%M:%S"),
            })

    # ═══════════════════════════════════════════════════════════
    #  🟡 HIGH ALERTS — TIGHTEN STOP-LOSS
    # ═══════════════════════════════════════════════════════════

    # 5. EMA CROSSOVER AGAINST POSITION
    if "EMA_9" in df.columns and "EMA_21" in df.columns:
        ema9 = float(last["EMA_9"])
        ema21 = float(last["EMA_21"])
        prev_ema9 = float(prev["EMA_9"])
        prev_ema21 = float(prev["EMA_21"])

        ema_crossed_bearish = ema9 < ema21 and prev_ema9 >= prev_ema21
        ema_crossed_bullish = ema9 > ema21 and prev_ema9 <= prev_ema21

        if has_position:
            if (is_long and ema_crossed_bearish) or (is_short and ema_crossed_bullish):
                alerts.append({
                    "severity": "HIGH",
                    "type": "EMA CROSS AGAINST",
                    "emoji": "⚠️",
                    "message": f"EMA 9/21 crossed {'BEARISH' if ema_crossed_bearish else 'BULLISH'} — against your position!",
                    "action": "Tighten SL to breakeven. Book 50% profit if in profit. Watch for 2 more candles.",
                    "timestamp": now.strftime("%H:%M:%S"),
                })

    # 6. VWAP BREAK
    if "VWAP" in df.columns and has_position:
        vwap = float(last["VWAP"])
        prev_vwap_side = float(prev["Close"]) > float(prev["VWAP"]) if "VWAP" in prev.index else None

        if prev_vwap_side is not None:
            now_above_vwap = price > vwap
            vwap_broken_down = prev_vwap_side and not now_above_vwap
            vwap_broken_up = not prev_vwap_side and now_above_vwap

            if (is_long and vwap_broken_down) or (is_short and vwap_broken_up):
                alerts.append({
                    "severity": "HIGH",
                    "type": "VWAP BROKEN",
                    "emoji": "⚠️",
                    "message": f"Price broke {'below' if vwap_broken_down else 'above'} VWAP ({vwap:.0f}) — institutional flow shifted!",
                    "action": "Move SL to VWAP level. If price sustains 2 candles below/above VWAP, EXIT.",
                    "timestamp": now.strftime("%H:%M:%S"),
                })

    # 7. ATR EXPANSION (Volatility Surge)
    if "ATR_7" in df.columns and "ATR_14" in df.columns:
        atr7 = float(last["ATR_7"]) if not pd.isna(last["ATR_7"]) else 0
        atr14 = float(last["ATR_14"]) if not pd.isna(last["ATR_14"]) else 0

        if atr14 > 0 and atr7 > atr14 * 2.0:
            alerts.append({
                "severity": "HIGH",
                "type": "VOLATILITY SURGE",
                "emoji": "🌊",
                "message": f"ATR expanded 2x normal — volatility surge in progress!",
                "action": "WIDEN your SL slightly (ATR-based). Book partial profits. Expect wild swings.",
                "timestamp": now.strftime("%H:%M:%S"),
            })

    # 8. CLOSING TIME ALERT
    if now.hour == 15 and now.minute >= 0:
        alerts.append({
            "severity": "HIGH",
            "type": "CLOSING TIME",
            "emoji": "⏰",
            "message": "Market closes in 30 minutes! Square off all INTRADAY positions.",
            "action": "EXIT all intraday trades. Only hold BTST/Swing positions overnight.",
            "timestamp": now.strftime("%H:%M:%S"),
        })
    elif now.hour == 14 and now.minute >= 45:
        alerts.append({
            "severity": "MEDIUM",
            "type": "CLOSING SOON",
            "emoji": "⏰",
            "message": "45 minutes to market close. Start planning exit.",
            "action": "Set limit orders for exit. Don't wait for last-minute rush.",
            "timestamp": now.strftime("%H:%M:%S"),
        })

    # ═══════════════════════════════════════════════════════════
    #  🔵 MEDIUM ALERTS — MONITOR CLOSELY
    # ═══════════════════════════════════════════════════════════

    # 9. RSI EXTREME
    if "RSI_7" in df.columns:
        rsi7 = float(last["RSI_7"]) if not pd.isna(last["RSI_7"]) else 50
        if rsi7 > 75:
            alerts.append({
                "severity": "MEDIUM",
                "type": "RSI OVERBOUGHT",
                "emoji": "📊",
                "message": f"RSI(7) at {rsi7:.1f} — EXTREMELY OVERBOUGHT",
                "action": "If holding CE, book 50% profit. Reversal probability increasing." if is_long else "Watch for PE buying opportunity.",
                "timestamp": now.strftime("%H:%M:%S"),
            })
        elif rsi7 < 25:
            alerts.append({
                "severity": "MEDIUM",
                "type": "RSI OVERSOLD",
                "emoji": "📊",
                "message": f"RSI(7) at {rsi7:.1f} — EXTREMELY OVERSOLD",
                "action": "If holding PE, book 50% profit. Bounce probability increasing." if is_short else "Watch for CE buying opportunity.",
                "timestamp": now.strftime("%H:%M:%S"),
            })

    # 10. PCR EXTREME
    if pcr_current > 0:
        if pcr_current > 1.5:
            alerts.append({
                "severity": "MEDIUM",
                "type": "PCR EXTREME HIGH",
                "emoji": "📈",
                "message": f"PCR at {pcr_current:.2f} — EXTREME put writing = very bullish",
                "action": "Strongly supports CE positions. Avoid PE buying.",
                "timestamp": now.strftime("%H:%M:%S"),
            })
        elif pcr_current < 0.4:
            alerts.append({
                "severity": "MEDIUM",
                "type": "PCR EXTREME LOW",
                "emoji": "📉",
                "message": f"PCR at {pcr_current:.2f} — EXTREME call writing = very bearish",
                "action": "Strongly supports PE positions. Avoid CE buying.",
                "timestamp": now.strftime("%H:%M:%S"),
            })

    # 11. HEIKIN ASHI REVERSAL
    if "HA_Bullish" in df.columns and len(df) >= 4:
        ha_now = bool(last["HA_Bullish"])
        ha_prev = bool(prev["HA_Bullish"])
        if ha_now != ha_prev:
            new_dir = "BULLISH" if ha_now else "BEARISH"
            alerts.append({
                "severity": "MEDIUM",
                "type": "HA REVERSAL",
                "emoji": "🕯️",
                "message": f"Heikin Ashi candle turned {new_dir} — early trend change signal",
                "action": "Wait for 1 more candle confirmation. Prepare to adjust position.",
                "timestamp": now.strftime("%H:%M:%S"),
            })

    # 12. CPR BREAK
    if cpr_levels and has_position:
        tc = cpr_levels.get("tc", 0)
        bc = cpr_levels.get("bc", 0)
        if bc > 0 and tc > 0:
            prev_price = float(prev["Close"])
            # Price broke above CPR (bullish) or below CPR (bearish)
            if prev_price >= bc and prev_price <= tc and price > tc:
                alerts.append({
                    "severity": "HIGH" if is_short else "MEDIUM",
                    "type": "CPR BREAKOUT UP",
                    "emoji": "📈",
                    "message": f"Price broke ABOVE CPR range ({tc:.0f}) — Trending day bullish!",
                    "action": "Bullish momentum confirmed. Hold CE. Exit PE." if is_long else "Consider exiting PE position.",
                    "timestamp": now.strftime("%H:%M:%S"),
                })
            elif prev_price >= bc and prev_price <= tc and price < bc:
                alerts.append({
                    "severity": "HIGH" if is_long else "MEDIUM",
                    "type": "CPR BREAKDOWN",
                    "emoji": "📉",
                    "message": f"Price broke BELOW CPR range ({bc:.0f}) — Trending day bearish!",
                    "action": "Bearish momentum confirmed. Exit CE. Hold PE." if is_short else "Consider exiting CE position.",
                    "timestamp": now.strftime("%H:%M:%S"),
                })

    # ── Sort by severity ─────────────────────────────────────
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda x: severity_order.get(x["severity"], 99))

    return alerts


def get_exit_recommendation(
    alerts: List[Dict],
    current_position: str,
    entry_premium: float = 0,
    current_premium: float = 0,
) -> Dict:
    """
    Based on alerts, give a clear EXIT or HOLD recommendation.
    """
    if not current_position or current_position == "NONE":
        return {
            "action": "NO POSITION",
            "message": "No active position to monitor.",
            "urgency": "NONE",
        }

    critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    high_count = sum(1 for a in alerts if a["severity"] == "HIGH")

    # P&L if known
    pnl_pct = 0
    if entry_premium > 0 and current_premium > 0:
        pnl_pct = ((current_premium - entry_premium) / entry_premium) * 100

    if critical_count >= 2:
        return {
            "action": "EXIT IMMEDIATELY",
            "message": f"Multiple CRITICAL alerts ({critical_count}). Exit all positions NOW.",
            "urgency": "CRITICAL",
            "color": "#ff1744",
        }
    elif critical_count == 1:
        return {
            "action": "EXIT / TIGHT SL",
            "message": "1 CRITICAL alert. Exit 50% now. Set tight SL on rest.",
            "urgency": "HIGH",
            "color": "#ff5722",
        }
    elif high_count >= 3:
        return {
            "action": "TIGHTEN SL",
            "message": f"Multiple HIGH alerts ({high_count}). Move SL to breakeven. Book partial profit.",
            "urgency": "HIGH",
            "color": "#ff9800",
        }
    elif high_count >= 1:
        return {
            "action": "WATCH CLOSELY",
            "message": "Some warning signs. Monitor next 2-3 candles before deciding.",
            "urgency": "MEDIUM",
            "color": "#ffc107",
        }
    else:
        if pnl_pct > 50:
            return {
                "action": "BOOK PARTIAL",
                "message": f"No warnings. Profit at {pnl_pct:.0f}%. Book 50% and trail SL.",
                "urgency": "LOW",
                "color": "#00e676",
            }
        return {
            "action": "HOLD",
            "message": "No significant warnings. Continue holding with original SL.",
            "urgency": "LOW",
            "color": "#00e676",
        }
