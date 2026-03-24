"""
╔══════════════════════════════════════════════════════════════╗
║  NIFTY SIGNALS PRO v2.0 — Configuration & Constants         ║
║  Enhanced: Global Indices + VIX + BTST + Real-Time Alerts   ║
╚══════════════════════════════════════════════════════════════╝
"""

NIFTY_LOT_SIZE = 65
BANKNIFTY_LOT_SIZE = 30

# ═══════════════════════════ ALL TICKERS ═════════════════════
TICKERS = {
    "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "INDIAVIX": "^INDIAVIX",
    "NIFTY_IT": "^CNXIT", "NIFTY_FIN": "^CNXFIN",
    "SP500": "^GSPC", "SP500_FUT": "ES=F", "DOW": "^DJI", "DOW_FUT": "YM=F",
    "NASDAQ": "^IXIC", "NASDAQ_FUT": "NQ=F", "VIX_US": "^VIX",
    "FTSE100": "^FTSE", "DAX": "^GDAXI", "CAC40": "^FCHI",
    "NIKKEI": "^N225", "HANGSENG": "^HSI", "SHANGHAI": "000001.SS",
    "KOSPI": "^KS11", "STRAITS": "^STI",
    "CRUDE": "CL=F", "BRENT": "BZ=F", "GOLD": "GC=F", "SILVER": "SI=F",
    "DXY": "DX-Y.NYB", "USDINR": "INR=X",
    "US10Y": "^TNX", "BITCOIN": "BTC-USD",
}

# ═══════════════ GLOBAL MARKET GROUPS FOR ANALYSIS ═══════════
GLOBAL_SIGNAL_MARKETS = {
    "US Futures": {"tickers": ["SP500_FUT", "DOW_FUT", "NASDAQ_FUT"], "weight": 0.30},
    "Asian Markets": {"tickers": ["NIKKEI", "HANGSENG", "SHANGHAI", "STRAITS"], "weight": 0.20},
    "European Markets": {"tickers": ["FTSE100", "DAX", "CAC40"], "weight": 0.10},
    "Commodities": {"tickers": ["CRUDE", "GOLD"], "weight": 0.15},
    "Currency": {"tickers": ["DXY", "USDINR"], "weight": 0.10},
    "Volatility": {"tickers": ["VIX_US", "INDIAVIX"], "weight": 0.15},
}

CORRELATION_DIRECTION = {
    "SP500_FUT": +1, "DOW_FUT": +1, "NASDAQ_FUT": +1,
    "NIKKEI": +1, "HANGSENG": +1, "SHANGHAI": +1, "STRAITS": +1,
    "FTSE100": +1, "DAX": +1, "CAC40": +1,
    "CRUDE": -1, "BRENT": -1, "GOLD": -0.3,
    "DXY": -1, "USDINR": -1,
    "VIX_US": -1, "INDIAVIX": -1, "US10Y": -0.5, "BITCOIN": +0.3,
}

# ═══════════════════ VIX ZONES ═══════════════════════════════
VIX_ZONES = {
    "EXTREME_LOW": {"range": (0, 11), "action": "BUY OPTIONS", "color": "#00e676"},
    "LOW": {"range": (11, 14), "action": "BUY OPTIONS", "color": "#66bb6a"},
    "NORMAL": {"range": (14, 18), "action": "EITHER", "color": "#ffc107"},
    "ELEVATED": {"range": (18, 22), "action": "SELL PREMIUM", "color": "#ff9800"},
    "HIGH": {"range": (22, 28), "action": "SELL + HEDGE", "color": "#f44336"},
    "EXTREME": {"range": (28, 100), "action": "CASH ONLY", "color": "#b71c1c"},
}
VIX_LOW = 12; VIX_NORMAL_LOW = 15; VIX_NORMAL_HIGH = 20; VIX_HIGH = 25; VIX_EXTREME = 30
VIX_SPIKE_PCT = 10; VIX_INTRADAY_SPIKE = 5

# ═══════════════════ BTST PREDICTOR WEIGHTS ══════════════════
BTST_WEIGHTS = {
    "us_futures": 0.25, "asian_close": 0.15, "fii_dii_flow": 0.15,
    "vix_trend": 0.10, "technical_trend": 0.15, "closing_pattern": 0.10,
    "oi_pcr_eod": 0.10,
}
GAP_STRONG_CONFIDENCE = 0.70; GAP_MODERATE_CONFIDENCE = 0.45

# ═══════════════════ ALERT CONFIG ════════════════════════════
ALERT_CONFIG = {
    "supertrend_flip": True, "ema_cross": True, "rsi_extreme": True,
    "vwap_break": True, "vix_spike": True, "atr_expansion": True,
    "support_break": True, "resistance_break": True,
    "oi_unwinding": True, "pcr_extreme": True, "breaking_news": True,
    "closing_time": True,
}
BREAKING_NEWS_KEYWORDS = [
    "crash", "circuit", "war", "attack", "rbi emergency", "sebi ban",
    "default", "crisis", "flash crash", "rate hike surprise",
    "sanctions", "pandemic", "lockdown",
]

# ═══════════════════ NSE URLs ════════════════════════════════
NSE_BASE = "https://www.nseindia.com"
NSE_OPTION_CHAIN_URL = NSE_BASE + "/api/option-chain-indices?symbol={symbol}"
NSE_INDEX_URL = NSE_BASE + "/api/allIndices"
NSE_FII_DII_URL = NSE_BASE + "/api/fiidiiTradeReact"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
}

# ═══════════════════ INDICATOR PARAMS ════════════════════════
ST_FAST = {"length": 5, "multiplier": 1.5}
ST_MEDIUM = {"length": 10, "multiplier": 3.0}
ST_SLOW = {"length": 14, "multiplier": 4.0}
RSI_FAST = 7; RSI_STANDARD = 14; RSI_OVERBOUGHT = 70; RSI_OVERSOLD = 30
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
EMA_SCALP = [5, 8, 13]; EMA_INTRADAY = [9, 21]; EMA_SWING = [20, 50]; EMA_POSITIONAL = [50, 200]
BB_LENGTH = 20; BB_STD = 2.0; ADX_LENGTH = 14; ADX_STRONG_TREND = 25; ADX_WEAK_TREND = 15

# ═══════════════════ STRATEGY WEIGHTS (v2 with global+vix) ══
STRATEGY_WEIGHTS = {
    "supertrend_rsi": 0.18, "ema_crossover": 0.10, "macd": 0.10,
    "vwap": 0.07, "bollinger": 0.07, "adx_trend": 0.05,
    "pcr_sentiment": 0.07, "oi_analysis": 0.06, "news_sentiment": 0.04,
    "global_cues": 0.14, "vix_analysis": 0.12,
}

STRONG_BUY_THRESHOLD = 0.45; BUY_THRESHOLD = 0.25
NEUTRAL_LOW = -0.25; STRONG_SELL_THRESHOLD = -0.45

# ═══════════════════ RISK MANAGEMENT ═════════════════════════
MAX_RISK_PER_TRADE_PCT = 2.0; DEFAULT_CAPITAL = 10000; MAX_DAILY_LOSS_PCT = 5.0
RISK_REWARD_MIN = 1.5; SL_OPTION_BUY_PCT = 30; SL_OPTION_SELL_MULT = 1.5; SL_ATR_MULTIPLIER = 1.5
PCR_EXTREME_BULLISH = 1.3; PCR_BULLISH = 1.0; PCR_BEARISH = 0.7; PCR_EXTREME_BEARISH = 0.5

# ═══════════════════ NEWS & FEEDS ════════════════════════════
RSS_FEEDS = {
    "ET Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "LiveMint": "https://www.livemint.com/rss/markets",
    "MoneyControl": "https://www.moneycontrol.com/rss/marketreports.xml",
    "BS Markets": "https://www.business-standard.com/rss/markets-106.rss",
    "Google Nifty": "https://news.google.com/rss/search?q=nifty+OR+banknifty+OR+sensex&hl=en-IN&gl=IN&ceid=IN:en",
}

TIMEFRAMES = {
    "Scalping": {"interval": "5m", "period": "5d", "label": "5-Min"},
    "Intraday": {"interval": "15m", "period": "10d", "label": "15-Min"},
    "Swing": {"interval": "1h", "period": "1mo", "label": "1-Hour"},
    "Positional": {"interval": "1d", "period": "6mo", "label": "Daily"},
}
STRIKE_STEP_NIFTY = 50; STRIKE_STEP_BANKNIFTY = 100
PREMIUM_TARGET_RANGE = (150, 300)
