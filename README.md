# 📈 NiftySignals Pro v2.0 — F&O Trading Signal System

> **Free, Open Source, High Win-Rate Trading Signal System for Indian Market F&O (Nifty50 & BankNifty)**

A professional-grade trading signal dashboard that combines **11 signal sources** — technical indicators, option chain analysis (OI, PCR, Max Pain), **Global Markets (US/Europe/Asia)**, **India VIX deep analysis**, news sentiment, **BTST gap-up/gap-down predictor**, and **real-time exit alerts** — into clear **BUY CE / BUY PE** recommendations with exact strike, SL, and targets.

### What's New in v2.0
- 🌍 **Global Markets Heatmap** — US Futures, Asian, European indices with Nifty correlation scoring
- 📊 **India VIX Deep Analysis** — Zone detection, spike alerts, strategy recommendation
- 🔮 **BTST Predictor** — Predicts tomorrow's gap-up/gap-down using 7 weighted factors
- 🚨 **Real-Time Exit Alerts** — Tells you when to exit: Supertrend flips, VIX spikes, VWAP breaks, breaking news
- 🧠 **Signal Explainer** — Full transparency on how every signal is calculated

---

## ⚡ Quick Start (3 commands)

```bash
# 1. Clone/download this folder
cd nifty_signals

# 2. Make the launch script executable
chmod +x run.sh

# 3. Run! (installs dependencies automatically)
./run.sh
```

The dashboard will open at **http://localhost:8501**

### Manual Setup (if run.sh doesn't work)

```bash
cd nifty_signals
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 What You Get

### Live Trading Signals
- **Clear BUY CE / BUY PE / NO TRADE** action signals
- **Exact strike price** to buy on Groww/Zerodha
- **Entry price** (estimated premium)
- **Stop-Loss** (both premium-based and underlying-based)
- **Target 1 & Target 2** for partial profit booking
- **Position sizing** — how many lots to trade based on your capital
- **Risk-reward ratio** calculated automatically
- **Step-by-step Groww app instructions**

### Technical Analysis Engine
| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| Supertrend | (5,1.5), (10,3), (14,4) | Trend direction — ~80% accuracy with RSI combo |
| RSI | 7 and 14 period | Overbought/oversold + Supertrend confirmation |
| MACD | 12/26/9 | Momentum + crossover signals |
| EMA Crossover | 9/21 | Short-term trend changes |
| VWAP | Intraday reset | Institutional buying/selling bias |
| Bollinger Bands | 20, 2σ | Volatility + mean reversion |
| ADX | 14 period | Trend strength filter |
| CPR | Daily pivots | Key support/resistance levels |
| ORB | 30-min range | Breakout trading levels |
| Heikin Ashi | Smoothed candles | Trend confirmation |
| Stochastic RSI | 14,3,3 | Momentum within trends |

### Options Analysis
- **Live Option Chain** from NSE (via web scraping)
- **PCR (Put-Call Ratio)** — OI-based, Volume-based, Change-based
- **Max Pain** calculation with visualization
- **OI-based Support & Resistance** levels
- **OI Buildup Analysis** (Long/Short buildup detection)

### Sentiment Analysis
- **RSS feed aggregation** from 5 Indian financial news sources
- **VADER sentiment scoring** enhanced with financial lexicon
- **Aggregate market mood** scoring

### Global Cues
- S&P 500 Futures, Dow Futures
- Dollar Index (DXY), Crude Oil, Gold
- Pre-market sentiment for gap prediction

---

## 📁 Project Structure

```
nifty_signals/
├── app.py                # Main Streamlit dashboard (run this)
├── config.py             # All tunable parameters & constants
├── data_fetcher.py       # Data retrieval (yfinance + NSE scraping)
├── indicators.py         # Technical indicator calculations
├── signal_engine.py      # Confluence scoring & trade recommendations
├── sentiment.py          # News sentiment analysis (VADER + RSS)
├── global_analysis.py    # NEW: Global indices + VIX analysis engine
├── btst_predictor.py     # NEW: BTST gap-up/gap-down predictor
├── realtime_alerts.py    # NEW: Real-time exit alerts & trend reversal
├── requirements.txt      # Python dependencies
├── run.sh                # One-click setup & launch script
└── README.md             # This file
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Capital & Risk
DEFAULT_CAPITAL = 200000        # Your trading capital in ₹
MAX_RISK_PER_TRADE_PCT = 2.0    # Max risk per trade (%)

# Lot Sizes (Jan 2026 onwards)
NIFTY_LOT_SIZE = 75
BANKNIFTY_LOT_SIZE = 30

# Signal Thresholds
STRONG_BUY_THRESHOLD = 0.65     # Score needed for STRONG BUY
BUY_THRESHOLD = 0.45            # Score needed for BUY

# Supertrend Parameters
ST_FAST = {"length": 5, "multiplier": 1.5}     # Scalping
ST_MEDIUM = {"length": 10, "multiplier": 3.0}   # Standard
ST_SLOW = {"length": 14, "multiplier": 4.0}     # Positional
```

---

## 📱 How to Trade Using These Signals

### For Groww App Users:

1. **Open the dashboard** during market hours (9:15 AM - 3:30 PM IST)
2. **Select your index** (NIFTY50 or BANKNIFTY) in the sidebar
3. **Select timeframe** (Intraday recommended for beginners)
4. **Check the main signal card** at the top
5. **If BUY CE/PE signal appears:**
   - Open Groww → F&O → Search for NIFTY/BANKNIFTY
   - Select Options → Choose nearest weekly expiry
   - Select the strike shown on the dashboard
   - BUY at the shown entry price
   - Immediately set SL at the shown Stop-Loss level
6. **Manage the trade:**
   - Book 50% profit at Target 1
   - Move SL to entry (breakeven) for remaining 50%
   - Book remaining at Target 2
7. **If NO TRADE signal** — wait and check back in 15-30 minutes

### 🚨 Mandatory Rules (Never Break These)

- **Never risk more than 2% of capital** on a single trade
- **Always place stop-loss** before looking at targets
- **Exit ALL intraday positions by 3:15 PM** — never carry overnight
- **Never add to a losing position** (no averaging down)
- **Take the signal as shown** — don't second-guess, don't modify
- **Paper trade for 1 month** before using real money
- **If daily loss reaches 5%** — stop trading for the day

---

## 📊 Strategy Win Rates (Backtested)

| Strategy | Win Rate | Source |
|----------|----------|--------|
| Supertrend(5,1.5) + RSI(7) | ~80% | Investar India backtest |
| VWAP Bounce Scalp | ~78% | TradingView community data |
| MACD + RSI combo | ~73% | Quantified Strategies |
| ORB (30-min) + RSI exit | ~71% | DailyBulls backtest (42 trades) |
| 9:20 Short Straddle | ~69% | AlgoTest 5-year backtest |
| Short Strangle (BankNifty) | ~68% | 2017-2020 backtest |
| Expiry Credit Spreads | 65-70% | Community consensus |

---

## 🛠️ Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Dashboard | Streamlit | Interactive web UI |
| Charts | Plotly | Candlestick & indicator charts |
| Price Data | yfinance | Free OHLCV data |
| Option Chain | requests + NSE API | Live option chain scraping |
| Indicators | Custom + pandas-ta | Technical analysis calculations |
| Sentiment | VADER + feedparser | News sentiment scoring |
| Data | pandas, numpy | Data processing |

---

## ⚠️ Disclaimer

This software is provided **for educational and research purposes only**. It is **NOT financial advice**.

- Trading F&O involves **significant risk of loss**
- Past performance and backtested results **do not guarantee** future results
- As per SEBI's 2023 study, **9 out of 10 individual F&O traders incur net losses**
- Always **paper trade first** before risking real money
- The developers are **not responsible** for any trading losses

**Use at your own risk. Always consult a qualified financial advisor.**

---

## 📜 License

MIT License — Free and Open Source. Use, modify, and distribute freely.
