"""
╔══════════════════════════════════════════════════════════════╗
║  SENTIMENT ANALYZER v3.0 — LLM-First + VADER Fallback      ║
║                                                              ║
║  Priority chain:                                             ║
║  1. NVIDIA NIM LLM (Llama 3.3 70B) → ~85% accuracy         ║
║  2. VADER + Financial Lexicon       → ~55% accuracy          ║
║                                                              ║
║  LLM adds:                                                   ║
║  • Context understanding (RBI holds ≠ neutral if cut was    ║
║    expected)                                                 ║
║  • Impact severity scoring (high/medium/low)                ║
║  • Affected instrument detection (NIFTY vs BANKNIFTY)       ║
║  • One-line reasoning for each headline                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import feedparser
import logging
from typing import Dict, List, Tuple

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

from config import RSS_FEEDS

logger = logging.getLogger(__name__)

# ═══════════════════ FINANCIAL LEXICON (VADER boost) ═════════
FINANCIAL_LEXICON = {
    "rally": 2.5, "surge": 2.5, "bull": 2.0, "bullish": 2.5,
    "breakout": 2.0, "all-time high": 3.0, "record high": 3.0,
    "uptick": 1.5, "gains": 1.5, "soars": 2.5, "jumps": 2.0,
    "buying": 1.0, "fii buying": 2.5, "rate cut": 2.0,
    "rbi cuts": 2.0, "dovish": 1.5, "upgrade": 1.5,
    "profit growth": 2.0, "beats estimates": 2.0,
    "crash": -3.0, "sell-off": -2.5, "selloff": -2.5, "bear": -2.0,
    "bearish": -2.5, "breakdown": -2.0, "plunge": -2.5,
    "tumble": -2.0, "slump": -2.0, "panic": -2.5, "correction": -1.5,
    "fii selling": -2.5, "rate hike": -2.0, "hawkish": -1.5,
    "inflation": -1.0, "recession": -2.5, "downgrade": -1.5,
    "miss estimates": -2.0, "profit warning": -2.0,
    "volatile": -0.5, "uncertainty": -1.0,
    "nifty": 0.0, "sensex": 0.0, "banknifty": 0.0,
    "sebi": 0.0, "rbi": 0.0, "budget": 0.5,
}


class VADERAnalyzer:
    """Fallback VADER analyzer with financial lexicon."""
    def __init__(self):
        if HAS_VADER:
            self.analyzer = SentimentIntensityAnalyzer()
            self.analyzer.lexicon.update(FINANCIAL_LEXICON)
        else:
            self.analyzer = None

    def score(self, text: str) -> float:
        if not self.analyzer: return 0.0
        return self.analyzer.polarity_scores(text.lower())["compound"]

_vader = VADERAnalyzer()


# ═══════════════════ RSS FEED FETCHER ════════════════════════
def fetch_news_headlines(max_per_feed: int = 10) -> List[Dict]:
    """Fetch headlines from all RSS feeds (no scoring yet)."""
    all_headlines = []
    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                if not title: continue
                all_headlines.append({
                    "title": title,
                    "source": source_name,
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                })
        except Exception as e:
            logger.warning(f"RSS failed for {source_name}: {e}")
    return all_headlines


# ═══════════════════ LLM-POWERED SENTIMENT ═══════════════════
def calculate_news_sentiment_llm(nim_client=None) -> Tuple[float, str, List[Dict]]:
    """
    Calculate sentiment using LLM (primary) or VADER (fallback).

    Returns: (score, label, headlines_with_scores)
    """
    raw_headlines = fetch_news_headlines()
    if not raw_headlines:
        return 0.0, "NO DATA", []

    # ── Try LLM first ────────────────────────────────────────
    llm_used = False
    if nim_client and nim_client.available:
        try:
            from llm_engine import llm_score_headlines
            titles = [h["title"] for h in raw_headlines]
            llm_results = llm_score_headlines(nim_client, titles, batch_size=5)

            if llm_results:
                for i, h in enumerate(raw_headlines):
                    if i < len(llm_results):
                        lr = llm_results[i]
                        h["sentiment"] = lr.get("score", 0.0)
                        h["llm_label"] = lr.get("sentiment", "neutral")
                        h["confidence"] = lr.get("confidence", 0.0)
                        h["impact"] = lr.get("impact", "low")
                        h["affected"] = lr.get("affected", [])
                        h["reasoning"] = lr.get("reasoning", "")
                        h["engine"] = "LLM"
                    else:
                        h["sentiment"] = _vader.score(h["title"])
                        h["engine"] = "VADER"
                llm_used = True
        except Exception as e:
            logger.warning(f"LLM sentiment failed, falling back to VADER: {e}")

    # ── Fallback to VADER ────────────────────────────────────
    if not llm_used:
        for h in raw_headlines:
            h["sentiment"] = _vader.score(h["title"])
            h["llm_label"] = ""
            h["confidence"] = 0.0
            h["impact"] = "low"
            h["affected"] = []
            h["reasoning"] = ""
            h["engine"] = "VADER"

    # Sort by impact strength
    raw_headlines.sort(key=lambda x: abs(x["sentiment"]), reverse=True)

    # ── Aggregate score ──────────────────────────────────────
    sentiments = [h["sentiment"] for h in raw_headlines]
    if not sentiments:
        return 0.0, "NO DATA", raw_headlines

    weights = [1.0 / (i + 1) for i in range(len(sentiments))]
    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(sentiments, weights)) / total_weight

    if weighted_score > 0.3: label = "VERY BULLISH"
    elif weighted_score > 0.1: label = "BULLISH"
    elif weighted_score < -0.3: label = "VERY BEARISH"
    elif weighted_score < -0.1: label = "BEARISH"
    else: label = "NEUTRAL"

    engine_tag = " (AI)" if llm_used else " (VADER)"
    return round(weighted_score, 3), label + engine_tag, raw_headlines


# ═══════════════════ BACKWARD COMPATIBLE ═════════════════════
def calculate_news_sentiment() -> Tuple[float, str, List[Dict]]:
    """Original function — uses VADER only (backward compatible)."""
    return calculate_news_sentiment_llm(nim_client=None)


# ═══════════════════ FILTER ══════════════════════════════════
def filter_relevant_headlines(headlines: List[Dict], keywords=None) -> List[Dict]:
    if not keywords:
        keywords = ["nifty", "banknifty", "bank nifty", "sensex", "market",
                     "rbi", "fii", "dii", "option", "f&o", "derivative"]
    return [h for h in headlines if any(kw in h["title"].lower() for kw in keywords)]
