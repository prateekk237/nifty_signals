"""
╔══════════════════════════════════════════════════════════════╗
║  LLM ENGINE — NVIDIA NIM Integration (Llama 3.3 70B)       ║
║                                                              ║
║  4 capabilities powered by LLM:                              ║
║  1. Financial News Sentiment (replaces VADER → 85%+ acc)    ║
║  2. Breaking News Interpreter (impact analysis)             ║
║  3. Trade Commentary (explains signals in plain English)    ║
║  4. BTST Narrative (gap prediction reasoning)               ║
║                                                              ║
║  Models:                                                     ║
║  • Primary:  meta/llama-3.3-70b-instruct (best accuracy)   ║
║  • Fallback: meta/llama-3.1-8b-instruct  (fastest)         ║
║  • Final:    VADER (offline fallback, no API needed)        ║
║                                                              ║
║  API: NVIDIA NIM — OpenAI-compatible format                  ║
║  Base URL: https://integrate.api.nvidia.com/v1               ║
║  Free tier: 40 requests/minute, all models accessible        ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Model Configuration ──────────────────────────────────────
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
PRIMARY_MODEL = "meta/llama-3.3-70b-instruct"
FAST_MODEL = "meta/llama-3.1-8b-instruct"
DEFAULT_TEMPERATURE = 0.1  # Low temp = consistent classification
DEFAULT_MAX_TOKENS = 512

# ── Try importing OpenAI client ──────────────────────────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai package not installed. Run: pip install openai")


# ═══════════════════════════════════════════════════════════════
#  NVIDIA NIM CLIENT
# ═══════════════════════════════════════════════════════════════
class NVIDIANimClient:
    """
    Client for NVIDIA NIM API with automatic fallback.

    Priority chain:
    1. Llama 3.3 70B (best accuracy)
    2. Llama 3.1 8B (fast fallback)
    3. None (caller falls back to VADER)
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.client = None
        self.available = False
        self._last_call_time = 0
        self._min_interval = 1.6  # ~40 req/min = 1 req per 1.5s
        self._call_count = 0
        self._error_count = 0

        if self.api_key and HAS_OPENAI:
            try:
                self.client = OpenAI(
                    base_url=NVIDIA_NIM_BASE_URL,
                    api_key=self.api_key,
                )
                self.available = True
                logger.info("NVIDIA NIM client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to init NVIDIA NIM client: {e}")
        elif not HAS_OPENAI:
            logger.warning("Install openai package: pip install openai")
        elif not self.api_key:
            logger.warning("Set NVIDIA_API_KEY environment variable or pass api_key")

    def _rate_limit(self):
        """Enforce rate limiting for free tier (40 req/min)."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Optional[str]:
        """
        Send a chat completion request to NVIDIA NIM.
        Returns the response text or None on failure.
        """
        if not self.available or not self.client:
            return None

        model = model or PRIMARY_MODEL
        self._rate_limit()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            self._call_count += 1
            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            self._error_count += 1
            error_msg = str(e)

            # If primary model fails, try fast model
            if model == PRIMARY_MODEL:
                logger.warning(f"Primary model failed ({error_msg}), trying fast model...")
                return self.chat(system_prompt, user_message, FAST_MODEL,
                                 temperature, max_tokens)

            logger.error(f"NVIDIA NIM call failed: {error_msg}")
            return None

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        model: str = None,
    ) -> Optional[dict]:
        """
        Chat and parse response as JSON.
        Handles markdown code blocks and invalid JSON gracefully.
        """
        raw = self.chat(system_prompt, user_message, model)
        if not raw:
            return None

        # Clean markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse JSON from LLM response: {cleaned[:200]}")
            return None

    @property
    def stats(self) -> dict:
        return {
            "available": self.available,
            "model": PRIMARY_MODEL,
            "fallback": FAST_MODEL,
            "calls": self._call_count,
            "errors": self._error_count,
        }


# ═══════════════════════════════════════════════════════════════
#  1. FINANCIAL SENTIMENT SCORING
# ═══════════════════════════════════════════════════════════════
SENTIMENT_SYSTEM_PROMPT = """You are an expert Indian stock market sentiment analyst specializing in NIFTY 50 and BankNifty F&O trading.

Analyze the given financial news headline and return ONLY a valid JSON object with these exact fields:
{
  "sentiment": "bullish" or "bearish" or "neutral",
  "score": float from -1.0 (extremely bearish) to +1.0 (extremely bullish),
  "confidence": float from 0.0 to 1.0,
  "impact": "high" or "medium" or "low",
  "affected": ["NIFTY", "BANKNIFTY"] or subset,
  "reasoning": "One sentence explaining why"
}

RULES:
- Consider Indian market context: RBI policy, SEBI regulations, FII/DII flows, monsoon, budget, elections
- "RBI holds rates" when cut was expected = bearish (disappointed expectations)
- "FII selling" = bearish for Nifty; "FII buying" = bullish
- Crude oil price rise = bearish for India (India imports 85% of oil)
- Dollar strength = bearish for Indian market (FII outflows)
- Rate cut = bullish; Rate hike = bearish
- Output ONLY the JSON. No markdown, no explanation outside JSON, no code blocks."""


def llm_score_headlines(
    client: NVIDIANimClient,
    headlines: List[str],
    batch_size: int = 5,
) -> List[Dict]:
    """
    Score multiple headlines using LLM in batches.
    Batching reduces API calls (5 headlines per call vs 1).
    """
    if not client.available:
        return []

    results = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        numbered = "\n".join([f"{j+1}. {h}" for j, h in enumerate(batch)])

        batch_prompt = f"""Analyze these {len(batch)} Indian financial news headlines.
Return a JSON array with one object per headline in the same order:

{numbered}

Return ONLY a valid JSON array like:
[{{"headline_num": 1, "sentiment": "bullish", "score": 0.6, "confidence": 0.8, "impact": "medium", "affected": ["NIFTY"], "reasoning": "..."}}, ...]"""

        result = client.chat_json(SENTIMENT_SYSTEM_PROMPT, batch_prompt)

        if result:
            # Handle both array and single object responses
            if isinstance(result, list):
                for j, item in enumerate(result):
                    if j < len(batch):
                        item["title"] = batch[j]
                        results.append(item)
            elif isinstance(result, dict):
                result["title"] = batch[0] if batch else ""
                results.append(result)
        else:
            # LLM failed for this batch — return empty scores
            for h in batch:
                results.append({
                    "title": h, "sentiment": "neutral", "score": 0.0,
                    "confidence": 0.0, "impact": "low", "affected": [],
                    "reasoning": "LLM analysis unavailable",
                })

    return results


def llm_score_single(client: NVIDIANimClient, headline: str) -> Optional[Dict]:
    """Score a single headline. Used for breaking news alerts."""
    if not client.available:
        return None
    return client.chat_json(SENTIMENT_SYSTEM_PROMPT, headline)


# ═══════════════════════════════════════════════════════════════
#  2. BREAKING NEWS INTERPRETER
# ═══════════════════════════════════════════════════════════════
BREAKING_NEWS_PROMPT = """You are an expert Indian F&O trading advisor. A breaking news event has occurred.

Analyze the news and return ONLY a valid JSON object:
{
  "severity": "critical" or "high" or "medium" or "low",
  "market_impact": "very_bearish" or "bearish" or "neutral" or "bullish" or "very_bullish",
  "nifty_impact_points": estimated Nifty point move (e.g., -200 or +150),
  "banknifty_impact_points": estimated BankNifty point move,
  "affected_sectors": ["banking", "IT", "pharma", etc.],
  "immediate_action": "What should a trader do RIGHT NOW",
  "option_advice": "Specific advice for F&O traders",
  "duration": "How long this impact will last (hours/days/weeks)",
  "similar_historical": "Brief mention of a similar past event and what happened"
}

Consider: FII reaction, RBI response probability, global contagion risk, sector-specific impact.
Output ONLY the JSON. No markdown."""


def interpret_breaking_news(client: NVIDIANimClient, news: str) -> Optional[Dict]:
    """Analyze breaking news for immediate trading impact."""
    if not client.available:
        return None
    return client.chat_json(BREAKING_NEWS_PROMPT, news)


# ═══════════════════════════════════════════════════════════════
#  3. TRADE COMMENTARY GENERATOR
# ═══════════════════════════════════════════════════════════════
TRADE_COMMENTARY_PROMPT = """You are a professional Indian F&O trading analyst writing a brief market commentary.

Given the current market data, write a 4-6 sentence actionable commentary. Be specific with numbers. Address:
1. What the indicators are saying (direction + strength)
2. What global markets suggest for today
3. The specific trade action and why
4. Key risk to watch out for

STYLE: Confident but cautious. Like a Bloomberg terminal note. Use ₹ for prices.
Write plain text only — no JSON, no bullet points, no headers."""


def generate_trade_commentary(
    client: NVIDIANimClient,
    symbol: str,
    price: float,
    signal: str,
    confluence: float,
    trade_action: str,
    strike: int,
    indicators: Dict,
    global_label: str,
    vix: float,
    pcr: float,
) -> str:
    """Generate LLM-powered trade commentary."""
    if not client.available:
        return ""

    # Build context message
    indicator_summary = []
    for name, data in indicators.items():
        indicator_summary.append(f"{name}: {data.get('label', 'N/A')} ({data.get('signal', 0):+.2f})")

    context = f"""MARKET DATA:
Symbol: {symbol} at ₹{price:,.2f}
Signal: {signal} (Confluence: {confluence:+.3f})
Trade: {trade_action} at strike {strike}
India VIX: {vix:.2f}
PCR: {pcr:.3f}
Global Markets: {global_label}

INDICATOR READINGS:
{chr(10).join(indicator_summary)}

Write a brief commentary for this trading setup."""

    result = client.chat(TRADE_COMMENTARY_PROMPT, context,
                         model=FAST_MODEL, max_tokens=300)
    return result or ""


# ═══════════════════════════════════════════════════════════════
#  4. BTST NARRATIVE GENERATOR
# ═══════════════════════════════════════════════════════════════
BTST_NARRATIVE_PROMPT = """You are an Indian market strategist explaining tomorrow's expected gap.

Given the BTST prediction data, write 3-4 sentences explaining:
1. WHY the gap-up or gap-down is expected (cite specific factors)
2. The STRENGTH of conviction
3. WHAT COULD GO WRONG (key risk)

STYLE: Clear, confident, specific. Mention numbers. Like a morning research note.
Write plain text only — no JSON."""


def generate_btst_narrative(
    client: NVIDIANimClient,
    prediction: str,
    score: float,
    confidence: float,
    factors: Dict,
) -> str:
    """Generate narrative explanation for BTST prediction."""
    if not client.available:
        return ""

    factor_lines = []
    for name, data in factors.items():
        factor_lines.append(f"- {name}: {data.get('detail', 'N/A')} (score: {data.get('score', 0):+.3f})")

    context = f"""BTST PREDICTION: {prediction}
Score: {score:+.3f} | Confidence: {confidence:.0f}%

FACTOR BREAKDOWN:
{chr(10).join(factor_lines)}

Explain this prediction to a retail F&O trader."""

    result = client.chat(BTST_NARRATIVE_PROMPT, context,
                         model=FAST_MODEL, max_tokens=250)
    return result or ""


# ═══════════════════════════════════════════════════════════════
#  5. ALERT CONTEXT EXPLAINER
# ═══════════════════════════════════════════════════════════════
ALERT_EXPLAIN_PROMPT = """You are an F&O risk advisor. A critical trading alert has triggered.

Explain in 2-3 sentences:
1. What this alert means for the trader's position
2. EXACT action to take right now
3. What happens if they don't act

Be urgent but clear. This is real money at stake.
Write plain text only."""


def explain_alert(
    client: NVIDIANimClient,
    alert_type: str,
    alert_message: str,
    current_position: str,
) -> str:
    """Generate LLM explanation for a trading alert."""
    if not client.available:
        return ""

    context = f"""ALERT: [{alert_type}] {alert_message}
Trader's current position: {current_position}

Explain what they should do."""

    result = client.chat(ALERT_EXPLAIN_PROMPT, context,
                         model=FAST_MODEL, max_tokens=150)
    return result or ""


# ═══════════════════════════════════════════════════════════════
#  GLOBAL INSTANCE (initialized in app.py with API key)
# ═══════════════════════════════════════════════════════════════
_nim_client: Optional[NVIDIANimClient] = None


def get_nim_client(api_key: str = None) -> NVIDIANimClient:
    """Get or create the global NIM client instance."""
    global _nim_client
    if _nim_client is None or (api_key and api_key != _nim_client.api_key):
        _nim_client = NVIDIANimClient(api_key)
    return _nim_client
