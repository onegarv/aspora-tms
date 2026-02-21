"""
Phase 5b — Sentiment Agent

Two paths (in priority order):
  1. Alpha Vantage NEWS_SENTIMENT — pre-scored, single API call, no LLM
  2. NewsAPI headlines + OpenRouter Claude Haiku — fallback

Sentiment score convention:
    +1.0 = extremely bullish INR (rate will DROP — fewer rupees per dollar)
    -1.0 = extremely bearish INR (rate will RISE — more rupees per dollar)
     0.0 = neutral

High-impact event detection forces act_on_signal=false in the safety gate.

NEVER raises an exception. Returns neutral fallback on any failure.
"""

import json
import os
import sys
from datetime import datetime, timezone

import requests as _requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Import from the news fetcher
from agents.fetch_news import fetch_headlines, fetch_alphavantage_sentiment, CACHE_FILE, AV_CACHE_FILE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Claude 3 Haiku on OpenRouter
OPENROUTER_MODEL_ID = "anthropic/claude-3-haiku"
MAX_HEADLINES = 20

# Haiku pricing (OpenRouter): $0.25 / 1M input, $1.25 / 1M output
HAIKU_INPUT_COST_PER_TOKEN = 0.25 / 1_000_000
HAIKU_OUTPUT_COST_PER_TOKEN = 1.25 / 1_000_000

# Neutral fallback — returned on any failure
NEUTRAL_FALLBACK = {
    "sentiment_score": 0.0,
    "confidence": 0.0,
    "explanation": "No sentiment data available — using neutral fallback",
    "bullish_inr_signals": [],
    "bearish_inr_signals": [],
    "high_impact_event_detected": False,
    "event_type": None,
    "event_description": None,
    "data_quality": "none",
}


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst specializing in USD/INR exchange rates.
Analyze the provided news headlines and return ONLY a valid JSON object with no preamble,
no markdown formatting, and no explanation outside the JSON.

The JSON must follow this exact schema:
{
  "sentiment_score": <float from -1.0 to +1.0>,
  "confidence": <float from 0.0 to 1.0>,
  "explanation": "<one sentence explaining the dominant driver>",
  "bullish_inr_signals": ["<relevant headline snippet>", ...],
  "bearish_inr_signals": ["<relevant headline snippet>", ...],
  "high_impact_event_detected": <true or false>,
  "event_type": "<RBI_MEETING|FED_MEETING|US_CPI|US_NFP|INDIA_BUDGET|GEOPOLITICAL|OTHER or null>",
  "event_description": "<string or null>",
  "data_quality": "<good|limited|none>"
}

IMPORTANT scoring rules:
- sentiment_score: +1.0 means extremely BULLISH for INR (INR strengthens, fewer rupees per dollar, rate goes DOWN)
- sentiment_score: -1.0 means extremely BEARISH for INR (INR weakens, more rupees per dollar, rate goes UP)
- sentiment_score: 0.0 means neutral
- confidence: how confident you are in the sentiment score (0.0 = no idea, 1.0 = very certain)
- high_impact_event_detected: true if any headline mentions RBI policy decision, RBI rate change,
  Federal Reserve decision, Fed rate hike/cut, India Union Budget, US CPI release,
  US Non-Farm Payrolls, or a major geopolitical shock
- data_quality: "good" if 5+ relevant headlines, "limited" if 1-4 relevant, "none" if 0 relevant

Respond ONLY with the JSON object."""


def _build_user_prompt(headlines: list[dict]) -> str:
    """Build the user message from headlines."""
    if not headlines:
        return "No news headlines available. Return neutral sentiment with data_quality='none'."

    lines = [f"Analyze these {len(headlines)} financial headlines from the last 24 hours:\n"]
    for i, h in enumerate(headlines[:MAX_HEADLINES], 1):
        source = h.get("source", "Unknown")
        title = h.get("title", "")
        desc = h.get("description", "")
        lines.append(f"{i}. [{source}] {title}")
        if desc:
            lines.append(f"   {desc[:150]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def _call_openrouter(headlines: list[dict]) -> dict:
    """
    Call Claude Haiku via OpenRouter with headlines.
    Returns parsed sentiment dict or raises on failure.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    user_prompt = _build_user_prompt(headlines)

    # Estimate tokens (rough: 1 token ≈ 4 chars)
    input_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    est_input_tokens = input_chars // 4
    est_output_tokens = 350  # structured JSON response is compact

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL_ID,
        "max_tokens": 512,
        "temperature": 0.1,  # low temperature for consistent structured output
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = _requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    response_body = response.json()
    raw_text = response_body["choices"][0]["message"]["content"]

    # Parse the JSON from model output
    # Strip any accidental markdown fences
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    result = json.loads(cleaned)

    # Compute actual token usage if available
    usage = response_body.get("usage", {})
    actual_input = usage.get("prompt_tokens", est_input_tokens)
    actual_output = usage.get("completion_tokens", est_output_tokens)
    cost = (actual_input * HAIKU_INPUT_COST_PER_TOKEN +
            actual_output * HAIKU_OUTPUT_COST_PER_TOKEN)

    result["_token_usage"] = {
        "input_tokens": actual_input,
        "output_tokens": actual_output,
        "estimated_cost_usd": round(cost, 6),
    }

    return result


# ---------------------------------------------------------------------------
# Main sentiment function
# ---------------------------------------------------------------------------

def _get_cache_info(source: str = "newsapi") -> dict:
    """Check if today's cache exists and return cache metadata."""
    cache_path = AV_CACHE_FILE if source == "alpha_vantage" else CACHE_FILE
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cache = json.load(f)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            cache_date = cache.get("date")
            if cache_date == today:
                return {"cache_hit": True, "cache_date": cache_date}
            return {"cache_hit": False, "cache_date": cache_date}
    except Exception:
        pass
    return {"cache_hit": False, "cache_date": None}


def get_sentiment(headlines: list[dict] = None) -> dict:
    """
    Full sentiment pipeline:
        1. Try Alpha Vantage pre-scored sentiment (if key exists, no explicit headlines)
        2. Fall back to NewsAPI headlines + OpenRouter scoring
        3. Return structured sentiment or neutral fallback

    Never raises an exception.
    """
    # -----------------------------------------------------------------
    # Path 1: Alpha Vantage (pre-scored, no LLM needed)
    # Only when caller didn't provide explicit headlines
    # -----------------------------------------------------------------
    if headlines is None:
        try:
            av_result = fetch_alphavantage_sentiment()
            if av_result is not None:
                cache_info = _get_cache_info(source="alpha_vantage")
                av_result.update(cache_info)
                return av_result
        except Exception as e:
            print(f"  [sentiment] WARNING: Alpha Vantage failed: {e}")
        # Alpha Vantage unavailable — fall through to NewsAPI + OpenRouter

    # -----------------------------------------------------------------
    # Path 2: NewsAPI headlines + OpenRouter scoring (fallback)
    # -----------------------------------------------------------------
    provided_headlines = headlines is not None
    if headlines is None:
        try:
            headlines = fetch_headlines()
        except Exception as e:
            print(f"  [sentiment] WARNING: fetch_headlines failed: {e}")
            headlines = []

    cache_info = _get_cache_info(source="newsapi") if not provided_headlines else {"cache_hit": False, "cache_date": None}

    if not headlines:
        print("  [sentiment] No headlines available — returning neutral fallback")
        result = NEUTRAL_FALLBACK.copy()
        result["_token_usage"] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}
        result.update(cache_info)
        return result

    # Call OpenRouter
    try:
        result = _call_openrouter(headlines)
        result["source"] = "newsapi_openrouter"

        # Validate required fields, fill in defaults
        result.setdefault("sentiment_score", 0.0)
        result.setdefault("confidence", 0.0)
        result.setdefault("explanation", "No explanation provided")
        result.setdefault("bullish_inr_signals", [])
        result.setdefault("bearish_inr_signals", [])
        result.setdefault("high_impact_event_detected", False)
        result.setdefault("event_type", None)
        result.setdefault("event_description", None)
        result.setdefault("data_quality", "limited")

        # Clamp sentiment score to [-1, 1]
        result["sentiment_score"] = max(-1.0, min(1.0, float(result["sentiment_score"])))
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

        result.update(cache_info)
        return result

    except ValueError as e:
        print(f"  [sentiment] ERROR: {e}")
    except _requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        if status == 401 or status == 403:
            print(f"  [sentiment] ERROR: OpenRouter auth failed (HTTP {status})")
            print("  Check OPENROUTER_API_KEY in .env")
        elif status == 404:
            print(f"  [sentiment] ERROR: Model {OPENROUTER_MODEL_ID} not found on OpenRouter")
        else:
            print(f"  [sentiment] ERROR: OpenRouter HTTP {status}: {e}")
    except json.JSONDecodeError as e:
        print(f"  [sentiment] ERROR: Failed to parse LLM JSON response: {e}")
    except _requests.exceptions.RequestException as e:
        print(f"  [sentiment] ERROR: Network error: {e}")
    except Exception as e:
        print(f"  [sentiment] ERROR: Unexpected error: {e}")

    # Fallback
    print("  [sentiment] Returning neutral fallback")
    result = NEUTRAL_FALLBACK.copy()
    result["_token_usage"] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}
    result.update(cache_info)
    return result


# ---------------------------------------------------------------------------
# Direction conversion helper
# ---------------------------------------------------------------------------

def sentiment_to_direction(sentiment_obj: dict) -> tuple[str, float]:
    """
    Convert sentiment score to directional signal for ensemble.

    Returns:
        (direction, confidence) where direction is 'UP', 'DOWN', or 'NEUTRAL'

    Convention:
        sentiment > +0.20 → bullish INR → rate falls → direction = 'DOWN'
        sentiment < -0.20 → bearish INR → rate rises → direction = 'UP'
        between -0.20 and +0.20 → 'NEUTRAL'
    """
    score = sentiment_obj.get("sentiment_score", 0.0)
    confidence = sentiment_obj.get("confidence", 0.0)

    if score > 0.20:
        return "DOWN", confidence   # bullish INR = rate drops
    elif score < -0.20:
        return "UP", confidence     # bearish INR = rate rises
    else:
        return "NEUTRAL", confidence


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SENTIMENT AGENT — Test Run")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: No headlines (fallback test)
    # ------------------------------------------------------------------
    print("\n--- Test 1: Empty headlines → neutral fallback ---")
    result = get_sentiment(headlines=[])
    direction, conf = sentiment_to_direction(result)
    print(f"  Direction: {direction}  Confidence: {conf:.2f}")
    print(f"  Sentiment score: {result['sentiment_score']}")
    print(f"  Data quality: {result['data_quality']}")
    print(f"  High impact event: {result['high_impact_event_detected']}")
    assert result["sentiment_score"] == 0.0, "Fallback should be neutral"
    assert result["data_quality"] == "none", "Fallback should be data_quality=none"
    print("  PASS\n")

    # ------------------------------------------------------------------
    # Test 2: With mock headlines (tests Bedrock integration)
    # ------------------------------------------------------------------
    print("--- Test 2: Mock headlines → OpenRouter sentiment analysis ---")
    mock_headlines = [
        {"title": "RBI holds interest rates steady, signals cautious approach",
         "source": "Reuters", "published_at": "2026-02-20T06:00:00Z",
         "description": "Reserve Bank of India kept repo rate unchanged at 6.5%"},
        {"title": "Crude oil prices surge 3% on supply concerns",
         "source": "Bloomberg", "published_at": "2026-02-20T05:00:00Z",
         "description": "Brent crude rose to $73 amid Middle East tensions"},
        {"title": "US Dollar weakens as Treasury yields fall",
         "source": "CNBC", "published_at": "2026-02-20T04:00:00Z",
         "description": "DXY index dropped 0.4% on weak economic data"},
        {"title": "India GDP growth forecast raised to 7.2% by IMF",
         "source": "Economic Times", "published_at": "2026-02-20T03:00:00Z",
         "description": "Strong domestic consumption drives upgrade"},
        {"title": "FII flows turn positive for India equities",
         "source": "Mint", "published_at": "2026-02-20T02:00:00Z",
         "description": "Foreign investors buy $500M in Indian stocks this week"},
    ]

    result = get_sentiment(headlines=mock_headlines)

    print(f"\n  Full sentiment output:")
    display = {k: v for k, v in result.items() if not k.startswith("_")}
    print(f"  {json.dumps(display, indent=2, default=str)}")

    token_usage = result.get("_token_usage", {})
    print(f"\n  Token usage:")
    print(f"    Input tokens:  {token_usage.get('input_tokens', 'N/A')}")
    print(f"    Output tokens: {token_usage.get('output_tokens', 'N/A')}")
    print(f"    Estimated cost: ${token_usage.get('estimated_cost_usd', 0):.6f}")

    daily_cost = token_usage.get("estimated_cost_usd", 0) * 1  # once per day
    monthly_cost = daily_cost * 22  # ~22 trading days
    print(f"\n  Estimated running cost:")
    print(f"    Per call:  ${token_usage.get('estimated_cost_usd', 0):.6f}")
    print(f"    Per month: ${monthly_cost:.4f} (22 trading days)")

    direction, conf = sentiment_to_direction(result)
    print(f"\n  Direction signal: {direction}")
    print(f"  Confidence:       {conf:.2f}")
    print(f"  High impact event: {result.get('high_impact_event_detected', False)}")

    # ------------------------------------------------------------------
    # Test 3: Live NewsAPI test (if key available)
    # ------------------------------------------------------------------
    newsapi_key = os.getenv("NEWSAPI_KEY", "")
    if newsapi_key and newsapi_key != "your_key_here":
        print("\n\n--- Test 3: Live NewsAPI → OpenRouter pipeline ---")
        live_result = get_sentiment()
        print(f"\n  Live sentiment output:")
        live_display = {k: v for k, v in live_result.items() if not k.startswith("_")}
        print(f"  {json.dumps(live_display, indent=2, default=str)}")
        live_dir, live_conf = sentiment_to_direction(live_result)
        print(f"\n  Live direction: {live_dir}  Confidence: {live_conf:.2f}")
    else:
        print("\n\n--- Test 3: Skipped (no NEWSAPI_KEY in .env) ---")

    print("\n--- Phase 5 complete ---")
