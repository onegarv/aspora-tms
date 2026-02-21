"""
Phase 5a — Financial News & Sentiment Fetcher

Two data sources (in priority order):
  1. Alpha Vantage NEWS_SENTIMENT — pre-scored sentiment for FOREX:USD
     Single API call, no LLM needed. Used when ALPHAVANTAGE_API_KEY is set.
  2. NewsAPI headlines — raw headlines sent to OpenRouter for scoring (fallback)

Both use daily file-based caching (one call per day).
Never raises an exception — returns empty/None on any failure.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

QUERIES = [
    "USD INR rupee",
    "RBI monetary policy",
    "India forex crude oil",
    "Federal Reserve dollar",
]

ARTICLES_PER_QUERY = 5

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CACHE_FILE = os.path.join(CACHE_DIR, "news_cache.json")
AV_CACHE_FILE = os.path.join(CACHE_DIR, "av_sentiment_cache.json")

# Alpha Vantage config
AV_URL = "https://www.alphavantage.co/query"
AV_RELEVANCE_THRESHOLD = 0.3  # only use articles with relevance > 0.3

# High-impact event keywords (scanned in article titles)
HIGH_IMPACT_KEYWORDS = {
    "RBI_MEETING": ["rbi meeting", "rbi policy", "rbi rate", "repo rate", "reserve bank of india"],
    "FED_MEETING": ["fomc", "federal reserve", "fed rate", "fed meeting", "fed decision",
                     "fed hike", "fed cut", "powell"],
    "US_CPI": ["us cpi", "consumer price index", "inflation data", "inflation report"],
    "US_NFP": ["non-farm payroll", "nonfarm payroll", "jobs report", "employment report"],
    "INDIA_BUDGET": ["india budget", "union budget", "fiscal deficit"],
    "GEOPOLITICAL": ["sanctions", "trade war", "tariff", "military", "invasion", "conflict"],
}


# ---------------------------------------------------------------------------
# Alpha Vantage sentiment (primary)
# ---------------------------------------------------------------------------

def _get_av_cache() -> dict | None:
    """Return today's cached Alpha Vantage sentiment, or None if expired."""
    if not os.path.exists(AV_CACHE_FILE):
        return None
    try:
        with open(AV_CACHE_FILE, "r") as f:
            cache = json.load(f)
        today = datetime.now().strftime("%Y-%m-%d")
        if cache.get("date") != today:
            print(f"  [alpha_vantage] Cache expired ({cache.get('date')}) — fetching fresh")
            return None
        print(f"  [alpha_vantage] Cache hit — using today's pre-scored sentiment")
        return cache.get("sentiment")
    except Exception as e:
        print(f"  [alpha_vantage] Cache read error: {e}")
        return None


def _save_av_cache(sentiment: dict) -> None:
    """Save Alpha Vantage sentiment result to daily cache."""
    cache = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "cached_at": datetime.now().isoformat(),
        "sentiment": sentiment,
    }
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(AV_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  [alpha_vantage] Cached sentiment for today")
    except Exception as e:
        print(f"  [alpha_vantage] Cache write error: {e}")


def _detect_high_impact(titles: list[str]) -> tuple[bool, str | None, str | None]:
    """
    Scan titles for high-impact event keywords.
    Returns (detected, event_type, event_description).
    """
    all_text = " ".join(titles).lower()
    for event_type, keywords in HIGH_IMPACT_KEYWORDS.items():
        for kw in keywords:
            if kw in all_text:
                # Find the matching title
                for title in titles:
                    if kw in title.lower():
                        return True, event_type, title
                return True, event_type, None
    return False, None, None


def fetch_alphavantage_sentiment() -> dict | None:
    """
    Fetch pre-scored sentiment from Alpha Vantage NEWS_SENTIMENT endpoint.

    Returns a structured sentiment dict matching get_sentiment() format:
        sentiment_score, confidence, explanation, bullish/bearish signals,
        high_impact_event_detected, event_type, data_quality, etc.

    Returns None if:
        - ALPHAVANTAGE_API_KEY not set
        - API call fails
        - No relevant articles found

    Score convention (Alpha Vantage → our convention):
        AV scores FOREX:USD: positive = bullish USD = bearish INR
        Our convention: positive = bullish INR
        Therefore: our_score = -av_score
    """
    # Check cache first
    cached = _get_av_cache()
    if cached is not None:
        return cached

    if not ALPHAVANTAGE_API_KEY:
        return None

    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": "FOREX:USD",
            "sort": "LATEST",
            "limit": "50",
            "apikey": ALPHAVANTAGE_API_KEY,
        }
        resp = requests.get(AV_URL, params=params, timeout=15)

        if resp.status_code != 200:
            print(f"  [alpha_vantage] WARNING: API returned {resp.status_code}")
            return None

        data = resp.json()

        # Check for error responses
        if "Information" in data:
            print(f"  [alpha_vantage] WARNING: {data['Information'][:100]}")
            return None
        if "Note" in data:
            print(f"  [alpha_vantage] WARNING: Rate limited — {data['Note'][:80]}")
            return None

        feed = data.get("feed", [])
        if not feed:
            print("  [alpha_vantage] WARNING: Empty feed returned")
            return None

        # Filter for FOREX:USD with relevance > threshold
        relevant_articles = []
        for art in feed:
            for ts in art.get("ticker_sentiment", []):
                if ts.get("ticker") == "FOREX:USD":
                    relevance = float(ts.get("relevance_score", "0"))
                    if relevance > AV_RELEVANCE_THRESHOLD:
                        score = float(ts.get("ticker_sentiment_score", "0"))
                        relevant_articles.append({
                            "title": art.get("title", ""),
                            "source": art.get("source", "Unknown"),
                            "summary": art.get("summary", ""),
                            "usd_score": score,       # raw AV score (bullish USD)
                            "relevance": relevance,
                            "label": ts.get("ticker_sentiment_label", "Neutral"),
                            "overall_score": float(art.get("overall_sentiment_score", "0")),
                        })

        print(f"  [alpha_vantage] {len(feed)} articles, "
              f"{len(relevant_articles)} relevant (FOREX:USD, rel>{AV_RELEVANCE_THRESHOLD})")

        if not relevant_articles:
            print("  [alpha_vantage] No relevant USD articles — falling back")
            return None

        # Compute weighted average score (weighted by relevance)
        scores = [a["usd_score"] for a in relevant_articles]
        weights = [a["relevance"] for a in relevant_articles]
        total_weight = sum(weights)
        wavg_usd_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Flip sign: AV bullish USD = our bearish INR
        inr_score = -wavg_usd_score

        # Confidence from article count + relevance spread
        n = len(relevant_articles)
        avg_relevance = total_weight / n
        if n >= 5:
            base_conf = 0.80
        elif n >= 3:
            base_conf = 0.65
        elif n >= 2:
            base_conf = 0.50
        else:
            base_conf = 0.35
        confidence = min(1.0, base_conf * avg_relevance + 0.1 * min(n, 5) / 5)

        # Separate bullish/bearish INR signals
        bullish_inr = []   # positive for INR = negative USD score
        bearish_inr = []   # negative for INR = positive USD score
        for a in relevant_articles:
            snippet = f"[{a['source']}] {a['title'][:80]}"
            if a["usd_score"] < -0.05:
                bullish_inr.append(snippet)   # weak USD = strong INR
            elif a["usd_score"] > 0.05:
                bearish_inr.append(snippet)   # strong USD = weak INR

        # High-impact event detection
        titles = [a["title"] for a in relevant_articles]
        hi_detected, hi_type, hi_desc = _detect_high_impact(titles)

        # Data quality
        if n >= 5:
            quality = "good"
        elif n >= 2:
            quality = "limited"
        else:
            quality = "limited"

        # Build the dominant driver explanation
        if abs(inr_score) < 0.05:
            direction_text = "mixed signals, neutral sentiment"
        elif inr_score > 0:
            direction_text = "USD weakness signals favour INR strength"
        else:
            direction_text = "USD strength signals suggest INR pressure"

        result = {
            "sentiment_score": max(-1.0, min(1.0, round(inr_score, 4))),
            "confidence": round(confidence, 4),
            "explanation": f"Alpha Vantage: {n} relevant articles — {direction_text}",
            "bullish_inr_signals": bullish_inr,
            "bearish_inr_signals": bearish_inr,
            "high_impact_event_detected": hi_detected,
            "event_type": hi_type,
            "event_description": hi_desc,
            "data_quality": quality,
            "source": "alpha_vantage",
            "article_count": n,
            "raw_usd_score": round(wavg_usd_score, 4),
            "_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        }

        _save_av_cache(result)
        return result

    except requests.RequestException as e:
        print(f"  [alpha_vantage] WARNING: Request failed: {e}")
        return None
    except Exception as e:
        print(f"  [alpha_vantage] WARNING: Unexpected error: {e}")
        return None


# ---------------------------------------------------------------------------
# NewsAPI daily cache (fallback path)
# ---------------------------------------------------------------------------

def get_cached_news() -> list[dict] | None:
    """Return today's cached articles, or None if cache is missing/expired."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        cache_date = cache.get("date")
        today = datetime.now().strftime("%Y-%m-%d")
        if cache_date == today:
            articles = cache.get("articles", [])
            print(f"  [fetch_news] Cache hit — using {len(articles)} headlines from today")
            return articles
        print(f"  [fetch_news] Cache expired ({cache_date}) — fetching fresh headlines")
        return None
    except Exception as e:
        print(f"  [fetch_news] Cache read error: {e}")
        return None


def save_news_cache(articles: list[dict]) -> None:
    """Save articles to daily cache file."""
    cache = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "articles": articles,
        "cached_at": datetime.now().isoformat(),
        "article_count": len(articles),
    }
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  [fetch_news] Cached {len(articles)} headlines for today")
    except Exception as e:
        print(f"  [fetch_news] Cache write error: {e}")


# ---------------------------------------------------------------------------
# NewsAPI fetcher (fallback)
# ---------------------------------------------------------------------------

def _fetch_from_newsapi(api_key: str = "", queries: list = None) -> list[dict]:
    """
    Fetch last 48 hours of financial headlines from NewsAPI.
    Returns list of dicts or empty list on failure — never raises.
    """
    key = api_key or NEWSAPI_KEY
    if not key or key == "your_key_here":
        print("  [fetch_news] WARNING: No valid NEWSAPI_KEY — returning empty headlines")
        return []

    if queries is None:
        queries = QUERIES

    from_date = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
    all_articles = []
    seen_titles = set()

    for query in queries:
        try:
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": ARTICLES_PER_QUERY,
                "language": "en",
                "apiKey": key,
            }
            resp = requests.get(NEWSAPI_URL, params=params, timeout=10)

            if resp.status_code != 200:
                print(f"  [fetch_news] WARNING: NewsAPI returned {resp.status_code} for query '{query}'")
                continue

            data = resp.json()
            articles = data.get("articles", [])

            for art in articles:
                title = (art.get("title") or "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                all_articles.append({
                    "title": title,
                    "source": (art.get("source") or {}).get("name", "Unknown"),
                    "published_at": art.get("publishedAt", ""),
                    "description": (art.get("description") or "")[:300],
                })

        except requests.RequestException as e:
            print(f"  [fetch_news] WARNING: Request failed for '{query}': {e}")
            continue
        except Exception as e:
            print(f"  [fetch_news] WARNING: Unexpected error for '{query}': {e}")
            continue

    print(f"  [fetch_news] Fetched {len(all_articles)} unique headlines from {len(queries)} queries")
    return all_articles


def fetch_headlines(api_key: str = "", queries: list = None) -> list[dict]:
    """
    Fetch headlines with daily file-based caching.

    1. Check cache — return if today's articles exist
    2. Fetch fresh from NewsAPI
    3. Cache if successful
    """
    # Step 1: Check cache
    cached = get_cached_news()
    if cached is not None:
        return cached

    # Step 2: Fetch fresh
    articles = _fetch_from_newsapi(api_key=api_key, queries=queries)

    # Step 3: Cache if we got articles
    if articles:
        save_news_cache(articles)

    return articles


if __name__ == "__main__":
    print("=" * 70)
    print("NEWS & SENTIMENT FETCHER — Test Run")
    print("=" * 70)

    # Test 1: Alpha Vantage
    if ALPHAVANTAGE_API_KEY:
        print("\n--- Test 1: Alpha Vantage pre-scored sentiment ---")
        result = fetch_alphavantage_sentiment()
        if result:
            print(f"\n  Score (INR): {result['sentiment_score']:+.4f}")
            print(f"  Raw USD score: {result['raw_usd_score']:+.4f}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Articles: {result['article_count']}")
            print(f"  Quality: {result['data_quality']}")
            print(f"  Explanation: {result['explanation']}")
            print(f"  High impact: {result['high_impact_event_detected']}")
            if result["bullish_inr_signals"]:
                print(f"  Bullish INR:")
                for s in result["bullish_inr_signals"]:
                    print(f"    + {s}")
            if result["bearish_inr_signals"]:
                print(f"  Bearish INR:")
                for s in result["bearish_inr_signals"]:
                    print(f"    - {s}")
        else:
            print("  No result (API issue or no relevant articles)")
    else:
        print("\n--- Test 1: Skipped (no ALPHAVANTAGE_API_KEY) ---")

    # Test 2: NewsAPI fallback
    if NEWSAPI_KEY and NEWSAPI_KEY != "your_key_here":
        print("\n--- Test 2: NewsAPI headlines (fallback) ---")
        headlines = fetch_headlines()
        print(f"  Headlines: {len(headlines)}")
        for h in headlines[:3]:
            print(f"    [{h['source']}] {h['title'][:70]}")
    else:
        print("\n--- Test 2: Skipped (no NEWSAPI_KEY) ---")

    print("\n--- Tests complete ---")
