"""
Central configuration loaded from .env file.
All tunable parameters are defined here — no hardcoded values in pipeline code.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class DBConfig:
    """PostgreSQL connection settings loaded from environment variables."""

    HOST = os.getenv("DB_HOST", "localhost")
    PORT = int(os.getenv("DB_PORT", "5432"))
    NAME = os.getenv("DB_NAME", "prediction_markets")
    USER = os.getenv("DB_USER", "postgres")
    PASSWORD = os.getenv("DB_PASSWORD", "postgres")

    @classmethod
    def connection_string(cls) -> str:
        """Build a psycopg2-compatible DSN string from the configured host, port, name, user, and password."""
        return f"host={cls.HOST} port={cls.PORT} dbname={cls.NAME} user={cls.USER} password={cls.PASSWORD}"


class LLMConfig:
    """LLM provider, model selection, and rate-limit / retry settings."""

    PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "gemini"
    MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    FR3_MODEL = os.getenv("FR3_LLM_MODEL", MODEL)
    FR4_MODEL = os.getenv("FR4_LLM_MODEL", MODEL)
    AVAILABLE_MODELS = [
        "allam-2-7b",
        "groq/compound",
        "groq/compound-mini",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
    ]
    RUNNER_DEFAULT_FR3_MODEL = os.getenv("RUNNER_DEFAULT_FR3_MODEL", "llama-3.1-8b-instant")
    RUNNER_DEFAULT_FR4_MODEL = os.getenv("RUNNER_DEFAULT_FR4_MODEL", "llama-3.1-8b-instant")
    API_KEY = os.getenv("LLM_API_KEY", "")
    MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    RATE_LIMIT_MAX_RETRIES = int(os.getenv("LLM_RATE_LIMIT_MAX_RETRIES", "-1"))
    TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
    MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("LLM_MIN_REQUEST_INTERVAL_SECONDS", "2.1"))
    RATE_LIMIT_BACKOFF_BASE_SECONDS = float(os.getenv("LLM_RATE_LIMIT_BACKOFF_BASE_SECONDS", "5.0"))
    RATE_LIMIT_BACKOFF_MAX_SECONDS = float(os.getenv("LLM_RATE_LIMIT_BACKOFF_MAX_SECONDS", "30.0"))


_DEFAULT_RSS_FEEDS = ",".join([
    # --- Major news outlets ---
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://www.ft.com/?format=rss",
    "https://feeds.skynews.com/feeds/rss/world.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.npr.org/1004/rss.xml",
    "https://rss.politico.com/politics-news.xml",
    "https://feeds.feedburner.com/TheAtlantic",
    "https://www.wired.com/feed/rss",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.nature.com/nature.rss",
    "https://rss.sciencedaily.com/all.xml",
    "https://feeds.feedburner.com/SpaceflightNow",
    # --- Reddit public RSS feeds (no API key needed) ---
    "https://www.reddit.com/r/worldnews/new.rss",
    "https://www.reddit.com/r/geopolitics/new.rss",
    "https://www.reddit.com/r/science/new.rss",
    "https://www.reddit.com/r/technology/new.rss",
    "https://www.reddit.com/r/MachineLearning/new.rss",
    "https://www.reddit.com/r/investing/new.rss",
    "https://www.reddit.com/r/economics/new.rss",
    "https://www.reddit.com/r/medicine/new.rss",
    "https://www.reddit.com/r/Futurology/new.rss",
    "https://www.reddit.com/r/space/new.rss",
    "https://www.reddit.com/r/energy/new.rss",
    "https://www.reddit.com/r/environment/new.rss",
    "https://www.reddit.com/r/law/new.rss",
    "https://www.reddit.com/r/business/new.rss",
    "https://www.reddit.com/r/GlobalPowers/new.rss",
])

_DEFAULT_GDELT_QUERY = (
    "election OR policy OR FDA OR merger OR acquisition OR "
    "climate OR AI OR sanctions OR NATO OR treaty OR "
    "IPO OR earnings OR trial OR space OR energy"
)


class IngestionConfig:
    """Source-specific settings for all FR1 ingestors (URLs, API keys, query terms, limits)."""

    RSS_FEEDS = [
        url.strip()
        for url in os.getenv("RSS_FEEDS", _DEFAULT_RSS_FEEDS).split(",")
        if url.strip()
    ]
    GDELT_QUERY = os.getenv("GDELT_QUERY", _DEFAULT_GDELT_QUERY)
    GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "100"))

    # Reddit (API-based ingestor — requires credentials for higher rate limits)
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "prediction-market-pipeline/1.0")
    REDDIT_SUBREDDITS = [
        s.strip() for s in os.getenv(
            "REDDIT_SUBREDDITS",
            "politics,worldnews,news,investing,CryptoCurrency,technology,Polymarket,"
            "geopolitics,science,MachineLearning,economics,medicine,Futurology,space,energy"
        ).split(",") if s.strip()
    ]
    REDDIT_POST_LIMIT = int(os.getenv("REDDIT_POST_LIMIT", "25"))

    # Hacker News
    HN_MAX_STORIES = int(os.getenv("HN_MAX_STORIES", "30"))

    # Wikipedia Pageviews
    WIKI_PAGES = [
        p.strip() for p in os.getenv("WIKI_PAGES", "").split(",") if p.strip()
    ]
    WIKI_PAGEVIEW_THRESHOLD = int(os.getenv("WIKI_PAGEVIEW_THRESHOLD", "10000"))

    # Federal Register
    FED_REGISTER_QUERY = os.getenv("FED_REGISTER_QUERY", "")
    FED_REGISTER_PER_PAGE = int(os.getenv("FED_REGISTER_PER_PAGE", "20"))

    # Congress.gov
    CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY", "")
    CONGRESS_LIMIT = int(os.getenv("CONGRESS_LIMIT", "20"))

    # SEC EDGAR
    SEC_EDGAR_USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT", "prediction-market-pipeline admin@example.com")
    SEC_EDGAR_FORM_TYPES = [
        t.strip() for t in os.getenv("SEC_EDGAR_FORM_TYPES", "8-K,4").split(",") if t.strip()
    ]
    SEC_EDGAR_LIMIT = int(os.getenv("SEC_EDGAR_LIMIT", "20"))

    # BLS
    BLS_API_KEY = os.getenv("BLS_API_KEY", "")
    BLS_SERIES_IDS = [
        s.strip() for s in os.getenv(
            "BLS_SERIES_IDS",
            "CUUR0000SA0,LNS14000000"  # CPI-U, Unemployment rate
        ).split(",") if s.strip()
    ]

    # FRED
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    FRED_SERIES_IDS = [
        s.strip() for s in os.getenv(
            "FRED_SERIES_IDS",
            "FEDFUNDS,T10Y2Y,UNRATE"  # Fed funds rate, 10Y-2Y spread, unemployment
        ).split(",") if s.strip()
    ]

    # EIA
    EIA_API_KEY = os.getenv("EIA_API_KEY", "")
    EIA_SERIES_IDS = [
        s.strip() for s in os.getenv(
            "EIA_SERIES_IDS",
            "PET.RWTC.W"  # WTI crude oil weekly
        ).split(",") if s.strip()
    ]

    # Kalshi
    KALSHI_LIMIT = int(os.getenv("KALSHI_LIMIT", "50"))


class ClusteringConfig:
    """DBSCAN hyperparameters, near-duplicate threshold, and per-source event weights for FR2."""

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.35"))
    DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "3"))
    CLUSTER_MIN_MENTIONS = int(os.getenv("CLUSTER_MIN_MENTIONS", "3"))
    NEAR_DUPLICATE_THRESHOLD = float(os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.92"))

    # Per-source weights: official sources count more than social posts
    SOURCE_WEIGHTS = {
        # Official / resolution sources
        "federal_register": 3.0,
        "congress": 3.0,
        "sec_edgar": 3.0,
        "bls": 3.0,
        "fred": 2.5,
        "eia": 2.5,
        # News / discovery sources
        "reuters": 2.0,
        "bbc": 2.0,
        "politico": 2.0,
        "gdelt": 1.5,
        # Market / benchmark sources
        "polymarket": 1.5,
        "kalshi": 1.5,
        # Social / attention sources
        "reddit": 1.0,
        "hackernews": 1.0,
        "wikipedia": 0.8,
    }
    DEFAULT_SOURCE_WEIGHT = float(os.getenv("DEFAULT_SOURCE_WEIGHT", "1.0"))


class PipelineConfig:
    """Top-level pipeline run settings: batch sizes, per-stage limits, and logging mode."""

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_MODE = os.getenv("LOG_MODE", "normal").strip().lower()
    FR3_MAX_CLUSTERS = int(os.getenv("FR3_MAX_CLUSTERS", "10"))
    FR4_MAX_EVENTS = int(os.getenv("FR4_MAX_EVENTS", "5"))
    FR4_MAX_QUESTIONS_PER_STORY = int(os.getenv("FR4_MAX_QUESTIONS_PER_STORY", "2"))
