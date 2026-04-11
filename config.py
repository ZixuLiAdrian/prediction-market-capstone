"""
Central configuration loaded from .env file.
All tunable parameters are defined here — no hardcoded values in pipeline code.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class DBConfig:
    HOST = os.getenv("DB_HOST", "localhost")
    PORT = int(os.getenv("DB_PORT", "5432"))
    NAME = os.getenv("DB_NAME", "prediction_markets")
    USER = os.getenv("DB_USER", "postgres")
    PASSWORD = os.getenv("DB_PASSWORD", "postgres")

    @classmethod
    def connection_string(cls):
        return f"host={cls.HOST} port={cls.PORT} dbname={cls.NAME} user={cls.USER} password={cls.PASSWORD}"


class LLMConfig:
    PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "gemini"
    MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    API_KEY = os.getenv("LLM_API_KEY", "")
    MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))


class IngestionConfig:
    RSS_FEEDS = [
        url.strip()
        for url in os.getenv(
            "RSS_FEEDS",
            "https://feeds.bbci.co.uk/news/world/rss.xml"
        ).split(",")
        if url.strip()
    ]
    GDELT_QUERY = os.getenv("GDELT_QUERY", "prediction market OR election OR policy")
    GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "100"))

    # Reddit
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "prediction-market-pipeline/1.0")
    REDDIT_SUBREDDITS = [
        s.strip() for s in os.getenv(
            "REDDIT_SUBREDDITS",
            "politics,worldnews,news,investing,CryptoCurrency,technology,Polymarket"
        ).split(",") if s.strip()
    ]
    REDDIT_POST_LIMIT = int(os.getenv("REDDIT_POST_LIMIT", "25"))

    # Hacker News
    HN_MAX_STORIES = int(os.getenv("HN_MAX_STORIES", "30"))

    # Wikipedia Pageviews
    WIKI_PAGES = [
        p.strip() for p in os.getenv(
            "WIKI_PAGES",
            ""
        ).split(",") if p.strip()
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
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
