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


# Default RSS feeds spanning diverse domains so FR4 has enough variety to
# generate novel markets beyond just finance and politics.
# Can be fully overridden via the RSS_FEEDS env var (comma-separated URLs).
_DEFAULT_RSS_FEEDS = ",".join([
    # ── World / geopolitics ───────────────────────────────────────────────
    "https://feeds.bbci.co.uk/news/world/rss.xml",               # BBC World
    "https://www.theguardian.com/world/rss",                     # Guardian World
    "https://www.politico.com/rss/politicopicks.xml",            # Politico
    # ── Business / finance ───────────────────────────────────────────────
    "https://feeds.bbci.co.uk/news/business/rss.xml",            # BBC Business
    "https://www.theguardian.com/business/rss",                  # Guardian Business
    # ── Technology ───────────────────────────────────────────────────────
    "https://feeds.bbci.co.uk/news/technology/rss.xml",          # BBC Technology
    "https://www.theguardian.com/technology/rss",                # Guardian Technology
    "https://techcrunch.com/feed/",                              # TechCrunch
    "https://feeds.arstechnica.com/arstechnica/index",           # Ars Technica
    # ── Health / biotech / pharma ─────────────────────────────────────────
    "https://feeds.bbci.co.uk/news/health/rss.xml",              # BBC Health
    "https://www.statnews.com/feed/",                            # STAT News (biotech/pharma)
    # ── Science / environment / climate ──────────────────────────────────
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",  # BBC Science
    "https://www.theguardian.com/science/rss",                   # Guardian Science
    "https://www.theguardian.com/environment/rss",               # Guardian Environment
    "https://www.sciencedaily.com/rss/all.xml",                  # Science Daily
    # ── Reddit — fast-moving & niche signals (free public RSS, no key) ───
    "https://www.reddit.com/r/worldnews/new.rss",                # World news & geopolitics
    "https://www.reddit.com/r/geopolitics/new.rss",              # Geopolitics analysis
    "https://www.reddit.com/r/science/new.rss",                  # Peer-reviewed science
    "https://www.reddit.com/r/technology/new.rss",               # General tech
    "https://www.reddit.com/r/MachineLearning/new.rss",          # AI/ML research & releases
    "https://www.reddit.com/r/investing/new.rss",                # Markets & corporate news
    "https://www.reddit.com/r/economics/new.rss",                # Macro economics
    "https://www.reddit.com/r/medicine/new.rss",                 # Clinical & medical research
    "https://www.reddit.com/r/Futurology/new.rss",               # Emerging tech & trends
    "https://www.reddit.com/r/space/new.rss",                    # Space exploration & missions
    "https://www.reddit.com/r/energy/new.rss",                   # Energy markets & policy
    "https://www.reddit.com/r/environment/new.rss",              # Climate & sustainability
    "https://www.reddit.com/r/law/new.rss",                      # Legal & regulatory decisions
    "https://www.reddit.com/r/business/new.rss",                 # Corporate & M&A news
    "https://www.reddit.com/r/GlobalPowers/new.rss",             # International relations
])


class IngestionConfig:
    RSS_FEEDS = [
        url.strip()
        for url in os.getenv("RSS_FEEDS", _DEFAULT_RSS_FEEDS).split(",")
        if url.strip()
    ]
    # Broad query so GDELT surfaces events across all domains, not just
    # elections and policy.  Override via GDELT_QUERY in .env if needed.
    GDELT_QUERY = os.getenv(
        "GDELT_QUERY",
        "election OR policy OR merger OR acquisition OR FDA OR "
        "sanctions OR trade OR climate OR AI OR earnings OR IPO OR "
        "treaty OR OPEC OR outbreak OR drug approval OR central bank",
    )
    GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "100"))


class ClusteringConfig:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.35"))
    DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "3"))
    CLUSTER_MIN_MENTIONS = int(os.getenv("CLUSTER_MIN_MENTIONS", "3"))


class PipelineConfig:
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
