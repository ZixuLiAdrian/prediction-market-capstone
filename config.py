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
    RSS_FEEDS = [
        url.strip()
        for url in os.getenv("RSS_FEEDS", _DEFAULT_RSS_FEEDS).split(",")
        if url.strip()
    ]
    GDELT_QUERY = os.getenv("GDELT_QUERY", _DEFAULT_GDELT_QUERY)
    GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "100"))


class ClusteringConfig:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.35"))
    DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "3"))
    CLUSTER_MIN_MENTIONS = int(os.getenv("CLUSTER_MIN_MENTIONS", "3"))


class PipelineConfig:
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
