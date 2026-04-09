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


class ClusteringConfig:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.35"))
    DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "3"))
    CLUSTER_MIN_MENTIONS = int(os.getenv("CLUSTER_MIN_MENTIONS", "3"))


class PipelineConfig:
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
