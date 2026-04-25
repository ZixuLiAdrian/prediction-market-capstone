"""
FR1: FRED (Federal Reserve Economic Data) Ingestion

Fetches recent observations from FRED API.
Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
Covers macro/rates/economic time series.
"""

import logging
from datetime import datetime, timezone
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

FRED_API = "https://api.stlouisfed.org/fred/series/observations"

SERIES_NAMES = {
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10-Year/2-Year Treasury Spread",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI (All Items)",
    "GDP": "Gross Domestic Product",
    "DGS10": "10-Year Treasury Yield",
}


class FREDIngestor(BaseIngestor):
    """Ingestor that fetches the most recent observation for each configured FRED series."""

    source_type = "official"

    def fetch(self) -> List[Event]:
        """Fetch the latest data point for each FRED series and return them as Events."""
        events = []

        if not IngestionConfig.FRED_API_KEY:
            logger.warning("FRED: No API key configured, skipping")
            return events

        for series_id in IngestionConfig.FRED_SERIES_IDS:
            try:
                params = {
                    "series_id": series_id,
                    "api_key": IngestionConfig.FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                }
                resp = requests.get(FRED_API, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                observations = data.get("observations", [])
                if not observations:
                    continue

                latest = observations[0]
                value = latest.get("value", "")
                date = latest.get("date", "")
                series_name = SERIES_NAMES.get(series_id, series_id)

                content = f"FRED {series_name}: {value} (as of {date})"
                title = f"{series_name} — {date}"

                timestamp = datetime.now(timezone.utc)
                if date:
                    try:
                        timestamp = datetime.strptime(date, "%Y-%m-%d")
                    except ValueError:
                        pass

                events.append(Event(
                    title=title,
                    content=content,
                    source="fred",
                    source_type=self.source_type,
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    timestamp=timestamp,
                    content_hash=compute_content_hash(f"fred-{series_id}-{date}"),
                    signal_role="resolution",
                ))

            except Exception as e:
                logger.warning(f"FRED: Failed to fetch {series_id}: {e}")

        logger.info(f"FRED: Retrieved data for {len(events)} series")
        return events
