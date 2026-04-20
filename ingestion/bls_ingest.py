"""
FR1: BLS (Bureau of Labor Statistics) Ingestion

Fetches recent data from the BLS public API.
No registration required for the public API (v1), but rate-limited.
Tracks CPI, unemployment, payrolls, and other economic indicators.
"""

import logging
from datetime import datetime
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

BLS_API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_API_V1 = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

# Human-readable names for common series
SERIES_NAMES = {
    "CUUR0000SA0": "CPI-U (All Urban Consumers)",
    "LNS14000000": "Unemployment Rate",
    "CES0000000001": "Total Nonfarm Payrolls",
    "JTS000000000000000JOR": "JOLTS Job Openings Rate",
}


class BLSIngestor(BaseIngestor):
    source_type = "official"

    def fetch(self) -> List[Event]:
        events = []

        try:
            api_url = BLS_API_V2 if IngestionConfig.BLS_API_KEY else BLS_API_V1
            payload = {
                "seriesid": IngestionConfig.BLS_SERIES_IDS,
                "latest": True,
            }
            if IngestionConfig.BLS_API_KEY:
                payload["registrationkey"] = IngestionConfig.BLS_API_KEY

            resp = requests.post(api_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for series in data.get("Results", {}).get("series", []):
                series_id = series.get("seriesID", "")
                series_name = SERIES_NAMES.get(series_id, series_id)
                series_data = series.get("data", [])

                if not series_data:
                    continue

                latest = series_data[0]
                value = latest.get("value", "")
                year = latest.get("year", "")
                period = latest.get("periodName", "")

                content = f"BLS {series_name}: {value} ({period} {year})"
                title = f"{series_name} — {period} {year}"

                events.append(Event(
                    title=title,
                    content=content,
                    source="bls",
                    source_type=self.source_type,
                    url=f"https://data.bls.gov/timeseries/{series_id}",
                    timestamp=datetime.utcnow(),
                    content_hash=compute_content_hash(f"bls-{series_id}-{year}-{period}"),
                    signal_role="resolution",
                ))

            logger.info(f"BLS: Retrieved data for {len(events)} series")

        except Exception as e:
            logger.warning(f"BLS: Failed to fetch: {e}")

        return events
