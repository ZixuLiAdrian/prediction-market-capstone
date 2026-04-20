"""
FR1: EIA (Energy Information Administration) Ingestion

Fetches energy market data from the EIA API v2.
Requires a free API key from https://www.eia.gov/opendata/
Covers oil, gas, electricity, inventories, and production data.
"""

import logging
from datetime import datetime
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

EIA_API_V2 = "https://api.eia.gov/v2/seriesid/{series_id}"

SERIES_NAMES = {
    "PET.RWTC.W": "WTI Crude Oil Price (Weekly)",
    "NG.RNGWHHD.W": "Henry Hub Natural Gas Spot Price (Weekly)",
    "PET.WCESTUS1.W": "U.S. Crude Oil Stocks (Weekly)",
}


class EIAIngestor(BaseIngestor):
    source_type = "official"

    def fetch(self) -> List[Event]:
        events = []

        if not IngestionConfig.EIA_API_KEY:
            logger.warning("EIA: No API key configured, skipping")
            return events

        for series_id in IngestionConfig.EIA_SERIES_IDS:
            try:
                url = f"https://api.eia.gov/v2/seriesid/{series_id}"
                params = {
                    "api_key": IngestionConfig.EIA_API_KEY,
                    "num": 1,
                }
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                series_data = data.get("response", {}).get("data", [])
                if not series_data:
                    continue

                latest = series_data[0]
                value = latest.get("value", "")
                period = latest.get("period", "")
                series_name = SERIES_NAMES.get(series_id, series_id)

                content = f"EIA {series_name}: {value} (period: {period})"
                title = f"{series_name} — {period}"

                timestamp = datetime.utcnow()
                if period:
                    try:
                        timestamp = datetime.strptime(period, "%Y-%m-%d")
                    except ValueError:
                        pass

                events.append(Event(
                    title=title,
                    content=content,
                    source="eia",
                    source_type=self.source_type,
                    url=f"https://www.eia.gov/opendata/browser/?id={series_id}",
                    timestamp=timestamp,
                    content_hash=compute_content_hash(f"eia-{series_id}-{period}"),
                    signal_role="resolution",
                ))

            except Exception as e:
                logger.warning(f"EIA: Failed to fetch {series_id}: {e}")

        logger.info(f"EIA: Retrieved data for {len(events)} series")
        return events
