"""
Pipeline Orchestrator

Runs the prediction market pipeline as a sequence of composable stages.
Each stage reads from the database and writes results back.

To add new stages (FR4-FR7), append to the STAGES list:
    STAGES.append(("FR4: Question Generation", run_question_gen))

Usage:
    python pipeline.py              # run all stages
    python pipeline.py --stage 1    # run only FR1
    python pipeline.py --stage 1-2  # run FR1 and FR2
"""

import argparse
import logging
import sys
import time
from typing import Callable, List, Tuple

from config import PipelineConfig
from db.connection import init_db, get_all_events, insert_cluster, get_clusters_for_extraction, insert_extracted_event
from ingestion.rss_ingest import RSSIngestor
from ingestion.gdelt_ingest import GDELTIngestor
from ingestion.market_ingest import MarketIngestor
from ingestion.reddit_ingest import RedditIngestor
from ingestion.hn_ingest import HackerNewsIngestor
from ingestion.wikipedia_ingest import WikipediaIngestor
from ingestion.federal_register_ingest import FederalRegisterIngestor
from ingestion.congress_ingest import CongressIngestor
from ingestion.sec_ingest import SECIngestor
from ingestion.bls_ingest import BLSIngestor
from ingestion.fred_ingest import FREDIngestor
from ingestion.eia_ingest import EIAIngestor
from ingestion.kalshi_ingest import KalshiIngestor
from clustering.embedder import Embedder
from clustering.cluster import ClusterEngine
from clustering.features import build_clusters, deduplicate_near_duplicates
from extraction.extractor import EventExtractor

logger = logging.getLogger(__name__)


# ---- Stage implementations ----

def run_ingestion():
    """FR1: Run all configured ingestors."""
    ingestors = [
        # News / discovery sources
        RSSIngestor(),
        GDELTIngestor(),
        # Social / attention sources
        RedditIngestor(),
        HackerNewsIngestor(),
        WikipediaIngestor(),
        # Official / resolution sources
        FederalRegisterIngestor(),
        CongressIngestor(),
        SECIngestor(),
        BLSIngestor(),
        FREDIngestor(),
        EIAIngestor(),
        # Market / benchmark sources
        MarketIngestor(),
        KalshiIngestor(),
    ]
    total_new = 0
    for ingestor in ingestors:
        try:
            total_new += ingestor.ingest()
        except Exception as e:
            logger.warning(f"Ingestor {ingestor.__class__.__name__} failed: {e}")
    logger.info(f"FR1 complete: {total_new} new events ingested")


def run_clustering():
    """FR2: Embed events, deduplicate near-duplicates, and cluster."""
    events = get_all_events()
    if not events:
        logger.warning("FR2: No events to cluster")
        return

    # Embed
    embedder = Embedder()
    texts = [e.content for e in events]
    embeddings = embedder.embed(texts)

    # Near-duplicate detection before clustering
    events = deduplicate_near_duplicates(events, embeddings)
    # Re-embed if deduplication removed items
    if len(events) < len(texts):
        texts = [e.content for e in events]
        embeddings = embedder.embed(texts)

    # Cluster (with embeddings for coherence computation)
    engine = ClusterEngine()
    label_to_events, label_to_embeddings = engine.cluster_with_embeddings(embeddings, events)

    # Compute features (including coherence, source_role_mix, weighted velocity) and filter
    clusters = build_clusters(label_to_events, label_to_embeddings=label_to_embeddings)

    # Save to DB
    for cluster in clusters:
        cluster_id = insert_cluster(cluster)
        logger.info(f"Saved cluster {cluster_id} with {len(cluster.events)} events")

    logger.info(f"FR2 complete: {len(clusters)} clusters saved")


def run_extraction():
    """FR3: Extract structured, market-ready event specs from clusters using LLM."""
    clusters = get_clusters_for_extraction()
    if not clusters:
        logger.warning("FR3: No clusters to extract (all already processed or none exist)")
        return

    extractor = EventExtractor()
    extracted_events = extractor.extract_batch(clusters)

    for extracted in extracted_events:
        event_id = insert_extracted_event(extracted)
        logger.info(f"Saved extracted event {event_id} for cluster {extracted.cluster_id}")

    logger.info(f"FR3 complete: {len(extracted_events)} events extracted")


# ---- Stage registry ----
# Teammates: append your stages here
STAGES: List[Tuple[str, Callable]] = [
    ("FR1: Event Ingestion", run_ingestion),
    ("FR2: Event Clustering", run_clustering),
    ("FR3: LLM Extraction", run_extraction),
    # ("FR4: Question Generation", run_question_gen),
    # ("FR5: Rule Validation", run_validation),
    # ("FR6: Heuristic Scoring", run_scoring),
]


def run_pipeline(start: int = 1, end: int = None):
    """Run pipeline stages from start to end (1-indexed)."""
    if end is None:
        end = len(STAGES)

    logging.basicConfig(
        level=getattr(logging, PipelineConfig.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("Prediction Market Pipeline — Starting")
    logger.info("=" * 60)

    # Initialize database tables
    init_db()

    pipeline_start = time.time()

    for i, (name, stage_fn) in enumerate(STAGES[start - 1 : end], start=start):
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Stage {i}/{len(STAGES)}: {name}")
        logger.info(f"{'─' * 40}")

        stage_start = time.time()
        try:
            stage_fn()
            elapsed = time.time() - stage_start
            logger.info(f"Stage {i} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - stage_start
            logger.error(f"Stage {i} FAILED after {elapsed:.1f}s: {e}", exc_info=True)
            logger.error("Pipeline stopped due to stage failure")
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline completed in {total_elapsed:.1f}s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prediction market pipeline")
    parser.add_argument(
        "--stage", type=str, default=None,
        help="Stage range to run, e.g. '1', '1-2', '2-3'. Default: run all."
    )
    args = parser.parse_args()

    if args.stage:
        if "-" in args.stage:
            start, end = args.stage.split("-")
            run_pipeline(start=int(start), end=int(end))
        else:
            stage = int(args.stage)
            run_pipeline(start=stage, end=stage)
    else:
        run_pipeline()
