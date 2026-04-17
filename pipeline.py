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
from db.connection import (
    init_db, get_all_events, insert_cluster,
    get_clusters_for_extraction, insert_extracted_event,
    get_extracted_events_for_generation, insert_candidate_question,
    get_candidate_questions_for_validation, insert_validation_result,
    get_validated_questions_for_scoring, insert_scored_candidate,
    get_all_candidate_question_texts,
)
from ingestion.rss_ingest import RSSIngestor
from ingestion.gdelt_ingest import GDELTIngestor
from ingestion.market_ingest import MarketIngestor
from clustering.embedder import Embedder
from clustering.cluster import ClusterEngine
from clustering.features import build_clusters
from extraction.extractor import EventExtractor
from generation.generator import QuestionGenerator
from validation.validator import validate_question
from scoring.scorer import score_questions_with_breakdown

logger = logging.getLogger(__name__)

# Limits number of events processed for faster demo runs.
# Set to None to process all events.
MAX_EVENTS = 5  # set to None for no limit


# ---- Stage implementations ----

def run_ingestion():
    """FR1: Run all configured ingestors."""
    ingestors = [RSSIngestor(), GDELTIngestor(), MarketIngestor()]
    total_new = 0
    for ingestor in ingestors:
        total_new += ingestor.ingest()
    logger.info(f"FR1 complete: {total_new} new events ingested")


def run_clustering():
    """FR2: Embed events and cluster them."""
    events = get_all_events()
    if not events:
        logger.warning("FR2: No events to cluster")
        return

    # Embed
    embedder = Embedder()
    texts = [e.content for e in events]
    embeddings = embedder.embed(texts)

    # Cluster
    engine = ClusterEngine()
    label_to_events = engine.cluster(embeddings, events)

    # Compute features and filter
    clusters = build_clusters(label_to_events)

    # Save to DB
    for cluster in clusters:
        cluster_id = insert_cluster(cluster)
        logger.info(f"Saved cluster {cluster_id} with {len(cluster.events)} events")

    logger.info(f"FR2 complete: {len(clusters)} clusters saved")


def run_extraction():
    """FR3: Extract structured events from clusters using LLM."""
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


def run_question_generation():
    """FR4: Generate candidate prediction market questions from extracted events."""
    events = get_extracted_events_for_generation()
    if MAX_EVENTS is not None:
        events = events[:MAX_EVENTS]
    if not events:
        logger.warning("FR4: No extracted events to generate questions for (all already processed or none exist)")
        return

    generator = QuestionGenerator()
    questions = generator.generate_batch(events)

    for q in questions:
        q_id = insert_candidate_question(q)
        logger.debug(f"Saved candidate question {q_id}: {q.question_text[:60]}...")

    logger.info(f"FR4 complete: {len(questions)} candidate questions generated from {len(events)} events")


def run_validation():
    """FR5: Run deterministic validation on candidate questions."""
    questions = get_candidate_questions_for_validation()
    if not questions:
        logger.warning("FR5: No candidate questions to validate (all already processed or none exist)")
        return

    for q in questions:
        result = validate_question(q)
        result_id = insert_validation_result(result)
        logger.debug(
            f"Saved validation result {result_id} for question {q.id}: "
            f"is_valid={result.is_valid}, flags={len(result.flags)}, clarity={result.clarity_score:.2f}"
        )

    logger.info(f"FR5 complete: {len(questions)} questions validated")


def run_scoring():
    """FR6: Score validated candidate questions with deterministic heuristics."""
    rows = get_validated_questions_for_scoring()
    if not rows:
        logger.warning("FR6: No validated questions to score (all already processed or none exist)")
        logger.info("FR6 complete: 0 validated questions scored")
        return

    all_question_texts_by_id = get_all_candidate_question_texts()
    scored_candidates, breakdowns = score_questions_with_breakdown(rows, all_question_texts_by_id)

    for scored in scored_candidates:
        breakdown = breakdowns.get(scored.question_id, {})
        component_scores = breakdown.get("component_scores", {})
        quality_flags = breakdown.get("quality_flags", {})

        logger.debug(
            f"[RANK {scored.rank}] Q{scored.question_id} | total={scored.total_score:.4f} | "
            f"{breakdown.get('question_text', '')}"
        )

        scored_id = insert_scored_candidate(scored)

        logger.debug(
            f"Saved scored candidate {scored_id} for question {scored.question_id}: "
            f"total={scored.total_score:.4f}, rank={scored.rank}"
        )
        logger.debug(
            "FR6 breakdown | "
            f"question_id={breakdown.get('question_id', scored.question_id)} | "
            f"question_text={breakdown.get('question_text', '')} | "
            f"rank={breakdown.get('rank', scored.rank)} | "
            f"total_score={breakdown.get('total_score', scored.total_score):.4f} | "
            f"clarity_score={component_scores.get('clarity_score', 0.0):.4f} | "
            f"mention_velocity_score={component_scores.get('mention_velocity_score', 0.0):.4f} | "
            f"source_diversity_score={component_scores.get('source_diversity_score', 0.0):.4f} | "
            f"novelty_score={component_scores.get('novelty_score', 0.0):.4f} | "
            f"market_interest_score={component_scores.get('market_interest_score', 0.0):.4f} | "
            f"resolution_strength_score={component_scores.get('resolution_strength_score', 0.0):.4f} | "
            f"time_horizon_score={component_scores.get('time_horizon_score', 0.0):.4f} | "
            f"homepage_source={quality_flags.get('homepage_source', False)} | "
            f"promo_event={quality_flags.get('promo_event', False)} | "
            f"retail_promo_event={quality_flags.get('retail_promo_event', False)} | "
            f"low_significance_event={quality_flags.get('low_significance_event', False)} | "
            f"near_duplicate_theme={quality_flags.get('near_duplicate_theme', False)} | "
            f"weather_event={quality_flags.get('weather_event', False)} | "
            f"final_clamped_score={breakdown.get('final_clamped_score', scored.total_score):.4f}"
        )

    logger.info(f"FR6 complete: {len(scored_candidates)} validated questions scored")


# ---- Stage registry ----
# Teammates: append your stages here
STAGES: List[Tuple[str, Callable]] = [
    ("FR1: Event Ingestion", run_ingestion),
    ("FR2: Event Clustering", run_clustering),
    ("FR3: LLM Extraction", run_extraction),
    ("FR4: Question Generation", run_question_generation),
    ("FR5: Rule Validation", run_validation),
    ("FR6: Heuristic Scoring", run_scoring),
]


def _root_log_level(debug: bool) -> int:
    if debug:
        return logging.DEBUG
    name = (PipelineConfig.LOG_LEVEL or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def _configure_third_party_loggers(debug: bool) -> None:
    level = logging.INFO if debug else logging.WARNING
    for log_name in ("httpx", "groq", "groq._base_client"):
        logging.getLogger(log_name).setLevel(level)


def run_pipeline(start: int = 1, end: int = None, debug: bool = False):
    """Run pipeline stages from start to end (1-indexed)."""
    if end is None:
        end = len(STAGES)

    logging.basicConfig(
        level=_root_log_level(debug),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log"),
        ],
    )
    _configure_third_party_loggers(debug)

    logger.info("=" * 60)
    logger.info("Prediction Market Pipeline — Starting")
    logger.info("=" * 60)

    # Initialize database tables
    init_db()

    pipeline_start = time.time()

    for i, (name, stage_fn) in enumerate(STAGES[start - 1 : end], start=start):
        logger.info(f"\n{'-' * 40}")
        logger.info(f"Stage {i}/{len(STAGES)}: {name}")
        logger.info(f"{'-' * 40}")

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging: DEBUG on root, per-item FR4/FR5/FR6 logs, and INFO on httpx/groq.",
    )
    args = parser.parse_args()

    if args.stage:
        if "-" in args.stage:
            start, end = args.stage.split("-")
            run_pipeline(start=int(start), end=int(end), debug=args.debug)
        else:
            stage = int(args.stage)
            run_pipeline(start=stage, end=stage, debug=args.debug)
    else:
        run_pipeline(debug=args.debug)
