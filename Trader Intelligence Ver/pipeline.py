"""
Pipeline Orchestrator

Runs the prediction market pipeline as a sequence of composable stages.
Each stage reads from the database and writes results back.

Usage:
    python pipeline.py              # run all stages
    python pipeline.py --stage 1    # run only FR1
    python pipeline.py --stage 1-2  # run FR1 and FR2
    python pipeline.py --stage 4    # run FR4 only
    python pipeline.py --stage 1-4  # run full pipeline
"""

import argparse
from collections import Counter
import logging
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from config import PipelineConfig
from db.connection import (
    init_db, get_all_events, insert_cluster,
    get_clusters_for_extraction, insert_extracted_event,
    get_extracted_events_for_generation, insert_candidate_question,
    get_candidate_questions_for_validation, insert_validation_result,
    get_validated_questions_for_scoring, insert_scored_candidate,
    get_all_candidate_question_texts,
    get_extracted_event_by_id, question_has_repair_child,
    mark_pipeline_run_started, mark_pipeline_run_completed,
    mark_pipeline_run_failed, update_pipeline_run_stage,
)
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
from generation.generator import QuestionGenerator
from models import Cluster
from ranking.story_dedupe import dedupe_extracted_events, questions_are_near_duplicates
from validation.validator import validate_question, is_salvageable_validation_flags
from scoring.scorer import score_questions_with_breakdown

logger = logging.getLogger(__name__)
STAGE_SEPARATOR = "-" * 40
THIRD_PARTY_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "huggingface_hub",
    "transformers",
    "groq",
    "google",
]


class _NormalConsoleFilter(logging.Filter):
    """Allow pipeline summaries plus warnings/errors from every logger."""

    def __init__(self, pipeline_logger_name: str):
        super().__init__()
        self.pipeline_logger_name = pipeline_logger_name

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name == self.pipeline_logger_name


def _resolve_log_mode(log_mode: str = None, debug: bool = False) -> str:
    """Resolve the effective console logging mode."""
    if debug:
        return "debug"

    mode = (log_mode or PipelineConfig.LOG_MODE or "normal").strip().lower()
    if mode not in {"normal", "debug"}:
        raise ValueError(f"Unsupported log mode: {log_mode}")
    return mode


def _set_external_logger_levels() -> None:
    """Keep third-party client chatter out of the terminal in both modes."""
    for logger_name in THIRD_PARTY_NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _configure_logging(log_mode: str) -> None:
    """Configure console + file logging for normal vs debug runs."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if log_mode == "debug" else logging.INFO)
    if log_mode == "debug":
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    else:
        console_handler.addFilter(_NormalConsoleFilter(logger.name))
        console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler("pipeline.log")
    file_handler.setLevel(logging.DEBUG if log_mode == "debug" else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    _set_external_logger_levels()


def _format_summary(summary: Dict) -> str:
    """Compact stage summary text for normal-mode console output."""
    parts: list[str] = []
    for key, value in summary.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else "no additional details"


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
    source_successes = 0
    source_failures = 0
    for ingestor in ingestors:
        try:
            new = ingestor.ingest()
            total_new += new
            source_successes += 1
        except Exception as e:
            source_failures += 1
            logger.warning(f"Ingestor {ingestor.__class__.__name__} failed: {e}")
    return {
        "sources": len(ingestors),
        "sources_ok": source_successes,
        "sources_failed": source_failures,
        "new_events": total_new,
    }


def run_clustering():
    """FR2: Embed events, deduplicate near-duplicates, and cluster."""
    events = get_all_events()
    if not events:
        logger.info("FR2: No events to cluster")
        return {"events_loaded": 0, "events_after_dedup": 0, "clusters": 0}

    events_loaded = len(events)

    # Embed
    embedder = Embedder()
    texts = [e.content for e in events]
    embeddings = embedder.embed(texts)

    # Deduplicate near-duplicates before clustering
    events, embeddings = deduplicate_near_duplicates(events, embeddings)
    logger.info(f"FR2: {len(events)} events after near-duplicate removal")

    # Cluster
    engine = ClusterEngine()
    label_to_events = engine.cluster(embeddings, events)

    # Compute features and filter
    clusters = build_clusters(label_to_events)

    # Save to DB
    for cluster in clusters:
        cluster_id = insert_cluster(cluster)
        logger.debug(f"Saved cluster {cluster_id} with {len(cluster.events)} events")

    return {
        "events_loaded": events_loaded,
        "events_after_dedup": len(events),
        "clusters": len(clusters),
    }


def run_extraction(
    max_clusters: int = None,
    model: str = None,
    progress_reporter: Optional[Callable[[Dict], None]] = None,
):
    """FR3: Extract structured, market-ready event specs from clusters using LLM."""
    clusters = get_clusters_for_extraction(limit=max_clusters)
    if not clusters:
        logger.info("FR3: No clusters to extract")
        return {"pending_clusters": 0, "extracted_events": 0, "saved_events": 0}

    extractor = EventExtractor(model=model)
    extracted_events = []
    total_clusters = len(clusters)

    for index, cluster_data in enumerate(clusters, start=1):
        cluster = Cluster(
            events=cluster_data["events"],
            features=cluster_data["features"],
            label=cluster_data["label"],
        )
        extracted = extractor.extract(cluster, cluster_data["cluster_id"])
        if extracted:
            event_id = insert_extracted_event(extracted)
            extracted.id = event_id
            extracted_events.append(extracted)
            logger.debug(f"Saved extracted event {event_id} for cluster {extracted.cluster_id}")

        if progress_reporter is not None:
            progress_reporter(
                {
                    "pending_clusters": total_clusters,
                    "clusters_processed": index,
                    "progress_pct": round((index / total_clusters) * 100, 1),
                    "extracted_events": len(extracted_events),
                    "saved_events": len(extracted_events),
                }
            )

    return {
        "pending_clusters": total_clusters,
        "clusters_processed": total_clusters,
        "progress_pct": 100.0,
        "extracted_events": len(extracted_events),
        "saved_events": len(extracted_events),
    }


def run_question_generation(
    max_events: int = None,
    model: str = None,
    progress_reporter: Optional[Callable[[Dict], None]] = None,
):
    """FR4: Generate candidate prediction market questions from extracted events."""
    raw_events = get_extracted_events_for_generation(limit=max_events)
    if not raw_events:
        logger.info("FR4: No extracted events to generate questions for")
        return {"eligible_events": 0, "questions_generated": 0}

    events = dedupe_extracted_events(raw_events)
    deduped_event_count = len(raw_events) - len(events)
    if deduped_event_count > 0:
        logger.info(
            f"FR4: deduplicated {deduped_event_count} overlapping extracted events before generation"
        )

    generator = QuestionGenerator(model=model)
    questions = []
    existing_question_texts = [text for _, text in get_all_candidate_question_texts()]
    total_events = len(events)

    for index, event in enumerate(events, start=1):
        generated = generator.generate(event)
        story_questions = []
        for q in generated:
            # One upstream story can still fan out into many slight variants even
            # after extraction dedupe. Cap and cross-check here so the final
            # queue stays diverse enough for human review instead of being
            # dominated by one ceasefire, earnings, or election narrative.
            if len(story_questions) >= PipelineConfig.FR4_MAX_QUESTIONS_PER_STORY:
                logger.info(
                    f"ExtractedEvent {event.id}: capped at "
                    f"{PipelineConfig.FR4_MAX_QUESTIONS_PER_STORY} questions for one story"
                )
                break

            if any(
                questions_are_near_duplicates(q.question_text, existing_text)
                for existing_text in existing_question_texts
            ):
                logger.info(
                    f"ExtractedEvent {event.id}: skipped near-duplicate question "
                    f"already present in queue: {q.question_text[:90]}"
                )
                continue

            story_questions.append(q)
            existing_question_texts.append(q.question_text)

        questions.extend(story_questions)
        for q in story_questions:
            q_id = insert_candidate_question(q)
            logger.debug(f"Saved candidate question {q_id}: {q.question_text[:80]}...")

        if progress_reporter is not None:
            progress_reporter(
                {
                    "eligible_events": total_events,
                    "raw_eligible_events": len(raw_events),
                    "events_processed": index,
                    "progress_pct": round((index / total_events) * 100, 1),
                    "questions_generated": len(questions),
                }
            )

    return {
        "eligible_events": total_events,
        "raw_eligible_events": len(raw_events),
        "events_processed": total_events,
        "progress_pct": 100.0,
        "questions_generated": len(questions),
    }


def run_validation():
    """FR5: Run deterministic rule validation on candidate questions."""
    questions = get_candidate_questions_for_validation()
    if not questions:
        logger.info("FR5: No candidate questions to validate")
        return {
            "questions_checked": 0,
            "passed": 0,
            "failed": 0,
            "top_flags": "-",
        }

    valid_count = 0
    invalid_count = 0
    repaired_count = 0
    repair_salvaged = 0
    flag_counts: Counter[str] = Counter()
    repair_generator = QuestionGenerator()
    for q in questions:
        result = validate_question(q)
        insert_validation_result(result)
        if result.is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            flag_counts.update(result.flags)
        logger.debug(
            f"Validated question {q.id}: is_valid={result.is_valid}, "
            f"flags={len(result.flags)}, clarity={result.clarity_score:.2f}"
        )

        if (
            not result.is_valid
            and q.id is not None
            and is_salvageable_validation_flags(result.flags)
            and not question_has_repair_child(q.id)
        ):
            extracted_event = get_extracted_event_by_id(q.extracted_event_id)
            if extracted_event is not None:
                repaired = repair_generator.repair_question(extracted_event, q, result.flags)
                if repaired is not None:
                    repaired_count += 1
                    repaired_question_id = insert_candidate_question(repaired)
                    repaired.id = repaired_question_id
                    repaired_validation = validate_question(repaired)
                    insert_validation_result(repaired_validation)
                    if repaired_validation.is_valid:
                        repair_salvaged += 1
                        valid_count += 1
                    else:
                        invalid_count += 1
                        flag_counts.update(repaired_validation.flags)
                    logger.debug(
                        f"Repaired question {repaired_question_id}: "
                        f"is_valid={repaired_validation.is_valid}, flags={repaired_validation.flags}"
                    )

    top_flags = ", ".join(
        f"{flag}({count})" for flag, count in flag_counts.most_common(3)
    ) or "-"
    return {
        "questions_checked": len(questions),
        "passed": valid_count,
        "failed": invalid_count,
        "repaired": repaired_count,
        "repair_salvaged": repair_salvaged,
        "top_flags": top_flags,
    }


def run_scoring():
    """FR6: Score validated candidate questions with deterministic heuristics."""
    rows = get_validated_questions_for_scoring()
    if not rows:
        logger.info("FR6: No validated questions to score")
        return {"validated_questions": 0, "scored": 0, "top_score": None}

    all_question_texts_by_id = get_all_candidate_question_texts()
    scored_candidates, _ = score_questions_with_breakdown(rows, all_question_texts_by_id)

    for scored in scored_candidates:
        insert_scored_candidate(scored)

    top_score = max((candidate.total_score for candidate in scored_candidates), default=None)
    return {
        "validated_questions": len(rows),
        "scored": len(scored_candidates),
        "top_score": top_score,
    }


# ---- Stage registry ----
STAGES: List[Tuple[str, Callable]] = [
    ("FR1: Event Ingestion", run_ingestion),
    ("FR2: Event Clustering", run_clustering),
    ("FR3: LLM Extraction", run_extraction),
    ("FR4: Question Generation", run_question_generation),
    ("FR5: Rule Validation", run_validation),
    ("FR6: Heuristic Scoring", run_scoring),
]


def run_pipeline(
    start: int = 1,
    end: int = None,
    fr3_limit: int = None,
    fr4_limit: int = None,
    fr3_model: str = None,
    fr4_model: str = None,
    log_mode: str = None,
    debug: bool = False,
    run_id: int = None,
):
    """Run pipeline stages from start to end (1-indexed)."""
    if end is None:
        end = len(STAGES)

    effective_log_mode = _resolve_log_mode(log_mode, debug=debug)
    _configure_logging(effective_log_mode)

    logger.info("=" * 60)
    logger.info("Prediction Market Pipeline - Starting")
    logger.info("=" * 60)
    logger.info(
        "Run config: "
        f"stages={start}-{end} | "
        f"log_mode={effective_log_mode} | "
        f"fr3_limit={fr3_limit if fr3_limit is not None else 'default'} | "
        f"fr4_limit={fr4_limit if fr4_limit is not None else 'default'} | "
        f"fr3_model={fr3_model or 'config/default'} | "
        f"fr4_model={fr4_model or 'config/default'}"
    )

    # Initialize database tables
    init_db()
    logger.info("Database initialized")
    if run_id is not None:
        mark_pipeline_run_started(run_id)

    pipeline_start = time.time()

    for i, (name, stage_fn) in enumerate(STAGES[start - 1 : end], start=start):
        logger.info(f"\n{STAGE_SEPARATOR}")
        logger.info(f"Stage {i}/{len(STAGES)}: {name}")
        logger.info(STAGE_SEPARATOR)
        if run_id is not None:
            update_pipeline_run_stage(run_id, i, name, "running")

        stage_start = time.time()
        try:
            progress_reporter = None
            if run_id is not None:
                def progress_reporter(summary: Dict, *, _run_id=run_id, _stage=i, _name=name):
                    update_pipeline_run_stage(_run_id, _stage, _name, "running", summary=summary)

            if stage_fn is run_extraction:
                summary = stage_fn(
                    max_clusters=fr3_limit,
                    model=fr3_model,
                    progress_reporter=progress_reporter,
                )
            elif stage_fn is run_question_generation:
                summary = stage_fn(
                    max_events=fr4_limit,
                    model=fr4_model,
                    progress_reporter=progress_reporter,
                )
            else:
                summary = stage_fn()
            elapsed = time.time() - stage_start
            if run_id is not None:
                update_pipeline_run_stage(run_id, i, name, "completed", summary=summary or {})
            logger.info(f"Summary: {_format_summary(summary or {})}")
            logger.info(f"Stage {i} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - stage_start
            if run_id is not None:
                update_pipeline_run_stage(
                    run_id,
                    i,
                    name,
                    "failed",
                    error_message=str(e),
                )
                mark_pipeline_run_failed(run_id, str(e))
            logger.error(f"Stage {i} FAILED after {elapsed:.1f}s: {e}", exc_info=True)
            logger.error("Pipeline stopped due to stage failure")
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start
    if run_id is not None:
        mark_pipeline_run_completed(run_id)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline completed in {total_elapsed:.1f}s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prediction market pipeline")
    parser.add_argument(
        "--stage", type=str, default=None,
        help="Stage range to run, e.g. '1', '1-2', '2-3', '1-4'. Default: run all."
    )
    parser.add_argument(
        "--fr3-limit", type=int, default=PipelineConfig.FR3_MAX_CLUSTERS,
        help=(
            "Maximum number of clusters to send through FR3 extraction. "
            "Default: FR3_MAX_CLUSTERS from config/.env."
        ),
    )
    parser.add_argument(
        "--fr4-limit", type=int, default=PipelineConfig.FR4_MAX_EVENTS,
        help=(
            "Maximum number of extracted events to send through FR4 question generation. "
            "Default: FR4_MAX_EVENTS from config/.env."
        ),
    )
    parser.add_argument(
        "--log-mode",
        choices=["normal", "debug"],
        default=None,
        help="Console logging mode. Default comes from LOG_MODE in config/.env.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shortcut for --log-mode debug.",
    )
    parser.add_argument(
        "--fr3-all",
        action="store_true",
        help="Process all pending clusters in FR3, ignoring the FR3 max-clusters cap.",
    )
    parser.add_argument(
        "--fr4-all",
        action="store_true",
        help="Process all eligible extracted events in FR4, ignoring the FR4 max-events cap.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Optional DB-backed pipeline run id used for Streamlit progress tracking.",
    )
    parser.add_argument(
        "--fr3-model",
        type=str,
        default=None,
        help="Optional model override for FR3 extraction.",
    )
    parser.add_argument(
        "--fr4-model",
        type=str,
        default=None,
        help="Optional model override for FR4 question generation / repair.",
    )
    args = parser.parse_args()
    effective_fr3_limit = None if args.fr3_all else args.fr3_limit
    effective_fr4_limit = None if args.fr4_all else args.fr4_limit

    if args.stage:
        if "-" in args.stage:
            start, end = args.stage.split("-")
            run_pipeline(
                start=int(start),
                end=int(end),
                fr3_limit=effective_fr3_limit,
                fr4_limit=effective_fr4_limit,
                fr3_model=args.fr3_model,
                fr4_model=args.fr4_model,
                log_mode=args.log_mode,
                debug=args.debug,
                run_id=args.run_id,
            )
        else:
            stage = int(args.stage)
            run_pipeline(
                start=stage,
                end=stage,
                fr3_limit=effective_fr3_limit,
                fr4_limit=effective_fr4_limit,
                fr3_model=args.fr3_model,
                fr4_model=args.fr4_model,
                log_mode=args.log_mode,
                debug=args.debug,
                run_id=args.run_id,
            )
    else:
        run_pipeline(
            fr3_limit=effective_fr3_limit,
            fr4_limit=effective_fr4_limit,
            fr3_model=args.fr3_model,
            fr4_model=args.fr4_model,
            log_mode=args.log_mode,
            debug=args.debug,
            run_id=args.run_id,
        )
