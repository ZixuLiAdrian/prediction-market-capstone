"""
Database connection helper and common query functions.

Provides a connection pool and typed helper functions for each table.
Downstream modules (FR4-FR7) can import and extend these helpers.
"""

import json
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from config import DBConfig
from ranking.market_priority import compute_cluster_priority, compute_extracted_event_priority
from models import Event, Cluster, ClusterFeatures, ExtractedEvent, CandidateQuestion, ValidationResult, ScoredCandidate

logger = logging.getLogger(__name__)

_connection = None


def get_connection():
    """Get or create a database connection."""
    global _connection
    if _connection is None or _connection.closed:
        _connection = psycopg2.connect(DBConfig.connection_string())
        _connection.autocommit = True
    return _connection


@contextmanager
def get_cursor():
    """Context manager for database cursor with auto-cleanup."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cursor
    finally:
        cursor.close()


def init_db():
    """Initialize database tables from schema.sql."""
    import os
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    with get_cursor() as cur:
        cur.execute(schema_sql)
    logger.info("Database tables initialized")


# ---- FR1: Event helpers ----

def insert_event(event: Event) -> Optional[int]:
    """Insert an event, skip if content_hash already exists. Returns event ID or None."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO events (title, content, source, source_type, url, entities, content_hash, signal_role, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id
            """,
            (event.title, event.content, event.source, event.source_type,
             event.url, event.entities, event.content_hash, event.signal_role,
             event.timestamp),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def get_all_events() -> List[Event]:
    """Retrieve all events from the database."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM events ORDER BY timestamp DESC")
        rows = cur.fetchall()
    return [
        Event(
            id=r["id"], title=r["title"], content=r["content"],
            source=r["source"], source_type=r["source_type"],
            url=r["url"], entities=r["entities"],
            content_hash=r["content_hash"],
            signal_role=r.get("signal_role", "discovery"),
            timestamp=r["timestamp"],
        )
        for r in rows
    ]


# ---- FR2: Cluster helpers ----

def insert_cluster(cluster: Cluster) -> int:
    """Insert a cluster and its event mappings. Returns cluster ID."""
    source_role_mix = {
        str(role): int(count)
        for role, count in (cluster.features.source_role_mix or {}).items()
    }

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO clusters (label, mention_velocity, source_diversity, recency, size,
                                  source_role_mix, coherence_score, weighted_mention_velocity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (int(cluster.label), float(cluster.features.mention_velocity),
             int(cluster.features.source_diversity), float(cluster.features.recency),
             len(cluster.events),
             json.dumps(source_role_mix),
             float(cluster.features.coherence_score),
             float(cluster.features.weighted_mention_velocity)),
        )
        cluster_id = cur.fetchone()["id"]

        for event in cluster.events:
            if event.id is not None:
                cur.execute(
                    "INSERT INTO cluster_events (cluster_id, event_id) VALUES (%s, %s)",
                    (cluster_id, event.id),
                )
    return cluster_id


def get_clusters_for_extraction(limit: Optional[int] = None) -> List[dict]:
    """Get clusters that haven't been extracted yet, optionally capped by signal strength."""
    with get_cursor() as cur:
        query = """
            SELECT c.id, c.label, c.mention_velocity, c.source_diversity, c.recency,
                   c.source_role_mix, c.coherence_score, c.weighted_mention_velocity
            FROM clusters c
            WHERE c.id NOT IN (SELECT cluster_id FROM extracted_events)
        """
        cur.execute(query)
        clusters = cur.fetchall()

        result = []
        for c in clusters:
            cur.execute(
                """
                SELECT e.* FROM events e
                JOIN cluster_events ce ON ce.event_id = e.id
                WHERE ce.cluster_id = %s
                """,
                (c["id"],),
            )
            events = cur.fetchall()

            source_role_mix = c.get("source_role_mix") or {}
            if isinstance(source_role_mix, str):
                source_role_mix = json.loads(source_role_mix)

            features = ClusterFeatures(
                mention_velocity=c["mention_velocity"],
                source_diversity=c["source_diversity"],
                recency=c["recency"],
                source_role_mix=source_role_mix,
                coherence_score=c.get("coherence_score", 0.0),
                weighted_mention_velocity=c.get("weighted_mention_velocity", 0.0),
            )
            event_objects = [
                Event(
                    id=e["id"], title=e["title"], content=e["content"],
                    source=e["source"], source_type=e["source_type"],
                    url=e["url"], entities=e["entities"],
                    content_hash=e["content_hash"],
                    signal_role=e.get("signal_role", "discovery"),
                    timestamp=e["timestamp"],
                )
                for e in events
            ]
            priority_topic, priority_score = compute_cluster_priority(features, event_objects)

            result.append({
                "cluster_id": c["id"],
                "label": c["label"],
                "features": features,
                "events": event_objects,
                "priority_topic": priority_topic,
                "priority_score": priority_score,
            })
    # Priority is computed in Python instead of SQL because it depends on
    # cross-cutting heuristics from `ranking/market_priority.py` that combine cluster
    # features with event text/source hints. That keeps the ranking logic
    # readable and testable without duplicating it inside a long CASE expression.
    result.sort(
        key=lambda item: (
            -item["priority_score"],
            -float(item["features"].weighted_mention_velocity or 0.0),
            -float(item["features"].mention_velocity or 0.0),
            -int(item["features"].source_diversity or 0),
            float(item["features"].recency or 0.0),
            int(item["cluster_id"]),
        )
    )
    if limit is not None:
        return result[: int(limit)]
    return result


# ---- FR3: Extracted event helpers ----

def insert_extracted_event(extracted: ExtractedEvent) -> int:
    """Insert an extracted event. Returns its ID."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO extracted_events
                (cluster_id, event_summary, entities, event_type, outcome_variable,
                 candidate_deadlines, resolution_sources, tradability, rejection_reason,
                 confidence, market_angle, contradiction_flag, contradiction_details,
                 time_horizon, resolution_hints, raw_llm_response)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (extracted.cluster_id, extracted.event_summary,
             json.dumps(extracted.entities), extracted.event_type,
             extracted.outcome_variable,
             json.dumps(extracted.candidate_deadlines),
             json.dumps(extracted.resolution_sources),
             extracted.tradability, extracted.rejection_reason,
             extracted.confidence, extracted.market_angle,
             extracted.contradiction_flag, extracted.contradiction_details,
             extracted.time_horizon,
             json.dumps(extracted.resolution_hints),
             extracted.raw_llm_response or ""),
        )
        return cur.fetchone()["id"]


def _parse_json_field(val, default):
    """Safely parse a JSON field that may already be a list or a JSON string."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


def get_extracted_events() -> List[ExtractedEvent]:
    """Retrieve all extracted events."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM extracted_events ORDER BY id")
        rows = cur.fetchall()
    return [
        ExtractedEvent(
            id=r["id"], cluster_id=r["cluster_id"],
            event_summary=r["event_summary"],
            entities=_parse_json_field(r["entities"], []),
            event_type=r.get("event_type", ""),
            outcome_variable=r.get("outcome_variable", ""),
            candidate_deadlines=_parse_json_field(r.get("candidate_deadlines"), []),
            resolution_sources=_parse_json_field(r.get("resolution_sources"), []),
            tradability=r.get("tradability", "suitable"),
            rejection_reason=r.get("rejection_reason", ""),
            confidence=r.get("confidence", 0.5),
            market_angle=r.get("market_angle", ""),
            contradiction_flag=r.get("contradiction_flag", False),
            contradiction_details=r.get("contradiction_details", ""),
            time_horizon=r.get("time_horizon", ""),
            resolution_hints=_parse_json_field(r.get("resolution_hints"), []),
            raw_llm_response=r.get("raw_llm_response"),
        )
        for r in rows
    ]


def get_extracted_events_for_generation(limit: Optional[int] = None) -> List[ExtractedEvent]:
    """
    Retrieve suitable extracted events that haven't had questions generated yet.

    FR4 prioritizes events backed by stronger cluster popularity signals:
    weighted mention velocity, mention velocity, source diversity, extraction
    confidence, and recency. An optional limit caps how many events are sent
    to the LLM in a single run.
    """
    with get_cursor() as cur:
        query = """
            SELECT
                ee.*,
                c.mention_velocity AS cluster_mention_velocity,
                c.source_diversity AS cluster_source_diversity,
                c.recency AS cluster_recency,
                c.source_role_mix AS cluster_source_role_mix,
                c.coherence_score AS cluster_coherence_score,
                c.weighted_mention_velocity AS cluster_weighted_mention_velocity
            FROM extracted_events ee
            JOIN clusters c ON c.id = ee.cluster_id
            WHERE ee.tradability = 'suitable'
              AND ee.id NOT IN (
                  SELECT DISTINCT extracted_event_id FROM candidate_questions
              )
        """
        cur.execute(query)
        rows = cur.fetchall()

    extracted_events_with_priority = []
    for r in rows:
        source_role_mix = r.get("cluster_source_role_mix") or {}
        if isinstance(source_role_mix, str):
            source_role_mix = json.loads(source_role_mix)

        cluster_features = ClusterFeatures(
            mention_velocity=r.get("cluster_mention_velocity") or 0.0,
            source_diversity=r.get("cluster_source_diversity") or 0,
            recency=r.get("cluster_recency") or 0.0,
            source_role_mix=source_role_mix,
            coherence_score=r.get("cluster_coherence_score") or 0.0,
            weighted_mention_velocity=r.get("cluster_weighted_mention_velocity") or 0.0,
        )
        extracted_event = ExtractedEvent(
            id=r["id"], cluster_id=r["cluster_id"],
            event_summary=r["event_summary"],
            entities=_parse_json_field(r["entities"], []),
            event_type=r.get("event_type", ""),
            outcome_variable=r.get("outcome_variable", ""),
            candidate_deadlines=_parse_json_field(r.get("candidate_deadlines"), []),
            resolution_sources=_parse_json_field(r.get("resolution_sources"), []),
            tradability=r.get("tradability", "suitable"),
            rejection_reason=r.get("rejection_reason", ""),
            confidence=r.get("confidence", 0.5),
            market_angle=r.get("market_angle", ""),
            contradiction_flag=r.get("contradiction_flag", False),
            contradiction_details=r.get("contradiction_details", ""),
            time_horizon=r.get("time_horizon", ""),
            resolution_hints=_parse_json_field(r.get("resolution_hints"), []),
            raw_llm_response=r.get("raw_llm_response"),
        )
        with get_cursor() as event_cur:
            event_cur.execute(
                """
                SELECT e.source
                FROM events e
                JOIN cluster_events ce ON ce.event_id = e.id
                WHERE ce.cluster_id = %s
                """,
                (int(extracted_event.cluster_id),),
            )
            source_rows = event_cur.fetchall()
        sources = [source_row["source"] for source_row in source_rows]
        priority_topic, priority_score = compute_extracted_event_priority(
            extracted_event,
            cluster_features,
            sources=sources,
        )
        extracted_events_with_priority.append((extracted_event, cluster_features, priority_topic, priority_score))

    # FR4 ordering is intentionally a second pass after FR3: once extraction has
    # produced event-type/confidence/tradability fields, we can rank the queue
    # more like a prediction-market editor would instead of just replaying raw
    # cluster order from FR2.
    extracted_events_with_priority.sort(
        key=lambda item: (
            -item[3],
            -float(item[1].weighted_mention_velocity or 0.0),
            -float(item[1].mention_velocity or 0.0),
            -float(item[0].confidence or 0.0),
            -int(item[1].source_diversity or 0),
            float(item[1].recency or 0.0),
            int(item[0].id or 0),
        )
    )
    extracted_events = [item[0] for item in extracted_events_with_priority]
    if limit is not None:
        return extracted_events[: int(limit)]
    return extracted_events


# ---- FR4: Candidate question helpers ----

def insert_candidate_question(q: CandidateQuestion) -> int:
    """Insert a candidate question. Returns its DB ID."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO candidate_questions
                (extracted_event_id, repair_parent_question_id, question_text, category, question_type, options,
                 deadline, deadline_source, resolution_source, resolution_criteria,
                 rationale, resolution_confidence, resolution_confidence_reason,
                 source_independence, timing_reliability, already_resolved, raw_llm_response)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                q.extracted_event_id, q.repair_parent_question_id, q.question_text, q.category, q.question_type,
                json.dumps(q.options), q.deadline, q.deadline_source,
                q.resolution_source, q.resolution_criteria, q.rationale,
                q.resolution_confidence, q.resolution_confidence_reason,
                q.source_independence, q.timing_reliability, q.already_resolved,
                q.raw_llm_response or "",
            ),
        )
        return cur.fetchone()["id"]


def get_candidate_questions(extracted_event_id: Optional[int] = None) -> List[CandidateQuestion]:
    """Retrieve candidate questions, optionally filtered by extracted_event_id."""
    with get_cursor() as cur:
        if extracted_event_id is not None:
            cur.execute(
                "SELECT * FROM candidate_questions WHERE extracted_event_id = %s ORDER BY id",
                (extracted_event_id,),
            )
        else:
            cur.execute("SELECT * FROM candidate_questions ORDER BY id")
        rows = cur.fetchall()
    return [
        CandidateQuestion(
            id=r["id"],
            extracted_event_id=r["extracted_event_id"],
            repair_parent_question_id=r.get("repair_parent_question_id"),
            question_text=r["question_text"],
            category=r["category"],
            question_type=r["question_type"],
            options=_parse_json_field(r["options"], []),
            deadline=r["deadline"],
            deadline_source=r["deadline_source"],
            resolution_source=r["resolution_source"],
            resolution_criteria=r["resolution_criteria"],
            rationale=r["rationale"],
            resolution_confidence=r.get("resolution_confidence") or 0.0,
            resolution_confidence_reason=r.get("resolution_confidence_reason") or "",
            source_independence=r.get("source_independence") or 0.0,
            timing_reliability=r.get("timing_reliability") or 0.0,
            already_resolved=r.get("already_resolved") or False,
            raw_llm_response=r.get("raw_llm_response"),
        )
        for r in rows
    ]


# ---- FR5: Validation helpers ----

def get_candidate_questions_for_validation() -> List[CandidateQuestion]:
    """Retrieve candidate questions that have not been validated yet (idempotent)."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT cq.*
            FROM candidate_questions cq
            WHERE NOT EXISTS (
                SELECT 1 FROM validation_results vr WHERE vr.question_id = cq.id
            )
            ORDER BY cq.id
            """
        )
        rows = cur.fetchall()
    return [
        CandidateQuestion(
            id=r["id"],
            extracted_event_id=r["extracted_event_id"],
            repair_parent_question_id=r.get("repair_parent_question_id"),
            question_text=r["question_text"],
            category=r["category"],
            question_type=r["question_type"],
            options=_parse_json_field(r["options"], []),
            deadline=r["deadline"],
            deadline_source=r["deadline_source"],
            resolution_source=r["resolution_source"],
            resolution_criteria=r["resolution_criteria"],
            rationale=r["rationale"],
            resolution_confidence=r.get("resolution_confidence") or 0.0,
            resolution_confidence_reason=r.get("resolution_confidence_reason") or "",
            source_independence=r.get("source_independence") or 0.0,
            timing_reliability=r.get("timing_reliability") or 0.0,
            already_resolved=r.get("already_resolved") or False,
            raw_llm_response=r.get("raw_llm_response"),
        )
        for r in rows
    ]


def insert_validation_result(result: ValidationResult) -> int:
    """Insert FR5 validation result for a candidate question."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO validation_results (question_id, is_valid, flags, clarity_score)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (result.question_id, result.is_valid, json.dumps(result.flags), result.clarity_score),
        )
        return cur.fetchone()["id"]


def get_extracted_event_by_id(extracted_event_id: int) -> Optional[ExtractedEvent]:
    """Retrieve a single extracted event by id."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM extracted_events WHERE id = %s", (int(extracted_event_id),))
        row = cur.fetchone()
    if not row:
        return None
    return ExtractedEvent(
        id=row["id"],
        cluster_id=row["cluster_id"],
        event_summary=row["event_summary"],
        entities=_parse_json_field(row["entities"], []),
        event_type=row.get("event_type", ""),
        outcome_variable=row.get("outcome_variable", ""),
        candidate_deadlines=_parse_json_field(row.get("candidate_deadlines"), []),
        resolution_sources=_parse_json_field(row.get("resolution_sources"), []),
        tradability=row.get("tradability", "suitable"),
        rejection_reason=row.get("rejection_reason", ""),
        confidence=row.get("confidence", 0.5),
        market_angle=row.get("market_angle", ""),
        contradiction_flag=row.get("contradiction_flag", False),
        contradiction_details=row.get("contradiction_details", ""),
        time_horizon=row.get("time_horizon", ""),
        resolution_hints=_parse_json_field(row.get("resolution_hints"), []),
        raw_llm_response=row.get("raw_llm_response"),
    )


def question_has_repair_child(question_id: int) -> bool:
    """Return True if this failed question already has a repaired child question."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1 FROM candidate_questions
                WHERE repair_parent_question_id = %s
            ) AS has_child
            """,
            (int(question_id),),
        )
        row = cur.fetchone()
    return bool(row["has_child"]) if row else False


# ---- FR6: Scoring helpers ----

def get_validated_questions_for_scoring() -> List[Dict]:
    """Get validated and unscored questions with cluster feature join data (idempotent)."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                cq.id AS question_id,
                cq.question_text,
                cq.category,
                cq.deadline,
                cq.resolution_source,
                cq.resolution_confidence,
                cq.source_independence,
                cq.timing_reliability,
                cq.already_resolved,
                c.mention_velocity,
                c.source_diversity,
                vr.clarity_score,
                vr.flags AS validation_flags
            FROM candidate_questions cq
            JOIN extracted_events ee ON ee.id = cq.extracted_event_id
            JOIN clusters c ON c.id = ee.cluster_id
            JOIN validation_results vr ON vr.question_id = cq.id
            WHERE vr.is_valid = TRUE
              AND NOT EXISTS (
                  SELECT 1 FROM scored_candidates sc WHERE sc.question_id = cq.id
              )
            ORDER BY cq.id
            """
        )
        return cur.fetchall()


def get_all_candidate_question_texts() -> List[Tuple[int, str]]:
    """Return all candidate question ids and texts for novelty comparisons in FR6."""
    with get_cursor() as cur:
        cur.execute("SELECT id, question_text FROM candidate_questions ORDER BY id")
        rows = cur.fetchall()
    return [(int(r["id"]), r["question_text"]) for r in rows]


def insert_scored_candidate(scored: ScoredCandidate) -> int:
    """Insert FR6 scored candidate row."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO scored_candidates
                (question_id, total_score, mention_velocity_score, source_diversity_score,
                 clarity_score, novelty_score, market_interest_score,
                 resolution_strength_score, time_horizon_score, rank)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (scored.question_id, scored.total_score, scored.mention_velocity_score,
             scored.source_diversity_score, scored.clarity_score, scored.novelty_score,
             scored.market_interest_score, scored.resolution_strength_score,
             scored.time_horizon_score, scored.rank),
        )
        return cur.fetchone()["id"]


def get_ranked_scored_questions() -> List[Dict]:
    """Retrieve all scored questions with full metadata for FR7 display.

    No LIMIT — the full ranked list is returned; filtering/pagination is
    handled in the Streamlit layer so the user always sees every question.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                sc.question_id, sc.rank, sc.total_score,
                sc.mention_velocity_score, sc.source_diversity_score,
                sc.clarity_score, sc.novelty_score,
                sc.market_interest_score, sc.resolution_strength_score,
                sc.time_horizon_score,
                cq.question_text, cq.category, cq.question_type,
                cq.options, cq.deadline, cq.resolution_source,
                cq.resolution_criteria, cq.rationale,
                cq.resolution_confidence, cq.source_independence,
                cq.timing_reliability
            FROM scored_candidates sc
            JOIN candidate_questions cq ON cq.id = sc.question_id
            ORDER BY sc.rank ASC, sc.question_id ASC
            """
        )
        return cur.fetchall()


# ---- Pipeline run tracking helpers ----

def create_pipeline_run(
    stage_start: int = 1,
    stage_end: int = 6,
    fr3_limit_mode: str = "default",
    fr3_limit_value: Optional[int] = None,
    fr4_limit_mode: str = "default",
    fr4_limit_value: Optional[int] = None,
    log_mode: str = "normal",
    fr3_model: str = "",
    fr4_model: str = "",
) -> int:
    """Create a pipeline run record plus placeholder rows for each stage."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs
                (status, stage_start, stage_end, fr3_limit_mode, fr3_limit_value,
                 fr4_limit_mode, fr4_limit_value, log_mode, fr3_model, fr4_model)
            VALUES ('queued', %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                int(stage_start),
                int(stage_end),
                fr3_limit_mode,
                int(fr3_limit_value) if fr3_limit_value is not None else None,
                fr4_limit_mode,
                int(fr4_limit_value) if fr4_limit_value is not None else None,
                log_mode,
                fr3_model or "",
                fr4_model or "",
            ),
        )
        run_id = int(cur.fetchone()["id"])

        stage_rows = [
            (run_id, stage_number, f"FR{stage_number}", "pending")
            for stage_number in range(int(stage_start), int(stage_end) + 1)
        ]
        cur.executemany(
            """
            INSERT INTO pipeline_run_stages (run_id, stage_number, stage_name, status)
            VALUES (%s, %s, %s, %s)
            """,
            stage_rows,
        )

    return run_id


def mark_pipeline_run_started(run_id: int, subprocess_pid: Optional[int] = None) -> None:
    """Mark a queued run as actively executing."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET status = 'running',
                subprocess_pid = COALESCE(%s, subprocess_pid),
                started_at = COALESCE(started_at, NOW()),
                finished_at = NULL,
                error_message = ''
            WHERE id = %s
            """,
            (subprocess_pid, int(run_id)),
        )


def mark_pipeline_run_completed(run_id: int) -> None:
    """Mark a run as completed."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET status = 'completed',
                finished_at = NOW(),
                error_message = ''
            WHERE id = %s
            """,
            (int(run_id),),
        )


def mark_pipeline_run_failed(run_id: int, error_message: str) -> None:
    """Mark a run as failed and persist the final error."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET status = 'failed',
                finished_at = NOW(),
                error_message = %s
            WHERE id = %s
            """,
            ((error_message or "").strip(), int(run_id)),
        )


def cancel_pipeline_run(run_id: int, reason: str = "Cancelled by user") -> None:
    """
    Mark a run as stopped by the user and close any unfinished stages.

    Uses the existing `failed` status so older local databases do not need a
    check-constraint migration just to support cancellation.
    """
    message = (reason or "Cancelled by user").strip()
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET status = 'failed',
                finished_at = NOW(),
                error_message = %s
            WHERE id = %s
            """,
            (message, int(run_id)),
        )
        cur.execute(
            """
            UPDATE pipeline_run_stages
            SET status = 'failed',
                finished_at = NOW(),
                error_message = %s
            WHERE run_id = %s
              AND status IN ('pending', 'running')
            """,
            (message, int(run_id)),
        )


def update_pipeline_run_stage(
    run_id: int,
    stage_number: int,
    stage_name: str,
    status: str,
    summary: Optional[Dict] = None,
    error_message: str = "",
) -> None:
    """Upsert stage progress details for a pipeline run."""
    normalized_status = (status or "").strip().lower()
    if normalized_status not in {"pending", "running", "completed", "failed"}:
        raise ValueError(f"Unsupported pipeline stage status: {status}")

    summary_payload = json.dumps(summary or {})

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_run_stages
                (run_id, stage_number, stage_name, status, summary, error_message, started_at, finished_at)
            VALUES (
                %s, %s, %s, %s, %s::jsonb, %s,
                CASE WHEN %s = 'running' THEN NOW() ELSE NULL END,
                CASE WHEN %s IN ('completed', 'failed') THEN NOW() ELSE NULL END
            )
            ON CONFLICT (run_id, stage_number) DO UPDATE SET
                stage_name = EXCLUDED.stage_name,
                status = EXCLUDED.status,
                summary = EXCLUDED.summary,
                error_message = EXCLUDED.error_message,
                started_at = CASE
                    WHEN EXCLUDED.status = 'running'
                    THEN COALESCE(pipeline_run_stages.started_at, NOW())
                    ELSE pipeline_run_stages.started_at
                END,
                finished_at = CASE
                    WHEN EXCLUDED.status IN ('completed', 'failed') THEN NOW()
                    ELSE NULL
                END
            """,
            (
                int(run_id),
                int(stage_number),
                stage_name,
                normalized_status,
                summary_payload,
                (error_message or "").strip(),
                normalized_status,
                normalized_status,
            ),
        )


def get_latest_pipeline_run() -> Optional[Dict]:
    """Return the most recent pipeline run, if any."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM pipeline_runs
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_active_pipeline_run() -> Optional[Dict]:
    """Return the newest queued/running pipeline run, if any."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM pipeline_runs
            WHERE status IN ('queued', 'running')
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_pipeline_run_stages(run_id: int) -> List[Dict]:
    """Return the stage rows for a pipeline run in stage order."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM pipeline_run_stages
            WHERE run_id = %s
            ORDER BY stage_number ASC
            """,
            (int(run_id),),
        )
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def get_dashboard_scored_questions() -> List[Dict]:
    """
    Retrieve all scored questions plus any persisted human review state.

    The dashboard derives the effective status and recomputes display ranks in
    the Streamlit layer so old batch-local FR6 ranks do not leak into the
    active review queue.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                sc.question_id, sc.rank, sc.total_score,
                sc.mention_velocity_score, sc.source_diversity_score,
                sc.clarity_score, sc.novelty_score,
                sc.market_interest_score, sc.resolution_strength_score,
                sc.time_horizon_score,
                cq.question_text, cq.category, cq.question_type,
                cq.options, cq.deadline, cq.deadline_source, cq.resolution_source,
                cq.resolution_criteria, cq.rationale,
                cq.resolution_confidence, cq.source_independence,
                cq.timing_reliability, cq.created_at AS question_created_at,
                vr.is_valid, vr.flags AS validation_flags,
                qrs.status AS review_status,
                qrs.reason AS review_reason,
                qrs.notes AS review_notes,
                qrs.changed_at AS review_changed_at
            FROM scored_candidates sc
            JOIN candidate_questions cq ON cq.id = sc.question_id
            LEFT JOIN validation_results vr ON vr.question_id = cq.id
            LEFT JOIN question_review_state qrs ON qrs.question_id = cq.id
            ORDER BY sc.total_score DESC, sc.question_id ASC
            """
        )
        return cur.fetchall()


def get_dashboard_topics() -> List[Dict]:
    """
    Retrieve topic-level discovery rows for the consumer dashboard.

    The query rolls up extracted events, linked source events, and generated
    candidate questions so Streamlit can compute a simple deterministic trend
    score without needing to know table relationships.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                ee.id,
                COALESCE(NULLIF(ee.event_summary, ''), 'Emerging topic') AS title,
                ee.event_summary AS summary,
                COALESCE(
                    (
                        ARRAY_AGG(
                            NULLIF(cq.category, '')
                            ORDER BY sc.total_score DESC NULLS LAST, cq.id ASC
                        )
                    )[1],
                    NULLIF(ee.event_type, ''),
                    'other'
                ) AS category,
                COUNT(DISTINCT ce.event_id) AS event_count,
                COUNT(DISTINCT e.source) AS source_count,
                COUNT(DISTINCT cq.id) AS suggested_market_count,
                AVG(sc.total_score) AS avg_candidate_score,
                MAX(e.timestamp) AS latest_event_at,
                (
                    ARRAY_AGG(
                        cq.question_text
                        ORDER BY sc.total_score DESC NULLS LAST, cq.id ASC
                    )
                )[1] AS example_question
            FROM extracted_events ee
            JOIN clusters c ON c.id = ee.cluster_id
            LEFT JOIN cluster_events ce ON ce.cluster_id = c.id
            LEFT JOIN events e ON e.id = ce.event_id
            LEFT JOIN candidate_questions cq ON cq.extracted_event_id = ee.id
            LEFT JOIN scored_candidates sc ON sc.question_id = cq.id
            GROUP BY
                ee.id,
                ee.event_summary,
                ee.event_type
            ORDER BY
                MAX(e.timestamp) DESC NULLS LAST,
                COUNT(DISTINCT ce.event_id) DESC,
                ee.id ASC
            """
        )
        return cur.fetchall()


def get_dashboard_repair_questions() -> List[Dict]:
    """
    Retrieve failed validation questions ranked by underlying cluster popularity.

    These rows power the "Needs Repair" dashboard queue so high-signal events are
    still visible even when the generated question needs cleanup.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                cq.id AS question_id,
                cq.extracted_event_id,
                cq.repair_parent_question_id,
                cq.question_text,
                cq.category,
                cq.question_type,
                cq.options,
                cq.deadline,
                cq.deadline_source,
                cq.resolution_source,
                cq.resolution_criteria,
                cq.rationale,
                cq.created_at AS question_created_at,
                vr.flags AS validation_flags,
                vr.created_at AS validation_created_at,
                ee.confidence AS extraction_confidence,
                c.mention_velocity,
                c.weighted_mention_velocity,
                c.source_diversity,
                c.recency,
                EXISTS (
                    SELECT 1
                    FROM candidate_questions child
                    WHERE child.repair_parent_question_id = cq.id
                ) AS has_repair_child
            FROM candidate_questions cq
            JOIN validation_results vr ON vr.question_id = cq.id
            JOIN extracted_events ee ON ee.id = cq.extracted_event_id
            JOIN clusters c ON c.id = ee.cluster_id
            WHERE vr.is_valid = FALSE
            ORDER BY
                c.weighted_mention_velocity DESC NULLS LAST,
                c.mention_velocity DESC NULLS LAST,
                c.source_diversity DESC NULLS LAST,
                ee.confidence DESC NULLS LAST,
                c.recency ASC NULLS LAST,
                cq.id ASC
            """
        )
        return cur.fetchall()


def set_question_review_status(
    question_id: int,
    status: str,
    reason: str = "",
    notes: str = "",
) -> None:
    """
    Persist a manual dashboard review decision for a scored question.

    `selected` and `removed` are stored in the DB. Moving a question back to
    `active` clears any persisted override so active remains the default state.
    """
    normalized = (status or "").strip().lower()
    if normalized not in {"active", "selected", "removed"}:
        raise ValueError(f"Unsupported review status: {status}")

    with get_cursor() as cur:
        if normalized == "active":
            cur.execute(
                "DELETE FROM question_review_state WHERE question_id = %s",
                (int(question_id),),
            )
            return

        cur.execute(
            """
            INSERT INTO question_review_state (question_id, status, reason, notes, changed_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (question_id) DO UPDATE SET
                status = EXCLUDED.status,
                reason = EXCLUDED.reason,
                notes = EXCLUDED.notes,
                changed_at = NOW()
            """,
            (int(question_id), normalized, reason, notes),
        )
