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
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO clusters (label, mention_velocity, source_diversity, recency, size,
                                  source_role_mix, coherence_score, weighted_mention_velocity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (cluster.label, cluster.features.mention_velocity,
             cluster.features.source_diversity, cluster.features.recency,
             len(cluster.events),
             json.dumps(cluster.features.source_role_mix),
             cluster.features.coherence_score,
             cluster.features.weighted_mention_velocity),
        )
        cluster_id = cur.fetchone()["id"]

        for event in cluster.events:
            if event.id is not None:
                cur.execute(
                    "INSERT INTO cluster_events (cluster_id, event_id) VALUES (%s, %s)",
                    (cluster_id, event.id),
                )
    return cluster_id


def get_clusters_for_extraction() -> List[dict]:
    """Get clusters that haven't been extracted yet, with their events."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.label, c.mention_velocity, c.source_diversity, c.recency,
                   c.source_role_mix, c.coherence_score, c.weighted_mention_velocity
            FROM clusters c
            WHERE c.id NOT IN (SELECT cluster_id FROM extracted_events)
            ORDER BY c.id
            """
        )
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

            result.append({
                "cluster_id": c["id"],
                "label": c["label"],
                "features": ClusterFeatures(
                    mention_velocity=c["mention_velocity"],
                    source_diversity=c["source_diversity"],
                    recency=c["recency"],
                    source_role_mix=source_role_mix,
                    coherence_score=c.get("coherence_score", 0.0),
                    weighted_mention_velocity=c.get("weighted_mention_velocity", 0.0),
                ),
                "events": [
                    Event(
                        id=e["id"], title=e["title"], content=e["content"],
                        source=e["source"], source_type=e["source_type"],
                        url=e["url"], entities=e["entities"],
                        content_hash=e["content_hash"],
                        signal_role=e.get("signal_role", "discovery"),
                        timestamp=e["timestamp"],
                    )
                    for e in events
                ],
            })
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

    def _parse_json_field(val, default):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default
        return default

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


def get_extracted_events_for_generation() -> List[ExtractedEvent]:
    """Retrieve suitable extracted events that haven't had questions generated yet (idempotent)."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM extracted_events
            WHERE tradability = 'suitable'
              AND id NOT IN (SELECT DISTINCT extracted_event_id FROM candidate_questions)
            ORDER BY id
            """
        )
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


# ---- FR4: Candidate question helpers ----

def insert_candidate_question(q: CandidateQuestion) -> int:
    """Insert a candidate question. Returns its DB ID."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO candidate_questions
                (extracted_event_id, question_text, category, question_type, options,
                 deadline, deadline_source, resolution_source, resolution_criteria,
                 rationale, resolution_confidence, resolution_confidence_reason,
                 source_independence, timing_reliability, already_resolved, raw_llm_response)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                q.extracted_event_id, q.question_text, q.category, q.question_type,
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
