"""
Database connection helper and common query functions.

Provides a connection pool and typed helper functions for each table.
Downstream modules (FR4-FR7) can import and extend these helpers.
"""

import json
import logging
from contextlib import contextmanager
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from config import DBConfig
from models import Event, Cluster, ClusterFeatures, ExtractedEvent, CandidateQuestion

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
            INSERT INTO events (title, content, source, source_type, url, entities, content_hash, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id
            """,
            (event.title, event.content, event.source, event.source_type,
             event.url, event.entities, event.content_hash, event.timestamp),
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
            content_hash=r["content_hash"], timestamp=r["timestamp"],
        )
        for r in rows
    ]


# ---- FR2: Cluster helpers ----

def insert_cluster(cluster: Cluster) -> int:
    """Insert a cluster and its event mappings. Returns cluster ID."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO clusters (label, mention_velocity, source_diversity, recency, size)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (cluster.label, cluster.features.mention_velocity,
             cluster.features.source_diversity, cluster.features.recency,
             len(cluster.events)),
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
            SELECT c.id, c.label, c.mention_velocity, c.source_diversity, c.recency
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
            result.append({
                "cluster_id": c["id"],
                "label": c["label"],
                "features": ClusterFeatures(
                    mention_velocity=c["mention_velocity"],
                    source_diversity=c["source_diversity"],
                    recency=c["recency"],
                ),
                "events": [
                    Event(
                        id=e["id"], title=e["title"], content=e["content"],
                        source=e["source"], source_type=e["source_type"],
                        url=e["url"], entities=e["entities"],
                        content_hash=e["content_hash"], timestamp=e["timestamp"],
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
                (cluster_id, event_summary, entities, time_horizon, resolution_hints, raw_llm_response)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (extracted.cluster_id, extracted.event_summary,
             json.dumps(extracted.entities), extracted.time_horizon,
             json.dumps(extracted.resolution_hints),
             extracted.raw_llm_response or ""),
        )
        return cur.fetchone()["id"]


def get_extracted_events() -> List[ExtractedEvent]:
    """Retrieve all extracted events. Used by FR4 downstream."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM extracted_events ORDER BY id")
        rows = cur.fetchall()
    return [
        ExtractedEvent(
            id=r["id"], cluster_id=r["cluster_id"],
            event_summary=r["event_summary"],
            entities=r["entities"] if isinstance(r["entities"], list) else json.loads(r["entities"]),
            time_horizon=r["time_horizon"],
            resolution_hints=r["resolution_hints"] if isinstance(r["resolution_hints"], list) else json.loads(r["resolution_hints"]),
            raw_llm_response=r.get("raw_llm_response"),
        )
        for r in rows
    ]


def get_extracted_events_for_generation() -> List[ExtractedEvent]:
    """Retrieve extracted events that haven't had questions generated yet (idempotent)."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM extracted_events
            WHERE id NOT IN (SELECT DISTINCT extracted_event_id FROM candidate_questions)
            ORDER BY id
            """
        )
        rows = cur.fetchall()
    return [
        ExtractedEvent(
            id=r["id"], cluster_id=r["cluster_id"],
            event_summary=r["event_summary"],
            entities=r["entities"] if isinstance(r["entities"], list) else json.loads(r["entities"]),
            time_horizon=r["time_horizon"],
            resolution_hints=r["resolution_hints"] if isinstance(r["resolution_hints"], list) else json.loads(r["resolution_hints"]),
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
                 rationale, raw_llm_response)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                q.extracted_event_id, q.question_text, q.category, q.question_type,
                json.dumps(q.options), q.deadline, q.deadline_source,
                q.resolution_source, q.resolution_criteria, q.rationale,
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
            options=r["options"] if isinstance(r["options"], list) else json.loads(r["options"]),
            deadline=r["deadline"],
            deadline_source=r["deadline_source"],
            resolution_source=r["resolution_source"],
            resolution_criteria=r["resolution_criteria"],
            rationale=r["rationale"],
            raw_llm_response=r.get("raw_llm_response"),
        )
        for r in rows
    ]
