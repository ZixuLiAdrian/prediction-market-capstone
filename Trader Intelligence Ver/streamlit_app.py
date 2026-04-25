"""
Prediction Market Discovery dashboard.

This Streamlit app reframes the old internal review dashboard into a
consumer-facing discovery page for active markets and emerging topics while
keeping a few legacy helper functions that existing tests still exercise.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import streamlit as st

from db.connection import (
    get_active_pipeline_run,
    get_dashboard_scored_questions,
    get_dashboard_topics,
    get_latest_pipeline_run,
    get_pipeline_run_stages,
    init_db,
)

UTC = timezone.utc
DISCOVERY_CATEGORIES = [
    "All",
    "Politics",
    "Economics",
    "Technology",
    "Sports",
    "Entertainment",
    "Crypto",
    "Science",
    "World News",
    "Health",
    "Other",
]

STATUS_OPTIONS = ["All", "Active", "Ending Soon", "Expired", "Tracked"]
SORT_OPTIONS = {
    "Score (Highest First)": lambda r: -float(r.get("total_score") or 0.0),
    "Deadline (Soonest)": lambda r: r.get("deadline") or "9999-99-99",
    "Category (A-Z)": lambda r: (r.get("category") or "").lower(),
}
DISCOVERY_SORT_OPTIONS = ["Highest Score", "Newest", "Ending Soon", "Emerging"]
PIPELINE_STAGE_LABELS = {
    1: "FR1 Event Ingestion",
    2: "FR2 Event Clustering",
    3: "FR3 LLM Extraction",
    4: "FR4 Question Generation",
    5: "FR5 Rule Validation",
    6: "FR6 Heuristic Scoring",
}
TOPIC_SOURCE_MAP = {
    "Politics": ["Reuters politics desk", "official election calendars", "major polling trackers"],
    "Economics": ["BLS and FRED releases", "company filings", "major financial press"],
    "Technology": ["company launch blogs", "earnings calls", "developer conferences"],
    "Sports": ["league schedules", "team injury reports", "official standings pages"],
    "Entertainment": ["studio release calendars", "awards schedules", "trade publications"],
    "Crypto": ["exchange listings", "ETF or policy filings", "project and regulator announcements"],
    "Science": ["journal publications", "lab or agency updates", "conference schedules"],
    "World News": ["major wire services", "government statements", "multilateral organizations"],
    "Health": ["FDA calendars", "trial registries", "company and regulator updates"],
    "Other": ["primary source announcements", "major wire services", "specialist trade press"],
}
ADJACENT_TOPIC_MAP = {
    "Politics": ["campaign momentum", "regulatory follow-through", "cross-border reaction"],
    "Economics": ["rates and inflation", "employment sentiment", "policy spillovers"],
    "Technology": ["AI competition", "enterprise adoption", "regulatory pressure"],
    "Sports": ["injury volatility", "title odds", "broadcast narrative"],
    "Entertainment": ["release timing", "awards momentum", "social buzz"],
    "Crypto": ["ETF flows", "regulatory shifts", "exchange liquidity"],
    "Science": ["funding outlook", "commercialization", "policy adoption"],
    "World News": ["sanctions risk", "commodity impact", "diplomatic signaling"],
    "Health": ["trial timelines", "regulatory approval", "healthcare market response"],
    "Other": ["policy knock-on effects", "industry sentiment", "follow-up milestones"],
}


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#0f766e"
    if score >= 0.55:
        return "#1d4ed8"
    if score >= 0.40:
        return "#b45309"
    return "#b91c1c"


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)

    text = str(value).strip()
    for parser in (
        lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        lambda s: datetime.strptime(s, "%Y-%m-%d"),
        lambda s: datetime.strptime(s, "%B %d, %Y"),
        lambda s: datetime.strptime(s, "%b %d, %Y"),
        lambda s: datetime.strptime(s, "%Y/%m/%d"),
        lambda s: datetime.strptime(s, "%d %B %Y"),
    ):
        try:
            parsed = parser(text)
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _days_until(deadline_str: str) -> Optional[int]:
    parsed = _parse_dt(deadline_str)
    if parsed is None:
        return None
    return (parsed.date() - datetime.now(UTC).date()).days


def _effective_status(row: dict) -> str:
    review_status = (row.get("review_status") or "").strip().lower()
    if review_status in {"selected", "removed"}:
        return review_status

    days = _days_until((row.get("deadline") or "").strip())
    if days is not None and days < 0:
        return "expired"
    return "active"


def _assign_display_ranks(rows: list[dict]) -> list[dict]:
    ranked = sorted(
        (dict(r) for r in rows),
        key=lambda r: (-float(r.get("total_score") or 0.0), int(r.get("question_id") or r.get("id") or 0)),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _filter_rows(
    rows: list[dict],
    search: str,
    selected_cats: list[str],
    type_choice: str,
    min_score_pct: int,
    sort_choice: str,
) -> list[dict]:
    filtered = list(rows)
    if search:
        needle = search.strip().lower()
        filtered = [r for r in filtered if needle in (r.get("question_text") or "").lower()]
    if selected_cats:
        filtered = [r for r in filtered if (r.get("category") or "other").lower() in selected_cats]
    if type_choice != "All":
        target = "binary" if type_choice == "Binary" else "multiple_choice"
        filtered = [r for r in filtered if (r.get("question_type") or "") == target]
    if min_score_pct > 0:
        filtered = [r for r in filtered if float(r.get("total_score") or 0.0) >= min_score_pct / 100]
    filtered.sort(key=SORT_OPTIONS[sort_choice])
    return filtered


def _build_pipeline_command(
    python_executable: str,
    pipeline_path: str,
    fr3_mode: str,
    fr3_custom_limit: Optional[int],
    fr4_mode: str,
    custom_limit: Optional[int],
    log_mode: str,
    run_id: int,
    fr3_model: str,
    fr4_model: str,
) -> list[str]:
    command = [
        python_executable,
        pipeline_path,
        "--run-id",
        str(int(run_id)),
        "--log-mode",
        log_mode,
    ]
    if fr3_model:
        command.extend(["--fr3-model", fr3_model])
    if fr4_model:
        command.extend(["--fr4-model", fr4_model])
    normalized_fr3_mode = (fr3_mode or "default").strip().lower()
    if normalized_fr3_mode == "custom" and fr3_custom_limit is not None:
        command.extend(["--fr3-limit", str(int(fr3_custom_limit))])
    elif normalized_fr3_mode == "all":
        command.append("--fr3-all")
    normalized_fr4_mode = (fr4_mode or "default").strip().lower()
    if normalized_fr4_mode == "custom" and custom_limit is not None:
        command.extend(["--fr4-limit", str(int(custom_limit))])
    elif normalized_fr4_mode == "all":
        command.append("--fr4-all")
    return command


def _format_stage_summary(summary) -> str:
    if not summary:
        return ""
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except Exception:
            return summary
    if not isinstance(summary, dict):
        return str(summary)
    parts = []
    for key, value in summary.items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}={value}")
    return " | ".join(parts)


def _pipeline_progress_value(stage_rows: list[dict]) -> float:
    if not stage_rows:
        return 0.0
    completed = sum(1 for row in stage_rows if row.get("status") == "completed")
    return completed / len(stage_rows)


def _stage_progress_value(stage_row: dict) -> float | None:
    summary = stage_row.get("summary")
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except Exception:
            return None
    if not isinstance(summary, dict):
        return None

    if summary.get("pending_clusters"):
        processed = summary.get("clusters_processed")
        total = summary.get("pending_clusters")
        if processed is not None and total:
            return min(max(float(processed) / float(total), 0.0), 1.0)

    if summary.get("eligible_events"):
        processed = summary.get("events_processed")
        total = summary.get("eligible_events")
        if processed is not None and total:
            return min(max(float(processed) / float(total), 0.0), 1.0)

    return None


def _stage_info_lines(stage_row: dict, run_row: dict | None) -> list[str]:
    lines: list[str] = []
    summary_text = _format_stage_summary(stage_row.get("summary"))
    if summary_text:
        lines.append(summary_text)

    stage_number = int(stage_row.get("stage_number") or 0)
    if run_row and stage_number == 3 and run_row.get("fr3_model"):
        lines.append(f"LLM model: {run_row['fr3_model']}")
    if run_row and stage_number == 4 and run_row.get("fr4_model"):
        lines.append(f"LLM model: {run_row['fr4_model']}")

    error_message = (stage_row.get("error_message") or "").strip()
    if error_message:
        lines.append(f"Error: {error_message}")
    return lines


def _terminate_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    subprocess.run(
        ["kill", "-TERM", str(int(pid))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _safe_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return [raw] if raw else []
    return []


def _format_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{int(round(float(value) * 100))}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.0f}%"


def _format_date_label(value: Any) -> str:
    parsed = _parse_dt(value)
    if parsed is None:
        return str(value) if value else "TBD"
    return parsed.strftime("%b %d, %Y")


def _category_label(value: str | None) -> str:
    normalized = (value or "Other").replace("_", " ").strip().lower()
    if not normalized:
        return "Other"

    aliases = {
        "finance": "Economics",
        "business": "Economics",
        "economics": "Economics",
        "politics": "Politics",
        "technology": "Technology",
        "tech": "Technology",
        "sports": "Sports",
        "entertainment": "Entertainment",
        "crypto": "Crypto",
        "science": "Science",
        "space": "Science",
        "health": "Health",
        "world news": "World News",
        "international": "World News",
        "geopolitics": "World News",
        "legal": "World News",
    }
    return aliases.get(normalized, normalized.title() if normalized.title() in DISCOVERY_CATEGORIES else "Other")


def _status_chip(value: str) -> str:
    status = (value or "active").lower()
    tone = {
        "active": "#1d4ed8",
        "tracked": "#7c3aed",
        "ending soon": "#b45309",
        "expired": "#b91c1c",
        "valid": "#0f766e",
        "needs review": "#b45309",
        "demo": "#64748b",
    }.get(status, "#475569")
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{tone}15;color:{tone};font-size:12px;font-weight:600;'>{value}</span>"
    )


def _category_chip(value: str) -> str:
    label = _category_label(value)
    return (
        "<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        "background:#e2e8f0;color:#0f172a;font-size:12px;font-weight:600;'>"
        f"{label}</span>"
    )


def _type_chip(value: str) -> str:
    label = "Multiple Choice" if (value or "").lower() == "multiple_choice" else "Binary"
    return (
        "<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        "background:#dbeafe;color:#1d4ed8;font-size:12px;font-weight:600;'>"
        f"{label}</span>"
    )


def _linkish_source(source: str | None) -> str:
    match = re.search(r"https?://[^\s)]+", source or "")
    if not match:
        return source or "Not specified"
    url = match.group(0).rstrip(".,;")
    label = urlparse(url).netloc.replace("www.", "")
    return f"[{label}]({url})"


def _topic_label_chip(value: str, tone: str = "research") -> str:
    palette = {
        "research": ("#7c3aed", "#f3e8ff"),
        "watch": ("#1d4ed8", "#dbeafe"),
    }
    text, bg = palette.get(tone, palette["research"])
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{text};font-size:12px;font-weight:700;'>{value}</span>"
    )


def _topic_focus_sentence(topic: dict) -> str:
    summary = (topic.get("summary") or "").strip()
    title = (topic.get("title") or "This topic").strip()
    if summary:
        first_sentence = summary.split(".")[0].strip()
        if first_sentence:
            if not first_sentence.endswith("."):
                first_sentence += "."
            return first_sentence
    return f"{title} is attracting enough cross-source attention to be worth active monitoring."


def _topic_what_is_uncertain(topic: dict) -> str:
    category = _category_label(topic.get("category"))
    title = topic.get("title") or "this topic"
    return {
        "Politics": f"Whether {title.lower()} turns into a measurable political shift or just a short-lived narrative spike.",
        "Economics": f"Whether {title.lower()} changes the economic outlook enough to move expectations, pricing, or policy.",
        "Technology": f"Whether {title.lower()} leads to a real launch, adoption milestone, or competitive response.",
        "Sports": f"Whether {title.lower()} changes the competitive landscape in a way markets will quickly price in.",
        "Entertainment": f"Whether {title.lower()} becomes a sustained audience and awards story rather than a one-cycle headline.",
        "Crypto": f"Whether {title.lower()} becomes a concrete catalyst for price, regulation, or market structure.",
        "Science": f"Whether {title.lower()} turns into a validated milestone with downstream commercial or policy consequences.",
        "Health": f"Whether {title.lower()} develops into a verifiable regulatory or clinical milestone on a tradable timeline.",
        "World News": f"Whether {title.lower()} escalates into a durable geopolitical development with clear downstream signals.",
    }.get(category, f"What matters most is whether {title.lower()} develops into a concrete, verifiable signal rather than remaining narrative noise.")


def _topic_watching_points(topic: dict) -> list[str]:
    category = _category_label(topic.get("category"))
    event_count = int(topic.get("event_count") or 0)
    source_count = int(topic.get("source_count") or 0)
    return [
        f"Watch whether mention velocity stays elevated beyond the current {event_count}-event burst.",
        f"Track whether coverage broadens beyond the current {source_count} distinct sources.",
        {
            "Politics": "Monitor official campaign, government, or election-calendar updates that convert narrative into a dated milestone.",
            "Economics": "Watch for scheduled data releases, policy meetings, or filings that turn the topic into a measurable macro signal.",
            "Technology": "Monitor launch dates, product notes, conference appearances, and executive comments for concrete confirmation.",
            "Sports": "Track official schedules, injury reports, standings changes, and matchup implications.",
            "Entertainment": "Watch release calendars, box-office reporting, ratings, and awards season catalysts.",
            "Crypto": "Monitor exchange, ETF, and regulator announcements that provide unambiguous market-moving confirmation.",
            "Science": "Track publication dates, conference readouts, and agency or lab statements.",
            "Health": "Watch trial readouts, FDA milestones, advisory-committee calendars, and sponsor guidance.",
            "World News": "Track sanctions, official statements, treaty votes, military posture changes, and multilateral responses.",
        }.get(category, "Monitor official source confirmation and follow-on second-order signals before treating the topic as durable."),
    ]


def _topic_key_dates(topic: dict) -> list[str]:
    latest = _parse_dt(topic.get("latest_event_at"))
    if latest is None:
        return ["Near-term follow-up timing is still unclear."]
    next_week = (latest + timedelta(days=7)).strftime("%b %d")
    next_month = (latest + timedelta(days=30)).strftime("%b %d")
    return [
        f"Near-term follow-up window: by {next_week}",
        f"If the story is durable, expect a clearer milestone before {next_month}",
    ]


def _topic_data_sources(topic: dict) -> list[str]:
    category = _category_label(topic.get("category"))
    return TOPIC_SOURCE_MAP.get(category, TOPIC_SOURCE_MAP["Other"])


def _topic_adjacent_topics(topic: dict) -> list[str]:
    category = _category_label(topic.get("category"))
    return ADJACENT_TOPIC_MAP.get(category, ADJACENT_TOPIC_MAP["Other"])


def enrich_topic_rows(topics: list[dict]) -> list[dict]:
    enriched = []
    for topic in topics:
        item = dict(topic)
        item["focus_sentence"] = _topic_focus_sentence(item)
        item["what_is_uncertain"] = _topic_what_is_uncertain(item)
        item["watching_points"] = _topic_watching_points(item)
        item["key_dates"] = _topic_key_dates(item)
        item["data_sources"] = _topic_data_sources(item)
        item["adjacent_topics"] = _topic_adjacent_topics(item)
        enriched.append(item)
    return enriched


def get_demo_markets() -> list[dict]:
    now = datetime.now(UTC)
    return [
        {
            "id": "demo-1",
            "question": "Will the SEC approve a spot Ethereum ETF before July 31, 2026?",
            "category": "Crypto",
            "question_type": "binary",
            "total_score": 0.92,
            "deadline": (now + timedelta(days=52)).strftime("%Y-%m-%d"),
            "deadline_source": "Regulatory filing calendar and issuer updates",
            "resolution_source": "SEC announcements and issuer filings",
            "resolution_criteria": "Resolves Yes if the SEC approves at least one spot Ethereum ETF before the deadline.",
            "rationale": "Crypto policy remains one of the most watched catalysts across retail prediction markets.",
            "score_breakdown": {
                "Market Interest": 0.96,
                "Resolution Strength": 0.91,
                "Clarity": 0.90,
                "Novelty": 0.81,
            },
            "validation_status": "Valid",
            "warnings": [],
            "created_at": (now - timedelta(days=1)).isoformat(),
        },
        {
            "id": "demo-2",
            "question": "Will OpenAI release GPT-6 publicly before September 30, 2026?",
            "category": "Technology",
            "question_type": "binary",
            "total_score": 0.84,
            "deadline": (now + timedelta(days=113)).strftime("%Y-%m-%d"),
            "deadline_source": "Company launch windows and public release statements",
            "resolution_source": "OpenAI product announcements",
            "resolution_criteria": "Resolves Yes if OpenAI publicly launches GPT-6 to paying or free users before the deadline.",
            "rationale": "Major AI model launches drive outsized interest and broad coverage across consumer tech audiences.",
            "score_breakdown": {
                "Market Interest": 0.90,
                "Resolution Strength": 0.82,
                "Clarity": 0.86,
                "Novelty": 0.70,
            },
            "validation_status": "Valid",
            "warnings": ["Release naming could change and should be monitored."],
            "created_at": (now - timedelta(days=3)).isoformat(),
        },
        {
            "id": "demo-3",
            "question": "Which studio will win Best Picture at the 2027 Oscars?",
            "category": "Entertainment",
            "question_type": "multiple_choice",
            "total_score": 0.76,
            "deadline": (now + timedelta(days=300)).strftime("%Y-%m-%d"),
            "deadline_source": "Academy Awards schedule",
            "resolution_source": "The Academy official winners list",
            "resolution_criteria": "Resolves to the studio behind the film that wins Best Picture at the 2027 Oscars.",
            "rationale": "Awards-season narratives create sticky recurring interest with clear public resolution.",
            "score_breakdown": {
                "Market Interest": 0.74,
                "Resolution Strength": 0.88,
                "Clarity": 0.78,
                "Novelty": 0.63,
            },
            "validation_status": "Needs Review",
            "warnings": ["Candidate studios may need normalization closer to the ceremony."],
            "created_at": (now - timedelta(days=5)).isoformat(),
        },
    ]


def get_demo_topics() -> list[dict]:
    now = datetime.now(UTC)
    topics = [
        {
            "id": "topic-1",
            "title": "AI product launches accelerate",
            "summary": "A cluster of product rumors, enterprise deals, and benchmark chatter is driving new AI market ideas.",
            "category": "Technology",
            "event_count": 14,
            "source_count": 9,
            "suggested_market_count": 6,
            "avg_candidate_score": 0.83,
            "latest_event_at": (now - timedelta(hours=8)).isoformat(),
        },
        {
            "id": "topic-2",
            "title": "Election polling volatility",
            "summary": "Polling swings and campaign announcements are creating fresh election-related market opportunities.",
            "category": "Politics",
            "event_count": 11,
            "source_count": 7,
            "suggested_market_count": 5,
            "avg_candidate_score": 0.79,
            "latest_event_at": (now - timedelta(days=1)).isoformat(),
        },
        {
            "id": "topic-3",
            "title": "Breakthrough biotech readouts",
            "summary": "Clinical milestones and FDA timelines are surfacing a steady stream of health and science markets.",
            "category": "Health",
            "event_count": 8,
            "source_count": 6,
            "suggested_market_count": 4,
            "avg_candidate_score": 0.74,
            "latest_event_at": (now - timedelta(days=2)).isoformat(),
        },
    ]
    for topic in topics:
        topic["trend_score"] = compute_trend_score(topic)
    return enrich_topic_rows(topics)


def normalize_market_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        warnings = _safe_json_list(row.get("validation_flags") or row.get("warnings"))
        validation_status = row.get("validation_status")
        if not validation_status:
            is_valid = row.get("is_valid")
            if is_valid is True:
                validation_status = "Valid"
            elif warnings:
                validation_status = "Needs Review"
            else:
                validation_status = "Unknown"

        score_breakdown = {
            "Market Interest": float(row.get("market_interest_score") or 0.0),
            "Resolution Strength": float(row.get("resolution_strength_score") or 0.0),
            "Clarity": float(row.get("clarity_score") or 0.0),
            "Mention Velocity": float(row.get("mention_velocity_score") or 0.0),
            "Novelty": float(row.get("novelty_score") or 0.0),
            "Time Horizon": float(row.get("time_horizon_score") or 0.0),
            "Source Diversity": float(row.get("source_diversity_score") or 0.0),
        }
        market = {
            "id": row.get("question_id") or row.get("id"),
            "question": row.get("question") or row.get("question_text") or "Untitled market",
            "question_text": row.get("question") or row.get("question_text") or "Untitled market",
            "category": _category_label(row.get("category")),
            "question_type": row.get("question_type") or "binary",
            "total_score": float(row.get("total_score") or 0.0),
            "deadline": row.get("deadline"),
            "deadline_source": row.get("deadline_source") or "Not provided",
            "resolution_source": row.get("resolution_source") or "Not provided",
            "resolution_criteria": row.get("resolution_criteria") or "Resolution criteria not available yet.",
            "rationale": row.get("rationale") or "This market is drawing attention because it sits at the intersection of news momentum and clear resolution.",
            "score_breakdown": score_breakdown,
            "validation_status": validation_status,
            "warnings": warnings,
            "created_at": row.get("question_created_at") or row.get("created_at"),
            "review_status": row.get("review_status"),
        }
        market["status"] = _market_status(market)
        normalized.append(market)
    return normalized


def compute_trend_score(topic: dict) -> float:
    event_count = float(topic.get("event_count") or 0.0)
    source_count = float(topic.get("source_count") or 0.0)
    suggested_market_count = float(topic.get("suggested_market_count") or 0.0)
    avg_candidate_score = float(topic.get("avg_candidate_score") or 0.0)

    latest_event_at = _parse_dt(topic.get("latest_event_at"))
    if latest_event_at is None:
        recency_points = 0.0
    else:
        age_days = max((datetime.now(UTC) - latest_event_at).total_seconds() / 86400, 0.0)
        recency_points = max(0.0, 25.0 - min(age_days, 25.0))

    trend_score = (
        min(event_count, 20.0) * 2.2
        + min(source_count, 15.0) * 2.0
        + min(suggested_market_count, 10.0) * 2.4
        + avg_candidate_score * 24.0
        + recency_points
    )
    return round(trend_score, 1)


def compute_summary_metrics(markets: list[dict], topics: list[dict]) -> dict:
    now = datetime.now(UTC)
    active_markets = [m for m in markets if _market_status(m) != "Expired"]
    avg_score = sum(float(m.get("total_score") or 0.0) for m in markets) / len(markets) if markets else 0.0
    new_this_week = sum(
        1
        for market in markets
        if (_parse_dt(market.get("created_at")) or now - timedelta(days=3650)) >= now - timedelta(days=7)
    )
    return {
        "Active Markets": len(active_markets),
        "Emerging Topics": len(topics),
        "Average Market Score": f"{avg_score * 100:.1f}",
        "New This Week": new_this_week,
    }


def _market_status(market: dict) -> str:
    if str(market.get("id")) in st.session_state.get("tracked_markets", set()):
        return "Tracked"
    days = _days_until(str(market.get("deadline") or ""))
    if days is not None and days < 0:
        return "Expired"
    if days is not None and days <= 14:
        return "Ending Soon"
    return "Active"


def filter_markets(markets: list[dict], filters: dict) -> list[dict]:
    filtered = list(markets)
    search_text = (filters.get("search_text") or "").strip().lower()
    if search_text:
        filtered = [
            market
            for market in filtered
            if search_text in market.get("question", "").lower()
            or search_text in market.get("rationale", "").lower()
            or search_text in market.get("category", "").lower()
        ]

    category = filters.get("category")
    if category and category != "All":
        filtered = [market for market in filtered if _category_label(market.get("category")) == category]

    explorer_category = filters.get("explorer_category")
    if explorer_category and explorer_category != "All":
        filtered = [market for market in filtered if _category_label(market.get("category")) == explorer_category]

    question_type = filters.get("question_type")
    if question_type and question_type != "All":
        normalized = "multiple_choice" if question_type == "Multiple Choice" else "binary"
        filtered = [market for market in filtered if market.get("question_type") == normalized]

    min_score = float(filters.get("minimum_score") or 0.0)
    filtered = [market for market in filtered if float(market.get("total_score") or 0.0) >= min_score]

    status = filters.get("status")
    if status and status != "All":
        filtered = [market for market in filtered if _market_status(market) == status]

    return sort_markets(filtered, filters.get("sort_order", "Highest Score"))


def sort_markets(markets: list[dict], sort_order: str) -> list[dict]:
    def deadline_key(market: dict) -> tuple[int, int]:
        days = _days_until(str(market.get("deadline") or ""))
        if days is None:
            return (1, 999999)
        if days < 0:
            return (2, abs(days))
        return (0, days)

    def created_key(market: dict) -> float:
        parsed = _parse_dt(market.get("created_at"))
        return parsed.timestamp() if parsed else 0.0

    def emerging_key(market: dict) -> tuple[float, float, float]:
        return (
            -float(market.get("total_score") or 0.0),
            -created_key(market),
            deadline_key(market)[1],
        )

    if sort_order == "Newest":
        return sorted(markets, key=lambda m: (-created_key(m), -float(m.get("total_score") or 0.0)))
    if sort_order == "Ending Soon":
        return sorted(markets, key=lambda m: (deadline_key(m), -float(m.get("total_score") or 0.0)))
    if sort_order == "Emerging":
        return sorted(markets, key=emerging_key)
    return sorted(
        markets,
        key=lambda m: (
            -float(m.get("total_score") or 0.0),
            deadline_key(m),
            -created_key(m),
        ),
    )


def initialize_session_state() -> None:
    if "tracked_markets" not in st.session_state:
        st.session_state.tracked_markets = set()
    if "saved_topics" not in st.session_state:
        st.session_state.saved_topics = set()
    if "category_explorer" not in st.session_state:
        st.session_state.category_explorer = "All"


def load_dashboard_data() -> dict:
    data = {
        "markets": [],
        "topics": [],
        "using_demo": False,
        "db_warning": None,
        "source_label": "Live data",
        "pipeline_run": None,
        "pipeline_stages": [],
    }

    try:
        init_db()
        raw_market_rows = [dict(row) for row in get_dashboard_scored_questions()]
        raw_topic_rows = [dict(row) for row in get_dashboard_topics()]
        data["pipeline_run"] = get_active_pipeline_run() or get_latest_pipeline_run()
        if data["pipeline_run"]:
            data["pipeline_stages"] = get_pipeline_run_stages(int(data["pipeline_run"]["id"]))

        markets = normalize_market_rows(raw_market_rows)
        topics = [dict(row) for row in raw_topic_rows]
        for topic in topics:
            topic.setdefault("category", _category_label(topic.get("category") or topic.get("event_type") or "Other"))
            topic["trend_score"] = compute_trend_score(topic)
        topics = enrich_topic_rows(topics)
        data["markets"] = markets
        data["topics"] = topics

        if not markets:
            data["db_warning"] = "No scored candidates are available yet. Showing demo discovery data until the pipeline produces markets."
            data["markets"] = get_demo_markets()
            data["topics"] = get_demo_topics()
            data["using_demo"] = True
            data["source_label"] = "Demo data"
    except Exception as exc:
        data["db_warning"] = f"Database connection failed. Showing demo discovery data instead. Details: {exc}"
        data["markets"] = get_demo_markets()
        data["topics"] = get_demo_topics()
        data["using_demo"] = True
        data["source_label"] = "Demo data"

    return data


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
          <div class="eyebrow">Prediction Market Discovery</div>
          <h1>Prediction Market Discovery</h1>
          <p>Track the market questions worth following now, and separate out emerging topics that still belong in a research watchlist.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(data: dict) -> None:
    metrics = compute_summary_metrics(data["markets"], data["topics"])
    columns = st.columns(4)
    for column, (label, value) in zip(columns, metrics.items()):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if data.get("db_warning"):
        st.warning(data["db_warning"])


def render_sidebar_filters(data: dict) -> dict:
    with st.sidebar:
        st.markdown("## Discover")
        st.caption(f"Source: {data.get('source_label', 'Live data')}")
        search_text = st.text_input("Search", placeholder="Search markets, categories, or themes")
        category = st.selectbox("Category", DISCOVERY_CATEGORIES, index=0)
        question_type = st.selectbox("Question Type", ["All", "Binary", "Multiple Choice"], index=0)
        minimum_score = st.slider("Minimum Score", min_value=0, max_value=100, value=40, step=5) / 100
        status = st.selectbox("Status", STATUS_OPTIONS, index=0)
        sort_order = st.selectbox("Sort Order", DISCOVERY_SORT_OPTIONS, index=0)

    return {
        "search_text": search_text,
        "category": category,
        "question_type": question_type,
        "minimum_score": minimum_score,
        "status": status,
        "sort_order": sort_order,
        "explorer_category": st.session_state.get("category_explorer", "All"),
    }


def render_topic_card(topic: dict) -> None:
    saved_topics = st.session_state.saved_topics
    is_saved = str(topic.get("id")) in saved_topics
    container = st.container(border=True)
    with container:
        top_left, top_right = st.columns([5, 1])
        with top_left:
            st.markdown(
                f"### {topic.get('title') or topic.get('summary') or 'Emerging topic'}\n\n"
                f"{topic.get('focus_sentence') or topic.get('summary') or 'This topic is starting to generate fresh market ideas.'}"
            )
        with top_right:
            st.markdown(
                f"""
                <div style="text-align:right;">
                  <div class="trend-label">Trend Score</div>
                  <div class="trend-value">{topic.get('trend_score', 0):.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"{_category_chip(topic.get('category'))} {_topic_label_chip('Research Signal Only')} {_topic_label_chip('No Market Listed Yet', 'watch')}",
            unsafe_allow_html=True,
        )
        stat_cols = st.columns(3)
        stat_cols[0].metric("Events", int(topic.get("event_count") or 0))
        stat_cols[1].metric("Sources", int(topic.get("source_count") or 0))
        stat_cols[2].metric("Coverage", int(topic.get("suggested_market_count") or 0))

        st.markdown(
            f"""
            <div class="example-callout">
              <strong>What is still uncertain:</strong> {topic.get("what_is_uncertain")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Research Brief"):
            st.markdown("**Watching Points**")
            for item in topic.get("watching_points", []):
                st.markdown(f"- {item}")

            st.markdown("**Key Dates**")
            for item in topic.get("key_dates", []):
                st.markdown(f"- {item}")

            st.markdown("**Data Sources To Check**")
            for item in topic.get("data_sources", []):
                st.markdown(f"- {item}")

            st.markdown("**Adjacent Topics**")
            for item in topic.get("adjacent_topics", []):
                st.markdown(f"- {item}")

        action_label = "Saved" if is_saved else "Save Topic"
        if st.button(action_label, key=f"save-topic-{topic.get('id')}", disabled=is_saved):
            saved_topics.add(str(topic.get("id")))
            st.rerun()


def render_emerging_topics(data: dict, filters: dict) -> None:
    st.markdown("## Emerging Topics")
    st.caption("Research intelligence only: topics with momentum but no listed market yet.")
    topics = list(data["topics"])

    search_text = (filters.get("search_text") or "").strip().lower()
    if search_text:
        topics = [
            topic
            for topic in topics
            if search_text in (topic.get("title") or "").lower()
            or search_text in (topic.get("summary") or "").lower()
            or search_text in (topic.get("what_is_uncertain") or "").lower()
        ]

    category = filters.get("category")
    if category and category != "All":
        topics = [topic for topic in topics if _category_label(topic.get("category")) == category]

    topics.sort(key=lambda topic: (-float(topic.get("trend_score") or 0.0), -(topic.get("event_count") or 0)))

    if not topics:
        st.info("No emerging topics match the current filters yet.")
        return

    for topic in topics[:6]:
        render_topic_card(topic)


def render_category_explorer(data: dict, filters: dict) -> None:
    st.markdown("## Category Explorer")
    available_counts: dict[str, int] = {}
    for market in data["markets"]:
        label = _category_label(market.get("category"))
        available_counts[label] = available_counts.get(label, 0) + 1

    option_labels = []
    for label in DISCOVERY_CATEGORIES:
        if label == "All":
            option_labels.append("All")
        else:
            option_labels.append(f"{label} ({available_counts.get(label, 0)})")

    current = st.session_state.get("category_explorer", "All")
    current_index = DISCOVERY_CATEGORIES.index(current) if current in DISCOVERY_CATEGORIES else 0
    selected_option = st.selectbox(
        "Browse categories",
        option_labels,
        index=current_index,
        key="category_explorer_select",
    )
    selected_category = selected_option.split(" (")[0]
    st.session_state.category_explorer = selected_category

    if selected_category == "All":
        st.caption("Browse the full market landscape or narrow the discovery feed to a category.")
    else:
        st.caption(f"Showing discovery results for {selected_category}.")


def render_market_card(market: dict) -> None:
    tracked_markets = st.session_state.tracked_markets
    market_id = str(market.get("id"))
    days_left = _days_until(str(market.get("deadline") or ""))
    deadline_label = _format_date_label(market.get("deadline"))
    if days_left is None:
        deadline_suffix = "TBD"
    elif days_left < 0:
        deadline_suffix = "Expired"
    elif days_left == 0:
        deadline_suffix = "Ends today"
    else:
        deadline_suffix = f"{days_left} days left"

    with st.container(border=True):
        header_left, header_right = st.columns([6, 1.3])
        with header_left:
            st.markdown(f"### {market.get('question')}")
            st.markdown(
                f"{_category_chip(market.get('category'))} {_type_chip(market.get('question_type'))} "
                f"{_status_chip(market.get('validation_status') or 'Unknown')} {_topic_label_chip('Trackable Now', 'watch')}",
                unsafe_allow_html=True,
            )
        with header_right:
            st.metric("Score", _format_score(market.get("total_score")))

        meta_cols = st.columns(4)
        meta_cols[0].markdown(f"**Deadline**  \n{deadline_label}")
        meta_cols[1].markdown(f"**Timing**  \n{deadline_suffix}")
        meta_cols[2].markdown(f"**Resolution Source**  \n{market.get('resolution_source') or 'Not specified'}")
        meta_cols[3].markdown(f"**Category**  \n{_category_label(market.get('category'))}")

        st.markdown(
            f"""
            <div class="rationale-box">
              <strong>Why traders care:</strong> {market.get("rationale")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        action_cols = st.columns([1, 6])
        with action_cols[0]:
            if market_id in tracked_markets:
                st.button("Tracking", key=f"track-{market_id}", disabled=True, use_container_width=True)
            else:
                if st.button("Track Market", key=f"track-{market_id}", use_container_width=True):
                    tracked_markets.add(market_id)
                    st.rerun()

        with st.expander("View Details"):
            st.markdown(f"**Resolution Criteria**  \n{market.get('resolution_criteria')}")
            st.markdown(f"**Deadline Source**  \n{market.get('deadline_source')}")
            st.markdown(f"**Generated Rationale**  \n{market.get('rationale')}")
            st.markdown(f"**Resolution Source Link**  \n{_linkish_source(market.get('resolution_source'))}")

            st.markdown("**Score Breakdown**")
            for label, value in market.get("score_breakdown", {}).items():
                st.progress(min(max(float(value), 0.0), 1.0), text=f"{label}: {_format_percent(value)}")

            warnings = market.get("warnings") or []
            if warnings:
                st.markdown("**Warnings**")
                for warning in warnings:
                    st.warning(str(warning))
            else:
                st.caption("No warnings recorded for this market.")


def render_active_markets(data: dict, filters: dict) -> None:
    st.markdown("## Active Markets")
    st.caption("Highest-conviction market questions worth tracking right now, ordered by score and near-term relevance.")
    filters = dict(filters)
    filters["explorer_category"] = st.session_state.get("category_explorer", "All")
    markets = filter_markets(data["markets"], filters)

    if not data["markets"]:
        st.markdown(
            """
            <div class="empty-state-card">
              <h3>No markets yet</h3>
              <p>Markets will appear here after the pipeline runs and scores candidate questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not markets:
        st.markdown(
            """
            <div class="empty-state-card">
              <h3>No markets match these filters</h3>
              <p>Try broadening the search, lowering the score threshold, or switching categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"{len(markets)} markets shown")
    for market in markets:
        render_market_card(market)


def render_saved_items_sidebar() -> None:
    with st.sidebar:
        st.markdown("---")
        st.markdown("## Saved Items")
        tracked = sorted(st.session_state.tracked_markets)
        saved_topics = sorted(st.session_state.saved_topics)

        if not tracked and not saved_topics:
            st.caption("Track markets or save topics to keep a lightweight watchlist here.")
        else:
            if tracked:
                st.markdown("**Tracked Markets**")
                for item in tracked:
                    st.caption(f"Market #{item}")
            if saved_topics:
                st.markdown("**Saved Topics**")
                for item in saved_topics:
                    st.caption(f"Topic #{item}")

        if st.button("Clear Saved Items", use_container_width=True):
            st.session_state.tracked_markets = set()
            st.session_state.saved_topics = set()
            st.rerun()


def render_admin_tools_collapsed() -> None:
    with st.sidebar:
        with st.expander("Admin Tools", expanded=False):
            st.caption("Hidden by default so the dashboard stays consumer-facing.")
            try:
                run_row = get_active_pipeline_run() or get_latest_pipeline_run()
                if not run_row:
                    st.write("No pipeline run metadata available.")
                    return

                st.write(f"Latest run: #{run_row.get('id')}")
                st.write(f"Status: {run_row.get('status', 'unknown')}")
                stages = get_pipeline_run_stages(int(run_row["id"]))
                if stages:
                    st.progress(_pipeline_progress_value(stages))
                    for stage in stages:
                        label = PIPELINE_STAGE_LABELS.get(stage.get("stage_number"), f"Stage {stage.get('stage_number')}")
                        lines = _stage_info_lines(stage, run_row)
                        st.markdown(f"**{label}**")
                        st.caption(" | ".join(lines) if lines else stage.get("status", "pending"))
            except Exception:
                st.write("Admin metadata is unavailable in demo mode.")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14,165,233,0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(250,204,21,0.15), transparent 25%),
                linear-gradient(180deg, #f8fafc 0%, #eff6ff 50%, #f8fafc 100%);
        }
        .main .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .hero-shell {
            padding: 1.25rem 0 1.75rem 0;
        }
        .hero-shell .eyebrow {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(29, 78, 216, 0.10);
            color: #1d4ed8;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 14px;
        }
        .hero-shell h1 {
            margin: 0;
            color: #0f172a;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        .hero-shell p {
            margin: 10px 0 0 0;
            color: #334155;
            font-size: 1.05rem;
            max-width: 760px;
            line-height: 1.7;
        }
        .metric-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(8px);
        }
        .metric-label {
            color: #475569;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-value {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 800;
            margin-top: 6px;
        }
        .example-callout {
            margin-top: 12px;
            padding: 12px 14px;
            border-radius: 16px;
            background: #fff7ed;
            color: #7c2d12;
            border: 1px solid #fdba74;
        }
        .rationale-box {
            margin-top: 14px;
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(148,163,184,0.25);
            color: #334155;
        }
        .empty-state-card {
            padding: 28px;
            border-radius: 24px;
            background: rgba(255,255,255,0.78);
            border: 1px dashed rgba(148,163,184,0.6);
            text-align: center;
            color: #475569;
        }
        .empty-state-card h3 {
            color: #0f172a;
            margin-bottom: 8px;
        }
        .trend-label {
            color: #64748b;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
        }
        .trend-value {
            color: #0f172a;
            font-size: 1.75rem;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Prediction Market Discovery",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    initialize_session_state()
    data = load_dashboard_data()
    render_header()
    render_summary_metrics(data)
    filters = render_sidebar_filters(data)
    render_emerging_topics(data, filters)
    render_category_explorer(data, filters)
    render_active_markets(data, filters)
    render_saved_items_sidebar()
    render_admin_tools_collapsed()


if __name__ == "__main__":
    main()
