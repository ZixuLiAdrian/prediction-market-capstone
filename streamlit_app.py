"""
FR7: Prediction Market Intelligence Dashboard

Review dashboard for scored prediction market candidates. The active queue is
re-ranked on every load so old batch-local FR6 ranks do not create duplicate
"#1" / "#2" entries after multiple runs.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import streamlit as st

from config import LLMConfig
from db.connection import (
    cancel_pipeline_run,
    create_pipeline_run,
    get_active_pipeline_run,
    get_dashboard_repair_questions,
    get_dashboard_scored_questions,
    get_latest_pipeline_run,
    get_pipeline_run_stages,
    init_db,
    mark_pipeline_run_failed,
    mark_pipeline_run_started,
    set_question_review_status,
)
from validation.validator import is_salvageable_validation_flags

UTC = timezone.utc

CAT_COLORS = {
    "politics": ("#60A5FA", "#1e3a5f"),
    "finance": ("#34D399", "#064e3b"),
    "crypto": ("#FBBF24", "#451a03"),
    "sports": ("#F87171", "#450a0a"),
    "health": ("#F472B6", "#500724"),
    "technology": ("#A78BFA", "#2e1065"),
    "environment": ("#4ADE80", "#052e16"),
    "science": ("#38BDF8", "#082f49"),
    "legal": ("#FB923C", "#431407"),
    "entertainment": ("#C084FC", "#3b0764"),
    "international": ("#818CF8", "#1e1b4b"),
    "economics": ("#2DD4BF", "#042f2e"),
    "business": ("#2DD4BF", "#042f2e"),
    "energy": ("#F59E0B", "#451a03"),
    "space": ("#38BDF8", "#082f49"),
    "geopolitics": ("#818CF8", "#1e1b4b"),
    "other": ("#94A3B8", "#1e293b"),
}

STATUS_META = {
    "active": ("#60A5FA", "#1e3a5f", "Active"),
    "selected": ("#34D399", "#064e3b", "Selected"),
    "removed": ("#F87171", "#450a0a", "Removed"),
    "expired": ("#FBBF24", "#451a03", "Expired"),
    "needs_repair": ("#F97316", "#431407", "Needs Repair"),
}

COMPONENT_META = [
    ("market_interest_score", "Market Interest", "25%"),
    ("resolution_strength_score", "Resolution Strength", "25%"),
    ("clarity_score", "Clarity", "20%"),
    ("mention_velocity_score", "Mention Velocity", "10%"),
    ("novelty_score", "Novelty", "10%"),
    ("time_horizon_score", "Time Horizon", "5%"),
    ("source_diversity_score", "Source Diversity", "5%"),
]

SORT_OPTIONS = {
    "Score (Highest First)": lambda r: -float(r.get("total_score") or 0.0),
    "Deadline (Soonest)": lambda r: r.get("deadline") or "9999-99-99",
    "Category (A-Z)": lambda r: (r.get("category") or "").lower(),
}

PIPELINE_STAGE_LABELS = {
    1: "FR1 Event Ingestion",
    2: "FR2 Event Clustering",
    3: "FR3 LLM Extraction",
    4: "FR4 Question Generation",
    5: "FR5 Rule Validation",
    6: "FR6 Heuristic Scoring",
}


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#34D399"
    if score >= 0.55:
        return "#60A5FA"
    if score >= 0.40:
        return "#F97316"
    return "#F87171"


def _days_until(deadline_str: str) -> Optional[int]:
    if not deadline_str:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d", "%d %B %Y"):
        try:
            deadline = datetime.strptime(deadline_str.strip(), fmt).date()
            return (deadline - datetime.now(UTC).date()).days
        except ValueError:
            continue
    return None


def _days_chip(days: Optional[int]) -> str:
    if days is None:
        return ""
    if days < 0:
        return '<span class="days-chip days-urgent">Expired</span>'
    if days == 0:
        return '<span class="days-chip days-urgent">Today</span>'
    if days <= 7:
        return f'<span class="days-chip days-urgent">{days}d left</span>'
    if days <= 30:
        return f'<span class="days-chip days-soon">{days}d left</span>'
    return f'<span class="days-chip days-ok">{days}d left</span>'


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
        key=lambda r: (-float(r.get("total_score") or 0.0), int(r.get("question_id") or 0)),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _parse_options(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


def _parse_flags(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(flag) for flag in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(flag) for flag in parsed]
        except Exception:
            return [raw]
    return []


def _cat_badge(category: str) -> str:
    text, bg = CAT_COLORS.get((category or "other").lower(), CAT_COLORS["other"])
    label = (category or "other").replace("_", " ").title()
    return (
        f'<span class="badge" style="color:{text};background:{bg}CC;'
        f'border:1px solid {text}55;">{label}</span>'
    )


def _type_badge(question_type: str) -> str:
    if (question_type or "").lower() == "binary":
        return (
            '<span class="badge" style="color:#F97316;background:#431407CC;'
            'border:1px solid #F9731655;">Binary</span>'
        )
    return (
        '<span class="badge" style="color:#A78BFA;background:#2e1065CC;'
        'border:1px solid #A78BFA55;">Multiple Choice</span>'
    )


def _status_badge(status: str) -> str:
    text, bg, label = STATUS_META.get((status or "active").lower(), STATUS_META["active"])
    return (
        f'<span class="badge" style="color:{text};background:{bg}CC;'
        f'border:1px solid {text}55;">{label}</span>'
    )


def _rank_badge(rank: int) -> str:
    if rank == 1:
        style = "background:linear-gradient(135deg,#F97316,#EA580C);"
    elif rank == 2:
        style = "background:linear-gradient(135deg,#FB923C,#C2410C);"
    elif rank == 3:
        style = "background:linear-gradient(135deg,#FDBA74,#EA580C);"
    else:
        style = "background:#1E293B;border:1px solid #334155;"
    return f'<div class="rank-badge" style="{style}">{rank}</div>'


def _assign_repair_ranks(rows: list[dict]) -> list[dict]:
    ranked = [dict(r) for r in rows]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _score_circle(score: float) -> str:
    color = _score_color(score)
    pct = int(score * 100)
    return (
        f'<div class="score-circle" style="border:3px solid {color};color:{color};">'
        f'<span class="sc-num">{pct}</span>'
        f'<span class="sc-sub">/ 100</span>'
        f"</div>"
    )


def _source_link(source: str) -> str:
    match = re.search(r"https?://[^\s)]+", source or "")
    if match:
        url = match.group(0).rstrip(".,;")
        label = urlparse(url).netloc.replace("www.", "")
        return f'<a href="{url}" target="_blank" rel="noopener">{label}</a>'
    source = (source or "-").strip()
    return source[:55] + ("..." if len(source) > 55 else "")


def _breakdown_row(label: str, weight: str, value: float) -> str:
    color = _score_color(value)
    pct = int(value * 100)
    return (
        '<div class="br-row">'
        f'<span class="br-label">{label}</span>'
        f'<span class="br-weight">{weight}</span>'
        '<div class="br-track">'
        f'<div class="br-fill" style="width:{pct}%;background:{color};"></div>'
        '</div>'
        f'<span class="br-val" style="color:{color};">{pct}%</span>'
        '</div>'
    )


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


def _filter_repair_rows(
    rows: list[dict],
    search: str,
    selected_cats: list[str],
    type_choice: str,
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

    if sort_choice == "Category (A-Z)":
        filtered.sort(key=lambda r: (r.get("category") or "").lower())
    elif sort_choice == "Deadline (Soonest)":
        filtered.sort(key=lambda r: r.get("deadline") or "9999-99-99")
    else:
        # Default repair queue order is already popularity-ranked from the DB.
        filtered.sort(key=lambda r: r.get("rank", 0))
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
    """Build the background pipeline command for Streamlit launches."""
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
    normalized_mode = (fr4_mode or "default").strip().lower()
    if normalized_mode == "custom" and custom_limit is not None:
        command.extend(["--fr4-limit", str(int(custom_limit))])
    elif normalized_mode == "all":
        command.append("--fr4-all")
    return command


def _format_run_timestamp(value) -> str:
    if not value:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


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


def _model_index(options: list[str], preferred: str) -> int:
    """Return a safe selectbox index for a preferred model name."""
    try:
        return options.index(preferred)
    except ValueError:
        return 0


def _stage_progress_value(stage_row: dict) -> float | None:
    """Return per-stage fractional progress when the summary exposes processed/total counts."""
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


def _terminate_process_tree(pid: int) -> None:
    """Terminate a background pipeline process and its children."""
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


def _stage_info_lines(stage_row: dict, run_row: dict | None) -> list[str]:
    """Build the compact info-popover lines for a pipeline stage."""
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


def _inject_css() -> None:
    st.markdown(
        """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", "Helvetica Neue", sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.main { background: #0F1117; }
.main .block-container { padding: 2rem 2.5rem 4rem 2.5rem; max-width: 1300px; }
section[data-testid="stSidebar"] { background: #14161E !important; border-right: 1px solid #1E2535; }

.app-title { font-size: 30px; font-weight: 800; color: #F1F5F9; letter-spacing: -0.8px; line-height: 1.15; margin: 0; }
.app-title span { color: #F97316; }
.app-subtitle { font-size: 15px; color: #94A3B8; margin-top: 6px; font-weight: 400; line-height: 1.6; }

.stat-grid { display: flex; gap: 12px; margin: 22px 0 30px 0; }
.stat-card { flex: 1; background: #14161E; border: 1px solid #1E2535; border-radius: 14px; padding: 18px 20px; }
.stat-value { font-size: 30px; font-weight: 800; color: #F1F5F9; line-height: 1; }
.stat-label { font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 6px; }

.q-card { background: #14161E; border: 1px solid #1E2535; border-radius: 16px; padding: 24px 26px 20px 26px; margin-bottom: 10px; }
.q-header { display: flex; align-items: flex-start; gap: 14px; }
.rank-badge { min-width: 36px; height: 36px; border-radius: 50%; color: #FFFFFF; font-size: 13px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 3px; }
.q-text { font-size: 17px; font-weight: 600; color: #F1F5F9; line-height: 1.5; flex: 1; }
.score-circle { min-width: 54px; height: 54px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; line-height: 1; }
.sc-num { font-size: 17px; font-weight: 800; }
.sc-sub { font-size: 9px; font-weight: 500; opacity: 0.6; margin-top: 1px; }
.badges-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 12px 0 0 50px; }
.badge { font-size: 12px; font-weight: 600; padding: 4px 11px; border-radius: 20px; white-space: nowrap; }
.score-bar-wrap { margin: 16px 0 12px 0; background: #1E293B; border-radius: 6px; height: 5px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 6px; }
.q-meta { display: flex; gap: 22px; flex-wrap: wrap; font-size: 13.5px; color: #64748B; margin-top: 4px; align-items: center; }
.meta-item a { color: #F97316; text-decoration: none; }
.days-chip { font-size: 11.5px; font-weight: 600; padding: 2px 9px; border-radius: 10px; margin-left: 5px; }
.days-urgent { background: #450a0a; color: #F87171; border: 1px solid #F8717140; }
.days-soon { background: #1c1400; color: #FBBF24; border: 1px solid #FBBF2440; }
.days-ok { background: #052e16; color: #4ADE80; border: 1px solid #4ADE8040; }
.option-pill { font-size: 13px; font-weight: 500; padding: 5px 14px; border-radius: 20px; background: #1E293B; border: 1px solid #334155; color: #CBD5E1; }

.results-bar { font-size: 14px; color: #64748B; font-weight: 500; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #1E2535; }
.results-bar strong { color: #94A3B8; }
.empty-state { text-align: center; padding: 70px 20px; }
.empty-title { font-size: 22px; font-weight: 700; color: #94A3B8; margin-bottom: 10px; }
.empty-sub { font-size: 15px; color: #475569; line-height: 1.7; max-width: 420px; margin: 0 auto; }

.br-section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #475569; margin: 0 0 14px 0; }
.br-row { display: flex; align-items: center; gap: 12px; margin-bottom: 11px; }
.br-label { font-size: 13px; color: #CBD5E1; min-width: 175px; flex-shrink: 0; font-weight: 500; }
.br-weight { font-size: 11px; color: #475569; width: 32px; text-align: right; flex-shrink: 0; font-weight: 600; }
.br-track { flex: 1; background: #1E293B; border-radius: 5px; height: 9px; overflow: hidden; }
.br-fill { height: 100%; border-radius: 5px; }
.br-val { font-size: 13px; font-weight: 700; width: 38px; text-align: right; flex-shrink: 0; }

.llm-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 4px 0; }
.llm-chip { font-size: 13px; padding: 8px 14px; border-radius: 10px; background: #1A1D27; border: 1px solid #2D3348; color: #94A3B8; }
.llm-label { font-size: 10px; color: #475569; display: block; margin-bottom: 2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.detail-header { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.9px; color: #475569; margin: 20px 0 8px 0; }
.detail-text { font-size: 14px; color: #CBD5E1; line-height: 1.7; padding: 14px 16px; background: #1A1D27; border-radius: 10px; border: 1px solid #2D3348; }
.detail-text.muted { color: #94A3B8; font-style: italic; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_card(row: dict) -> None:
    rank = int(row["rank"])
    score = float(row.get("total_score") or 0.0)
    color = _score_color(score)
    pct = int(score * 100)
    days = _days_until((row.get("deadline") or "").strip())
    effective_status = row.get("effective_status", "active")

    options = _parse_options(row.get("options"))
    options_html = ""
    if options:
        pills = "".join(f'<span class="option-pill">{option}</span>' for option in options)
        options_html = f'<div style="display:flex;gap:7px;flex-wrap:wrap;margin-top:12px;">{pills}</div>'

    status_html = ""
    if effective_status != "active":
        status_html = _status_badge(effective_status)

    st.markdown(
        f"""
<div class="q-card">
  <div class="q-header">
    {_rank_badge(rank)}
    <div class="q-text">{row.get('question_text', '')}</div>
    {_score_circle(score)}
  </div>
  <div class="badges-row">
    {_cat_badge(row.get('category', 'other'))}
    {_type_badge(row.get('question_type', 'binary'))}
    {status_html}
  </div>
  <div class="score-bar-wrap">
    <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
  </div>
  <div class="q-meta">
    <span class="meta-item">Deadline: {row.get('deadline', '-') or '-'}{_days_chip(days)}</span>
    <span class="meta-item">Source: {_source_link(row.get('resolution_source', ''))}</span>
  </div>
  {options_html}
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Score breakdown and details", expanded=False):
        bars = '<div class="br-section-title">Score Components</div>'
        for key, label, weight in COMPONENT_META:
            bars += _breakdown_row(label, weight, float(row.get(key) or 0.0))

        res_conf = float(row.get("resolution_confidence") or 0.0)
        src_ind = float(row.get("source_independence") or 0.0)
        timing = float(row.get("timing_reliability") or 0.0)
        chips = f"""
<div class="llm-grid">
  <div class="llm-chip"><span class="llm-label">Resolution Confidence</span><strong style="color:{_score_color(res_conf)};">{int(res_conf*100)}%</strong></div>
  <div class="llm-chip"><span class="llm-label">Source Independence</span><strong style="color:{_score_color(src_ind)};">{int(src_ind*100)}%</strong></div>
  <div class="llm-chip"><span class="llm-label">Timing Reliability</span><strong style="color:{_score_color(timing)};">{int(timing*100)}%</strong></div>
</div>
        """
        st.markdown(bars + chips, unsafe_allow_html=True)

        resolution_criteria = (row.get("resolution_criteria") or "").strip()
        rationale = (row.get("rationale") or "").strip()
        if resolution_criteria:
            st.markdown(
                '<div class="detail-header">Resolution Criteria</div>'
                f'<div class="detail-text">{resolution_criteria}</div>',
                unsafe_allow_html=True,
            )
        if rationale:
            st.markdown(
                '<div class="detail-header">Why This Market?</div>'
                f'<div class="detail-text muted">{rationale}</div>',
                unsafe_allow_html=True,
            )


def _render_actions(row: dict, status: str) -> None:
    question_id = int(row["question_id"])
    if status == "active":
        left, mid, _ = st.columns([1, 1, 6])
        with left:
            if st.button("Select", key=f"select_{question_id}", use_container_width=True):
                set_question_review_status(question_id, "selected")
                st.rerun()
        with mid:
            if st.button("Remove", key=f"remove_{question_id}", use_container_width=True):
                set_question_review_status(question_id, "removed")
                st.rerun()
        return

    if status in {"selected", "removed"}:
        left, _ = st.columns([1, 7])
        with left:
            if st.button("Restore", key=f"restore_{question_id}", use_container_width=True):
                set_question_review_status(question_id, "active")
                st.rerun()


def _render_repair_card(row: dict) -> None:
    rank = int(row["rank"])
    flags = _parse_flags(row.get("validation_flags"))
    flag_badges = "".join(
        f'<span class="badge" style="color:#F97316;background:#431407CC;border:1px solid #F9731655;">{flag}</span>'
        for flag in flags
    )
    status_html = _status_badge("needs_repair")
    popularity_html = (
        f"Weighted velocity: {float(row.get('weighted_mention_velocity') or 0.0):.2f}"
        f" · Sources: {int(row.get('source_diversity') or 0)}"
        f" · Extraction confidence: {float(row.get('extraction_confidence') or 0.0):.2f}"
    )

    st.markdown(
        f"""
<div class="q-card">
  <div class="q-header">
    {_rank_badge(rank)}
    <div class="q-text">{row.get('question_text', '')}</div>
  </div>
  <div class="badges-row">
    {_cat_badge(row.get('category', 'other'))}
    {_type_badge(row.get('question_type', 'binary'))}
    {status_html}
  </div>
  <div class="badges-row">{flag_badges}</div>
  <div class="q-meta" style="margin-top:14px;">
    <span class="meta-item">Repair priority: {popularity_html}</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Repair details", expanded=False):
        st.markdown(
            '<div class="detail-header">Why It Failed</div>'
            f'<div class="detail-text">{", ".join(flags) if flags else "No validation flags recorded."}</div>',
            unsafe_allow_html=True,
        )
        if row.get("resolution_criteria"):
            st.markdown(
                '<div class="detail-header">Current Resolution Criteria</div>'
                f'<div class="detail-text">{row["resolution_criteria"]}</div>',
                unsafe_allow_html=True,
            )
        if row.get("rationale"):
            st.markdown(
                '<div class="detail-header">Original Rationale</div>'
                f'<div class="detail-text muted">{row["rationale"]}</div>',
                unsafe_allow_html=True,
            )


def _render_view(
    label: str,
    rows: list[dict],
    search: str,
    selected_cats: list[str],
    type_choice: str,
    min_score_pct: int,
    sort_choice: str,
) -> None:
    if not rows:
        st.markdown(
            f"""
<div class="empty-state">
  <div class="empty-title">No {label.lower()} markets</div>
  <div class="empty-sub">This section is currently empty.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    filtered = _filter_rows(rows, search, selected_cats, type_choice, min_score_pct, sort_choice)
    if not filtered:
        st.markdown(
            """
<div class="empty-state">
  <div class="empty-title">No matches found</div>
  <div class="empty-sub">Try broadening your search or relaxing the filters.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    suffix = f" &nbsp;&middot;&nbsp; filtered from <strong>{len(rows)}</strong>" if len(filtered) < len(rows) else ""
    st.markdown(
        f'<div class="results-bar"><strong>{len(filtered)}</strong> {label.lower()} market{"s" if len(filtered) != 1 else ""}{suffix}</div>',
        unsafe_allow_html=True,
    )

    for row in filtered:
        _render_card(row)
        _render_actions(row, row.get("effective_status", "active"))


def _render_repair_view(
    rows: list[dict],
    search: str,
    selected_cats: list[str],
    type_choice: str,
    sort_choice: str,
) -> None:
    if not rows:
        st.markdown(
            """
<div class="empty-state">
  <div class="empty-title">No repair opportunities</div>
  <div class="empty-sub">High-signal failed questions will appear here when they look salvageable.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    filtered = _filter_repair_rows(rows, search, selected_cats, type_choice, sort_choice)
    if not filtered:
        st.markdown(
            """
<div class="empty-state">
  <div class="empty-title">No matches found</div>
  <div class="empty-sub">Try broadening your search or relaxing the filters.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    suffix = f" &nbsp;&middot;&nbsp; filtered from <strong>{len(rows)}</strong>" if len(filtered) < len(rows) else ""
    st.markdown(
        f'<div class="results-bar"><strong>{len(filtered)}</strong> repair opportunit{"ies" if len(filtered) != 1 else "y"}{suffix}</div>',
        unsafe_allow_html=True,
    )

    for row in filtered:
        _render_repair_card(row)


def main() -> None:
    st.set_page_config(
        page_title="Prediction Market Intelligence",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_db()
    _inject_css()

    raw_rows = [dict(r) for r in get_dashboard_scored_questions()]
    raw_repair_rows = [dict(r) for r in get_dashboard_repair_questions()]
    all_rows = []
    grouped_rows = {status: [] for status in STATUS_META}
    for row in raw_rows:
        item = dict(row)
        item["effective_status"] = _effective_status(item)
        all_rows.append(item)
        grouped_rows[item["effective_status"]].append(item)

    for status in grouped_rows:
        grouped_rows[status] = _assign_display_ranks(grouped_rows[status])

    active_rows = grouped_rows["active"]
    selected_rows = grouped_rows["selected"]
    removed_rows = grouped_rows["removed"]
    expired_rows = grouped_rows["expired"]
    repair_rows = _assign_repair_ranks(
        [
            dict(row)
            for row in raw_repair_rows
            if is_salvageable_validation_flags(_parse_flags(row.get("validation_flags")))
            and not bool(row.get("has_repair_child"))
        ]
    )

    hdr_left, hdr_right = st.columns([5, 1])
    with hdr_left:
        st.markdown(
            """
<div class="app-header">
  <div class="app-title"><span>Prediction</span> Market Intelligence</div>
  <div class="app-subtitle">Review scored questions, keep the best ones, and track selected, removed, and expired ideas separately.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with hdr_right:
        st.write("")
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    repo_root = Path(__file__).resolve().parent
    pipeline_path = str(repo_root / "pipeline.py")
    python_executable = sys.executable
    current_run = get_active_pipeline_run() or get_latest_pipeline_run()
    current_run_id = int(current_run["id"]) if current_run else None
    current_stage_rows = get_pipeline_run_stages(current_run_id) if current_run_id is not None else []

    st.markdown("### Market Generator")
    runner_left, runner_right = st.columns([3, 2])

    active_run_in_progress = bool(current_run and current_run.get("status") in {"queued", "running"})
    with runner_left:
        fr3_mode_label = st.radio(
            "Event Extraction Scope (FR3)",
            ["Default", "Custom", "Process everything"],
            horizontal=True,
            help="Choose whether FR3 uses the default cluster cap, a custom cap, or no cap at all.",
        )
        fr3_custom_limit = None
        if fr3_mode_label == "Custom":
            fr3_custom_limit = st.number_input(
                "FR3 max clusters",
                min_value=1,
                value=10,
                step=1,
                help="Maximum number of pending clusters to send through FR3 during this run.",
            )

        fr4_mode_label = st.radio(
            "Question Generation Scope (FR4)",
            ["Default", "Custom", "Process everything"],
            horizontal=True,
            help="Choose whether FR4 uses the default cap, a custom cap, or no cap at all.",
        )
        custom_limit = None
        if fr4_mode_label == "Custom":
            custom_limit = st.number_input(
                "FR4 max events",
                min_value=1,
                value=5,
                step=1,
                help="Maximum number of extracted events to send through FR4 during this run.",
            )

        available_models = list(LLMConfig.AVAILABLE_MODELS)
        selected_fr3_model = st.selectbox(
            "FR3 model",
            options=available_models,
            index=_model_index(available_models, LLMConfig.RUNNER_DEFAULT_FR3_MODEL),
            help="Model used for FR3 structured event extraction. The selector defaults to the configured runner model.",
        )
        selected_fr4_model = st.selectbox(
            "FR4 model",
            options=available_models,
            index=_model_index(available_models, LLMConfig.RUNNER_DEFAULT_FR4_MODEL),
            help="Model used for FR4 question generation and repair. The selector defaults to the configured runner model.",
        )

        run_clicked = st.button(
            "Run Pipeline (FR1-FR6)",
            type="primary",
            disabled=active_run_in_progress,
            help="Launch the full pipeline in the background and track progress here.",
        )

        if run_clicked:
            fr3_mode = {
                "Default": "default",
                "Custom": "custom",
                "Process everything": "all",
            }[fr3_mode_label]
            fr4_mode = {
                "Default": "default",
                "Custom": "custom",
                "Process everything": "all",
            }[fr4_mode_label]
            run_id = create_pipeline_run(
                stage_start=1,
                stage_end=6,
                fr3_limit_mode=fr3_mode,
                fr3_limit_value=int(fr3_custom_limit) if fr3_custom_limit is not None else None,
                fr4_limit_mode=fr4_mode,
                fr4_limit_value=int(custom_limit) if custom_limit is not None else None,
                log_mode="normal",
                fr3_model=selected_fr3_model,
                fr4_model=selected_fr4_model,
            )
            command = _build_pipeline_command(
                python_executable=python_executable,
                pipeline_path=pipeline_path,
                fr3_mode=fr3_mode,
                fr3_custom_limit=int(fr3_custom_limit) if fr3_custom_limit is not None else None,
                fr4_mode=fr4_mode,
                custom_limit=int(custom_limit) if custom_limit is not None else None,
                log_mode="normal",
                run_id=run_id,
                fr3_model=selected_fr3_model,
                fr4_model=selected_fr4_model,
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(repo_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                mark_pipeline_run_started(run_id, subprocess_pid=process.pid)
                st.session_state["current_pipeline_run_id"] = run_id
                st.success(f"Started pipeline run #{run_id}. Click Refresh to update progress.")
                st.rerun()
            except Exception as exc:
                mark_pipeline_run_failed(run_id, str(exc))
                st.error(f"Unable to start the pipeline run: {exc}")

    with runner_right:
        if current_run:
            progress_value = _pipeline_progress_value(current_stage_rows)
            st.progress(progress_value, text=f"Run #{current_run['id']} - {str(current_run.get('status', 'queued')).title()}")
            st.caption(
                f"Started: {_format_run_timestamp(current_run.get('started_at'))} | "
                f"Finished: {_format_run_timestamp(current_run.get('finished_at'))}"
            )
            st.caption(
                f"FR3 mode: {current_run.get('fr3_limit_mode', 'default')} | "
                f"FR3 limit: {current_run.get('fr3_limit_value') if current_run.get('fr3_limit_value') is not None else 'default/all'}"
            )
            st.caption(
                f"FR4 mode: {current_run.get('fr4_limit_mode', 'default')} | "
                f"FR4 limit: {current_run.get('fr4_limit_value') if current_run.get('fr4_limit_value') is not None else 'default/all'} | "
                f"Log mode: {current_run.get('log_mode', 'normal')}"
            )
            if current_run.get("error_message"):
                st.error(current_run["error_message"])
            if active_run_in_progress:
                cancel_clicked = st.button(
                    "Cancel Run",
                    use_container_width=True,
                    help="Stop the background pipeline process and mark this run as cancelled.",
                )
                if cancel_clicked:
                    pid = current_run.get("subprocess_pid")
                    try:
                        if pid:
                            _terminate_process_tree(int(pid))
                        cancel_pipeline_run(int(current_run["id"]), "Cancelled by user from dashboard")
                        st.warning(f"Pipeline run #{current_run['id']} was cancelled.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Unable to cancel pipeline run #{current_run['id']}: {exc}")
        else:
            st.info("No pipeline runs yet. Start one here to generate fresh questions directly from the dashboard.")

    if active_run_in_progress:
        st.info("A pipeline run is in progress in the background. Use Refresh to pull the latest stage progress.")

    if current_stage_rows:
        stage_columns = st.columns(len(current_stage_rows))
        for index, stage_row in enumerate(current_stage_rows):
            stage_label = PIPELINE_STAGE_LABELS.get(
                int(stage_row.get("stage_number") or 0),
                stage_row.get("stage_name") or "Stage",
            )
            status = str(stage_row.get("status") or "pending").title()
            with stage_columns[index]:
                st.markdown(f"**{stage_label}**")
                st.caption(status)
                stage_progress = _stage_progress_value(stage_row)
                if stage_progress is not None:
                    st.progress(stage_progress)
                info_lines = _stage_info_lines(stage_row, current_run)
                if info_lines:
                    with st.popover("i", help="Stage details"):
                        for line in info_lines:
                            st.caption(line)

    st.markdown("---")

    if all_rows:
        categories = set((row.get("category") or "other").lower() for row in all_rows)
        st.markdown(
            f"""
<div class="stat-grid">
  <div class="stat-card"><div class="stat-value">{len(active_rows)}</div><div class="stat-label">Active Queue</div></div>
  <div class="stat-card"><div class="stat-value">{len(selected_rows)}</div><div class="stat-label">Selected</div></div>
  <div class="stat-card"><div class="stat-value">{len(removed_rows)}</div><div class="stat-label">Removed</div></div>
  <div class="stat-card"><div class="stat-value">{len(expired_rows)}</div><div class="stat-label">Expired</div></div>
  <div class="stat-card"><div class="stat-value" style="color:#F97316;">{len(categories)}</div><div class="stat-label">Categories</div></div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown(
            '<div style="font-size:20px;font-weight:800;color:#F1F5F9;letter-spacing:-0.4px;">🔧 Filters</div>'
            '<div style="font-size:13px;color:#64748B;margin-top:4px;margin-bottom:20px;">Refine the visible market list</div>',
            unsafe_allow_html=True,
        )

        search = st.text_input("Search", placeholder="e.g. Fed, Bitcoin, election", label_visibility="collapsed")
        st.caption("Search question text")

        st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.9px;color:#475569;margin:22px 0 8px 0;">Category</div>', unsafe_allow_html=True)
        all_cats = sorted(set((row.get("category") or "other").lower() for row in all_rows))
        selected_cats = st.multiselect(
            "category",
            options=all_cats,
            default=all_cats,
            format_func=lambda value: value.replace("_", " ").title(),
            label_visibility="collapsed",
        )

        st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.9px;color:#475569;margin:22px 0 8px 0;">Question Type</div>', unsafe_allow_html=True)
        type_choice = st.radio("type", ["All", "Binary", "Multiple Choice"], horizontal=True, label_visibility="collapsed")

        st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.9px;color:#475569;margin:22px 0 8px 0;">Minimum Score</div>', unsafe_allow_html=True)
        min_score_pct = st.slider(
            "min_score",
            0,
            100,
            0,
            step=5,
            label_visibility="collapsed",
            help="Hide markets with a total score below this threshold",
        )

        st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.9px;color:#475569;margin:22px 0 8px 0;">Sort By</div>', unsafe_allow_html=True)
        sort_choice = st.selectbox("sort", list(SORT_OPTIONS.keys()), label_visibility="collapsed")

        st.markdown(
            '<div style="margin-top:32px;padding-top:20px;border-top:1px solid #1E2535;">'
            '<div style="font-size:11px;color:#334155;font-weight:500;line-height:1.6;">'
            'Scores are weighted composites of market interest, resolution quality, '
            'clarity, velocity, novelty, time horizon, and source diversity.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    if not all_rows:
        st.markdown(
            """
<div class="empty-state">
  <div class="empty-title">No markets scored yet</div>
  <div class="empty-sub">Use the Pipeline Runner above to launch FR1-FR6, then refresh here to review the scored questions.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    active_tab, repair_tab, selected_tab, removed_tab, expired_tab = st.tabs(
        ["Active", "Needs Repair", "Selected", "Removed", "Expired"]
    )
    with active_tab:
        _render_view("Active", active_rows, search, selected_cats, type_choice, min_score_pct, sort_choice)
    with repair_tab:
        _render_repair_view(repair_rows, search, selected_cats, type_choice, sort_choice)
    with selected_tab:
        _render_view("Selected", selected_rows, search, selected_cats, type_choice, min_score_pct, sort_choice)
    with removed_tab:
        _render_view("Removed", removed_rows, search, selected_cats, type_choice, min_score_pct, sort_choice)
    with expired_tab:
        _render_view("Expired", expired_rows, search, selected_cats, type_choice, min_score_pct, sort_choice)


if __name__ == "__main__":
    main()
