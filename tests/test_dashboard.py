"""Tests for dashboard lifecycle and ranking helpers."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streamlit_app import (
    _assign_display_ranks,
    _build_pipeline_command,
    _effective_status,
    _filter_rows,
    _pipeline_progress_value,
    _stage_progress_value,
    _stage_info_lines,
    _terminate_process_tree,
)


def test_effective_status_prefers_manual_review_state():
    assert _effective_status({"review_status": "selected", "deadline": "2000-01-01"}) == "selected"
    assert _effective_status({"review_status": "removed", "deadline": "2999-01-01"}) == "removed"


def test_effective_status_marks_expired_when_deadline_has_passed():
    assert _effective_status({"review_status": None, "deadline": "2000-01-01"}) == "expired"
    assert _effective_status({"review_status": None, "deadline": "2999-01-01"}) == "active"


def test_assign_display_ranks_recomputes_sequential_ranks_from_score_then_id():
    rows = [
        {"question_id": 20, "total_score": 0.70},
        {"question_id": 11, "total_score": 0.95},
        {"question_id": 10, "total_score": 0.95},
    ]

    ranked = _assign_display_ranks(rows)
    assert [row["question_id"] for row in ranked] == [10, 11, 20]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_filter_rows_applies_search_category_type_and_score():
    rows = [
        {
            "rank": 1,
            "question_id": 10,
            "question_text": "Will Bitcoin close above $100,000 by January 1, 2999?",
            "category": "finance",
            "question_type": "binary",
            "total_score": 0.92,
            "deadline": "2999-01-01",
        },
        {
            "rank": 2,
            "question_id": 11,
            "question_text": "Which party will win the election?",
            "category": "politics",
            "question_type": "multiple_choice",
            "total_score": 0.55,
            "deadline": "2999-01-01",
        },
    ]

    filtered = _filter_rows(
        rows=rows,
        search="bitcoin",
        selected_cats=["finance"],
        type_choice="Binary",
        min_score_pct=60,
        sort_choice="Score (Highest First)",
    )

    assert [row["question_id"] for row in filtered] == [10]


def test_build_pipeline_command_supports_default_custom_and_all_modes():
    default_cmd = _build_pipeline_command(
        python_executable="python",
        pipeline_path="pipeline.py",
        fr3_mode="default",
        fr3_custom_limit=None,
        fr4_mode="default",
        custom_limit=None,
        log_mode="normal",
        run_id=5,
        fr3_model="qwen/qwen3-32b",
        fr4_model="groq/compound-mini",
    )
    custom_cmd = _build_pipeline_command(
        python_executable="python",
        pipeline_path="pipeline.py",
        fr3_mode="custom",
        fr3_custom_limit=4,
        fr4_mode="custom",
        custom_limit=7,
        log_mode="debug",
        run_id=6,
        fr3_model="llama-3.1-8b-instant",
        fr4_model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    all_cmd = _build_pipeline_command(
        python_executable="python",
        pipeline_path="pipeline.py",
        fr3_mode="all",
        fr3_custom_limit=None,
        fr4_mode="all",
        custom_limit=None,
        log_mode="normal",
        run_id=7,
        fr3_model="qwen/qwen3-32b",
        fr4_model="groq/compound-mini",
    )

    assert default_cmd == [
        "python",
        "pipeline.py",
        "--run-id",
        "5",
        "--log-mode",
        "normal",
        "--fr3-model",
        "qwen/qwen3-32b",
        "--fr4-model",
        "groq/compound-mini",
    ]
    assert custom_cmd == [
        "python",
        "pipeline.py",
        "--run-id",
        "6",
        "--log-mode",
        "debug",
        "--fr3-model",
        "llama-3.1-8b-instant",
        "--fr4-model",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "--fr3-limit",
        "4",
        "--fr4-limit",
        "7",
    ]
    assert all_cmd == [
        "python",
        "pipeline.py",
        "--run-id",
        "7",
        "--log-mode",
        "normal",
        "--fr3-model",
        "qwen/qwen3-32b",
        "--fr4-model",
        "groq/compound-mini",
        "--fr3-all",
        "--fr4-all",
    ]


def test_pipeline_progress_value_uses_completed_stage_fraction():
    rows = [
        {"status": "completed"},
        {"status": "running"},
        {"status": "completed"},
        {"status": "pending"},
    ]

    assert _pipeline_progress_value(rows) == 0.5
    assert _pipeline_progress_value([]) == 0.0


def test_stage_progress_value_uses_processed_totals_for_fr3_and_fr4():
    fr3 = {"summary": {"pending_clusters": 8, "clusters_processed": 2}}
    fr4 = {"summary": {"eligible_events": 5, "events_processed": 3}}

    assert _stage_progress_value(fr3) == 0.25
    assert _stage_progress_value(fr4) == 0.6
    assert _stage_progress_value({"summary": {"passed": 2}}) is None


def test_stage_info_lines_include_summary_models_and_errors():
    fr3_lines = _stage_info_lines(
        {"stage_number": 3, "summary": {"pending_clusters": 5}, "error_message": ""},
        {"fr3_model": "model-fr3", "fr4_model": "model-fr4"},
    )
    fr4_lines = _stage_info_lines(
        {"stage_number": 4, "summary": {"questions_generated": 8}, "error_message": "retrying"},
        {"fr3_model": "model-fr3", "fr4_model": "model-fr4"},
    )

    assert fr3_lines == ["pending_clusters=5", "LLM model: model-fr3"]
    assert fr4_lines == [
        "questions_generated=8",
        "LLM model: model-fr4",
        "Error: retrying",
    ]


def test_terminate_process_tree_uses_taskkill_on_windows(monkeypatch):
    calls = []

    monkeypatch.setattr("streamlit_app.os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        "streamlit_app.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    _terminate_process_tree(321)

    assert calls == [
        (
            ["taskkill", "/PID", "321", "/T", "/F"],
            {"check": True, "stdout": __import__("subprocess").DEVNULL, "stderr": __import__("subprocess").DEVNULL},
        )
    ]
