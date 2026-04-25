"""Focused tests for pipeline stage wiring."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pipeline
from models import CandidateQuestion, ExtractedEvent, ValidationResult


def test_resolve_log_mode_defaults_to_normal():
    assert pipeline._resolve_log_mode() == "normal"
    assert pipeline._resolve_log_mode("normal") == "normal"


def test_resolve_log_mode_debug_flag_overrides_explicit_mode():
    assert pipeline._resolve_log_mode("normal", debug=True) == "debug"
    assert pipeline._resolve_log_mode("debug") == "debug"


def test_normal_console_filter_allows_pipeline_info_and_all_warnings():
    filter_ = pipeline._NormalConsoleFilter("pipeline")

    pipeline_record = pipeline.logging.LogRecord(
        name="pipeline",
        level=pipeline.logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="pipeline summary",
        args=(),
        exc_info=None,
    )
    warning_record = pipeline.logging.LogRecord(
        name="httpx",
        level=pipeline.logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="external warning",
        args=(),
        exc_info=None,
    )
    external_info = pipeline.logging.LogRecord(
        name="httpx",
        level=pipeline.logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="noisy info",
        args=(),
        exc_info=None,
    )

    assert filter_.filter(pipeline_record) is True
    assert filter_.filter(warning_record) is True
    assert filter_.filter(external_info) is False


def test_run_question_generation_passes_limit_and_model(monkeypatch):
    """FR4 should fetch with the configured limit and pass the selected model through."""
    captured = {"limit": None, "model": None}

    sample_events = [
        ExtractedEvent(id=1, cluster_id=1, event_summary="Fed rate cut story", entities=["Fed"], event_type="macro_release"),
        ExtractedEvent(id=2, cluster_id=2, event_summary="Bitcoin ETF story", entities=["Bitcoin"], event_type="crypto"),
    ]

    def fake_get_extracted_events_for_generation(limit=None):
        captured["limit"] = limit
        return sample_events

    class DummyQuestionGenerator:
        def __init__(self, model=None):
            captured["model"] = model

        def generate(self, event):
            assert event in sample_events
            return []

    monkeypatch.setattr(pipeline, "get_extracted_events_for_generation", fake_get_extracted_events_for_generation)
    monkeypatch.setattr(pipeline, "QuestionGenerator", DummyQuestionGenerator)

    pipeline.run_question_generation(max_events=3, model="custom-fr4-model")

    assert captured["limit"] == 3
    assert captured["model"] == "custom-fr4-model"


def test_run_extraction_passes_limit_and_model(monkeypatch):
    """FR3 should fetch with the configured limit and pass the selected model through."""
    captured = {"limit": None, "model": None}
    sample_clusters = [{"cluster_id": 1, "label": 0, "features": None, "events": []}]

    def fake_get_clusters_for_extraction(limit=None):
        captured["limit"] = limit
        return sample_clusters

    class DummyEventExtractor:
        def __init__(self, model=None):
            captured["model"] = model

        def extract(self, cluster, cluster_id):
            assert cluster_id == 1
            return None

    monkeypatch.setattr(pipeline, "get_clusters_for_extraction", fake_get_clusters_for_extraction)
    monkeypatch.setattr(pipeline, "EventExtractor", DummyEventExtractor)

    pipeline.run_extraction(max_clusters=4, model="custom-fr3-model")

    assert captured["limit"] == 4
    assert captured["model"] == "custom-fr3-model"


def test_run_extraction_passes_model(monkeypatch):
    """FR3 should pass the selected model into EventExtractor."""
    captured = {"model": None}
    sample_clusters = [{"cluster_id": 1, "label": 0, "features": None, "events": []}]

    def fake_get_clusters_for_extraction(limit=None):
        assert limit is None
        return sample_clusters

    class DummyEventExtractor:
        def __init__(self, model=None):
            captured["model"] = model

        def extract(self, cluster, cluster_id):
            assert cluster_id == 1
            return None

    monkeypatch.setattr(pipeline, "get_clusters_for_extraction", fake_get_clusters_for_extraction)
    monkeypatch.setattr(pipeline, "EventExtractor", DummyEventExtractor)

    pipeline.run_extraction(model="custom-fr3-model")

    assert captured["model"] == "custom-fr3-model"


def test_run_extraction_reports_incremental_progress(monkeypatch):
    sample_clusters = [
        {"cluster_id": 1, "label": 0, "features": None, "events": []},
        {"cluster_id": 2, "label": 1, "features": None, "events": []},
    ]
    progress_updates = []

    def fake_get_clusters_for_extraction(limit=None):
        return sample_clusters

    class DummyEventExtractor:
        def __init__(self, model=None):
            pass

        def extract(self, cluster, cluster_id):
            return ExtractedEvent(cluster_id=cluster_id, event_summary="summary", entities=["A"])

    inserted = []
    monkeypatch.setattr(pipeline, "get_clusters_for_extraction", fake_get_clusters_for_extraction)
    monkeypatch.setattr(pipeline, "EventExtractor", DummyEventExtractor)
    monkeypatch.setattr(pipeline, "insert_extracted_event", lambda extracted: inserted.append(extracted.cluster_id) or len(inserted))

    summary = pipeline.run_extraction(progress_reporter=lambda summary: progress_updates.append(dict(summary)))

    assert inserted == [1, 2]
    assert progress_updates == [
        {
            "pending_clusters": 2,
            "clusters_processed": 1,
            "progress_pct": 50.0,
            "extracted_events": 1,
            "saved_events": 1,
        },
        {
            "pending_clusters": 2,
            "clusters_processed": 2,
            "progress_pct": 100.0,
            "extracted_events": 2,
            "saved_events": 2,
        },
    ]
    assert summary["progress_pct"] == 100.0


def test_run_question_generation_reports_incremental_progress(monkeypatch):
    sample_events = [
        ExtractedEvent(id=1, cluster_id=1, event_summary="Fed rate cut story", entities=["Fed"], event_type="macro_release"),
        ExtractedEvent(id=2, cluster_id=2, event_summary="Bitcoin ETF story", entities=["Bitcoin"], event_type="crypto"),
    ]
    progress_updates = []

    def fake_get_extracted_events_for_generation(limit=None):
        return sample_events

    class DummyQuestionGenerator:
        def __init__(self, model=None):
            pass

        def generate(self, event):
            return [CandidateQuestion(
                extracted_event_id=event.id,
                question_text=f"Question for {event.event_summary}?",
                category="finance",
                question_type="binary",
                options=["Yes", "No"],
                deadline="January 1, 2027",
                deadline_source="Official calendar at https://example.com/calendar",
                resolution_source="Official source at https://example.com/source",
                resolution_criteria="Resolves YES if event happens. Resolves NO otherwise.",
                rationale="Test rationale",
            )]

    inserted = []
    monkeypatch.setattr(pipeline, "get_extracted_events_for_generation", fake_get_extracted_events_for_generation)
    monkeypatch.setattr(pipeline, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(pipeline, "insert_candidate_question", lambda question: inserted.append(question.question_text) or len(inserted))

    summary = pipeline.run_question_generation(progress_reporter=lambda summary: progress_updates.append(dict(summary)))

    assert len(inserted) == 2
    assert progress_updates == [
        {
            "eligible_events": 2,
            "raw_eligible_events": 2,
            "events_processed": 1,
            "progress_pct": 50.0,
            "questions_generated": 1,
        },
        {
            "eligible_events": 2,
            "raw_eligible_events": 2,
            "events_processed": 2,
            "progress_pct": 100.0,
            "questions_generated": 2,
        },
    ]
    assert summary["progress_pct"] == 100.0


def test_run_question_generation_dedupes_story_events_and_caps_questions(monkeypatch):
    ceasefire_a = ExtractedEvent(
        id=10,
        cluster_id=1,
        event_summary="A 10-day ceasefire between Israel and Lebanon has gone into effect, with Hezbollah's adherence uncertain.",
        entities=["Israel", "Lebanon", "Hezbollah"],
        event_type="geopolitics",
    )
    ceasefire_b = ExtractedEvent(
        id=11,
        cluster_id=2,
        event_summary="A 10-day ceasefire between Israel and Lebanon has begun, and Hezbollah has not yet confirmed it will comply.",
        entities=["Israel", "Lebanon", "Hezbollah"],
        event_type="geopolitics",
    )
    finance = ExtractedEvent(
        id=12,
        cluster_id=3,
        event_summary="Archimedes Tech SPAC Partners II announced a business combination.",
        entities=["Archimedes Tech SPAC Partners II"],
        event_type="business",
    )

    monkeypatch.setattr(
        pipeline,
        "get_extracted_events_for_generation",
        lambda limit=None: [ceasefire_a, ceasefire_b, finance],
    )
    monkeypatch.setattr(pipeline, "get_all_candidate_question_texts", lambda: [])

    class DummyQuestionGenerator:
        def __init__(self, model=None):
            pass

        def generate(self, event):
            if event.id == 12:
                return [
                    CandidateQuestion(
                        extracted_event_id=12,
                        question_text="Will Archimedes Tech SPAC Partners II complete its business combination by June 30, 2026?",
                        category="finance",
                        question_type="binary",
                        options=["Yes", "No"],
                        deadline="June 30, 2026",
                        deadline_source="Official SEC timeline at https://example.com/sec",
                        resolution_source="SEC filing at https://example.com/sec-filing",
                        resolution_criteria="Resolves YES if the company announces completion. Resolves NO otherwise.",
                        rationale="Distinct finance question.",
                    )
                ]

            return [
                CandidateQuestion(
                    extracted_event_id=event.id,
                    question_text="Will the ceasefire between Israel and Lebanon be extended beyond its initial term?",
                    category="geopolitics",
                    question_type="binary",
                    options=["Yes", "No"],
                    deadline="April 30, 2026",
                    deadline_source="Official statement at https://example.com/ceasefire",
                    resolution_source="Official source at https://example.com/source",
                    resolution_criteria="Resolves YES if the ceasefire is formally extended. Resolves NO otherwise.",
                    rationale="Ceasefire extension question.",
                ),
                CandidateQuestion(
                    extracted_event_id=event.id,
                    question_text="Will Hezbollah adhere to the ceasefire for the full 10-day duration?",
                    category="geopolitics",
                    question_type="binary",
                    options=["Yes", "No"],
                    deadline="April 30, 2026",
                    deadline_source="Official statement at https://example.com/ceasefire",
                    resolution_source="Official source at https://example.com/source",
                    resolution_criteria="Resolves YES if Hezbollah fully complies. Resolves NO otherwise.",
                    rationale="Ceasefire adherence question.",
                ),
                CandidateQuestion(
                    extracted_event_id=event.id,
                    question_text="What will be the outcome of the current ceasefire between Israel and Lebanon by the end of the 10-day period?",
                    category="geopolitics",
                    question_type="multiple_choice",
                    options=["Extended", "Ends without extension", "Broken early"],
                    deadline="April 30, 2026",
                    deadline_source="Official statement at https://example.com/ceasefire",
                    resolution_source="Official source at https://example.com/source",
                    resolution_criteria="Resolves to the outcome that occurs by the deadline.",
                    rationale="Outcome question.",
                ),
            ]

    inserted = []
    monkeypatch.setattr(pipeline, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(pipeline, "insert_candidate_question", lambda question: inserted.append(question.question_text) or len(inserted))

    summary = pipeline.run_question_generation(max_events=3)

    assert summary["raw_eligible_events"] == 3
    assert summary["eligible_events"] == 2
    assert summary["questions_generated"] == 3
    assert len(inserted) == 3
    assert inserted.count("Will the ceasefire between Israel and Lebanon be extended beyond its initial term?") == 1


def test_run_pipeline_updates_db_backed_stage_tracking(monkeypatch):
    """A tracked run should mark the run/stages started and completed in order."""
    calls = {
        "started": [],
        "completed": [],
        "failed": [],
        "stages": [],
    }

    monkeypatch.setattr(pipeline, "_configure_logging", lambda mode: None)
    monkeypatch.setattr(pipeline, "init_db", lambda: None)

    def stage_one():
        return {"saved": 3}

    def stage_two():
        return {"saved": 1}

    monkeypatch.setattr(pipeline, "STAGES", [("Stage One", stage_one), ("Stage Two", stage_two)])
    monkeypatch.setattr(pipeline, "mark_pipeline_run_started", lambda run_id: calls["started"].append(run_id))
    monkeypatch.setattr(pipeline, "mark_pipeline_run_completed", lambda run_id: calls["completed"].append(run_id))
    monkeypatch.setattr(
        pipeline,
        "mark_pipeline_run_failed",
        lambda run_id, error_message: calls["failed"].append((run_id, error_message)),
    )
    monkeypatch.setattr(
        pipeline,
        "update_pipeline_run_stage",
        lambda run_id, stage_number, stage_name, status, summary=None, error_message="": calls["stages"].append(
            (run_id, stage_number, stage_name, status, summary, error_message)
        ),
    )

    pipeline.run_pipeline(start=1, end=2, run_id=77, log_mode="normal")

    assert calls["started"] == [77]
    assert calls["completed"] == [77]
    assert calls["failed"] == []
    assert calls["stages"] == [
        (77, 1, "Stage One", "running", None, ""),
        (77, 1, "Stage One", "completed", {"saved": 3}, ""),
        (77, 2, "Stage Two", "running", None, ""),
        (77, 2, "Stage Two", "completed", {"saved": 1}, ""),
    ]


def test_run_validation_repairs_salvageable_failures(monkeypatch):
    question = CandidateQuestion(
        id=10,
        extracted_event_id=5,
        question_text="Will the event happen by January 1, 2020?",
        category="finance",
        question_type="binary",
        options=["Yes", "No"],
        deadline="January 1, 2020",
        deadline_source="Official calendar at https://example.com/calendar",
        resolution_source="Official source at https://example.com/source",
        resolution_criteria="Resolves YES if event happens. Resolves NO otherwise.",
        rationale="Test rationale",
    )
    repaired = CandidateQuestion(
        extracted_event_id=5,
        repair_parent_question_id=10,
        question_text="Will the event happen by January 1, 2027?",
        category="finance",
        question_type="binary",
        options=["Yes", "No"],
        deadline="January 1, 2027",
        deadline_source="Official calendar at https://example.com/calendar",
        resolution_source="Official source at https://example.com/source",
        resolution_criteria="Resolves YES if event happens. Resolves NO otherwise.",
        rationale="Repaired rationale",
    )

    validation_results = [
        ValidationResult(question_id=10, is_valid=False, flags=["invalid_deadline_window"], clarity_score=0.8),
        ValidationResult(question_id=11, is_valid=True, flags=[], clarity_score=1.0),
    ]
    inserted_ids = []

    class DummyRepairGenerator:
        def repair_question(self, extracted_event, failed_question, validation_flags):
            assert extracted_event.id == 5
            assert failed_question.id == 10
            assert validation_flags == ["invalid_deadline_window"]
            return repaired

    monkeypatch.setattr(pipeline, "get_candidate_questions_for_validation", lambda: [question])
    monkeypatch.setattr(pipeline, "QuestionGenerator", lambda: DummyRepairGenerator())
    monkeypatch.setattr(
        pipeline,
        "validate_question",
        lambda q: validation_results.pop(0),
    )
    monkeypatch.setattr(pipeline, "insert_validation_result", lambda result: 1)
    monkeypatch.setattr(pipeline, "question_has_repair_child", lambda question_id: False)
    monkeypatch.setattr(
        pipeline,
        "get_extracted_event_by_id",
        lambda extracted_event_id: ExtractedEvent(id=5, cluster_id=1, event_summary="summary", entities=[]),
    )
    monkeypatch.setattr(pipeline, "insert_candidate_question", lambda q: inserted_ids.append(q) or 11)

    summary = pipeline.run_validation()

    assert summary["questions_checked"] == 1
    assert summary["repaired"] == 1
    assert summary["repair_salvaged"] == 1
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert inserted_ids[0].repair_parent_question_id == 10
