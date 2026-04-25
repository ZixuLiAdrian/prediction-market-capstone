"""Tests for shared LLM client pacing and rate-limit handling."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extraction.llm_client import LLMClient


def test_wait_for_rate_limit_slot_sleeps_when_called_too_soon(monkeypatch):
    """Provider-wide pacing should sleep when another request just ran."""
    client = LLMClient(provider="groq", model="test-model", api_key="test-key")
    client._next_allowed_request_time_by_provider = {"groq": 10.0}

    sleeps = []
    monkeypatch.setattr("time.monotonic", lambda: 9.0)
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    client._wait_for_rate_limit_slot()

    assert sleeps == [1.0]
    assert client._next_allowed_request_time_by_provider["groq"] == 11.1


def test_call_retries_after_rate_limit_with_backoff(monkeypatch):
    """A 429 should trigger backoff and then allow a successful retry."""
    now = {"value": 0.0}
    sleeps = []
    attempts = {"count": 0}

    class RateLimitError(Exception):
        status_code = 429

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    def fake_call(system_prompt, user_prompt):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("429 Too Many Requests")
        return '{"ok": true}'

    client = LLMClient(
        provider="groq",
        model="test-model",
        api_key="test-key",
        max_retries=1,
    )
    client._call_groq = fake_call
    monkeypatch.setattr("time.monotonic", lambda: now["value"])
    monkeypatch.setattr("time.sleep", fake_sleep)

    result = client.call(
        system_prompt="system",
        user_prompt="user",
        response_schema=None,
    )

    assert result == {"ok": True}
    assert attempts["count"] == 2
    assert sleeps == [5.0]


def test_call_can_retry_rate_limits_beyond_normal_retry_budget(monkeypatch):
    """Unlimited rate-limit retries should keep going even when max_retries is zero."""
    now = {"value": 0.0}
    sleeps = []
    attempts = {"count": 0}

    class RateLimitError(Exception):
        status_code = 429

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    def fake_call(system_prompt, user_prompt):
        attempts["count"] += 1
        if attempts["count"] < 4:
            raise RateLimitError("429 Too Many Requests")
        return '{"ok": true}'

    client = LLMClient(
        provider="groq",
        model="test-model",
        api_key="test-key",
        max_retries=0,
        rate_limit_max_retries=-1,
    )
    client._call_groq = fake_call
    monkeypatch.setattr("time.monotonic", lambda: now["value"])
    monkeypatch.setattr("time.sleep", fake_sleep)

    result = client.call(
        system_prompt="system",
        user_prompt="user",
        response_schema=None,
    )

    assert result == {"ok": True}
    assert attempts["count"] == 4
    assert sleeps == [5.0, 10.0, 20.0]
