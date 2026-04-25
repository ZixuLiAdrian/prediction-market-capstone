"""
Reusable LLM client wrapper.

Supports Groq and Gemini providers. Handles retry logic, timeout, and
structured JSON output parsing. FR4 (question generation) should import
and reuse this same class with a different prompt.

Usage:
    from extraction.llm_client import LLMClient
    client = LLMClient()
    result = client.call(system_prompt="...", user_prompt="...", response_schema={...})
"""

import json
import logging
import time
from typing import Optional

import jsonschema

from config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM API client with retry and schema validation."""

    _next_allowed_request_time_by_provider: dict[str, float] = {}

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        max_retries: int = None,
        rate_limit_max_retries: int = None,
        timeout: int = None,
    ):
        self.provider = provider or LLMConfig.PROVIDER
        self.model = model or LLMConfig.MODEL
        self.api_key = api_key or LLMConfig.API_KEY
        self.max_retries = max_retries if max_retries is not None else LLMConfig.MAX_RETRIES
        self.rate_limit_max_retries = (
            rate_limit_max_retries
            if rate_limit_max_retries is not None
            else LLMConfig.RATE_LIMIT_MAX_RETRIES
        )
        self.timeout = timeout or LLMConfig.TIMEOUT
        self.min_request_interval_seconds = LLMConfig.MIN_REQUEST_INTERVAL_SECONDS
        self.rate_limit_backoff_base_seconds = LLMConfig.RATE_LIMIT_BACKOFF_BASE_SECONDS
        self.rate_limit_backoff_max_seconds = LLMConfig.RATE_LIMIT_BACKOFF_MAX_SECONDS
        self._client = None

    def _wait_for_rate_limit_slot(self):
        """Throttle requests with a small provider-wide gap to avoid bursty 429s."""
        next_allowed = self._next_allowed_request_time_by_provider.get(self.provider, 0.0)
        now = time.monotonic()
        if now < next_allowed:
            sleep_for = next_allowed - now
            logger.debug(
                f"Rate limit pacing: sleeping {sleep_for:.2f}s before next {self.provider} request"
            )
            time.sleep(sleep_for)

        self._next_allowed_request_time_by_provider[self.provider] = (
            time.monotonic() + self.min_request_interval_seconds
        )

    def _register_backoff(self, delay_seconds: float):
        """Push the next allowed request time forward after a rate limit response."""
        target_time = time.monotonic() + max(0.0, delay_seconds)
        existing = self._next_allowed_request_time_by_provider.get(self.provider, 0.0)
        self._next_allowed_request_time_by_provider[self.provider] = max(existing, target_time)

    @staticmethod
    def _get_status_code(error: Exception) -> Optional[int]:
        """Best-effort extraction of HTTP status code from provider exceptions."""
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            return int(status_code)

        response = getattr(error, "response", None)
        if response is not None:
            resp_status = getattr(response, "status_code", None)
            if resp_status is not None:
                return int(resp_status)

        return None

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Identify rate limit responses from Groq/Gemini SDK wrappers."""
        status_code = self._get_status_code(error)
        if status_code == 429:
            return True

        lowered = str(error).lower()
        return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered

    @staticmethod
    def _get_retry_after_seconds(error: Exception) -> Optional[float]:
        """Read Retry-After from provider response headers when available."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        retry_after = headers.get("retry-after")
        if retry_after is None:
            return None

        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return None

    def _get_client(self):
        """Lazy-initialize the provider-specific client."""
        if self._client is not None:
            return self._client

        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return self._client

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq API and return raw response text."""
        self._wait_for_rate_limit_slot()
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini API and return raw response text."""
        self._wait_for_rate_limit_slot()
        client = self._get_client()
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = client.generate_content(full_prompt)
        return response.text

    def _parse_json(self, raw_response: str) -> dict:
        """Extract and parse JSON from LLM response, handling markdown code blocks."""
        json_str = raw_response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[1]  # remove ```json line
            json_str = json_str.rsplit("```", 1)[0]  # remove closing ```
        return json.loads(json_str)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[dict] = None,
    ) -> dict:
        """
        Call the LLM and return parsed JSON response.

        On schema validation failure, sends the error back to the LLM as a
        repair prompt instead of blindly retrying the same request.

        Args:
            system_prompt: System-level instructions for the LLM.
            user_prompt: The user-facing prompt with actual content.
            response_schema: Optional JSON schema to validate the response against.

        Returns:
            Parsed dict from the LLM's JSON response.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        call_fn = self._call_groq if self.provider == "groq" else self._call_gemini
        last_error = None
        current_user_prompt = user_prompt
        attempt = 0
        schema_failures = 0
        api_failures = 0
        rate_limit_failures = 0

        while True:
            attempt += 1
            try:
                raw_response = call_fn(system_prompt, current_user_prompt)
                parsed = self._parse_json(raw_response)

                # Validate against schema if provided
                if response_schema:
                    jsonschema.validate(instance=parsed, schema=response_schema)

                logger.debug(f"LLM call succeeded on attempt {attempt}")
                return parsed

            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                last_error = e
                schema_failures += 1
                logger.warning(
                    f"LLM response validation failed "
                    f"(attempt {attempt}, schema retry {schema_failures}/{self.max_retries + 1}): {e}"
                )

                if schema_failures > self.max_retries:
                    break

                # Smart retry: send the error back to the LLM for repair
                error_msg = str(e)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."

                current_user_prompt = (
                    f"Your previous response had a validation error:\n{error_msg}\n\n"
                    f"Please fix the JSON and respond again. Original request:\n\n{user_prompt}"
                )
                time.sleep(1)

            except Exception as e:
                last_error = e
                current_user_prompt = user_prompt  # reset to original on API errors

                if self._is_rate_limit_error(e):
                    rate_limit_failures += 1
                    budget_label = (
                        "unbounded"
                        if self.rate_limit_max_retries < 0
                        else str(self.rate_limit_max_retries + 1)
                    )
                    logger.warning(
                        f"LLM API call hit rate limit "
                        f"(attempt {attempt}, rate-limit retry {rate_limit_failures}/{budget_label}): {e}"
                    )
                    if (
                        self.rate_limit_max_retries >= 0
                        and rate_limit_failures > self.rate_limit_max_retries
                    ):
                        break

                    retry_after = self._get_retry_after_seconds(e)
                    delay = retry_after
                    if delay is None:
                        delay = min(
                            self.rate_limit_backoff_max_seconds,
                            self.rate_limit_backoff_base_seconds * (2 ** min(rate_limit_failures - 1, 8)),
                        )
                    self._register_backoff(delay)
                    logger.warning(
                        f"Rate limit encountered for {self.provider}/{self.model}; "
                        f"backing off for {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    api_failures += 1
                    logger.warning(
                        f"LLM API call failed "
                        f"(attempt {attempt}, api retry {api_failures}/{self.max_retries + 1}): {e}"
                    )
                    if api_failures > self.max_retries:
                        break
                    time.sleep(2)

        raise RuntimeError(
            f"LLM call failed after {attempt} attempts. Last error: {last_error}"
        )
