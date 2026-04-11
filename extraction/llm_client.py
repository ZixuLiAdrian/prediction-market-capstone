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

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        max_retries: int = None,
        timeout: int = None,
    ):
        self.provider = provider or LLMConfig.PROVIDER
        self.model = model or LLMConfig.MODEL
        self.api_key = api_key or LLMConfig.API_KEY
        self.max_retries = max_retries if max_retries is not None else LLMConfig.MAX_RETRIES
        self.timeout = timeout or LLMConfig.TIMEOUT
        self._client = None

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

        for attempt in range(1, self.max_retries + 2):  # +1 for initial attempt
            try:
                raw_response = call_fn(system_prompt, current_user_prompt)
                parsed = self._parse_json(raw_response)

                # Validate against schema if provided
                if response_schema:
                    jsonschema.validate(instance=parsed, schema=response_schema)

                logger.info(f"LLM call succeeded on attempt {attempt}")
                return parsed

            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                last_error = e
                logger.warning(f"LLM response validation failed (attempt {attempt}/{self.max_retries + 1}): {e}")

                if attempt <= self.max_retries:
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
                logger.warning(f"LLM API call failed (attempt {attempt}/{self.max_retries + 1}): {e}")
                if attempt <= self.max_retries:
                    current_user_prompt = user_prompt  # reset to original on API errors
                    time.sleep(2)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        )
