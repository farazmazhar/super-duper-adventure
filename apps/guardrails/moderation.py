"""App 5 — moderation pre-check (the one LLM call in the guardrail path).

Calls `meta-llama/llama-guard-4-12b` via OpenRouter on the **user prompt** at
chat entry. Catches semantic unsafe intent that heuristics miss (G6).

Togglable via env `MODERATION_ENABLED` (default on). **Fail-open**: if the
provider is unavailable (rate limit / outage / no key), log a clear warning and
return a `pass` result — the deterministic checks still run, and the chat stays
usable. This is the documented exception to G1 (deterministic guardrails).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from apps.common.config import settings
from apps.guardrails.models import GuardrailResult

logger = logging.getLogger(__name__)

MODERATION_SYSTEM_PROMPT = (
    "You are a content moderation classifier. Classify the user's message as "
    "either 'safe' or 'unsafe'. Reply with exactly one word."
)


class ModerationClient:
    """Thin wrapper around the OpenAI SDK (OpenAI-compatible endpoint).

    `client_factory` is injectable for tests (a fake OpenAI client).
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or settings.moderation_model
        self._client = None
        self._client_factory = client_factory

    @property
    def client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key or "missing-key",
                    base_url=self.base_url,
                )
        return self._client

    def close(self) -> None:
        self._client = None

    def is_unsafe(self, text: str) -> bool:
        """Ask llama-guard whether the text is unsafe. Raises on provider failure."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MODERATION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        label = (resp.choices[0].message.content or "").strip().lower()
        return "unsafe" in label


def check_moderation(
    text: str,
    client: ModerationClient | None = None,
    enabled: bool | None = None,
) -> GuardrailResult:
    """Moderation pre-check. Togglable; fail-open on provider failure.

    `client` injectable for tests; `enabled` overrides settings for tests.
    """
    if enabled is None:
        enabled = settings.moderation_enabled
    if not enabled:
        return GuardrailResult(
            passed=True, severity="pass", rule="moderation",
            message="Moderation pre-check disabled (MODERATION_ENABLED=false).",
            detail={"enabled": False},
        )

    own_client = client is None
    client = client or ModerationClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    try:
        if client.is_unsafe(text):
            return GuardrailResult(
                passed=False, severity="block", rule="moderation",
                message="Moderation flagged the input as unsafe.",
                detail={"model": client.model},
            )
        return GuardrailResult(
            passed=True, severity="pass", rule="moderation",
            message="Moderation passed.",
            detail={"model": client.model},
        )
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("moderation skipped (unavailable): %s: %s", type(exc).__name__, exc)
        return GuardrailResult(
            passed=True, severity="pass", rule="moderation",
            message="Moderation skipped (unavailable); deterministic checks still apply.",
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )
    finally:
        if own_client:
            client.close()
