"""Thin OpenAI wrapper for Sprint-3 LLM-powered coaching.

All LLM calls in the pipeline go through ``LLMClient`` so model selection,
retry logic, and graceful fallback are centralised.

Usage::

    from vocal_coach.llm import LLMClient
    from vocal_coach.coaching_config import CoachingConfig

    cfg = CoachingConfig.load(...)
    client = LLMClient.from_config(cfg.llm)
    if client is None:
        # OPENAI_API_KEY not set — pipeline continues without LLM
        ...

    # Structured JSON response (Pydantic model)
    result: MyModel | None = client.chat_json(
        system="You are a vocal coach.",
        user="Describe this performance.",
        schema=MyModel,
    )

    # Free-text response
    text: str | None = client.chat_text(
        system="You are a vocal coach.",
        user="Summarise this performance in 3 sentences.",
    )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Module-level singleton so the key is read once.
# ---------------------------------------------------------------------------

_ENV_KEY = "OPENAI_API_KEY"


def _load_dotenv() -> None:
    """Best-effort load of a .env file from the repo root."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(override=False)
    except Exception:
        pass


class LLMClient:
    """Sync OpenAI client with structured-output helpers and graceful fallback.

    Instantiate via ``LLMClient.from_config()`` rather than directly so the
    fallback-to-``None`` path is handled for you.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None  # lazy-init; set in from_config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: "LLMConfig") -> Optional["LLMClient"]:  # noqa: F821
        """Return an ``LLMClient`` ready to use, or ``None`` if unavailable.

        Returns ``None`` when:
        - ``cfg.enabled`` is ``False``
        - ``OPENAI_API_KEY`` is not set
        - The ``openai`` package is not installed
        """
        if not cfg.enabled:
            logger.debug("[llm] disabled by config")
            return None

        _load_dotenv()
        api_key = os.environ.get(_ENV_KEY, "").strip()
        if not api_key:
            logger.warning(
                "[llm] OPENAI_API_KEY not set — LLM features will be skipped."
            )
            return None

        try:
            import openai  # type: ignore  # noqa: F401
        except ImportError:
            logger.warning(
                "[llm] 'openai' package not installed — LLM features will be skipped."
            )
            return None

        inst = cls(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        inst._client = _build_openai_client(api_key)
        return inst

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
    ) -> Optional[T]:
        """Send a chat request and parse the JSON response into ``schema``.

        Returns ``None`` on any error (network, quota, parse failure) so
        callers can fall back to deterministic output gracefully.
        """
        if self._client is None:
            return None
        try:
            raw = self._raw_chat(system=system, user=user, json_mode=True)
            if raw is None:
                return None
            data = json.loads(raw)
            return schema.model_validate(data)
        except Exception as exc:
            logger.warning("[llm] chat_json failed: %s", exc)
            return None

    def chat_text(
        self,
        *,
        system: str,
        user: str,
    ) -> Optional[str]:
        """Send a chat request and return the response as plain text.

        Returns ``None`` on any error.
        """
        if self._client is None:
            return None
        try:
            return self._raw_chat(system=system, user=user, json_mode=False)
        except Exception as exc:
            logger.warning("[llm] chat_text failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _raw_chat(
        self,
        *,
        system: str,
        user: str,
        json_mode: bool,
    ) -> Optional[str]:
        """Make the OpenAI ChatCompletion call; return the content string."""
        import openai  # type: ignore

        kwargs: dict = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            logger.warning("[llm] rate limit hit — skipping LLM call")
            return None
        except openai.APIConnectionError as exc:
            logger.warning("[llm] connection error: %s", exc)
            return None
        except openai.APIStatusError as exc:
            logger.warning("[llm] API error %s: %s", exc.status_code, exc.message)
            return None

        choice = resp.choices[0]
        return choice.message.content


def _build_openai_client(api_key: str):
    import openai  # type: ignore

    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Convenience: module-level singleton for scripts that don't want to manage
# the client lifecycle themselves.
# ---------------------------------------------------------------------------

_singleton: Optional[LLMClient] = None


def get_client(cfg: Optional["LLMConfig"] = None) -> Optional[LLMClient]:  # noqa: F821
    """Return the module-level singleton, creating it from *cfg* if needed.

    Passing *cfg* on first call initialises the singleton; subsequent calls
    ignore *cfg* and return the cached client.
    """
    global _singleton
    if _singleton is None and cfg is not None:
        _singleton = LLMClient.from_config(cfg)
    return _singleton


__all__ = ["LLMClient", "get_client"]
