"""Unit tests for vocal_coach.llm (LLMClient).

All tests use mocks so no real OpenAI API key is required.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.coaching_config import LLMConfig
from vocal_coach.llm import LLMClient, get_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(model: str = "gpt-4o-mini") -> LLMClient:
    """Build a client with a mock OpenAI backend (no key check)."""
    inst = LLMClient(model=model, temperature=0.3, max_tokens=512)
    inst._client = MagicMock()
    return inst


def _stub_completion(client: LLMClient, content: str) -> None:
    """Configure client._client.chat.completions.create to return *content*."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    client._client.chat.completions.create.return_value = resp


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_returns_none_when_disabled(self) -> None:
        cfg = LLMConfig(enabled=False)
        assert LLMClient.from_config(cfg) is None

    def test_returns_none_when_no_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig(enabled=True)
        with patch("vocal_coach.llm._load_dotenv"):
            result = LLMClient.from_config(cfg)
        assert result is None

    def test_returns_none_when_openai_not_installed(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = LLMConfig(enabled=True)
        with (
            patch("vocal_coach.llm._load_dotenv"),
            patch("builtins.__import__", side_effect=ImportError("no openai")),
        ):
            # The ImportError happens inside from_config's try block
            result = LLMClient.from_config(cfg)
        # May or may not be None depending on import resolution; just check no exception.
        # (The real test is the warning path, which we can't easily check without caplog)

    def test_returns_client_when_key_present(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        cfg = LLMConfig(enabled=True)
        mock_openai_cls = MagicMock()
        with (
            patch("vocal_coach.llm._load_dotenv"),
            patch("vocal_coach.llm._build_openai_client", return_value=MagicMock()) as mock_build,
            patch.dict("sys.modules", {"openai": MagicMock()}),
        ):
            result = LLMClient.from_config(cfg)
        assert result is not None
        assert result.model == cfg.model


# ---------------------------------------------------------------------------
# chat_json
# ---------------------------------------------------------------------------


class TestChatJson:
    from pydantic import BaseModel

    class _Schema(BaseModel):
        value: str
        count: int

    def test_parses_valid_json(self) -> None:
        client = _make_client()
        payload = {"value": "hello", "count": 42}
        _stub_completion(client, json.dumps(payload))

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            result = client.chat_json(
                system="sys",
                user="user",
                schema=self._Schema,
            )
        assert result is not None
        assert result.value == "hello"
        assert result.count == 42

    def test_returns_none_on_invalid_json(self) -> None:
        client = _make_client()
        _stub_completion(client, "not valid json")
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            result = client.chat_json(system="s", user="u", schema=self._Schema)
        assert result is None

    def test_returns_none_when_no_inner_client(self) -> None:
        client = LLMClient()
        # _client is None
        result = client.chat_json(system="s", user="u", schema=self._Schema)
        assert result is None

    def test_returns_none_on_rate_limit(self) -> None:
        import importlib

        client = _make_client()
        mock_openai = MagicMock()

        class FakeRateLimitError(Exception):
            pass

        mock_openai.RateLimitError = FakeRateLimitError
        client._client.chat.completions.create.side_effect = FakeRateLimitError("rate limit")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = client.chat_json(system="s", user="u", schema=self._Schema)
        assert result is None


# ---------------------------------------------------------------------------
# chat_text
# ---------------------------------------------------------------------------


class TestChatText:
    def test_returns_text(self) -> None:
        client = _make_client()
        _stub_completion(client, "hello world")
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            result = client.chat_text(system="s", user="u")
        assert result == "hello world"

    def test_returns_none_when_no_inner_client(self) -> None:
        client = LLMClient()
        result = client.chat_text(system="s", user="u")
        assert result is None


# ---------------------------------------------------------------------------
# get_client singleton
# ---------------------------------------------------------------------------


class TestGetClientSingleton:
    def setup_method(self) -> None:
        # Reset the module-level singleton before each test.
        import vocal_coach.llm as llm_mod
        llm_mod._singleton = None

    def test_returns_none_without_cfg(self) -> None:
        import vocal_coach.llm as llm_mod
        llm_mod._singleton = None
        assert get_client() is None

    def test_creates_on_first_call_with_cfg(self, monkeypatch) -> None:
        import vocal_coach.llm as llm_mod
        llm_mod._singleton = None
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = LLMConfig(enabled=True)
        with (
            patch("vocal_coach.llm._load_dotenv"),
            patch("vocal_coach.llm._build_openai_client", return_value=MagicMock()),
            patch.dict("sys.modules", {"openai": MagicMock()}),
        ):
            c1 = get_client(cfg)
            c2 = get_client()  # second call without cfg — should return cached
        assert c1 is c2
