"""Unit tests for vocal_coach.vocal_profile.

Uses mocked LLM calls — no real OpenAI API key required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.schemas import VocalProfile
from vocal_coach.vocal_profile import (
    _DETECTOR_DESCRIPTIONS,
    generate_vocal_profile,
    load_vocal_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_manifest(
    title: str = "Yesterday",
    artist: str = "The Beatles",
    language: str = "English",
    song_id: str = "yesterday",
) -> MagicMock:
    m = MagicMock()
    m.title = title
    m.artist = artist
    m.language = language
    m.song_id = song_id
    return m


def _make_mock_llm(response_payload: dict) -> MagicMock:
    """Return a mock LLMClient whose chat_json returns a VocalProfile."""
    llm = MagicMock()
    # chat_json is called with schema=_VocalProfileRaw; return a parsed instance
    from vocal_coach.vocal_profile import _VocalProfileRaw
    llm.chat_json.return_value = _VocalProfileRaw.model_validate(response_payload)
    llm.model = "gpt-4o-mini"
    return llm


_VALID_PAYLOAD = {
    "genre_tags": ["soft-rock", "ballad"],
    "vocal_style": "Smooth, intimate delivery with controlled vibrato.",
    "key_techniques": ["vibrato", "legato", "dynamic contrast"],
    "emphasize_highlights": ["vibrato_quality", "steady_sustain"],
    "deemphasize_highlights": ["scoop_habit"],
    "highlight_notes": {
        "vibrato_quality": "Vibrato is central to the artist's style here.",
        "steady_sustain": "Long notes are a signature of this song.",
        "scoop_habit": "Scooping is part of McCartney's phrasing style.",
    },
    "coaching_context": "Focus on legato phrasing and dynamic shaping.",
}


# ---------------------------------------------------------------------------
# generate_vocal_profile
# ---------------------------------------------------------------------------


class TestGenerateVocalProfile:
    def test_returns_none_when_llm_is_none(self) -> None:
        manifest = _make_mock_manifest()
        result = generate_vocal_profile(manifest, None)
        assert result is None

    def test_returns_none_when_manifest_has_no_title_or_artist(self) -> None:
        manifest = _make_mock_manifest(title="", artist="")
        llm = MagicMock()
        result = generate_vocal_profile(manifest, llm)
        assert result is None

    def test_returns_valid_profile(self) -> None:
        manifest = _make_mock_manifest()
        llm = _make_mock_llm(_VALID_PAYLOAD)
        profile = generate_vocal_profile(manifest, llm)

        assert profile is not None
        assert isinstance(profile, VocalProfile)
        assert profile.song_id == "yesterday"
        assert "soft-rock" in profile.genre_tags
        assert "vibrato_quality" in profile.emphasize_highlights
        assert "scoop_habit" in profile.deemphasize_highlights

    def test_drops_unknown_detector_names(self) -> None:
        payload = dict(_VALID_PAYLOAD)
        payload["emphasize_highlights"] = ["vibrato_quality", "FAKE_DETECTOR_XYZ"]
        payload["highlight_notes"]["FAKE_DETECTOR_XYZ"] = "should be dropped"

        manifest = _make_mock_manifest()
        llm = _make_mock_llm(payload)
        profile = generate_vocal_profile(manifest, llm)

        assert profile is not None
        assert "FAKE_DETECTOR_XYZ" not in profile.emphasize_highlights
        assert "vibrato_quality" in profile.emphasize_highlights

    def test_drops_note_for_dropped_detector(self) -> None:
        payload = dict(_VALID_PAYLOAD)
        payload["deemphasize_highlights"] = ["scoop_habit", "NOT_REAL"]
        payload["highlight_notes"]["NOT_REAL"] = "should be gone"

        manifest = _make_mock_manifest()
        llm = _make_mock_llm(payload)
        profile = generate_vocal_profile(manifest, llm)

        assert "NOT_REAL" not in profile.highlight_notes

    def test_returns_none_when_llm_returns_none(self) -> None:
        manifest = _make_mock_manifest()
        llm = MagicMock()
        llm.chat_json.return_value = None
        result = generate_vocal_profile(manifest, llm)
        assert result is None

    def test_stamps_model_and_generated_at(self) -> None:
        manifest = _make_mock_manifest()
        llm = _make_mock_llm(_VALID_PAYLOAD)
        profile = generate_vocal_profile(manifest, llm)

        assert profile.model_used == "gpt-4o-mini"
        assert profile.generated_at is not None


# ---------------------------------------------------------------------------
# load_vocal_profile
# ---------------------------------------------------------------------------


class TestLoadVocalProfile:
    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "vocal_profile.json"
        result = load_vocal_profile(path)
        assert result is None

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "vocal_profile.json"
        profile_data = VocalProfile(
            song_id="yesterday",
            genre_tags=["pop"],
        )
        path.write_text(profile_data.model_dump_json(), encoding="utf-8")

        loaded = load_vocal_profile(path)
        assert loaded is not None
        assert loaded.song_id == "yesterday"
        assert "pop" in loaded.genre_tags

    def test_returns_none_for_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "vocal_profile.json"
        path.write_text("not valid json", encoding="utf-8")
        result = load_vocal_profile(path)
        assert result is None


# ---------------------------------------------------------------------------
# Detector descriptions completeness
# ---------------------------------------------------------------------------


class TestDetectorDescriptions:
    def test_known_detectors_in_descriptions(self) -> None:
        """Key detectors should be described so the LLM has full context."""
        required = {
            "scoop_habit",
            "vibrato_quality",
            "steady_sustain",
            "pitch_instability",
            "best_pitch_phrase",
            "breath_support_issue",
        }
        for det in required:
            assert det in _DETECTOR_DESCRIPTIONS, f"Missing: {det}"

    def test_all_descriptions_non_empty(self) -> None:
        for k, v in _DETECTOR_DESCRIPTIONS.items():
            assert v.strip(), f"Empty description for {k}"
