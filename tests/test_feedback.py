"""Unit tests for vocal_coach.feedback (Phase B LLM feedback generation).

All tests use mocked LLM clients — no real OpenAI API key is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.feedback import (
    _CardSummaryBatch,
    _CardSummary,
    _PerformanceSummaryRaw,
    _build_card_evidence,
    _lyric_for_moment,
    _topics_from_moments,
    generate_performance_summary,
    rewrite_card_summaries,
)
from vocal_coach.schemas import CoachingMoment, PerformanceSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SENTINEL = object()


def _make_moment(
    id: str = "sharp_flat_note:0",
    type: str = "sharp_flat_note",
    summary: str = "This note was sharp.",
    title: str = "Note on 'yesterday' is sharp",
    feedback_basis: str = "absolute",
    confidence: str = "high",
    scope: str = "local",
    note_indices=_SENTINEL,  # use sentinel so empty list is respected
    detail=_SENTINEL,         # use sentinel so empty dict is respected
) -> CoachingMoment:
    actual_note_indices = [0] if note_indices is _SENTINEL else note_indices
    actual_detail = (
        {"median_cents": 38.0, "direction": "sharp", "section": "Verse 1"}
        if detail is _SENTINEL
        else detail
    )
    return CoachingMoment(
        id=id,
        type=type,
        title=title,
        summary=summary,
        start_s=1.0,
        end_s=2.0,
        score=0.8,
        note_indices=actual_note_indices,
        feedback_basis=feedback_basis,
        confidence=confidence,
        scope=scope,
        detail=actual_detail,
    )


def _make_llm(batch_response=None, summary_response=None):
    """Create a mock LLMClient."""
    llm = MagicMock()
    llm.model = "gpt-4o-mini"
    llm.max_tokens = 1024

    if batch_response is not None:
        llm.chat_json.return_value = batch_response
    else:
        llm.chat_json.return_value = None

    return llm


def _make_rag(passages=None):
    rag = MagicMock()
    rag.query_detector.return_value = passages or ["Playbook: sharp_flat_note guidance."]
    rag.query_topics.return_value = passages or ["Pedagogy: pitch accuracy context."]
    return rag


def _make_reference(lyric_words=None):
    """Create a minimal mock ReferenceAnnotation."""
    ref = MagicMock()
    notes = []
    words = lyric_words or ["yesterday"]
    for i, w in enumerate(words):
        note = MagicMock()
        note.lyric_word = w
        note.word_index = i
        notes.append(note)
    ref.notes = notes
    return ref


def _make_overview(mimic_score=45.2, pct_in_tune=0.39):
    ov = MagicMock()
    ov.mimic_score = mimic_score
    ov.pct_in_tune = pct_in_tune
    ov.median_cents = -5.0
    ov.voiced_coverage = 0.78
    ov.arrival_offset_ms_mean = 42.0
    ov.expressive_density = 0.3
    ov.technique_match_rate = 0.62
    ov.note_count = 196
    ov.strongest_section = "Verse 1"
    return ov


# ---------------------------------------------------------------------------
# _lyric_for_moment
# ---------------------------------------------------------------------------


class TestLyricForMoment:
    def test_returns_lyric_word(self) -> None:
        moment = _make_moment(note_indices=[0])
        ref = _make_reference(["yesterday"])
        result = _lyric_for_moment(moment, ref)
        assert result == "yesterday"

    def test_returns_empty_when_no_reference(self) -> None:
        moment = _make_moment(note_indices=[0])
        result = _lyric_for_moment(moment, None)
        assert result == ""

    def test_returns_empty_when_no_note_indices(self) -> None:
        moment = _make_moment(note_indices=[])
        ref = _make_reference(["yesterday"])
        result = _lyric_for_moment(moment, ref)
        assert result == ""

    def test_deduplicates_words(self) -> None:
        # Two notes with same word_index = same word, should appear once.
        moment = _make_moment(note_indices=[0, 0])
        ref = _make_reference(["yesterday"])
        result = _lyric_for_moment(moment, ref)
        assert result == "yesterday"

    def test_caps_at_8_words(self) -> None:
        moment = _make_moment(note_indices=list(range(20)))
        words = [f"word{i}" for i in range(20)]
        ref = _make_reference(words)
        result = _lyric_for_moment(moment, ref)
        assert len(result.split()) <= 8


# ---------------------------------------------------------------------------
# _build_card_evidence
# ---------------------------------------------------------------------------


class TestBuildCardEvidence:
    def test_includes_required_keys(self) -> None:
        moment = _make_moment()
        ref = _make_reference()
        ev = _build_card_evidence(moment, ref, "playbook text")
        assert ev["id"] == moment.id
        assert ev["type"] == moment.type
        assert ev["feedback_basis"] == moment.feedback_basis
        assert ev["confidence"] == moment.confidence
        assert "measurements" in ev
        assert ev["playbook"] == "playbook text"

    def test_extracts_known_detail_keys(self) -> None:
        moment = _make_moment(detail={"median_cents": 38.0, "direction": "sharp", "evidence_strength": 0.9})
        ev = _build_card_evidence(moment, None, "")
        # evidence_strength is NOT in _DETAIL_KEYS_INCLUDE — should be excluded
        assert "evidence_strength" not in ev["measurements"]
        assert "median_cents" in ev["measurements"]

    def test_section_from_section_names(self) -> None:
        moment = _make_moment(detail={})
        moment.section_names = ["Chorus 1"]
        ev = _build_card_evidence(moment, None, "")
        assert ev["section"] == "Chorus 1"


# ---------------------------------------------------------------------------
# _topics_from_moments
# ---------------------------------------------------------------------------


class TestTopicsFromMoments:
    def test_derives_topics_from_types(self) -> None:
        moments = [
            _make_moment(type="breath_support_issue"),
            _make_moment(type="vibrato_quality"),
        ]
        topics = _topics_from_moments(moments)
        assert "breath_support" in topics
        assert "vibrato" in topics

    def test_deduplicates_topics(self) -> None:
        moments = [
            _make_moment(type="best_pitch_phrase"),
            _make_moment(type="pitch_struggle"),
        ]
        topics = _topics_from_moments(moments)
        # Both types map to "pitch" — should appear only once.
        assert topics.count("pitch") == 1

    def test_empty_moments_returns_empty(self) -> None:
        assert _topics_from_moments([]) == []

    def test_unknown_type_returns_empty_list(self) -> None:
        moments = [_make_moment(type="totally_unknown_type")]
        topics = _topics_from_moments(moments)
        assert topics == []


# ---------------------------------------------------------------------------
# rewrite_card_summaries
# ---------------------------------------------------------------------------


class TestRewriteCardSummaries:
    def test_returns_moments_unchanged_when_llm_none(self) -> None:
        moments = [_make_moment(summary="Original.")]
        result = rewrite_card_summaries(moments, llm=None, rag=None)
        assert result is moments
        assert result[0].summary == "Original."

    def test_returns_empty_list_unchanged(self) -> None:
        result = rewrite_card_summaries([], llm=MagicMock(), rag=None)
        assert result == []

    def test_rewrites_summary_on_success(self) -> None:
        moment = _make_moment(id="sharp_flat_note:0", summary="Original.")
        batch = _CardSummaryBatch(cards=[
            _CardSummary(id="sharp_flat_note:0", summary="LLM-rewritten summary.")
        ])
        llm = _make_llm(batch_response=batch)
        rag = _make_rag()
        result = rewrite_card_summaries([moment], llm=llm, rag=rag, reference=_make_reference())
        assert result[0].summary == "LLM-rewritten summary."

    def test_preserves_deterministic_summary_in_detail(self) -> None:
        moment = _make_moment(id="sharp_flat_note:0", summary="Original summary.")
        batch = _CardSummaryBatch(cards=[
            _CardSummary(id="sharp_flat_note:0", summary="LLM summary.")
        ])
        llm = _make_llm(batch_response=batch)
        rewrite_card_summaries([moment], llm=llm, rag=None)
        assert moment.detail["deterministic_summary"] == "Original summary."

    def test_stores_llm_evidence_in_detail(self) -> None:
        moment = _make_moment(id="sharp_flat_note:0")
        batch = _CardSummaryBatch(cards=[
            _CardSummary(id="sharp_flat_note:0", summary="LLM summary.")
        ])
        llm = _make_llm(batch_response=batch)
        rewrite_card_summaries([moment], llm=llm, rag=None)
        assert "llm_evidence" in moment.detail
        assert moment.detail["llm_evidence"]["id"] == moment.id

    def test_fallback_on_llm_none_response(self) -> None:
        moment = _make_moment(summary="Original.")
        llm = _make_llm(batch_response=None)  # LLM returns None
        rewrite_card_summaries([moment], llm=llm, rag=None)
        assert moment.summary == "Original."
        assert "deterministic_summary" not in moment.detail

    def test_keeps_original_when_id_missing_in_response(self) -> None:
        moment = _make_moment(id="sharp_flat_note:0", summary="Original.")
        # LLM responds with a different id — no match.
        batch = _CardSummaryBatch(cards=[
            _CardSummary(id="WRONG_ID", summary="Should not be applied.")
        ])
        llm = _make_llm(batch_response=batch)
        rewrite_card_summaries([moment], llm=llm, rag=None)
        assert moment.summary == "Original."

    def test_multiple_moments_all_rewritten(self) -> None:
        moments = [
            _make_moment(id="m1", summary="Original 1."),
            _make_moment(id="m2", summary="Original 2."),
        ]
        batch = _CardSummaryBatch(cards=[
            _CardSummary(id="m1", summary="New 1."),
            _CardSummary(id="m2", summary="New 2."),
        ])
        llm = _make_llm(batch_response=batch)
        rewrite_card_summaries(moments, llm=llm, rag=None)
        assert moments[0].summary == "New 1."
        assert moments[1].summary == "New 2."

    def test_uses_rag_for_playbook_retrieval(self) -> None:
        moment = _make_moment()
        batch = _CardSummaryBatch(cards=[_CardSummary(id=moment.id, summary="New.")])
        llm = _make_llm(batch_response=batch)
        rag = _make_rag(passages=["custom playbook text"])
        rewrite_card_summaries([moment], llm=llm, rag=rag)
        # RAG was queried for the moment type.
        rag.query_detector.assert_called_once_with(moment.type, k=1)

    def test_skips_rag_when_none(self) -> None:
        moment = _make_moment()
        batch = _CardSummaryBatch(cards=[_CardSummary(id=moment.id, summary="New.")])
        llm = _make_llm(batch_response=batch)
        # Should not raise even without RAG.
        rewrite_card_summaries([moment], llm=llm, rag=None)

    def test_injects_coaching_context_from_vocal_profile(self) -> None:
        moment = _make_moment()
        batch = _CardSummaryBatch(cards=[_CardSummary(id=moment.id, summary="New.")])
        llm = _make_llm(batch_response=batch)

        profile = MagicMock()
        profile.vocal_style = "Soft, emotive vocal delivery."
        profile.coaching_context = "This is a Beatles ballad."

        rewrite_card_summaries([moment], llm=llm, rag=None, vocal_profile=profile)
        call_kwargs = llm.chat_json.call_args[1]
        assert "Beatles ballad" in call_kwargs["system"]


# ---------------------------------------------------------------------------
# generate_performance_summary
# ---------------------------------------------------------------------------


class TestGeneratePerformanceSummary:
    def test_returns_none_when_llm_none(self) -> None:
        result = generate_performance_summary([], None, [], llm=None, rag=None)
        assert result is None

    def test_returns_valid_summary(self) -> None:
        moments = [_make_moment()]
        overview = _make_overview()
        raw = _PerformanceSummaryRaw(
            narrative="Good performance overall.",
            takeaways=["Focus on breath support."],
            trends=["Pitch accuracy dropped in each chorus."],
        )
        llm = _make_llm(batch_response=raw)
        result = generate_performance_summary(moments, overview, [], llm=llm, rag=None)
        assert result is not None
        assert isinstance(result, PerformanceSummary)
        assert result.narrative == "Good performance overall."
        assert len(result.takeaways) == 1
        assert len(result.trends) == 1

    def test_stamps_model_and_generated_at(self) -> None:
        raw = _PerformanceSummaryRaw(narrative="Good.", takeaways=[], trends=[])
        llm = _make_llm(batch_response=raw)
        result = generate_performance_summary([], None, [], llm=llm, rag=None)
        assert result is not None
        assert result.model_used == "gpt-4o-mini"
        assert result.generated_at is not None

    def test_returns_none_when_llm_returns_none(self) -> None:
        llm = _make_llm(batch_response=None)
        result = generate_performance_summary([], None, [], llm=llm, rag=None)
        assert result is None

    def test_returns_none_on_empty_narrative(self) -> None:
        raw = _PerformanceSummaryRaw(narrative="", takeaways=[], trends=[])
        llm = _make_llm(batch_response=raw)
        result = generate_performance_summary([], None, [], llm=llm, rag=None)
        assert result is None

    def test_queries_rag_topics(self) -> None:
        moments = [_make_moment(type="breath_support_issue")]
        raw = _PerformanceSummaryRaw(narrative="Good.", takeaways=[], trends=[])
        llm = _make_llm(batch_response=raw)
        rag = _make_rag(passages=["pedagogy text"])
        generate_performance_summary(moments, None, [], llm=llm, rag=rag)
        rag.query_topics.assert_called_once()
        # Topics should include breath_support since moment type is breath_support_issue.
        topics_arg = rag.query_topics.call_args[0][0]
        assert "breath_support" in topics_arg

    def test_skips_rag_when_none(self) -> None:
        raw = _PerformanceSummaryRaw(narrative="Good.", takeaways=[], trends=[])
        llm = _make_llm(batch_response=raw)
        # Should not raise.
        result = generate_performance_summary([], None, [], llm=llm, rag=None)
        assert result is not None

    def test_injects_vocal_profile_context(self) -> None:
        raw = _PerformanceSummaryRaw(narrative="Good.", takeaways=[], trends=[])
        llm = _make_llm(batch_response=raw)
        profile = MagicMock()
        profile.vocal_style = "Soft, emotive vocal delivery."
        profile.coaching_context = "Pop ballad with legato emphasis."
        profile.genre_tags = ["pop", "ballad"]
        generate_performance_summary([], None, [], llm=llm, rag=None, vocal_profile=profile)
        system = llm.chat_json.call_args[1]["system"]
        assert "Pop ballad with legato emphasis" in system
        assert "pop" in system

    def test_includes_overview_in_prompt(self) -> None:
        raw = _PerformanceSummaryRaw(narrative="Good.", takeaways=[], trends=[])
        llm = _make_llm(batch_response=raw)
        overview = _make_overview(mimic_score=72.5)
        generate_performance_summary([], overview, [], llm=llm, rag=None)
        user_prompt = llm.chat_json.call_args[1]["user"]
        assert "72.5" in user_prompt

    def test_includes_section_trends_in_prompt(self) -> None:
        raw = _PerformanceSummaryRaw(narrative="Good.", takeaways=[], trends=[])
        llm = _make_llm(batch_response=raw)
        section = MagicMock()
        section.name = "Chorus 1"
        section.kind = "chorus"
        section.pct_in_tune = 0.42
        section.median_cents = -8.0
        section.arrival_offset_ms_mean = 30.0
        section.mean_rms_db = -12.0
        section.technique_density = {"vibrato": 0.5}
        generate_performance_summary([], None, [section], llm=llm, rag=None)
        user_prompt = llm.chat_json.call_args[1]["user"]
        assert "Chorus 1" in user_prompt


# ---------------------------------------------------------------------------
# PerformanceSummary schema
# ---------------------------------------------------------------------------


class TestPerformanceSummarySchema:
    def test_valid_schema(self) -> None:
        ps = PerformanceSummary(
            narrative="Good performance.",
            takeaways=["Focus on breath."],
            trends=["Pitch drops in chorus."],
            model_used="gpt-4o-mini",
        )
        assert ps.narrative == "Good performance."
        assert len(ps.takeaways) == 1
        assert len(ps.trends) == 1
        assert ps.model_used == "gpt-4o-mini"
        assert ps.generated_at is None

    def test_empty_defaults(self) -> None:
        ps = PerformanceSummary(narrative="Good.")
        assert ps.takeaways == []
        assert ps.trends == []

    def test_serialises_to_json(self) -> None:
        ps = PerformanceSummary(narrative="Good.", takeaways=["a"], trends=["b"])
        j = ps.model_dump_json()
        assert "Good." in j
        assert '"takeaways"' in j
