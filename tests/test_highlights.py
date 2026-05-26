"""Tests for vocal_coach.highlights."""

from __future__ import annotations

from vocal_coach.coaching_config import CoachingConfig
from vocal_coach.highlights import (
    TECH_LABELS,
    detect_best_pitch_phrase,
    detect_best_pitch_phrases,
    detect_expressive_match,
    detect_missed_expression,
    detect_pitch_struggle,
    detect_pitch_struggles,
    select_highlights,
)
from vocal_coach.schemas import (
    NoteMeasurementV2,
    NoteTechniqueComparison,
    ReferenceAnnotation,
    ReferenceNote,
    ReferenceSection,
)


def _ref_with_n_notes(n: int) -> ReferenceAnnotation:
    notes = [
        ReferenceNote(
            index=i,
            start_s=float(i),
            end_s=float(i) + 0.5,
            midi_pitch=60,
            note_name="C4",
            lyric_word=f"w{i}",
            word_index=i,
            phonemes=[],
        )
        for i in range(n)
    ]
    return ReferenceAnnotation(
        sample_id="t",
        audio_path="t.wav",
        sample_rate=16000,
        duration_s=float(n) + 0.5,
        sections=[ReferenceSection(name="Full", start_s=0.0, end_s=float(n) + 0.5)],
        words=[f"w{i}" for i in range(n)],
        phones=[],
        ph2word=[],
        notes=notes,
    )


def _measurement(
    note: ReferenceNote,
    *,
    pct_in_tune: float | None,
    arrival_offset_ms: float | None = 0.0,
    median_cents: float | None = 0.0,
) -> NoteMeasurementV2:
    return NoteMeasurementV2(
        note_index=note.index,
        start_s=note.start_s,
        end_s=note.end_s,
        midi_pitch=note.midi_pitch,
        note_name=note.note_name,
        lyric_word=note.lyric_word,
        voiced_coverage=1.0,
        median_cents=median_cents,
        pct_in_tune=pct_in_tune,
        drift_cents_per_s=0.0,
        arrival_offset_ms=arrival_offset_ms,
        core_start_s=note.start_s,
        core_end_s=note.end_s,
    )


def test_best_phrase_picks_highest_pct_in_tune_window() -> None:
    ref = _ref_with_n_notes(20)
    cfg = CoachingConfig()
    # Notes 4..11 are clean (1.0); the rest are noisy (0.5).
    notes = [
        _measurement(n, pct_in_tune=1.0 if 4 <= i < 12 else 0.5)
        for i, n in enumerate(ref.notes)
    ]
    moment = detect_best_pitch_phrase(ref, notes, config=cfg)
    assert moment is not None
    assert 4 in moment.note_indices and 11 in moment.note_indices
    assert moment.score > 0.95


def test_pitch_struggle_picks_lowest_pct_in_tune() -> None:
    ref = _ref_with_n_notes(12)
    cfg = CoachingConfig()
    # Notes 0..4 are bad (0.1), the rest are fine (0.9).
    notes = [
        _measurement(n, pct_in_tune=0.1 if i < 5 else 0.9)
        for i, n in enumerate(ref.notes)
    ]
    moment = detect_pitch_struggle(ref, notes, config=cfg)
    assert moment is not None
    assert 0 in moment.note_indices
    assert moment.start_s == 0.0


def test_expressive_match_for_shared_vibrato() -> None:
    ref = _ref_with_n_notes(8)
    cfg = CoachingConfig()
    notes = [_measurement(n, pct_in_tune=0.7) for n in ref.notes]
    techs = [
        NoteTechniqueComparison(
            note_index=i,
            reference_techniques=["vibrato"] if 2 <= i < 6 else [],
            user_techniques=["vibrato"] if 2 <= i < 6 else [],
            matched=["vibrato"] if 2 <= i < 6 else [],
            missed=[],
            user_added=[],
        )
        for i in range(len(ref.notes))
    ]
    moment = detect_expressive_match(ref, notes, techs, config=cfg)
    assert moment is not None
    assert moment.techniques == ["vibrato"]
    assert set(moment.note_indices) >= {2, 3, 4, 5}


def test_missed_expression_when_user_skips_reference_technique() -> None:
    ref = _ref_with_n_notes(8)
    cfg = CoachingConfig()
    notes = [_measurement(n, pct_in_tune=0.7) for n in ref.notes]
    techs = [
        NoteTechniqueComparison(
            note_index=i,
            reference_techniques=["vibrato"] if 1 <= i < 5 else [],
            user_techniques=[],
            matched=[],
            missed=["vibrato"] if 1 <= i < 5 else [],
            user_added=[],
        )
        for i in range(len(ref.notes))
    ]
    moment = detect_missed_expression(ref, notes, techs, config=cfg)
    assert moment is not None
    assert moment.type == "missed_expression"
    assert moment.techniques == ["vibrato"]


def test_multiple_pitch_phrases_when_configured() -> None:
    ref = _ref_with_n_notes(24)
    cfg = CoachingConfig()
    cfg.highlights.pitch_window_min = 8
    cfg.highlights.pitch_phrases_per_type = 2
    # Two clean islands: notes 2..9 and 14..21.
    notes = [
        _measurement(
            n,
            pct_in_tune=1.0 if (2 <= i < 10) or (14 <= i < 22) else 0.3,
        )
        for i, n in enumerate(ref.notes)
    ]
    phrases = detect_best_pitch_phrases(ref, notes, config=cfg)
    assert len(phrases) == 2
    assert not _windows_overlap(phrases[0].note_indices, phrases[1].note_indices)


def _windows_overlap(idxs_a: list[int], idxs_b: list[int]) -> bool:
    a, b = set(idxs_a), set(idxs_b)
    return bool(a & b)


def test_pharyngeal_uses_friendly_copy() -> None:
    ref = _ref_with_n_notes(8)
    cfg = CoachingConfig()
    notes = [_measurement(n, pct_in_tune=0.7) for n in ref.notes]
    techs = [
        NoteTechniqueComparison(
            note_index=i,
            reference_techniques=["pharyngeal"] if 2 <= i < 6 else [],
            user_techniques=["pharyngeal"] if 2 <= i < 6 else [],
            matched=["pharyngeal"] if 2 <= i < 6 else [],
            missed=[],
            user_added=[],
        )
        for i in range(len(ref.notes))
    ]
    moment = detect_expressive_match(ref, notes, techs, config=cfg)
    assert moment is not None
    assert "pharyngeal" not in moment.title.lower()
    assert "pharyngeal" not in moment.summary.lower()
    assert TECH_LABELS["pharyngeal"] in moment.title or TECH_LABELS["pharyngeal"] in moment.summary


def test_select_highlights_caps_total_count() -> None:
    ref = _ref_with_n_notes(20)
    cfg = CoachingConfig()
    cfg.highlights.cap = 3
    # Mixed-quality notes so multiple detectors fire.
    notes = []
    for i, n in enumerate(ref.notes):
        if i < 5:
            notes.append(_measurement(n, pct_in_tune=0.05))
        elif 6 <= i < 11:
            notes.append(_measurement(n, pct_in_tune=1.0))
        else:
            notes.append(_measurement(n, pct_in_tune=0.5, arrival_offset_ms=200.0))
    techs = [
        NoteTechniqueComparison(
            note_index=i,
            reference_techniques=["vibrato"] if i < 4 else [],
            user_techniques=["vibrato"] if i < 4 else [],
            matched=["vibrato"] if i < 4 else [],
            missed=[],
            user_added=[],
        )
        for i in range(len(ref.notes))
    ]
    report = select_highlights(ref, notes, techs, config=cfg)
    assert len(report.moments) <= 3
    assert report.moments == sorted(report.moments, key=lambda m: m.start_s)
