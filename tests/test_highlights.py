"""Tests for vocal_coach.highlights."""

from __future__ import annotations

from vocal_coach.coaching_config import CoachingConfig
from vocal_coach.highlights import (
    MOMENT_FEEDBACK_BASIS,
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
    moments = detect_expressive_match(ref, notes, techs, config=cfg)
    assert moments
    moment = moments[0]
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
    moments = detect_missed_expression(ref, notes, techs, config=cfg)
    assert moments
    moment = moments[0]
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
    moments = detect_expressive_match(ref, notes, techs, config=cfg)
    assert moments
    moment = moments[0]
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


# ---------------------------------------------------------------------------
# Sprint 1: confidence and feedback_basis tests
# ---------------------------------------------------------------------------


def _measurement_with_voicing(
    note: ReferenceNote,
    *,
    pct_in_tune: float | None,
    voiced_coverage: float = 1.0,
    mean_voicing_confidence: float | None = None,
    ref_voiced_coverage: float | None = None,
) -> NoteMeasurementV2:
    return NoteMeasurementV2(
        note_index=note.index,
        start_s=note.start_s,
        end_s=note.end_s,
        midi_pitch=note.midi_pitch,
        note_name=note.note_name,
        lyric_word=note.lyric_word,
        voiced_coverage=voiced_coverage,
        median_cents=0.0,
        pct_in_tune=pct_in_tune,
        drift_cents_per_s=0.0,
        arrival_offset_ms=0.0,
        core_start_s=note.start_s,
        core_end_s=note.end_s,
        mean_voicing_confidence=mean_voicing_confidence,
        ref_voiced_coverage=ref_voiced_coverage,
    )


def test_feedback_basis_absolute_for_pitch_types() -> None:
    """Pitch-type moments should be classified as absolute."""
    assert MOMENT_FEEDBACK_BASIS["best_pitch_phrase"] == "absolute"
    assert MOMENT_FEEDBACK_BASIS["pitch_struggle"] == "absolute"
    assert MOMENT_FEEDBACK_BASIS["sharp_flat_note"] == "absolute"
    assert MOMENT_FEEDBACK_BASIS["timing_consistency"] == "absolute"
    assert MOMENT_FEEDBACK_BASIS["late_entrance"] == "absolute"
    assert MOMENT_FEEDBACK_BASIS["fade_within_notes"] == "absolute"


def test_feedback_basis_comparative_for_technique_types() -> None:
    """Technique-matching moment types should be classified as comparative."""
    assert MOMENT_FEEDBACK_BASIS["expressive_match"] == "comparative"
    assert MOMENT_FEEDBACK_BASIS["missed_expression"] == "comparative"
    assert MOMENT_FEEDBACK_BASIS["expressive_moment"] == "comparative"
    assert MOMENT_FEEDBACK_BASIS["vocal_texture"] == "comparative"


def test_select_highlights_stamps_feedback_basis() -> None:
    """select_highlights should set feedback_basis on every returned moment."""
    ref = _ref_with_n_notes(20)
    cfg = CoachingConfig()
    notes = [_measurement(n, pct_in_tune=1.0 if 4 <= i < 12 else 0.1)
             for i, n in enumerate(ref.notes)]
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
    report = select_highlights(ref, notes, techs, config=cfg)
    for m in report.moments:
        assert m.feedback_basis in ("absolute", "comparative"), (
            f"moment {m.type!r} has invalid feedback_basis {m.feedback_basis!r}"
        )


def test_select_highlights_stamps_confidence() -> None:
    """All returned moments should carry a valid confidence tier."""
    ref = _ref_with_n_notes(20)
    cfg = CoachingConfig()
    notes = [_measurement(n, pct_in_tune=0.6) for n in ref.notes]
    techs = [
        NoteTechniqueComparison(
            note_index=i, reference_techniques=[], user_techniques=[],
            matched=[], missed=[], user_added=[],
        )
        for i in range(len(ref.notes))
    ]
    report = select_highlights(ref, notes, techs, config=cfg)
    for m in report.moments:
        assert m.confidence in ("low", "medium", "high"), (
            f"moment {m.type!r} has invalid confidence {m.confidence!r}"
        )


def test_low_confidence_moments_suppressed_by_default() -> None:
    """Moments with very sparse voiced coverage should be suppressed."""
    ref = _ref_with_n_notes(20)
    cfg = CoachingConfig()
    cfg.confidence.suppress_low = True
    # Notes with near-zero voiced coverage → very low evidence strength.
    notes = [
        _measurement_with_voicing(
            n,
            pct_in_tune=0.05 if i < 10 else 1.0,
            voiced_coverage=0.02,  # essentially silent
            mean_voicing_confidence=0.02,
        )
        for i, n in enumerate(ref.notes)
    ]
    techs = [
        NoteTechniqueComparison(
            note_index=i, reference_techniques=[], user_techniques=[],
            matched=[], missed=[], user_added=[],
        )
        for i in range(len(ref.notes))
    ]
    report = select_highlights(ref, notes, techs, config=cfg)
    # No "low" confidence moments should appear in the output.
    for m in report.moments:
        assert m.confidence != "low", f"Low-confidence moment {m.type!r} was not suppressed"


def test_missed_expression_uses_softer_title() -> None:
    """missed_expression title should NOT say 'Try more' (softer copy)."""
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
    from vocal_coach.highlights import detect_missed_expression
    moments = detect_missed_expression(ref, notes, techs, config=cfg)
    assert moments, "Expected at least one missed_expression moment"
    m = moments[0]
    assert "Try more" not in m.title, f"Title should not say 'Try more': {m.title!r}"
    assert "reference" in m.summary.lower(), f"Summary should mention reference: {m.summary!r}"


# ---------------------------------------------------------------------------
# Sprint 2: ADSR + continuous feature tests
# ---------------------------------------------------------------------------


def _adsr_measurement(
    note: ReferenceNote,
    *,
    pct_in_tune: float | None = 0.8,
    median_cents: float | None = 0.0,
    scoop_cents: float | None = None,
    ref_scoop_cents: float | None = None,
    sustain_pitch_std_cents: float | None = None,
    sustain_rms_slope_db_per_s: float | None = None,
    release_pitch_slope_cents_per_s: float | None = None,
    vibrato_rate_hz: float | None = None,
    vibrato_extent_cents: float | None = None,
    ref_vibrato_rate_hz: float | None = None,
    attack_rms_slope_db_per_s: float | None = None,
    ref_attack_rms_slope_db_per_s: float | None = None,
    envelope_shape: str | None = None,
    ref_envelope_shape: str | None = None,
    sustain_rms_std_db: float | None = None,
    release_rms_slope_db_per_s: float | None = None,
    user_rms_db: float | None = None,
    user_rms_relative_db: float | None = None,
    rms_delta_db: float | None = None,
    arrival_offset_ms: float | None = 0.0,
    voiced_coverage: float = 1.0,
    attack_duration_s: float | None = None,
    midi_pitch: int | None = None,
) -> NoteMeasurementV2:
    return NoteMeasurementV2(
        note_index=note.index,
        start_s=note.start_s,
        end_s=note.end_s,
        midi_pitch=midi_pitch if midi_pitch is not None else note.midi_pitch,
        note_name=note.note_name,
        lyric_word=note.lyric_word,
        voiced_coverage=voiced_coverage,
        median_cents=median_cents,
        pct_in_tune=pct_in_tune,
        drift_cents_per_s=0.0,
        arrival_offset_ms=arrival_offset_ms,
        core_start_s=note.start_s,
        core_end_s=note.end_s,
        scoop_cents=scoop_cents,
        ref_scoop_cents=ref_scoop_cents,
        sustain_pitch_std_cents=sustain_pitch_std_cents,
        sustain_rms_slope_db_per_s=sustain_rms_slope_db_per_s,
        release_pitch_slope_cents_per_s=release_pitch_slope_cents_per_s,
        vibrato_rate_hz=vibrato_rate_hz,
        vibrato_extent_cents=vibrato_extent_cents,
        ref_vibrato_rate_hz=ref_vibrato_rate_hz,
        attack_rms_slope_db_per_s=attack_rms_slope_db_per_s,
        ref_attack_rms_slope_db_per_s=ref_attack_rms_slope_db_per_s,
        envelope_shape=envelope_shape,
        ref_envelope_shape=ref_envelope_shape,
        sustain_rms_std_db=sustain_rms_std_db,
        release_rms_slope_db_per_s=release_rms_slope_db_per_s,
        user_rms_db=user_rms_db,
        user_rms_relative_db=user_rms_relative_db,
        rms_delta_db=rms_delta_db,
        attack_duration_s=attack_duration_s,
    )


def test_detect_scoop_habit_fires_on_repeated_scoops() -> None:
    from vocal_coach.highlights import detect_scoop_habit
    ref = _ref_with_n_notes(10)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, scoop_cents=40.0 if i < 4 else 5.0)
        for i, n in enumerate(ref.notes)
    ]
    moments = detect_scoop_habit(ref, notes, config=cfg)
    assert moments, "Expected a scoop_habit moment"
    m = moments[0]
    assert m.type == "scoop_habit"
    assert m.feedback_basis in ("absolute", None)  # not yet stamped by pipeline
    assert m.detail["count"] == 4


def test_detect_scoop_habit_comparative_when_ref_also_scoops() -> None:
    from vocal_coach.highlights import detect_scoop_habit
    ref = _ref_with_n_notes(10)
    cfg = CoachingConfig()
    # All user scoops have matching ref scoops
    notes = [
        _adsr_measurement(n, scoop_cents=40.0 if i < 4 else 5.0,
                          ref_scoop_cents=40.0 if i < 4 else 5.0)
        for i, n in enumerate(ref.notes)
    ]
    moments = detect_scoop_habit(ref, notes, config=cfg)
    assert moments
    assert moments[0].detail["feedback_basis_override"] == "comparative"


def test_detect_pitch_overshoot_fires() -> None:
    from vocal_coach.highlights import detect_pitch_overshoot
    ref = _ref_with_n_notes(10)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, scoop_cents=-40.0 if i < 4 else 5.0)
        for i, n in enumerate(ref.notes)
    ]
    moments = detect_pitch_overshoot(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "pitch_overshoot"


def test_detect_clean_attack_affirming() -> None:
    from vocal_coach.highlights import detect_clean_attack
    ref = _ref_with_n_notes(10)
    cfg = CoachingConfig()
    notes = [_adsr_measurement(n, scoop_cents=5.0) for n in ref.notes]
    moments = detect_clean_attack(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "clean_attack"


def test_detect_falling_release_fires() -> None:
    from vocal_coach.highlights import detect_falling_release
    ref = _ref_with_n_notes(10)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, release_pitch_slope_cents_per_s=-200.0 if i < 3 else 0.0)
        for i, n in enumerate(ref.notes)
    ]
    moments = detect_falling_release(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "falling_release"


def test_detect_steady_sustain_picks_most_stable() -> None:
    from vocal_coach.highlights import detect_steady_sustain
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(
            n,
            sustain_pitch_std_cents=2.0 if i == 2 else 25.0,
            voiced_coverage=1.0,
        )
        for i, n in enumerate(ref.notes)
    ]
    # Make note 2 long enough
    notes[2] = NoteMeasurementV2(
        **{
            **notes[2].model_dump(),
            "start_s": 0.0,
            "end_s": 0.7,
        }
    )
    moments = detect_steady_sustain(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "steady_sustain"
    assert moments[0].detail["sustain_pitch_std_cents"] == 2.0


def test_detect_pitch_instability_ignores_vibrato_notes() -> None:
    from vocal_coach.highlights import detect_pitch_instability
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(
            n,
            sustain_pitch_std_cents=40.0,
            vibrato_rate_hz=5.5 if i == 0 else None,  # note 0 has vibrato → should be excluded
        )
        for i, n in enumerate(ref.notes)
    ]
    moments = detect_pitch_instability(ref, notes, config=cfg)
    assert moments
    # Note 0 (vibrato) must not be in note_indices
    assert 0 not in moments[0].note_indices


def test_detect_vibrato_quality_healthy_range() -> None:
    from vocal_coach.highlights import detect_vibrato_quality
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, vibrato_rate_hz=5.5, vibrato_extent_cents=60.0)
        for n in ref.notes
    ]
    moments = detect_vibrato_quality(ref, notes, config=cfg)
    types = {m.type for m in moments}
    assert "consistent_vibrato" in types


def test_detect_vibrato_quality_wide_vibrato() -> None:
    from vocal_coach.highlights import detect_vibrato_quality
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, vibrato_rate_hz=5.5, vibrato_extent_cents=150.0)
        for n in ref.notes
    ]
    moments = detect_vibrato_quality(ref, notes, config=cfg)
    types = {m.type for m in moments}
    assert "wide_vibrato" in types


def test_detect_breath_support_fires_on_flat_and_fade() -> None:
    from vocal_coach.highlights import detect_breath_support_issues
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(
            n,
            median_cents=-30.0,
            sustain_rms_slope_db_per_s=-5.0,
        )
        for n in ref.notes
    ]
    m = detect_breath_support_issues(ref, notes, config=cfg)
    assert m is not None
    assert m.type == "breath_support_issue"


def test_detect_breath_support_does_not_fire_without_fade() -> None:
    from vocal_coach.highlights import detect_breath_support_issues
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, median_cents=-30.0, sustain_rms_slope_db_per_s=0.0)
        for n in ref.notes
    ]
    m = detect_breath_support_issues(ref, notes, config=cfg)
    assert m is None


def test_detect_registration_strain_fires_on_high_loud_sharp() -> None:
    from vocal_coach.highlights import detect_registration_strain
    ref = _ref_with_n_notes(3)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(
            n,
            median_cents=30.0,
            user_rms_relative_db=5.0,
            midi_pitch=72,  # high note above passaggio
        )
        for n in ref.notes
    ]
    m = detect_registration_strain(ref, notes, config=cfg)
    assert m is not None
    assert m.type == "registration_strain"


def test_detect_registration_strain_no_fire_below_passaggio() -> None:
    from vocal_coach.highlights import detect_registration_strain
    ref = _ref_with_n_notes(3)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(
            n,
            median_cents=30.0,
            user_rms_relative_db=5.0,
            midi_pitch=60,  # below passaggio
        )
        for n in ref.notes
    ]
    m = detect_registration_strain(ref, notes, config=cfg)
    assert m is None


def test_detect_controlled_crescendo_fires() -> None:
    from vocal_coach.highlights import detect_controlled_crescendo
    ref = _ref_with_n_notes(12)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, rms_delta_db=3.0 if i < 6 else 0.0, pct_in_tune=0.75)
        for i, n in enumerate(ref.notes)
    ]
    moments = detect_controlled_crescendo(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "controlled_crescendo"


def test_detect_phrase_timing_bias_rushed() -> None:
    from vocal_coach.highlights import detect_phrase_timing_bias
    ref = _ref_with_n_notes(12)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, arrival_offset_ms=-40.0)
        for n in ref.notes
    ]
    moments = detect_phrase_timing_bias(ref, notes, config=cfg)
    types = {m.type for m in moments}
    assert "rushed_phrase" in types


def test_detect_phrase_timing_bias_dragged() -> None:
    from vocal_coach.highlights import detect_phrase_timing_bias
    ref = _ref_with_n_notes(12)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, arrival_offset_ms=50.0)
        for n in ref.notes
    ]
    moments = detect_phrase_timing_bias(ref, notes, config=cfg)
    types = {m.type for m in moments}
    assert "dragged_phrase" in types


def test_detect_rhythmic_precision_affirming() -> None:
    from vocal_coach.highlights import detect_rhythmic_precision
    ref = _ref_with_n_notes(8)
    cfg = CoachingConfig()
    notes = [_adsr_measurement(n, arrival_offset_ms=5.0) for n in ref.notes]
    moments = detect_rhythmic_precision(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "rhythmic_precision"


def test_detect_phrase_pitch_arc_fires_on_degrading_accuracy() -> None:
    from vocal_coach.highlights import detect_phrase_pitch_arc
    ref = _ref_with_n_notes(12)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, pct_in_tune=0.9 if i < 6 else 0.4)
        for i, n in enumerate(ref.notes)
    ]
    moments = detect_phrase_pitch_arc(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "phrase_pitch_arc"
    assert moments[0].detail["first_half_pct"] > moments[0].detail["second_half_pct"]


def test_detect_support_fade_fires() -> None:
    from vocal_coach.highlights import detect_support_fade
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, sustain_rms_slope_db_per_s=-5.0, sustain_rms_std_db=3.0)
        for n in ref.notes
    ]
    moments = detect_support_fade(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "support_fade"


def test_detect_scoop_with_fade_fires() -> None:
    from vocal_coach.highlights import detect_scoop_with_fade
    ref = _ref_with_n_notes(5)
    cfg = CoachingConfig()
    notes = [
        _adsr_measurement(n, scoop_cents=40.0, sustain_rms_slope_db_per_s=-5.0)
        for n in ref.notes
    ]
    moments = detect_scoop_with_fade(ref, notes, config=cfg)
    assert moments
    assert moments[0].type == "scoop_with_fade"


def test_feedback_basis_override_applied_in_select_highlights() -> None:
    """Dual-basis moments should have their override reflected after select_highlights."""
    ref = _ref_with_n_notes(12)
    cfg = CoachingConfig()
    cfg.highlights.cap = 20
    # Build notes that trigger scoop_habit and reference also scoops (comparative)
    notes = [
        _adsr_measurement(n, scoop_cents=40.0, ref_scoop_cents=40.0, pct_in_tune=0.7)
        for n in ref.notes
    ]
    techs = [
        NoteTechniqueComparison(
            note_index=i,
            reference_techniques=[],
            user_techniques=[],
            matched=[], missed=[], user_added=[],
        )
        for i in range(len(ref.notes))
    ]
    report = select_highlights(ref, notes, techs, config=cfg)
    scoop_moments = [m for m in report.moments if m.type == "scoop_habit"]
    if scoop_moments:
        assert scoop_moments[0].feedback_basis == "comparative"


def test_moment_category_covers_all_sprint2_types() -> None:
    """Every Sprint 2 type must appear in MOMENT_CATEGORY."""
    from vocal_coach.highlights import MOMENT_CATEGORY
    sprint2_types = [
        "scoop_habit", "pitch_overshoot", "clean_attack", "falling_release",
        "steady_sustain", "pitch_instability", "vibrato_quality", "consistent_vibrato",
        "wide_vibrato", "delayed_vibrato", "straight_tone_control",
        "breathy_onset", "clean_onset", "sforzando_attack", "note_crescendo",
        "note_swell", "support_fade", "dynamic_sustain", "release_cutoff",
        "breath_support_issue", "registration_strain", "controlled_crescendo",
        "loud_pitch_instability", "soft_passage_control", "vibrato_with_support",
        "scoop_with_fade", "technique_accuracy_tradeoff", "expressive_stability",
        "high_note_control",
        "rushed_phrase", "dragged_phrase", "rhythmic_precision", "phrase_pitch_arc",
        "section_improvement", "section_regression", "section_vibrato_contrast",
    ]
    missing = [t for t in sprint2_types if t not in MOMENT_CATEGORY]
    assert not missing, f"Types missing from MOMENT_CATEGORY: {missing}"
