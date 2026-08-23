"""Tests for vocal_coach.align_v2 dual-track measurement."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from vocal_coach.align_v2 import (
    PitchArrays,
    alignment_sanity_check,
    core_window_s,
    estimate_global_offset_s,
    estimate_octave_shift_semitones,
    measure_song,
    validate_chart_alignment,
)
from vocal_coach.coaching_config import CoachingConfig
from vocal_coach.schemas import (
    NoteMeasurementV2,
    PitchFrame,
    PitchTrack,
    ReferenceAnnotation,
    ReferenceNote,
    ReferenceSection,
    StarsPhoneme,
    StarsStyle,
    StarsTrack,
)


def _make_reference(notes: list[tuple[float, float, int, str]]) -> ReferenceAnnotation:
    """Build a tiny ReferenceAnnotation from (start, end, midi, lyric)."""
    ref_notes = [
        ReferenceNote(
            index=i,
            start_s=s,
            end_s=e,
            midi_pitch=midi,
            note_name=f"M{midi}",
            lyric_word=lyric,
            word_index=i,
            phonemes=[],
        )
        for i, (s, e, midi, lyric) in enumerate(notes)
    ]
    duration = max(e for _, e, _, _ in notes) + 0.5
    return ReferenceAnnotation(
        sample_id="t",
        audio_path="t.wav",
        sample_rate=16000,
        duration_s=duration,
        words=[lyric for *_, lyric in notes],
        phones=[],
        ph2word=[],
        sections=[ReferenceSection(name="Full", start_s=0.0, end_s=duration)],
        notes=ref_notes,
    )


def _synthetic_pitch(
    reference: ReferenceAnnotation,
    *,
    cents_offset: float = 0.0,
    time_offset_s: float = 0.0,
    semitone_offset: int = 0,
    hop: float = 0.01,
) -> PitchTrack:
    """Build a perfectly-voiced PitchTrack with the user singing the reference notes.

    ``time_offset_s`` shifts the user vocal in time (positive = the user
    starts the song late in user-time). ``cents_offset`` is added to every
    voiced frame's MIDI target. ``semitone_offset`` shifts the entire user
    range up/down by an integer number of semitones (use multiples of 12 to
    simulate an octave-low / octave-high singer).
    """
    n = int(reference.duration_s / hop) + 1
    frames = []
    for i in range(n):
        t = i * hop
        f0 = 0.0
        v = 0.0
        for note in reference.notes:
            song_t = t - time_offset_s
            if note.start_s <= song_t < note.end_s:
                target = note.midi_pitch + semitone_offset + cents_offset / 100.0
                f0 = 440.0 * (2 ** ((target - 69) / 12.0))
                v = 0.95
                break
        frames.append(PitchFrame(time=t, f0_hz=f0, voicing_confidence=v))
    return PitchTrack(sample_id="user", frames=frames, checkpoint="synthetic")


def test_perfect_user_is_in_tune() -> None:
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    pitch = _synthetic_pitch(ref)
    cfg = CoachingConfig()
    notes, techniques, offset, shift = measure_song(ref, pitch_user=pitch, config=cfg)
    assert offset == pytest.approx(0.0, abs=cfg.global_offset.step_s)
    assert shift == 0
    for n in notes:
        assert n.pct_in_tune is not None and n.pct_in_tune > 0.95
        assert n.median_cents is not None and abs(n.median_cents) < 5.0
        assert "flat" not in n.pitch_tags
        assert "sharp" not in n.pitch_tags
        assert n.note_octave_offset == 0


def test_flat_user_gets_flat_tag() -> None:
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    pitch = _synthetic_pitch(ref, cents_offset=-60.0)
    cfg = CoachingConfig()
    notes, _, _, _ = measure_song(ref, pitch_user=pitch, config=cfg)
    medians = [n.median_cents for n in notes if n.median_cents is not None]
    assert all(m < -40.0 for m in medians)
    flat_tags = sum(1 for n in notes for tag in n.pitch_tags if "flat" in tag)
    assert flat_tags >= len(notes)


def test_global_offset_recovered_when_user_late() -> None:
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    # user starts singing 200 ms late
    pitch = _synthetic_pitch(ref, time_offset_s=0.20)
    cfg = CoachingConfig()
    offset = estimate_global_offset_s(ref, pitch, config=cfg)
    # Expect offset close to +0.20 (user_time = song_time + 0.20s)
    assert offset == pytest.approx(0.20, abs=0.05)


def test_core_window_clamps_to_minimum() -> None:
    note = ReferenceNote(
        index=0,
        start_s=1.0,
        end_s=1.05,  # only 50 ms long
        midi_pitch=60,
        note_name="C4",
        lyric_word="x",
        word_index=0,
        phonemes=[],
    )
    cfg = CoachingConfig()
    cs, ce = core_window_s(note, arrival_offset_ms=0.0, config=cfg)
    # Float roundoff may shave a fraction of a microsecond off the result;
    # we just need the window to be within an FP epsilon of the min.
    assert ce - cs == pytest.approx(cfg.core_window.min_core_s, abs=1e-9)


def _make_stars_track(
    spans: list[tuple[float, float, str, dict[str, int]]],
) -> StarsTrack:
    phonemes = [
        StarsPhoneme(
            index=i,
            phoneme=ph,
            word=ph,
            word_index=i,
            start_s=s,
            end_s=e,
            techniques={
                "vibrato": tech.get("vibrato", 0),
                "glissando": tech.get("glissando", 0),
                "falsetto": tech.get("falsetto", 0),
                "breathe": tech.get("breathe", 0),
                "pharyngeal": tech.get("pharyngeal", 0),
                "mixed": tech.get("mixed", 0),
                "weak": tech.get("weak", 0),
                "strong": tech.get("strong", 0),
                "bubble": tech.get("bubble", 0),
            },
        )
        for i, (s, e, ph, tech) in enumerate(spans)
    ]
    return StarsTrack(
        sample_id="t",
        style=StarsStyle(
            language="EN", gender="x", emotion="x",
            method="x", pace="x", range="x", technique_group="x",
        ),
        phonemes=phonemes,
        notes=[],
    )


def test_technique_comparison_matches_and_misses() -> None:
    # Use clearly disjoint note windows + a >50ms gap between STARS spans so
    # the per-note technique attribution is not contaminated by the slack
    # (~20 ms) we deliberately apply when overlapping STARS phones with notes.
    ref = _make_reference([(0.0, 0.4, 69, "a"), (0.6, 1.0, 71, "b")])
    pitch = _synthetic_pitch(ref)
    stars_ref = _make_stars_track(
        [
            (0.0, 0.4, "AH", {"vibrato": 1}),
            (0.6, 1.0, "EE", {"glissando": 1}),
        ]
    )
    stars_user = _make_stars_track(
        [
            (0.0, 0.4, "AH", {"vibrato": 1, "breathe": 1}),
            (0.6, 1.0, "EE", {"falsetto": 1}),
        ]
    )
    cfg = CoachingConfig()
    _, techniques, _, _ = measure_song(
        ref, pitch_user=pitch, stars_ref=stars_ref, stars_user=stars_user, config=cfg
    )
    assert techniques[0].matched == ["vibrato"]
    assert "breathe" in techniques[0].user_added
    assert techniques[1].missed == ["glissando"]
    assert techniques[1].user_added == ["falsetto"]


def test_octave_low_user_is_auto_transposed_in_tune() -> None:
    """A user singing every note exactly one octave below the chart should be
    detected as -12 semitones and then scored as in-tune by the folded cents."""
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    pitch = _synthetic_pitch(ref, semitone_offset=-12)
    cfg = CoachingConfig()
    notes, _, _, shift = measure_song(ref, pitch_user=pitch, config=cfg)
    assert shift == -12
    for n in notes:
        assert n.pct_in_tune is not None and n.pct_in_tune > 0.95
        assert n.median_cents is not None and abs(n.median_cents) < 5.0
        assert n.note_octave_offset == 0
        assert "octave below" not in n.pitch_tags
        assert "octave above" not in n.pitch_tags


def test_octave_shift_recovers_large_chart_mismatch() -> None:
    """When the user is two octaves below the chart, the chart-based detector
    should round the median residual to -24 semitones."""
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    pitch_user = _synthetic_pitch(ref, semitone_offset=-24)
    cfg = CoachingConfig()
    shift = estimate_octave_shift_semitones(ref, pitch_user, config=cfg)
    assert shift == -24


def test_octave_shift_clamps_runaway_detection() -> None:
    """A user singing 5 octaves above the chart should be clamped, not propagated."""
    ref = _make_reference([(0.0, 0.5, 60, "a"), (0.5, 1.0, 60, "b"), (1.0, 1.5, 60, "c")])
    pitch_user = _synthetic_pitch(ref, semitone_offset=60)
    cfg = CoachingConfig()
    shift = estimate_octave_shift_semitones(ref, pitch_user, config=cfg, max_octaves=3)
    assert shift == 36


def test_single_outlier_note_gets_octave_tag_without_shifting_global() -> None:
    """One note bumped up an octave should be tagged 'octave above' but the
    global shift should stay 0 (median across notes wins)."""
    ref = _make_reference(
        [(0.0, 0.5, 60, "a"), (0.5, 1.0, 60, "b"), (1.0, 1.5, 60, "c"), (1.5, 2.0, 60, "d")]
    )
    # Build a pitch track where note index 2 is sung an octave high but the
    # rest are perfect. We do this by hand-stitching two synthetic tracks.
    base = _synthetic_pitch(ref)
    high = _synthetic_pitch(ref, semitone_offset=12)
    frames = []
    for f_base, f_high in zip(base.frames, high.frames):
        t = f_base.time
        if 1.0 <= t < 1.5:
            frames.append(f_high)
        else:
            frames.append(f_base)
    pitch = PitchTrack(sample_id="user", frames=frames, checkpoint="synthetic")
    cfg = CoachingConfig()
    notes, _, _, shift = measure_song(ref, pitch_user=pitch, config=cfg)
    assert shift == 0
    assert notes[2].note_octave_offset == 1
    assert "octave above" in notes[2].pitch_tags
    # Folded cents stay near zero so pct_in_tune is still high on the outlier.
    assert notes[2].pct_in_tune is not None and notes[2].pct_in_tune > 0.95
    # Other notes are unaffected.
    for i in (0, 1, 3):
        assert notes[i].note_octave_offset == 0
        assert "octave above" not in notes[i].pitch_tags


# ---------------------------------------------------------------------------
# New tests: center_offset_s, alignment_sanity_check, validate_chart_alignment
# ---------------------------------------------------------------------------


def test_center_offset_shifts_grid_recovers_late_user() -> None:
    """estimate_global_offset_s with center_offset_s finds users starting well
    outside the default +/-1.5 s window."""
    # Build a reference that has notes starting at 0, 0.5, 1.0 s in song time.
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    # User starts 5 s late in user-time (song_time = user_time - 5.0).
    pitch = _synthetic_pitch(ref, time_offset_s=5.0)
    cfg = CoachingConfig()
    # Default +/-1.5 s window centred on 0 should fail to find the offset.
    offset_default = estimate_global_offset_s(ref, pitch, config=cfg)
    assert abs(offset_default - 5.0) > 1.0, "Default window should not recover a 5 s shift"

    # Using center_offset_s=5.0 with a narrow refine window should nail it.
    offset_centred = estimate_global_offset_s(
        ref, pitch, config=cfg,
        search_range_s=1.0,
        center_offset_s=5.0,
    )
    assert offset_centred == pytest.approx(5.0, abs=0.1)


def test_estimate_global_offset_s_with_explicit_search_range() -> None:
    """The search_range_s parameter overrides the config value."""
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    # User is 3 s late — outside the default +/-1.5 s window, inside +/-5 s.
    # Build a manually extended pitch track so voiced frames exist at user-time 3-4.5 s.
    hop = 0.01
    n = int(5.0 / hop) + 1
    frames = []
    for i in range(n):
        t = i * hop
        f0 = 0.0
        v = 0.0
        for note in ref.notes:
            song_t = t - 3.0  # user vocal starts 3 s late
            if note.start_s <= song_t < note.end_s:
                f0 = 440.0 * (2 ** ((note.midi_pitch - 69) / 12.0))
                v = 0.95
                break
        frames.append(PitchFrame(time=t, f0_hz=f0, voicing_confidence=v))
    pitch = PitchTrack(sample_id="user", frames=frames, checkpoint="synthetic")
    cfg = CoachingConfig()
    offset = estimate_global_offset_s(ref, pitch, config=cfg, search_range_s=5.0)
    assert offset == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# alignment_sanity_check
# ---------------------------------------------------------------------------


def _make_note_measurement(
    *,
    pct_in_tune: Optional[float],
    voiced_coverage: float,
) -> NoteMeasurementV2:
    """Minimal NoteMeasurementV2 for sanity-check tests."""
    return NoteMeasurementV2(
        note_index=0,
        start_s=0.0,
        end_s=0.5,
        midi_pitch=69,
        note_name="A4",
        lyric_word="a",
        voiced_coverage=voiced_coverage,
        pct_in_tune=pct_in_tune,
    )


def test_sanity_check_passes_good_alignment() -> None:
    """Well-aligned performance (high pct_in_tune) should pass."""
    notes = [_make_note_measurement(pct_in_tune=0.80, voiced_coverage=0.90)] * 10
    assert alignment_sanity_check(notes) is True


def test_sanity_check_flags_misalignment() -> None:
    """Low pct_in_tune with reasonable voiced_coverage should be flagged."""
    notes = [_make_note_measurement(pct_in_tune=0.05, voiced_coverage=0.80)] * 10
    assert alignment_sanity_check(notes) is False


def test_sanity_check_passes_silent_user() -> None:
    """Low voiced_coverage should NOT trigger the flag (user barely sang)."""
    notes = [_make_note_measurement(pct_in_tune=0.05, voiced_coverage=0.10)] * 10
    assert alignment_sanity_check(notes) is True


def test_sanity_check_passes_no_notes() -> None:
    """Empty note list returns True (can't detect misalignment)."""
    assert alignment_sanity_check([]) is True


def test_sanity_check_passes_no_pct_in_tune() -> None:
    """Notes with pct_in_tune=None (unscored) return True."""
    notes = [_make_note_measurement(pct_in_tune=None, voiced_coverage=0.80)] * 5
    assert alignment_sanity_check(notes) is True


def test_sanity_check_passes_just_above_threshold() -> None:
    """pct_in_tune clearly above sanity_min_pct_in_tune should pass."""
    cfg = CoachingConfig()
    threshold = cfg.global_offset.sanity_min_pct_in_tune  # 0.15
    notes = [_make_note_measurement(pct_in_tune=threshold + 0.01, voiced_coverage=0.80)] * 10
    assert alignment_sanity_check(notes, config=cfg) is True


def test_sanity_check_fails_just_below_threshold() -> None:
    """pct_in_tune clearly below sanity_min_pct_in_tune should be flagged."""
    cfg = CoachingConfig()
    threshold = cfg.global_offset.sanity_min_pct_in_tune
    notes = [_make_note_measurement(pct_in_tune=threshold - 0.01, voiced_coverage=0.80)] * 10
    assert alignment_sanity_check(notes, config=cfg) is False


# ---------------------------------------------------------------------------
# validate_chart_alignment
# ---------------------------------------------------------------------------


def test_validate_chart_alignment_well_aligned() -> None:
    """Reference vocal perfectly in sync with chart should report near-zero offset."""
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    # The synthetic pitch generator builds a track that IS in sync with ref.
    pitch = _synthetic_pitch(ref, time_offset_s=0.0)
    cfg = CoachingConfig()
    offset, score = validate_chart_alignment(ref, pitch, config=cfg)
    assert abs(offset) < 0.1, f"Well-aligned chart should have near-zero offset, got {offset}"
    assert score > 0.0


def test_validate_chart_alignment_detects_gap_error() -> None:
    """A 400 ms GAP error in the chart should be detected."""
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    # Simulate the audio starting 400 ms earlier than the chart thinks (GAP too large).
    pitch = _synthetic_pitch(ref, time_offset_s=0.4)
    cfg = CoachingConfig()
    offset, score = validate_chart_alignment(ref, pitch, config=cfg)
    # The detected offset should be approximately 0.4 s.
    assert offset == pytest.approx(0.4, abs=0.08)


def test_validate_chart_alignment_correction_below_threshold_no_action_needed() -> None:
    """An offset below chart_correction_threshold_s should be within tolerance."""
    ref = _make_reference([(0.0, 0.5, 69, "a"), (0.5, 1.0, 71, "b"), (1.0, 1.5, 72, "c")])
    pitch = _synthetic_pitch(ref, time_offset_s=0.02)  # 20 ms — below 50 ms threshold
    cfg = CoachingConfig()
    offset, _score = validate_chart_alignment(ref, pitch, config=cfg)
    assert abs(offset) < cfg.global_offset.chart_correction_threshold_s
