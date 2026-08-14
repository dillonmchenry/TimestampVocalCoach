"""Tests for vocal_coach.overview.compute_overview.

Covers the three root causes of the timing-score zero-out documented in the
timing_score_zero-out plan:

  1. Scoring ceiling — ArrivalConfig.late_ms (50 ms) was being used as the
     full-penalty ceiling.  With overview_arrival_late_ms=150, a consistent
     +57 ms take must yield a non-zero arrival_consistency.

  2. Mean vs median — a handful of ±300–500 ms window-edge outliers must not
     collapse arrival_consistency to 0 (median is robust; mean is not).

  3. Window-edge rejection — arrival detections near ±search_back or
     +search_forward are treated as missed entrances by the detector, and the
     overview filter should exclude them from the median calculation.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

import pytest

from vocal_coach.overview import compute_overview, _W_PITCH, _W_TECH, _W_ARRIVAL
from vocal_coach.schemas import NoteMeasurementV2, NoteTechniqueComparison


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _note(
    index: int,
    *,
    pct_in_tune: float = 0.8,
    arrival_offset_ms: Optional[float] = 0.0,
    rms_delta_db: Optional[float] = None,
) -> NoteMeasurementV2:
    start = float(index)
    return NoteMeasurementV2(
        note_index=index,
        start_s=start,
        end_s=start + 0.5,
        midi_pitch=60,
        note_name="C4",
        lyric_word="la",
        voiced_coverage=1.0,
        median_cents=0.0,
        pct_in_tune=pct_in_tune,
        drift_cents_per_s=0.0,
        arrival_offset_ms=arrival_offset_ms,
        core_start_s=start,
        core_end_s=start + 0.5,
        rms_delta_db=rms_delta_db,
    )


def _notes(
    n: int,
    *,
    pct_in_tune: float = 0.8,
    arrival_offset_ms: Optional[float] = 0.0,
    rms_delta_db: Optional[float] = None,
) -> list[NoteMeasurementV2]:
    return [
        _note(i, pct_in_tune=pct_in_tune,
              arrival_offset_ms=arrival_offset_ms,
              rms_delta_db=rms_delta_db)
        for i in range(n)
    ]


def _tech(index: int, *, ref: list[str], matched: list[str]) -> NoteTechniqueComparison:
    return NoteTechniqueComparison(
        note_index=index,
        reference_techniques=ref,
        matched=matched,
        user_added=[],
    )


# ---------------------------------------------------------------------------
# Cause 1: 50 ms ceiling was zeroing out reasonable timing
# ---------------------------------------------------------------------------

class TestScoringThreshold:
    def test_consistent_57ms_late_is_nonzero_at_150ms_ceiling(self) -> None:
        """A take consistently 57 ms late should score > 0 with 150 ms ceiling."""
        notes = _notes(20, arrival_offset_ms=57.0)
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency is not None
        assert ov.arrival_consistency > 0.0
        # Expected: 1 - 57/150 ≈ 0.62
        assert ov.arrival_consistency == pytest.approx(1.0 - 57.0 / 150.0, abs=0.01)

    def test_consistent_57ms_late_was_zero_at_50ms_ceiling(self) -> None:
        """Document the old broken behaviour: 50 ms ceiling gives 0."""
        notes = _notes(20, arrival_offset_ms=57.0)
        ov = compute_overview(notes, [], arrival_late_ms=50.0)
        assert ov is not None
        assert ov.arrival_consistency == pytest.approx(0.0)

    def test_zero_offset_scores_1(self) -> None:
        notes = _notes(10, arrival_offset_ms=0.0)
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency == pytest.approx(1.0)

    def test_at_ceiling_scores_0(self) -> None:
        notes = _notes(10, arrival_offset_ms=150.0)
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency == pytest.approx(0.0)

    def test_mimic_score_includes_timing_at_57ms(self) -> None:
        """Mimic score must increase when arrival contributes."""
        notes_timed = _notes(20, pct_in_tune=0.5, arrival_offset_ms=57.0)
        notes_none = [_note(i, pct_in_tune=0.5, arrival_offset_ms=None) for i in range(20)]

        ov_timed = compute_overview(notes_timed, [], arrival_late_ms=150.0)
        ov_none  = compute_overview(notes_none,  [], arrival_late_ms=150.0)
        assert ov_timed is not None and ov_none is not None
        # arrival_consistency ≈ 0.62 > 0 so timed score should exceed pitch-only
        assert ov_timed.mimic_score is not None and ov_none.mimic_score is not None
        assert ov_timed.mimic_score > ov_none.mimic_score


# ---------------------------------------------------------------------------
# Cause 2: mean-absolute collapsed on outliers; median is robust
# ---------------------------------------------------------------------------

class TestMedianRobustness:
    def test_outliers_do_not_collapse_consistency(self) -> None:
        """8 notes at 20 ms + 2 notes at 400 ms — median path stays healthy."""
        normal   = [_note(i, arrival_offset_ms=20.0) for i in range(8)]
        outliers = [_note(8 + i, arrival_offset_ms=400.0) for i in range(2)]
        notes = normal + outliers
        ov = compute_overview(notes, [], arrival_late_ms=150.0, arrival_edge_margin_ms=0.0)
        assert ov is not None
        # Median |offset| = 20 ms → consistency = 1 - 20/150 ≈ 0.867
        # Mean |offset|   = (8*20 + 2*400) / 10 = 96 ms → 1 - 96/150 ≈ 0.36 (would still be >0)
        # But at 80 ms ceiling mean would give 0: 1 - 96/80 < 0 → clamped 0
        assert ov.arrival_consistency == pytest.approx(1.0 - 20.0 / 150.0, abs=0.01)

    def test_median_vs_mean_difference_at_old_ceiling(self) -> None:
        """Demonstrate that mean would zero at 80 ms while median does not."""
        normal   = [_note(i, arrival_offset_ms=20.0) for i in range(8)]
        outliers = [_note(8 + i, arrival_offset_ms=400.0) for i in range(2)]
        notes = normal + outliers

        ov_median = compute_overview(notes, [], arrival_late_ms=80.0, arrival_edge_margin_ms=0.0)
        assert ov_median is not None
        # Median = 20 ms → 1 - 20/80 = 0.75 (non-zero)
        assert ov_median.arrival_consistency == pytest.approx(0.75, abs=0.01)

        # If mean-abs were used: (8*20+2*400)/10 = 96 → 1 - 96/80 < 0 → 0
        mean_abs = (8 * 20 + 2 * 400) / 10
        mean_consistency = max(0.0, 1.0 - mean_abs / 80.0)
        assert mean_consistency == pytest.approx(0.0)

    def test_all_null_gives_none_consistency(self) -> None:
        notes = [_note(i, arrival_offset_ms=None) for i in range(10)]
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency is None

    def test_arrival_consistency_stored_on_overview(self) -> None:
        """arrival_consistency is present as a field on PerformanceOverview."""
        notes = _notes(5, arrival_offset_ms=30.0)
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        assert hasattr(ov, "arrival_consistency")
        assert ov.arrival_consistency is not None


# ---------------------------------------------------------------------------
# Cause 3: window-edge arrival detections are excluded from scoring
# ---------------------------------------------------------------------------

class TestEdgeRejection:
    # The overview uses hard-coded window constants (200 ms back, 500 ms fwd).
    # arrival_edge_margin_ms defaults to 20 ms.
    _SEARCH_BACK_MS = 200.0
    _SEARCH_FWD_MS  = 500.0
    _MARGIN         = 20.0

    def test_near_back_edge_excluded(self) -> None:
        """Offsets near -search_back_ms should be excluded."""
        edge = -(self._SEARCH_BACK_MS - self._MARGIN / 2)  # ≈ -190 ms, inside margin
        safe = 30.0

        notes_edge = [_note(i, arrival_offset_ms=edge) for i in range(10)]
        notes_safe = [_note(i, arrival_offset_ms=safe) for i in range(10)]

        ov_edge = compute_overview(notes_edge, [], arrival_late_ms=150.0)
        ov_safe = compute_overview(notes_safe, [], arrival_late_ms=150.0)

        assert ov_edge is not None and ov_safe is not None
        # Edge notes are all excluded → arrival_consistency is None (no scoreable values)
        assert ov_edge.arrival_consistency is None
        # Safe notes give a real consistency
        assert ov_safe.arrival_consistency is not None

    def test_near_forward_edge_excluded(self) -> None:
        """Offsets near +search_forward_ms should be excluded."""
        edge = self._SEARCH_FWD_MS - self._MARGIN / 2  # ≈ +490 ms, inside margin

        notes_edge = [_note(i, arrival_offset_ms=edge) for i in range(10)]
        ov = compute_overview(notes_edge, [], arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency is None

    def test_far_from_edge_retained(self) -> None:
        """Offsets well inside the window are retained."""
        # +100 ms is far from both ±200 ms and +500 ms boundaries
        notes = [_note(i, arrival_offset_ms=100.0) for i in range(10)]
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency is not None
        assert ov.arrival_consistency == pytest.approx(1.0 - 100.0 / 150.0, abs=0.01)

    def test_mixed_edge_and_real(self) -> None:
        """Edge detections are excluded; remaining real ones are scored."""
        real    = [_note(i, arrival_offset_ms=40.0) for i in range(8)]
        edges   = [_note(8 + i, arrival_offset_ms=490.0) for i in range(4)]
        notes   = real + edges
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        # Only the 8 real notes (40 ms) contribute → consistency = 1 - 40/150
        assert ov.arrival_consistency == pytest.approx(1.0 - 40.0 / 150.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration: full mimic score formula
# ---------------------------------------------------------------------------

class TestMimicScoreIntegration:
    def test_all_components_present(self) -> None:
        notes = _notes(10, pct_in_tune=0.8, arrival_offset_ms=30.0)
        techs = [_tech(i, ref=["vibrato"], matched=["vibrato"]) for i in range(10)]
        ov = compute_overview(notes, techs, arrival_late_ms=150.0)
        assert ov is not None
        # pitch=0.8, tech=1.0, arrival=1-30/150=0.8
        expected = (_W_PITCH * 0.8 + _W_TECH * 1.0 + _W_ARRIVAL * 0.8) * 100
        assert ov.mimic_score == pytest.approx(expected, abs=0.2)

    def test_no_arrival_redistributes_weight(self) -> None:
        """When arrival_consistency is None, its weight normalises out."""
        notes = [_note(i, pct_in_tune=0.8, arrival_offset_ms=None) for i in range(10)]
        techs = [_tech(i, ref=["vibrato"], matched=["vibrato"]) for i in range(10)]
        ov = compute_overview(notes, techs, arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency is None
        # Only pitch + tech; their weights normalise to sum to 1.0
        expected = (_W_PITCH * 0.8 + _W_TECH * 1.0) / (_W_PITCH + _W_TECH) * 100
        assert ov.mimic_score == pytest.approx(expected, abs=0.2)

    def test_arrival_consistency_field_matches_used_value(self) -> None:
        """The stored arrival_consistency must equal what went into mimic_score."""
        notes = _notes(10, pct_in_tune=0.6, arrival_offset_ms=60.0)
        ov = compute_overview(notes, [], arrival_late_ms=150.0)
        assert ov is not None
        assert ov.arrival_consistency is not None
        # Back-calculate: score = (W_PITCH*0.6 + W_ARRIVAL*arrival_consistency) / (W_PITCH+W_ARRIVAL) * 100
        denom = _W_PITCH + _W_ARRIVAL
        implied = (ov.mimic_score / 100.0 * denom - _W_PITCH * 0.6) / _W_ARRIVAL
        assert ov.arrival_consistency == pytest.approx(implied, abs=0.01)
