"""Sprint-3 top-level performance overview computation.

``compute_overview`` distils the full note-level measurement list into a
single ``PerformanceOverview`` suitable for the summary stat-tile block at
the top of the results panel.

The ``mimic_score`` (0-100) blends:
    - pitch accuracy  (pct_in_tune)              weight 0.60
    - technique match rate                        weight 0.25
    - arrival consistency (low variance = good)   weight 0.15
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

from vocal_coach.highlights import _section_blended_score
from vocal_coach.schemas import (
    NoteMeasurementV2,
    NoteTechniqueComparison,
    PerformanceOverview,
    SectionTrend,
)


# Mimic-score blend weights (must sum to 1.0).
_W_PITCH = 0.60
_W_TECH = 0.25
_W_ARRIVAL = 0.15

_EXPRESSIVE_TECHS = frozenset(
    ["vibrato", "glissando", "falsetto", "breathe", "pharyngeal", "mixed", "bubble", "weak", "strong"]
)


def _safe_mean(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.mean(clean) if clean else None


def _safe_median(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(clean) if clean else None


def _pick_strongest_section(
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    min_notes: int = 4,
) -> Optional[str]:
    scored: list[tuple[float, str]] = []
    for section in sections:
        if section.note_count < min_notes:
            continue
        val = _section_blended_score(section, techniques, notes)
        if val is not None:
            scored.append((val, section.name))
    if not scored:
        return None
    return max(scored, key=lambda t: t[0])[1]


def compute_overview(
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    sections: Optional[list[SectionTrend]] = None,
    section_best_overall_min_notes: int = 4,
    octave_shift_semitones: int = 0,
    arrival_late_ms: float = 80.0,
) -> Optional[PerformanceOverview]:
    """Build a ``PerformanceOverview`` from note-level measurements.

    Args:
        notes: Per-note measurements from ``align_v2.measure_song``.
        techniques: Per-note technique comparisons from ``align_v2.measure_song``.
        octave_shift_semitones: Auto-detected integer-semitone register shift.
        arrival_late_ms: Threshold (ms) used to define "consistent" timing;
            arrival variance is normalised against this to produce the arrival
            component of the mimic score.

    Returns:
        ``None`` when there are no scoreable notes (pitch track absent, etc.).
    """
    if not notes:
        return None

    # --- Pitch stats ---
    pct_in_tune = _safe_mean(
        [n.pct_in_tune for n in notes if n.pct_in_tune is not None]
    )
    median_cents = _safe_median(
        [n.median_cents for n in notes if n.median_cents is not None]
    )
    voiced_coverage = _safe_mean([n.voiced_coverage for n in notes]) or 0.0

    # --- Arrival stats ---
    arrival_values = [n.arrival_offset_ms for n in notes if n.arrival_offset_ms is not None]
    arrival_offset_ms_mean = _safe_mean(arrival_values)

    # Arrival consistency: 1.0 = perfectly on time, 0.0 = all notes maximally late.
    arrival_consistency: Optional[float] = None
    if arrival_values:
        mean_abs = statistics.mean(abs(v) for v in arrival_values)
        # Normalise: 0 ms late -> 1.0; arrival_late_ms or more -> 0.0, clamped.
        arrival_consistency = max(0.0, 1.0 - mean_abs / max(1.0, arrival_late_ms))

    # --- Technique stats ---
    total_ref = 0
    total_matched = 0
    expressive_notes = 0

    for comp in techniques:
        total_ref += len(comp.reference_techniques)
        total_matched += len(comp.matched)
        if set(comp.matched) | set(comp.user_added):
            if (set(comp.matched) | set(comp.user_added)) & _EXPRESSIVE_TECHS:
                expressive_notes += 1

    technique_match_rate: Optional[float] = (
        total_matched / total_ref if total_ref > 0 else None
    )
    expressive_density = expressive_notes / len(notes) if notes else 0.0

    # --- Mimic score (0-100) ---
    mimic_score: Optional[float] = None
    components: list[tuple[float, float]] = []
    if pct_in_tune is not None:
        components.append((_W_PITCH, pct_in_tune))
    if technique_match_rate is not None:
        components.append((_W_TECH, technique_match_rate))
    elif pct_in_tune is not None:
        # Redistribute tech weight to pitch when no reference STARS.
        components = [(c[0] + _W_TECH * c[0] / _W_PITCH, c[1]) if c[0] == _W_PITCH else c
                      for c in components]
    if arrival_consistency is not None:
        components.append((_W_ARRIVAL, arrival_consistency))

    if components:
        total_weight = sum(w for w, _ in components)
        if total_weight > 0:
            raw = sum(w * v for w, v in components) / total_weight
            mimic_score = round(raw * 100, 1)

    strongest_section: Optional[str] = None
    if sections:
        strongest_section = _pick_strongest_section(
            sections,
            notes,
            techniques,
            min_notes=section_best_overall_min_notes,
        )

    return PerformanceOverview(
        pct_in_tune=pct_in_tune,
        median_cents=median_cents,
        octave_shift_semitones=octave_shift_semitones,
        voiced_coverage=voiced_coverage,
        arrival_offset_ms_mean=arrival_offset_ms_mean,
        expressive_density=expressive_density,
        technique_match_rate=technique_match_rate,
        mimic_score=mimic_score,
        note_count=len(notes),
        strongest_section=strongest_section,
    )


__all__ = ["compute_overview"]
