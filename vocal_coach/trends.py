"""Sprint-3 section-trend aggregation.

``compute_section_trends`` maps the note-level measurements produced by
``align_v2`` onto the coarser section grid from ``ReferenceAnnotation.sections``
(either the auto-derived "Phrase N" set or a hand-labelled ``sections.yaml``
sidecar).  The resulting ``SectionTrend`` list feeds the section-level
highlight detectors in ``highlights.py`` and the section ribbon in the UI.
"""

from __future__ import annotations

import statistics
from typing import Optional

from vocal_coach.schemas import (
    NoteMeasurementV2,
    NoteTechniqueComparison,
    ReferenceAnnotation,
    SectionTrend,
)


def _safe_median(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _safe_mean(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def _safe_variance(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.variance(clean) if len(clean) >= 2 else None


def _note_midpoint(note: NoteMeasurementV2) -> float:
    return 0.5 * (note.start_s + note.end_s)


def compute_section_trends(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
) -> list[SectionTrend]:
    """Aggregate per-note measurements into per-section trend rows.

    Notes whose midpoint falls inside ``[section.start_s, section.end_s)``
    are assigned to that section. Notes that don't fall in any section
    (rare edge case with gap sections) are ignored.

    Args:
        reference: The reference annotation whose ``sections`` define the grid.
        notes: Per-note measurements from ``align_v2.measure_song``.
        techniques: Per-note technique comparisons from ``align_v2.measure_song``.

    Returns:
        One ``SectionTrend`` per entry in ``reference.sections``.  Empty list
        when ``reference.sections`` is empty.
    """
    if not reference.sections:
        return []

    # Build a fast lookup: note_index -> NoteTechniqueComparison
    tech_by_note: dict[int, NoteTechniqueComparison] = {
        t.note_index: t for t in techniques
    }

    expressive_techs = {
        "vibrato", "glissando", "falsetto", "breathe",
        "pharyngeal", "mixed", "bubble", "weak", "strong",
    }

    trends: list[SectionTrend] = []
    for section in reference.sections:
        # Gather notes in this section.
        sec_notes = [
            n for n in notes
            if section.start_s <= _note_midpoint(n) < section.end_s
        ]
        note_count = len(sec_notes)

        if note_count == 0:
            trends.append(
                SectionTrend(
                    name=section.name,
                    kind=section.kind,
                    start_s=section.start_s,
                    end_s=section.end_s,
                    note_count=0,
                    voiced_coverage=0.0,
                )
            )
            continue

        # Scalar aggregates.
        voiced_coverage = _safe_mean([n.voiced_coverage for n in sec_notes]) or 0.0
        median_cents = _safe_median(
            [n.median_cents for n in sec_notes if n.median_cents is not None]
        )
        cents_variance = _safe_variance(
            [n.median_cents for n in sec_notes if n.median_cents is not None]
        )
        pct_in_tune = _safe_mean(
            [n.pct_in_tune for n in sec_notes if n.pct_in_tune is not None]
        )
        arrival_offset_ms_mean = _safe_mean(
            [n.arrival_offset_ms for n in sec_notes if n.arrival_offset_ms is not None]
        )

        # Per-technique density for user side (matched + added) and reference side.
        user_tech_counts: dict[str, int] = {}
        ref_tech_counts: dict[str, int] = {}
        for n in sec_notes:
            comp = tech_by_note.get(n.note_index)
            if comp is None:
                continue
            for tech in set(comp.matched) | set(comp.user_added):
                user_tech_counts[tech] = user_tech_counts.get(tech, 0) + 1
            for tech in set(comp.reference_techniques):
                ref_tech_counts[tech] = ref_tech_counts.get(tech, 0) + 1

        technique_density = {
            tech: count / note_count
            for tech, count in user_tech_counts.items()
        }
        reference_technique_density = {
            tech: count / note_count
            for tech, count in ref_tech_counts.items()
        }

        # Loudness aggregates (None when loudness tracks were not available).
        mean_rms_db = _safe_mean(
            [n.user_rms_db for n in sec_notes if n.user_rms_db is not None]
        )
        ref_mean_rms_db = _safe_mean(
            [n.ref_rms_db for n in sec_notes if n.ref_rms_db is not None]
        )
        rms_delta_db = _safe_mean(
            [n.rms_delta_db for n in sec_notes if n.rms_delta_db is not None]
        )

        trends.append(
            SectionTrend(
                name=section.name,
                kind=section.kind,
                start_s=section.start_s,
                end_s=section.end_s,
                note_count=note_count,
                voiced_coverage=voiced_coverage,
                median_cents=median_cents,
                cents_variance=cents_variance,
                pct_in_tune=pct_in_tune,
                arrival_offset_ms_mean=arrival_offset_ms_mean,
                technique_density=technique_density,
                reference_technique_density=reference_technique_density,
                mean_rms_db=mean_rms_db,
                ref_mean_rms_db=ref_mean_rms_db,
                rms_delta_db=rms_delta_db,
            )
        )

    return trends


__all__ = ["compute_section_trends"]
