"""Deterministic highlight detectors.

Sprint 2 ships five highlight types. Each is a function that scans the
``NoteMeasurementV2`` rows (and ``NoteTechniqueComparison`` rows) and emits
zero or more ``CoachingMoment`` entries. The top-level ``select_highlights``
function ranks them and trims to the configured cap.

Highlight types:

    best_pitch_phrase     -- best mean pct_in_tune over configurable note windows
    pitch_struggle        -- worst mean pct_in_tune over configurable note windows
    expressive_match      -- ref + user share STARS techniques in the window
    expressive_moment     -- user produced strong techniques (with or without ref)
    missed_expression     -- ref had a technique the user lacked
    late_entrance         -- worst arrival window (still pitch-ish)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from vocal_coach.coaching_config import CoachingConfig
from vocal_coach.schemas import (
    CoachingMoment,
    HighlightsReport,
    NoteMeasurementV2,
    NoteTechniqueComparison,
    ReferenceAnnotation,
    ReferenceNote,
)


# User-facing copy for STARS technique keys (short label + coaching explanation).
TECH_LABELS: dict[str, str] = {
    "vibrato": "gentle vibrato",
    "glissando": "pitch slides",
    "falsetto": "light head voice",
    "pharyngeal": "deep, resonant tone",
    "breathe": "breathy tone",
    "bubble": "creaky / bubble tone",
    "weak": "soft, delicate tone",
    "strong": "powerful, full tone",
    "mixed": "mixed chest/head voice",
}

# Longer hints woven into highlight summaries (avoid jargon like "pharyngeal").
TECH_HINTS: dict[str, str] = {
    "vibrato": "a gentle waver in pitch",
    "glissando": "smooth slides between notes",
    "falsetto": "a lighter, head-voice color",
    "pharyngeal": "a deeper, rounded tone toward the back of the mouth",
    "breathe": "extra air in the tone",
    "bubble": "a creaky, textured edge",
    "weak": "a softer, more delicate delivery",
    "strong": "more power and fullness",
    "mixed": "a blend of chest and head voice",
}


def _label(tech: str) -> str:
    return TECH_LABELS.get(tech, tech.replace("_", " "))


def _hint(tech: str) -> str:
    return TECH_HINTS.get(tech, _label(tech))


def _section_for(reference: ReferenceAnnotation, t: float) -> Optional[str]:
    for sec in reference.sections:
        if sec.start_s <= t < sec.end_s:
            return sec.name
    return reference.sections[-1].name if reference.sections else None


def _windowed_indices(
    note_count: int,
    window_min: int,
    window_max: int,
) -> Iterable[tuple[int, int]]:
    """Yield ``(start_idx, end_idx_exclusive)`` for every window of size in [min, max]."""
    if note_count <= 0:
        return
    for size in range(window_min, window_max + 1):
        if size > note_count:
            break
        for start in range(0, note_count - size + 1):
            yield start, start + size


def _mean(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _phrase_window_for_note(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    start: int,
    end: int,
) -> tuple[float, float, list[int]]:
    """Compute (start_s, end_s, note_indices) for a contiguous note range."""
    if not notes:
        return 0.0, 0.0, []
    span = notes[start:end]
    note_indices = [n.note_index for n in span]
    return span[0].start_s, span[-1].end_s, note_indices


def _windows_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if two half-open note index ranges share any note."""
    return not (a[1] <= b[0] or b[1] <= a[0])


def _collect_top_pitch_phrases(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
    moment_type: str,
    window_min: int,
    window_max: int,
    qualify,
    rank_key,
    build_moment,
) -> list[CoachingMoment]:
    """Scan pitch windows, rank by ``rank_key``, return up to N non-overlapping phrases."""
    cfg = config.highlights
    ranked: list[tuple[float, int, int, float]] = []
    for start, end in _windowed_indices(len(notes), window_min, window_max):
        mean = _mean([notes[i].pct_in_tune for i in range(start, end)])
        if mean is None or not qualify(mean):
            continue
        ranked.append((rank_key(mean), start, end, mean))
    ranked.sort(key=lambda row: row[0], reverse=True)

    chosen: list[CoachingMoment] = []
    used_windows: list[tuple[int, int]] = []
    for _rank, start, end, mean_pct in ranked:
        if len(chosen) >= cfg.pitch_phrases_per_type:
            break
        window = (start, end)
        if any(_windows_overlap(window, used) for used in used_windows):
            continue
        moment = build_moment(reference, notes, start, end, mean_pct)
        if moment is None:
            continue
        chosen.append(moment)
        used_windows.append(window)
    return chosen


# ---------------------------------------------------------------------------
# Pitch-based detectors
# ---------------------------------------------------------------------------


def detect_best_pitch_phrases(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Top phrases by mean ``pct_in_tune`` over rolling pitch windows."""
    cfg = config.highlights

    def build_moment(ref, note_rows, start, end, mean_pct):
        start_s, end_s, idxs = _phrase_window_for_note(ref, note_rows, start, end)
        note_count = end - start
        return CoachingMoment(
            id=f"best_pitch_phrase:{idxs[0]}-{idxs[-1]}",
            type="best_pitch_phrase",
            title="Cleanest pitch run",
            summary=(
                f"You stayed in tune for {mean_pct * 100:.0f}% of sung frames "
                f"across {note_count} notes in this passage."
            ),
            start_s=start_s,
            end_s=end_s,
            score=mean_pct,
            note_indices=idxs,
            detail={
                "mean_pct_in_tune": mean_pct,
                "window_size": note_count,
                "section": _section_for(ref, 0.5 * (start_s + end_s)),
            },
        )

    return _collect_top_pitch_phrases(
        reference,
        notes,
        config=config,
        moment_type="best_pitch_phrase",
        window_min=cfg.pitch_window_min,
        window_max=cfg.pitch_window_max,
        qualify=lambda m: m >= cfg.best_phrase_min_pct_in_tune,
        rank_key=lambda m: m,
        build_moment=build_moment,
    )


def detect_pitch_struggles(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Worst phrases by mean ``pct_in_tune`` over rolling pitch windows."""
    cfg = config.highlights

    def build_moment(ref, note_rows, start, end, mean_pct):
        start_s, end_s, idxs = _phrase_window_for_note(ref, note_rows, start, end)
        note_count = end - start
        salience = 1.0 - mean_pct
        return CoachingMoment(
            id=f"pitch_struggle:{idxs[0]}-{idxs[-1]}",
            type="pitch_struggle",
            title="Tricky pitch passage",
            summary=(
                f"Only {mean_pct * 100:.0f}% of sung frames were in tune "
                f"across {note_count} notes — worth a focused practice pass."
            ),
            start_s=start_s,
            end_s=end_s,
            score=salience,
            note_indices=idxs,
            detail={
                "mean_pct_in_tune": mean_pct,
                "window_size": note_count,
                "section": _section_for(ref, 0.5 * (start_s + end_s)),
            },
        )

    return _collect_top_pitch_phrases(
        reference,
        notes,
        config=config,
        moment_type="pitch_struggle",
        window_min=cfg.pitch_window_min,
        window_max=cfg.pitch_window_max,
        qualify=lambda m: m <= cfg.pitch_struggle_max_pct_in_tune,
        rank_key=lambda m: 1.0 - m,
        build_moment=build_moment,
    )


def detect_best_pitch_phrase(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Return the single best pitch phrase (backward-compatible wrapper)."""
    phrases = detect_best_pitch_phrases(reference, notes, config=config)
    return phrases[0] if phrases else None


def detect_pitch_struggle(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Return the single worst pitch phrase (backward-compatible wrapper)."""
    phrases = detect_pitch_struggles(reference, notes, config=config)
    return phrases[0] if phrases else None


def detect_late_entrance(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Worst note by arrival offset (only if it's significantly late or early)."""
    cfg = config.arrival
    worst: Optional[NoteMeasurementV2] = None
    for n in notes:
        if n.arrival_offset_ms is None:
            continue
        if abs(n.arrival_offset_ms) < cfg.late_ms:
            continue
        if worst is None or abs(n.arrival_offset_ms) > abs(worst.arrival_offset_ms or 0.0):
            worst = n
    if worst is None or worst.arrival_offset_ms is None:
        return None
    direction = "late" if worst.arrival_offset_ms > 0 else "early"
    return CoachingMoment(
        id=f"late_entrance:{worst.note_index}",
        type="late_entrance",
        title=f"Watch your entrance on {worst.lyric_word!r}",
        summary=(
            f"You came in {abs(worst.arrival_offset_ms):.0f}ms {direction}."
        ),
        start_s=worst.start_s,
        end_s=worst.end_s,
        score=abs(worst.arrival_offset_ms) / 1000.0,
        note_indices=[worst.note_index],
        detail={
            "arrival_offset_ms": worst.arrival_offset_ms,
        },
    )


# ---------------------------------------------------------------------------
# STARS-driven detectors
# ---------------------------------------------------------------------------


def _phrase_techniques(
    techniques: list[NoteTechniqueComparison],
    start: int,
    end: int,
) -> dict[str, set[int]]:
    """Return ``{technique: {note_indices that have it}}`` for matched/missed/added."""
    out: dict[str, set[int]] = {}
    for i in range(start, end):
        comp = techniques[i]
        for tech in comp.matched:
            out.setdefault(f"matched:{tech}", set()).add(comp.note_index)
        for tech in comp.missed:
            out.setdefault(f"missed:{tech}", set()).add(comp.note_index)
        for tech in comp.user_added:
            out.setdefault(f"added:{tech}", set()).add(comp.note_index)
    return out


def detect_expressive_match(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Phrase with the most matched expressive technique notes."""
    cfg = config.highlights
    expressive = set(cfg.expressive_techniques)
    if not techniques:
        return None
    best_score: float = 0.0
    best: Optional[tuple[int, int, str]] = None
    for start, end in _windowed_indices(len(notes), cfg.window_min, cfg.window_max):
        for tech in expressive:
            count = sum(1 for i in range(start, end) if tech in techniques[i].matched)
            if count == 0:
                continue
            score = count / max(1, end - start)
            if score > best_score:
                best_score = score
                best = (start, end, tech)
    if best is None:
        return None
    s, e, tech = best
    start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
    return CoachingMoment(
        id=f"expressive_match:{tech}:{idxs[0]}-{idxs[-1]}",
        type="expressive_match",
        title=f"Matched the reference style: {_label(tech)}",
        summary=(
            f"You and the reference both used {_hint(tech)} in this phrase."
        ),
        start_s=start_s,
        end_s=end_s,
        score=best_score,
        note_indices=idxs,
        techniques=[tech],
        detail={
            "technique": tech,
            "matched_note_count": int(round(best_score * (e - s))),
            "window_size": e - s,
            "section": _section_for(reference, 0.5 * (start_s + end_s)),
        },
    )


def detect_expressive_moment(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Phrase where the user used the most expressive technique flags overall."""
    cfg = config.highlights
    expressive = set(cfg.expressive_techniques)
    if not techniques:
        return None
    best_score: float = 0.0
    best: Optional[tuple[int, int, str, int]] = None
    for start, end in _windowed_indices(len(notes), cfg.window_min, cfg.window_max):
        # For each tech, count notes where the user has it (matched OR added).
        for tech in expressive:
            count = sum(
                1
                for i in range(start, end)
                if tech in techniques[i].matched or tech in techniques[i].user_added
            )
            if count == 0:
                continue
            score = count / max(1, end - start)
            if score > best_score:
                best_score = score
                best = (start, end, tech, count)
    if best is None:
        return None
    s, e, tech, count = best
    start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
    return CoachingMoment(
        id=f"expressive_moment:{tech}:{idxs[0]}-{idxs[-1]}",
        type="expressive_moment",
        title=f"Strong {_label(tech)}",
        summary=(
            f"You brought {_hint(tech)} on {count} of {e - s} notes here — "
            "nice expressive choice."
        ),
        start_s=start_s,
        end_s=end_s,
        score=best_score,
        note_indices=idxs,
        techniques=[tech],
        detail={
            "technique": tech,
            "user_note_count": count,
            "window_size": e - s,
            "section": _section_for(reference, 0.5 * (start_s + end_s)),
        },
    )


def detect_missed_expression(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Phrase where the reference had an expressive technique the user did not match."""
    cfg = config.highlights
    expressive = set(cfg.expressive_techniques)
    if not techniques:
        return None
    best_score: float = 0.0
    best: Optional[tuple[int, int, str, int]] = None
    for start, end in _windowed_indices(len(notes), cfg.window_min, cfg.window_max):
        for tech in expressive:
            count = sum(1 for i in range(start, end) if tech in techniques[i].missed)
            if count == 0:
                continue
            score = count / max(1, end - start)
            if score > best_score:
                best_score = score
                best = (start, end, tech, count)
    if best is None:
        return None
    s, e, tech, count = best
    start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
    return CoachingMoment(
        id=f"missed_expression:{tech}:{idxs[0]}-{idxs[-1]}",
        type="missed_expression",
        title=f"Try more {_label(tech)}",
        summary=(
            f"The reference leans on {_hint(tech)} on {count} notes here; "
            "your take stays a bit straighter in tone."
        ),
        start_s=start_s,
        end_s=end_s,
        score=best_score,
        note_indices=idxs,
        techniques=[tech],
        detail={
            "technique": tech,
            "missed_note_count": count,
            "window_size": e - s,
            "section": _section_for(reference, 0.5 * (start_s + end_s)),
        },
    )


# ---------------------------------------------------------------------------
# Top-level selection
# ---------------------------------------------------------------------------


def _dedupe_overlap(
    moments: list[CoachingMoment],
    *,
    cap: int,
    max_per_type: int = 2,
) -> list[CoachingMoment]:
    """Cap and lightly dedupe overlapping moments of the same type."""
    chosen: list[CoachingMoment] = []
    by_type: dict[str, int] = {}
    for moment in moments:
        if len(chosen) >= cap:
            break
        if by_type.get(moment.type, 0) >= max_per_type:
            continue
        # Reject highlights overlapping any already-chosen one of the same type.
        same_type_overlap = any(
            (m.type == moment.type)
            and not (moment.end_s <= m.start_s or moment.start_s >= m.end_s)
            for m in chosen
        )
        if same_type_overlap:
            continue
        chosen.append(moment)
        by_type[moment.type] = by_type.get(moment.type, 0) + 1
    return chosen


def select_highlights(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: Optional[CoachingConfig] = None,
) -> HighlightsReport:
    """Run every detector, rank them, and trim to ``config.highlights.cap``."""
    cfg = config or CoachingConfig()
    candidates: list[CoachingMoment] = []

    candidates.extend(detect_best_pitch_phrases(reference, notes, config=cfg))
    candidates.extend(detect_pitch_struggles(reference, notes, config=cfg))
    late = detect_late_entrance(reference, notes, config=cfg)
    if late is not None:
        candidates.append(late)

    stars_detectors = [
        detect_expressive_match,
        detect_expressive_moment,
        detect_missed_expression,
    ]
    for fn in stars_detectors:
        moment = fn(reference, notes, techniques, config=cfg)
        if moment is not None:
            candidates.append(moment)

    # Diversity-aware ranking: alternate types so the UI shows a mix.
    candidates.sort(key=lambda m: m.score, reverse=True)
    chosen = _dedupe_overlap(
        candidates,
        cap=cfg.highlights.cap,
        max_per_type=cfg.highlights.max_per_type,
    )
    chosen.sort(key=lambda m: m.start_s)
    return HighlightsReport(moments=chosen, cap=cfg.highlights.cap)


__all__ = [
    "TECH_HINTS",
    "TECH_LABELS",
    "detect_best_pitch_phrase",
    "detect_best_pitch_phrases",
    "detect_expressive_match",
    "detect_expressive_moment",
    "detect_late_entrance",
    "detect_missed_expression",
    "detect_pitch_struggle",
    "detect_pitch_struggles",
    "select_highlights",
]
