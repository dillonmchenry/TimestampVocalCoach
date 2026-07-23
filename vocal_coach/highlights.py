"""Deterministic highlight detectors.

Each detector scans ``NoteMeasurementV2`` rows (and ``NoteTechniqueComparison``
rows) and emits zero or more ``CoachingMoment`` entries. The top-level
``select_highlights`` function ranks them and trims to the configured cap.

Highlight types — local:

    best_pitch_phrase     -- best mean pct_in_tune over configurable note windows
    pitch_struggle        -- worst mean pct_in_tune over configurable note windows
    sharp_flat_note       -- single notes notably sharp or flat
    late_entrance         -- notes with significantly late/early arrival
    timing_consistency    -- phrases with consistently off timing
    expressive_match      -- ref + user share STARS techniques in the window
    expressive_moment     -- user produced strong techniques (with or without ref)
    missed_expression     -- ref had a technique the user lacked
    vocal_texture         -- per-technique callouts (breathy, vibrato, …)
    fade_within_notes     -- user's voice fades within consecutive notes
    dynamic_drop          -- user notably quieter than reference
    dynamic_surge         -- user notably louder than reference

Highlight types — section:

    section_strength / section_weakness  -- single best/worst section by pitch
    best_overall_section / weakest_overall_section -- blended pitch+expression+timing
    section_delta         -- cross-kind pitch/cents/technique deltas
    section_dynamic_contrast -- verse vs chorus volume contrast
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
    SectionTrend,
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
# Single-note pitch callouts
# ---------------------------------------------------------------------------


def detect_sharp_flat_notes(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Individual notes whose median_cents are far from target."""
    cfg = config.highlights
    threshold = cfg.sharp_flat_note_min_cents
    candidates: list[tuple[float, NoteMeasurementV2]] = []
    for n in notes:
        if n.median_cents is None or n.voiced_coverage < config.pitch.min_voiced_coverage:
            continue
        if abs(n.median_cents) >= threshold:
            candidates.append((abs(n.median_cents), n))
    candidates.sort(key=lambda t: t[0], reverse=True)

    out: list[CoachingMoment] = []
    for _, n in candidates[: cfg.sharp_flat_note_max]:
        direction = "sharp" if n.median_cents > 0 else "flat"  # type: ignore[operator]
        cents = abs(n.median_cents)  # type: ignore[arg-type]
        out.append(
            CoachingMoment(
                id=f"sharp_flat_note:{n.note_index}",
                type="sharp_flat_note",
                title=f"Note on '{n.lyric_word}' is {direction}",
                summary=(
                    f"This {n.note_name} sat {cents:.0f}c {direction} of target — "
                    f"{'ease off the pressure a touch' if direction == 'sharp' else 'support the note with a bit more energy'}."
                ),
                start_s=n.start_s,
                end_s=n.end_s,
                score=cents / 100.0,
                note_indices=[n.note_index],
                detail={
                    "median_cents": n.median_cents,
                    "direction": direction,
                    "note_name": n.note_name,
                    "section": _section_for(reference, 0.5 * (n.start_s + n.end_s)),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Multiple entrance-timing callouts
# ---------------------------------------------------------------------------


def detect_entrance_timing_notes(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Top N notes by |arrival_offset_ms| (late or early)."""
    cfg_a = config.arrival
    cfg_h = config.highlights
    candidates: list[tuple[float, NoteMeasurementV2]] = []
    for n in notes:
        if n.arrival_offset_ms is None:
            continue
        if abs(n.arrival_offset_ms) < cfg_a.late_ms:
            continue
        candidates.append((abs(n.arrival_offset_ms), n))
    candidates.sort(key=lambda t: t[0], reverse=True)

    out: list[CoachingMoment] = []
    for _, n in candidates[: cfg_h.entrance_timing_max]:
        direction = "late" if n.arrival_offset_ms > 0 else "early"  # type: ignore[operator]
        ms = abs(n.arrival_offset_ms)  # type: ignore[arg-type]
        out.append(
            CoachingMoment(
                id=f"late_entrance:{n.note_index}",
                type="late_entrance",
                title=f"Watch your entrance on '{n.lyric_word}'",
                summary=f"You came in {ms:.0f}ms {direction} on this {n.note_name}.",
                start_s=n.start_s,
                end_s=n.end_s,
                score=ms / 1000.0,
                note_indices=[n.note_index],
                detail={
                    "arrival_offset_ms": n.arrival_offset_ms,
                    "section": _section_for(reference, 0.5 * (n.start_s + n.end_s)),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Timing-consistency detector (phrase-level)
# ---------------------------------------------------------------------------


def detect_timing_consistency(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Phrase where arrival is consistently off (mean |offset| above threshold)."""
    cfg = config.highlights
    w_min = cfg.timing_consistency_window_min
    w_max = cfg.timing_consistency_window_max
    threshold = cfg.timing_consistency_mean_ms

    worst_score = 0.0
    worst: Optional[tuple[int, int, float, float]] = None
    for start, end in _windowed_indices(len(notes), w_min, w_max):
        offsets = [
            notes[i].arrival_offset_ms
            for i in range(start, end)
            if notes[i].arrival_offset_ms is not None
        ]
        if len(offsets) < w_min:
            continue
        mean_abs = sum(abs(o) for o in offsets) / len(offsets)
        mean_signed = sum(offsets) / len(offsets)
        if mean_abs >= threshold and mean_abs > worst_score:
            worst_score = mean_abs
            worst = (start, end, mean_abs, mean_signed)

    if worst is None:
        return None

    s, e, mean_abs, mean_signed = worst
    start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
    direction = "late" if mean_signed > 0 else "early"
    return CoachingMoment(
        id=f"timing_consistency:{idxs[0]}-{idxs[-1]}",
        type="timing_consistency",
        title="Timing drifts in this passage",
        summary=(
            f"Your entrances run an average of {mean_abs:.0f}ms {direction} across "
            f"{e - s} notes here — try locking in with the beat."
        ),
        start_s=start_s,
        end_s=end_s,
        score=mean_abs / 1000.0,
        note_indices=idxs,
        detail={
            "mean_abs_offset_ms": mean_abs,
            "mean_signed_offset_ms": mean_signed,
            "direction": direction,
            "window_size": e - s,
            "section": _section_for(reference, 0.5 * (start_s + end_s)),
        },
    )


# ---------------------------------------------------------------------------
# Per-technique vocal-texture highlights
# ---------------------------------------------------------------------------


def detect_vocal_texture_moments(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Short, per-technique callouts like 'Breathy colour on these notes'.

    For each vocal-texture technique, find the longest run of consecutive
    notes where the user produced it.
    """
    cfg = config.highlights
    if not techniques:
        return []
    out: list[CoachingMoment] = []
    for tech in cfg.vocal_texture_techniques:
        has_tech = [
            tech in techniques[i].matched or tech in techniques[i].user_added
            for i in range(len(notes))
        ]
        best_run: list[int] = []
        current_run: list[int] = []
        for i, present in enumerate(has_tech):
            if present:
                current_run.append(i)
            else:
                if len(current_run) > len(best_run):
                    best_run = current_run
                current_run = []
        if len(current_run) > len(best_run):
            best_run = current_run

        if len(best_run) < cfg.vocal_texture_min_notes:
            continue

        s, e = best_run[0], best_run[-1] + 1
        start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
        label = _label(tech)
        hint = _hint(tech)
        out.append(
            CoachingMoment(
                id=f"vocal_texture:{tech}:{idxs[0]}-{idxs[-1]}",
                type="vocal_texture",
                title=f"{label.capitalize()} on this passage",
                summary=(
                    f"You used {hint} across {len(best_run)} notes here — "
                    f"{'great colour choice.' if tech in ('vibrato', 'breathe', 'falsetto') else 'adding real character.'}"
                ),
                start_s=start_s,
                end_s=end_s,
                score=len(best_run) / max(1, len(notes)),
                note_indices=idxs,
                techniques=[tech],
                detail={
                    "technique": tech,
                    "consecutive_notes": len(best_run),
                    "section": _section_for(reference, 0.5 * (start_s + end_s)),
                },
            )
        )
    return out


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
) -> list[CoachingMoment]:
    """Best phrase per expressive technique where ref and user both used it."""
    cfg = config.highlights
    expressive = set(cfg.expressive_techniques)
    if not techniques:
        return []
    best_by_tech: dict[str, tuple[float, int, int]] = {}
    for start, end in _windowed_indices(len(notes), cfg.window_min, cfg.window_max):
        for tech in expressive:
            count = sum(1 for i in range(start, end) if tech in techniques[i].matched)
            if count == 0:
                continue
            score = count / max(1, end - start)
            prev = best_by_tech.get(tech)
            if prev is None or score > prev[0]:
                best_by_tech[tech] = (score, start, end)
    out: list[CoachingMoment] = []
    for tech, (score, s, e) in sorted(best_by_tech.items(), key=lambda t: t[1][0], reverse=True):
        start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
        out.append(
            CoachingMoment(
                id=f"expressive_match:{tech}:{idxs[0]}-{idxs[-1]}",
                type="expressive_match",
                title=f"Matched the reference style: {_label(tech)}",
                summary=(
                    f"You and the reference both used {_hint(tech)} in this phrase."
                ),
                start_s=start_s,
                end_s=end_s,
                score=score,
                note_indices=idxs,
                techniques=[tech],
                detail={
                    "technique": tech,
                    "matched_note_count": int(round(score * (e - s))),
                    "window_size": e - s,
                    "section": _section_for(reference, 0.5 * (start_s + end_s)),
                },
            )
        )
    return out


def detect_expressive_moment(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Best phrase per expressive technique where the user brought it."""
    cfg = config.highlights
    expressive = set(cfg.expressive_techniques)
    if not techniques:
        return []
    best_by_tech: dict[str, tuple[float, int, int, int]] = {}
    for start, end in _windowed_indices(len(notes), cfg.window_min, cfg.window_max):
        for tech in expressive:
            count = sum(
                1
                for i in range(start, end)
                if tech in techniques[i].matched or tech in techniques[i].user_added
            )
            if count == 0:
                continue
            score = count / max(1, end - start)
            prev = best_by_tech.get(tech)
            if prev is None or score > prev[0]:
                best_by_tech[tech] = (score, start, end, count)
    out: list[CoachingMoment] = []
    for tech, (score, s, e, count) in sorted(best_by_tech.items(), key=lambda t: t[1][0], reverse=True):
        start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
        out.append(
            CoachingMoment(
                id=f"expressive_moment:{tech}:{idxs[0]}-{idxs[-1]}",
                type="expressive_moment",
                title=f"Strong {_label(tech)}",
                summary=(
                    f"You brought {_hint(tech)} on {count} of {e - s} notes here — "
                    "nice expressive choice."
                ),
                start_s=start_s,
                end_s=end_s,
                score=score,
                note_indices=idxs,
                techniques=[tech],
                detail={
                    "technique": tech,
                    "user_note_count": count,
                    "window_size": e - s,
                    "section": _section_for(reference, 0.5 * (start_s + end_s)),
                },
            )
        )
    return out


def detect_missed_expression(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Best phrase per expressive technique where the reference had it but user didn't."""
    cfg = config.highlights
    expressive = set(cfg.expressive_techniques)
    if not techniques:
        return []
    best_by_tech: dict[str, tuple[float, int, int, int]] = {}
    for start, end in _windowed_indices(len(notes), cfg.window_min, cfg.window_max):
        for tech in expressive:
            count = sum(1 for i in range(start, end) if tech in techniques[i].missed)
            if count == 0:
                continue
            score = count / max(1, end - start)
            prev = best_by_tech.get(tech)
            if prev is None or score > prev[0]:
                best_by_tech[tech] = (score, start, end, count)
    out: list[CoachingMoment] = []
    for tech, (score, s, e, count) in sorted(best_by_tech.items(), key=lambda t: t[1][0], reverse=True):
        start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
        out.append(
            CoachingMoment(
                id=f"missed_expression:{tech}:{idxs[0]}-{idxs[-1]}",
                type="missed_expression",
                title=f"Reference style: {_label(tech)}",
                summary=(
                    f"The reference uses {_hint(tech)} on {count} notes here — "
                    "your take takes a different stylistic approach."
                ),
                start_s=start_s,
                end_s=end_s,
                score=score,
                note_indices=idxs,
                techniques=[tech],
                detail={
                    "technique": tech,
                    "missed_note_count": count,
                    "window_size": e - s,
                    "section": _section_for(reference, 0.5 * (start_s + end_s)),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Sprint 3 — section-level detectors
# ---------------------------------------------------------------------------


def _note_indices_in_section(
    notes: list[NoteMeasurementV2],
    section_start_s: float,
    section_end_s: float,
) -> list[int]:
    """Return note_indices for notes whose midpoint lives inside the section."""
    out: list[int] = []
    for n in notes:
        mid = 0.5 * (n.start_s + n.end_s)
        if section_start_s <= mid < section_end_s:
            out.append(n.note_index)
    return out


def detect_section_strength(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Highlight the section with the highest mean pct_in_tune (above threshold)."""
    cfg = config.highlights
    qualifying = [
        s for s in sections
        if s.note_count >= cfg.section_min_notes
        and s.pct_in_tune is not None
        and s.pct_in_tune >= cfg.section_strength_min_pct_in_tune
    ]
    if not qualifying:
        return None
    best = max(qualifying, key=lambda s: s.pct_in_tune or 0.0)
    pct = best.pct_in_tune or 0.0
    idxs = _note_indices_in_section(notes, best.start_s, best.end_s)
    return CoachingMoment(
        id=f"section_strength:{best.name}",
        type="section_strength",
        scope="section",
        title=f"Strongest section: {best.name}",
        summary=(
            f"Your {best.name} held tune {pct * 100:.0f}% of the time across "
            f"{best.note_count} notes — your cleanest stretch."
        ),
        start_s=best.start_s,
        end_s=best.end_s,
        score=float(pct),
        note_indices=idxs,
        section_names=[best.name],
        detail={
            "pct_in_tune": pct,
            "median_cents": best.median_cents,
            "note_count": best.note_count,
            "kind": best.kind,
        },
    )


def detect_section_weakness(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Highlight the section with the lowest mean pct_in_tune (below threshold)."""
    cfg = config.highlights
    qualifying = [
        s for s in sections
        if s.note_count >= cfg.section_min_notes
        and s.pct_in_tune is not None
        and s.pct_in_tune <= cfg.section_weakness_max_pct_in_tune
    ]
    if not qualifying:
        return None
    worst = min(qualifying, key=lambda s: s.pct_in_tune or 0.0)
    pct = worst.pct_in_tune or 0.0
    idxs = _note_indices_in_section(notes, worst.start_s, worst.end_s)
    median_cents = worst.median_cents
    cents_hint = ""
    if median_cents is not None:
        direction = "flat" if median_cents < 0 else "sharp"
        cents_hint = f" Your median pitch in this section runs {abs(median_cents):.0f}c {direction}."
    return CoachingMoment(
        id=f"section_weakness:{worst.name}",
        type="section_weakness",
        scope="section",
        title=f"Toughest section: {worst.name}",
        summary=(
            f"Your {worst.name} held tune only {pct * 100:.0f}% of the time across "
            f"{worst.note_count} notes — a focused practice candidate.{cents_hint}"
        ),
        start_s=worst.start_s,
        end_s=worst.end_s,
        score=1.0 - float(pct),
        note_indices=idxs,
        section_names=[worst.name],
        detail={
            "pct_in_tune": pct,
            "median_cents": median_cents,
            "note_count": worst.note_count,
            "kind": worst.kind,
        },
    )


def _section_blended_score(
    section: SectionTrend,
    techniques: list[NoteTechniqueComparison],
    notes: list[NoteMeasurementV2],
) -> Optional[float]:
    """Blended 0-1 score for a section: pitch + expressiveness + timing."""
    pct = section.pct_in_tune
    if pct is None:
        return None

    # Expressiveness: fraction of notes with at least one user technique.
    sec_note_indices = set()
    for n in notes:
        mid = 0.5 * (n.start_s + n.end_s)
        if section.start_s <= mid < section.end_s:
            sec_note_indices.add(n.note_index)
    expressive_count = 0
    for t in techniques:
        if t.note_index in sec_note_indices and (t.matched or t.user_added):
            expressive_count += 1
    expr_density = expressive_count / max(1, len(sec_note_indices))

    # Timing: arrival consistency (1 = all on time, 0 = all off).
    arrival_vals = [
        n.arrival_offset_ms for n in notes
        if n.arrival_offset_ms is not None
        and n.note_index in sec_note_indices
    ]
    if arrival_vals:
        mean_abs = sum(abs(v) for v in arrival_vals) / len(arrival_vals)
        timing = max(0.0, 1.0 - mean_abs / 200.0)
    else:
        timing = 0.5

    return 0.50 * pct + 0.30 * expr_density + 0.20 * timing


def detect_best_overall_section(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Best section by blended pitch + expressiveness + timing score."""
    cfg = config.highlights
    scored: list[tuple[float, SectionTrend]] = []
    for s in sections:
        if s.note_count < cfg.section_best_overall_min_notes:
            continue
        val = _section_blended_score(s, techniques, notes)
        if val is not None:
            scored.append((val, s))
    if not scored:
        return None
    best_val, best = max(scored, key=lambda t: t[0])
    idxs = _note_indices_in_section(notes, best.start_s, best.end_s)

    parts: list[str] = []
    if best.pct_in_tune is not None:
        parts.append(f"{best.pct_in_tune * 100:.0f}% in tune")
    if best.arrival_offset_ms_mean is not None:
        parts.append(f"avg timing {abs(best.arrival_offset_ms_mean):.0f}ms off")
    detail_str = ", ".join(parts) if parts else "strong across the board"

    return CoachingMoment(
        id=f"best_overall_section:{best.name}",
        type="best_overall_section",
        scope="section",
        title=f"Your best section: {best.name}",
        summary=(
            f"Pitch, expression, and timing all come together in your {best.name} "
            f"({detail_str})."
        ),
        start_s=best.start_s,
        end_s=best.end_s,
        score=best_val,
        note_indices=idxs,
        section_names=[best.name],
        detail={
            "blended_score": best_val,
            "pct_in_tune": best.pct_in_tune,
            "arrival_offset_ms_mean": best.arrival_offset_ms_mean,
            "note_count": best.note_count,
            "kind": best.kind,
        },
    )


def detect_weakest_overall_section(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Weakest section by blended pitch + expressiveness + timing score."""
    cfg = config.highlights
    scored: list[tuple[float, SectionTrend]] = []
    for s in sections:
        if s.note_count < cfg.section_best_overall_min_notes:
            continue
        val = _section_blended_score(s, techniques, notes)
        if val is not None:
            scored.append((val, s))
    if not scored:
        return None
    worst_val, worst = min(scored, key=lambda t: t[0])
    idxs = _note_indices_in_section(notes, worst.start_s, worst.end_s)

    parts: list[str] = []
    if worst.pct_in_tune is not None:
        parts.append(f"only {worst.pct_in_tune * 100:.0f}% in tune")
    if worst.arrival_offset_ms_mean is not None:
        direction = "late" if worst.arrival_offset_ms_mean > 0 else "early"
        parts.append(f"avg {abs(worst.arrival_offset_ms_mean):.0f}ms {direction}")
    detail_str = ", ".join(parts) if parts else "room to grow"

    return CoachingMoment(
        id=f"weakest_overall_section:{worst.name}",
        type="weakest_overall_section",
        scope="section",
        title=f"Focus area: {worst.name}",
        summary=(
            f"Your {worst.name} could use the most attention ({detail_str}). "
            "Try isolating this section for focused practice."
        ),
        start_s=worst.start_s,
        end_s=worst.end_s,
        score=1.0 - worst_val,
        note_indices=idxs,
        section_names=[worst.name],
        detail={
            "blended_score": worst_val,
            "pct_in_tune": worst.pct_in_tune,
            "arrival_offset_ms_mean": worst.arrival_offset_ms_mean,
            "note_count": worst.note_count,
            "kind": worst.kind,
        },
    )


def _avg(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _kind_summary(
    sections: list[SectionTrend],
    kind: str,
) -> tuple[Optional[float], Optional[float], list[SectionTrend]]:
    """Return (avg pct_in_tune, avg median_cents, matching sections) for a kind."""
    matching = [s for s in sections if (s.kind or "").lower() == kind.lower()]
    if not matching:
        return None, None, matching
    pct = _avg([s.pct_in_tune for s in matching if s.pct_in_tune is not None])
    med = _avg([s.median_cents for s in matching if s.median_cents is not None])
    return pct, med, matching


def detect_section_pitch_deltas(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Cross-kind comparisons: 'verses 30c flatter than choruses', etc."""
    cfg = config.highlights
    out: list[CoachingMoment] = []
    for kind_a, kind_b in cfg.section_kind_pairs:
        pct_a, med_a, secs_a = _kind_summary(sections, kind_a)
        pct_b, med_b, secs_b = _kind_summary(sections, kind_b)
        if not secs_a or not secs_b:
            continue

        # Pitch-accuracy delta
        if pct_a is not None and pct_b is not None:
            delta = pct_b - pct_a
            if delta >= cfg.section_delta_min_pct_in_tune:
                span_start = min(s.start_s for s in secs_a)
                span_end = max(s.end_s for s in secs_a)
                out.append(
                    CoachingMoment(
                        id=f"section_delta_pct:{kind_a}_vs_{kind_b}",
                        type="section_delta",
                        scope="section",
                        title=f"{kind_a.capitalize()}s trail your {kind_b}s",
                        summary=(
                            f"Across {len(secs_a)} {kind_a}(s) you stayed in tune "
                            f"{pct_a * 100:.0f}% of the time vs {pct_b * 100:.0f}% "
                            f"across {len(secs_b)} {kind_b}(s)."
                        ),
                        start_s=span_start,
                        end_s=span_end,
                        score=delta,
                        section_names=[s.name for s in secs_a] + [s.name for s in secs_b],
                        detail={
                            "kind_a": kind_a,
                            "kind_b": kind_b,
                            "pct_in_tune_a": pct_a,
                            "pct_in_tune_b": pct_b,
                            "delta": delta,
                        },
                    )
                )

        # Median-cents delta (e.g. "verses run 30c flatter than choruses")
        if med_a is not None and med_b is not None:
            cents_delta = med_a - med_b
            if abs(cents_delta) >= cfg.section_delta_min_cents:
                direction = "flatter" if cents_delta < 0 else "sharper"
                span_start = min(s.start_s for s in secs_a)
                span_end = max(s.end_s for s in secs_a)
                out.append(
                    CoachingMoment(
                        id=f"section_delta_cents:{kind_a}_vs_{kind_b}",
                        type="section_delta",
                        scope="section",
                        title=f"{kind_a.capitalize()}s ran {direction} than {kind_b}s",
                        summary=(
                            f"Your median pitch on {kind_a}s sat "
                            f"{abs(cents_delta):.0f}c {direction} than on {kind_b}s "
                            "across the song."
                        ),
                        start_s=span_start,
                        end_s=span_end,
                        score=abs(cents_delta) / 100.0,
                        section_names=[s.name for s in secs_a] + [s.name for s in secs_b],
                        detail={
                            "kind_a": kind_a,
                            "kind_b": kind_b,
                            "median_cents_a": med_a,
                            "median_cents_b": med_b,
                            "cents_delta": cents_delta,
                        },
                    )
                )
    return out


def detect_section_technique_drops(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Surface "kept vibrato in chorus, dropped in verse" patterns.

    For each kind pair (a, b) and each expressive technique we compare the
    *user-side* technique_density between matching sections. A large drop
    going from kind_b -> kind_a (e.g. chorus->verse) is surfaced.
    """
    cfg = config.highlights
    out: list[CoachingMoment] = []
    expressive = set(cfg.expressive_techniques)
    for kind_a, kind_b in cfg.section_kind_pairs:
        _pct_a, _med_a, secs_a = _kind_summary(sections, kind_a)
        _pct_b, _med_b, secs_b = _kind_summary(sections, kind_b)
        if not secs_a or not secs_b:
            continue
        for tech in expressive:
            dens_a = _avg([s.technique_density.get(tech, 0.0) for s in secs_a]) or 0.0
            dens_b = _avg([s.technique_density.get(tech, 0.0) for s in secs_b]) or 0.0
            gap = dens_b - dens_a
            if gap < cfg.section_technique_min_density_gap:
                continue
            span_start = min(s.start_s for s in secs_a)
            span_end = max(s.end_s for s in secs_a)
            out.append(
                CoachingMoment(
                    id=f"section_tech_drop:{tech}:{kind_a}_vs_{kind_b}",
                    type="section_delta",
                    scope="section",
                    title=f"Carry {_label(tech)} into the {kind_a}s",
                    summary=(
                        f"You used {_hint(tech)} on {dens_b * 100:.0f}% of "
                        f"{kind_b} notes but only {dens_a * 100:.0f}% of "
                        f"{kind_a} notes — try carrying it across if you want "
                        "a more consistent style."
                    ),
                    start_s=span_start,
                    end_s=span_end,
                    score=gap,
                    section_names=[s.name for s in secs_a],
                    techniques=[tech],
                    detail={
                        "technique": tech,
                        "kind_a": kind_a,
                        "kind_b": kind_b,
                        "density_a": dens_a,
                        "density_b": dens_b,
                        "gap": gap,
                    },
                )
            )
    return out


def detect_section_moments(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Run all section detectors and return the merged list."""
    out: list[CoachingMoment] = []
    if not sections:
        return out

    # Blended best/weakest overall section
    best_overall = detect_best_overall_section(
        reference, sections, notes, techniques, config=config,
    )
    if best_overall is not None:
        out.append(best_overall)
    weakest_overall = detect_weakest_overall_section(
        reference, sections, notes, techniques, config=config,
    )
    if weakest_overall is not None:
        out.append(weakest_overall)

    # Pitch-only strength/weakness
    strength = detect_section_strength(reference, sections, notes, config=config)
    if strength is not None:
        out.append(strength)
    weakness = detect_section_weakness(reference, sections, notes, config=config)
    if weakness is not None:
        out.append(weakness)

    out.extend(detect_section_pitch_deltas(reference, sections, notes, config=config))
    out.extend(detect_section_technique_drops(reference, sections, notes, config=config))
    out.extend(detect_section_dynamic_contrast(reference, sections, notes, config=config))
    return out


# ---------------------------------------------------------------------------
# Loudness / dynamics detectors
# ---------------------------------------------------------------------------


def detect_fade_within_notes(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Find a phrase where the user's voice fades out on multiple notes.

    Looks for the longest consecutive run of notes where ``rms_fade_db_per_s``
    is at or below the configured threshold and voiced_coverage is adequate.
    Returns the worst contiguous window if it meets the minimum-notes bar.
    """
    cfg = config.highlights
    threshold = cfg.loudness_fade_threshold_db_per_s
    min_coverage = cfg.loudness_fade_min_voiced_coverage
    min_notes = cfg.loudness_fade_min_window_notes

    # Find runs of consecutive fading notes.
    is_fading = [
        (
            n.rms_fade_db_per_s is not None
            and n.rms_fade_db_per_s <= threshold
            and n.voiced_coverage >= min_coverage
        )
        for n in notes
    ]

    # Walk runs; keep the longest one.
    best_run: list[int] = []
    current_run: list[int] = []
    for i, fading in enumerate(is_fading):
        if fading:
            current_run.append(i)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = []
    if len(current_run) > len(best_run):
        best_run = current_run

    if len(best_run) < min_notes:
        return None

    s, e = best_run[0], best_run[-1] + 1
    start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
    mean_fade = _mean([notes[i].rms_fade_db_per_s for i in best_run if notes[i].rms_fade_db_per_s is not None])
    fade_str = f"{mean_fade:.1f}" if mean_fade is not None else "noticeably"
    return CoachingMoment(
        id=f"fade_within_notes:{idxs[0]}-{idxs[-1]}",
        type="fade_within_notes",
        title="Voice fades on note endings",
        summary=(
            f"Your level drops {fade_str} dB/s across {len(best_run)} consecutive "
            "notes here — try sustaining the tone fully through the end of each note."
        ),
        start_s=start_s,
        end_s=end_s,
        score=abs(mean_fade) if mean_fade is not None else float(len(best_run)),
        note_indices=idxs,
        detail={
            "mean_fade_db_per_s": mean_fade,
            "fading_note_count": len(best_run),
            "section": _section_for(reference, 0.5 * (start_s + end_s)),
        },
    )


def detect_dynamic_drop(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Find the phrase where the user sang notably quieter than the reference.

    Uses normalised ``rms_delta_db`` (user relative level − reference relative
    level), so mic-gain differences don't trigger false positives.
    Only runs when reference loudness was available (otherwise all
    ``rms_delta_db`` are None).
    """
    cfg = config.highlights
    min_delta = cfg.loudness_dynamic_delta_db  # use as magnitude threshold
    w_min = cfg.loudness_dynamic_window_min
    w_max = cfg.loudness_dynamic_window_max

    if not any(n.rms_delta_db is not None for n in notes):
        return None

    worst_score = 0.0
    worst: Optional[tuple[int, int]] = None
    for start, end in _windowed_indices(len(notes), w_min, w_max):
        deltas = [notes[i].rms_delta_db for i in range(start, end) if notes[i].rms_delta_db is not None]
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta < -min_delta and abs(mean_delta) > worst_score:
            worst_score = abs(mean_delta)
            worst = (start, end)

    if worst is None:
        return None

    s, e = worst
    start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
    db_str = f"{worst_score:.1f}"
    return CoachingMoment(
        id=f"dynamic_drop:{idxs[0]}-{idxs[-1]}",
        type="dynamic_drop",
        title="Pulling back in this phrase",
        summary=(
            f"Your voice is about {db_str} dB quieter (relative to your overall "
            "level) than the reference is here — try matching the reference's energy."
        ),
        start_s=start_s,
        end_s=end_s,
        score=worst_score,
        note_indices=idxs,
        detail={
            "mean_rms_delta_db": -worst_score,
            "section": _section_for(reference, 0.5 * (start_s + end_s)),
        },
    )


def detect_dynamic_surge(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Find the phrase where the user sang notably louder than the reference."""
    cfg = config.highlights
    min_delta = cfg.loudness_dynamic_delta_db
    w_min = cfg.loudness_dynamic_window_min
    w_max = cfg.loudness_dynamic_window_max

    if not any(n.rms_delta_db is not None for n in notes):
        return None

    best_score = 0.0
    best: Optional[tuple[int, int]] = None
    for start, end in _windowed_indices(len(notes), w_min, w_max):
        deltas = [notes[i].rms_delta_db for i in range(start, end) if notes[i].rms_delta_db is not None]
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta > min_delta and mean_delta > best_score:
            best_score = mean_delta
            best = (start, end)

    if best is None:
        return None

    s, e = best
    start_s, end_s, idxs = _phrase_window_for_note(reference, notes, s, e)
    db_str = f"{best_score:.1f}"
    return CoachingMoment(
        id=f"dynamic_surge:{idxs[0]}-{idxs[-1]}",
        type="dynamic_surge",
        title="Great power in this phrase",
        summary=(
            f"You pushed {db_str} dB above the reference's relative level here — "
            "lots of energy in this stretch."
        ),
        start_s=start_s,
        end_s=end_s,
        score=best_score,
        note_indices=idxs,
        detail={
            "mean_rms_delta_db": best_score,
            "section": _section_for(reference, 0.5 * (start_s + end_s)),
        },
    )


def detect_section_dynamic_contrast(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Detect meaningful dynamic contrast in the user's own performance.

    Compares the user's mean loudness level (``mean_rms_db``) between chorus
    and verse sections (or other configured kind-pairs).  Does NOT compare
    against the reference, so mic-gain normalisation is not required —
    chorus vs. verse are on the same absolute scale for the same recording.

    Surfaces an affirming highlight when the chorus is notably louder than
    the verse (good contrast), or a coaching note when contrast is near zero.
    Both are section-scope moments so they don't crowd out local highlights.
    """
    cfg = config.highlights
    min_db = cfg.section_dynamic_contrast_min_db
    out: list[CoachingMoment] = []

    for kind_a, kind_b in cfg.section_kind_pairs:  # (kind_a=verse, kind_b=chorus)
        secs_a = [s for s in sections if s.kind == kind_a and s.note_count >= cfg.section_min_notes and s.mean_rms_db is not None]
        secs_b = [s for s in sections if s.kind == kind_b and s.note_count >= cfg.section_min_notes and s.mean_rms_db is not None]
        if not secs_a or not secs_b:
            continue

        rms_a = sum(s.mean_rms_db for s in secs_a) / len(secs_a)  # type: ignore[operator]
        rms_b = sum(s.mean_rms_db for s in secs_b) / len(secs_b)  # type: ignore[operator]
        delta = rms_b - rms_a  # positive = kind_b (chorus) is louder than kind_a (verse)

        if abs(delta) < min_db:
            continue

        # Span covers the kind_a sections (where the contrast / lack thereof is felt).
        span_start = min(s.start_s for s in secs_a)
        span_end = max(s.end_s for s in secs_a)
        idxs = _note_indices_in_section(notes, span_start, span_end)

        if delta > 0:
            # Chorus louder than verse — affirming.
            title = f"Good dynamic lift into the {kind_b}"
            summary = (
                f"Your {kind_b}s average {delta:.1f} dB louder than your {kind_a}s "
                "— that contrast gives the song real energy."
            )
        else:
            # Verse louder or equal — coaching note.
            title = f"Try building more into the {kind_b}"
            summary = (
                f"Your {kind_a}s are actually {abs(delta):.1f} dB louder than "
                f"your {kind_b}s on average — a bit more volume in the {kind_b} "
                "would give it a stronger lift."
            )

        out.append(
            CoachingMoment(
                id=f"section_dynamic_contrast:{kind_a}_vs_{kind_b}",
                type="section_dynamic_contrast",
                scope="section",
                title=title,
                summary=summary,
                start_s=span_start,
                end_s=span_end,
                score=abs(delta),
                note_indices=idxs,
                section_names=[s.name for s in secs_a] + [s.name for s in secs_b],
                detail={
                    "kind_a": kind_a,
                    "kind_b": kind_b,
                    "rms_db_a": rms_a,
                    "rms_db_b": rms_b,
                    "delta_db": delta,
                },
            )
        )

    return out


# ---------------------------------------------------------------------------
# Sprint 2: Continuous-pitch detectors
# ---------------------------------------------------------------------------


def detect_scoop_habit(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Repeated upward approach into notes (scoop_cents > threshold)."""
    cfg = config.adsr
    threshold = cfg.scoop_min_cents
    user_scoops = [n for n in notes if n.scoop_cents is not None and n.scoop_cents > threshold]
    if len(user_scoops) < cfg.scoop_min_notes:
        return []
    ref_match = sum(
        1 for n in user_scoops
        if n.ref_scoop_cents is not None and n.ref_scoop_cents > threshold
    )
    ref_ratio = ref_match / len(user_scoops)
    mean_scoop = _mean([n.scoop_cents for n in user_scoops]) or 0.0
    if ref_ratio > 0.5:
        basis = "comparative"
        title = "Matching the artist's scoops"
        summary = (
            f"You approach {len(user_scoops)} notes with the same upward glide as the original "
            f"({mean_scoop:.0f}c on average) — a stylistic choice that fits the recording."
        )
    else:
        basis = "absolute"
        title = "Scooping into notes"
        summary = (
            f"You approach {len(user_scoops)} notes with a {mean_scoop:.0f}c upward glide before "
            f"landing on pitch — try attacking the target directly for a cleaner onset."
        )
    return [CoachingMoment(
        id="scoop_habit:0",
        type="scoop_habit",
        title=title,
        summary=summary,
        start_s=user_scoops[0].start_s,
        end_s=user_scoops[-1].end_s,
        score=mean_scoop / 100.0,
        note_indices=[n.note_index for n in user_scoops],
        detail={"mean_scoop_cents": round(mean_scoop, 1), "count": len(user_scoops),
                "feedback_basis_override": basis},
    )]


def detect_pitch_overshoot(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Repeated downward approach into notes (scoop_cents < -threshold)."""
    cfg = config.adsr
    threshold = cfg.scoop_min_cents
    overshoots = [n for n in notes if n.scoop_cents is not None and n.scoop_cents < -threshold]
    if len(overshoots) < cfg.scoop_min_notes:
        return []
    ref_match = sum(
        1 for n in overshoots
        if n.ref_scoop_cents is not None and n.ref_scoop_cents < -threshold
    )
    ref_ratio = ref_match / len(overshoots)
    mean_over = _mean([abs(n.scoop_cents) for n in overshoots]) or 0.0
    if ref_ratio > 0.5:
        basis = "comparative"
        title = "Matching the artist's downward approach"
        summary = (
            f"You approach {len(overshoots)} notes from above — the original does the same, "
            f"so this reads as a stylistic choice."
        )
    else:
        basis = "absolute"
        title = "Approaching notes from above"
        summary = (
            f"You overshoot {len(overshoots)} notes by {mean_over:.0f}c on average — "
            f"the pitch dips in before settling. Try landing on target from the start."
        )
    return [CoachingMoment(
        id="pitch_overshoot:0",
        type="pitch_overshoot",
        title=title,
        summary=summary,
        start_s=overshoots[0].start_s,
        end_s=overshoots[-1].end_s,
        score=mean_over / 100.0,
        note_indices=[n.note_index for n in overshoots],
        detail={"mean_overshoot_cents": round(mean_over, 1), "count": len(overshoots),
                "feedback_basis_override": basis},
    )]


def detect_clean_attack(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: consecutive notes with very small scoop_cents."""
    cfg = config.adsr
    max_abs = cfg.clean_attack_max_cents
    min_notes = cfg.clean_attack_min_notes
    # Collect runs of consecutive clean notes
    runs: list[list[NoteMeasurementV2]] = []
    current: list[NoteMeasurementV2] = []
    for n in notes:
        if n.scoop_cents is not None and abs(n.scoop_cents) <= max_abs:
            current.append(n)
        else:
            if len(current) >= min_notes:
                runs.append(current)
            current = []
    if len(current) >= min_notes:
        runs.append(current)
    if not runs:
        return []
    best = max(runs, key=len)
    mean_abs = _mean([abs(n.scoop_cents) for n in best]) or 0.0
    return [CoachingMoment(
        id="clean_attack:0",
        type="clean_attack",
        title="Clean, direct attacks",
        summary=(
            f"You land right on target for {len(best)} notes in a row with under {mean_abs:.0f}c "
            f"of approach — precise and confident."
        ),
        start_s=best[0].start_s,
        end_s=best[-1].end_s,
        score=float(len(best)) / max(1, len(notes)),
        note_indices=[n.note_index for n in best],
        detail={"mean_abs_scoop_cents": round(mean_abs, 1), "run_length": len(best)},
    )]


def detect_falling_release(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Notes where pitch drops sharply at release."""
    cfg = config.adsr
    threshold = cfg.release_slope_drop_threshold
    falling = [
        n for n in notes
        if n.release_pitch_slope_cents_per_s is not None
        and n.release_pitch_slope_cents_per_s < threshold
    ]
    if len(falling) < cfg.release_min_notes:
        return []
    ref_match = sum(
        1 for n in falling
        if n.ref_scoop_cents is not None  # no ref release slope field; skip ref comparison
    )
    mean_slope = _mean([n.release_pitch_slope_cents_per_s for n in falling]) or 0.0
    return [CoachingMoment(
        id="falling_release:0",
        type="falling_release",
        title="Pitch drops at note ends",
        summary=(
            f"Pitch falls at the end of {len(falling)} notes "
            f"({mean_slope:.0f} cents/s on average) — "
            f"try sustaining the target through the full note length."
        ),
        start_s=falling[0].start_s,
        end_s=falling[-1].end_s,
        score=abs(mean_slope) / 300.0,
        note_indices=[n.note_index for n in falling],
        detail={"mean_release_slope_cents_per_s": round(mean_slope, 1), "count": len(falling)},
    )]


def detect_steady_sustain(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: long notes with very low pitch std (stable sustain)."""
    cfg = config.adsr
    max_std = cfg.sustain_steady_max_pitch_std_cents
    min_dur = cfg.sustain_steady_min_duration_s
    steady = [
        n for n in notes
        if n.sustain_pitch_std_cents is not None
        and n.sustain_pitch_std_cents <= max_std
        and (n.end_s - n.start_s) >= min_dur
        and n.voiced_coverage >= 0.5
    ]
    if not steady:
        return []
    best = min(steady, key=lambda n: n.sustain_pitch_std_cents)
    return [CoachingMoment(
        id=f"steady_sustain:{best.note_index}",
        type="steady_sustain",
        title="Rock-solid sustain",
        summary=(
            f"Your pitch on '{best.lyric_word}' ({best.note_name}) stays within "
            f"{best.sustain_pitch_std_cents:.1f}c of centre — "
            f"impressively controlled."
        ),
        start_s=best.start_s,
        end_s=best.end_s,
        score=max(0.0, (max_std - best.sustain_pitch_std_cents) / max_std),
        note_indices=[best.note_index],
        detail={"sustain_pitch_std_cents": round(best.sustain_pitch_std_cents, 2),
                "note_name": best.note_name},
    )]


def detect_pitch_instability(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Notes with high pitch std but no vibrato detected."""
    cfg = config.adsr
    min_std = cfg.sustain_unstable_min_pitch_std_cents
    unstable = [
        n for n in notes
        if n.sustain_pitch_std_cents is not None
        and n.sustain_pitch_std_cents > min_std
        and n.vibrato_rate_hz is None  # not vibrato
        and n.voiced_coverage >= 0.4
    ]
    if not unstable:
        return []
    worst = max(unstable, key=lambda n: n.sustain_pitch_std_cents)
    mean_std = _mean([n.sustain_pitch_std_cents for n in unstable]) or 0.0
    return [CoachingMoment(
        id=f"pitch_instability:{worst.note_index}",
        type="pitch_instability",
        title="Unstable pitch on sustained notes",
        summary=(
            f"Pitch wobbles {mean_std:.0f}c on average during {len(unstable)} sustained note(s) — "
            f"check breath support and jaw tension to stabilise."
        ),
        start_s=unstable[0].start_s,
        end_s=unstable[-1].end_s,
        score=mean_std / 100.0,
        note_indices=[n.note_index for n in unstable],
        detail={"mean_sustain_pitch_std_cents": round(mean_std, 1), "count": len(unstable)},
    )]


def detect_vibrato_quality(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Aggregate vibrato characterisation across the performance."""
    cfg = config.adsr
    vib_notes = [n for n in notes if n.vibrato_rate_hz is not None]
    if not vib_notes:
        return []

    rates = [n.vibrato_rate_hz for n in vib_notes]
    extents = [n.vibrato_extent_cents for n in vib_notes if n.vibrato_extent_cents is not None]
    mean_rate = _mean(rates) or 0.0
    std_rate = float(__import__("statistics").pstdev([r for r in rates if r is not None]))
    mean_extent = _mean(extents) or 0.0

    ref_rates = [n.ref_vibrato_rate_hz for n in vib_notes if n.ref_vibrato_rate_hz is not None]
    ref_mean_rate = _mean(ref_rates)
    moments: list[CoachingMoment] = []

    # Consistent healthy vibrato (affirming)
    if (cfg.vibrato_healthy_min_hz <= mean_rate <= cfg.vibrato_healthy_max_hz
            and std_rate < cfg.vibrato_rate_consistency_hz):
        moments.append(CoachingMoment(
            id="consistent_vibrato:0",
            type="consistent_vibrato",
            title="Consistent, healthy vibrato",
            summary=(
                f"Your vibrato averages {mean_rate:.1f} Hz with very little variation — "
                f"right in the natural singing range."
            ),
            start_s=vib_notes[0].start_s,
            end_s=vib_notes[-1].end_s,
            score=0.9,
            note_indices=[n.note_index for n in vib_notes],
            detail={"mean_rate_hz": round(mean_rate, 2), "std_rate_hz": round(std_rate, 3)},
        ))
    # Wide vibrato
    if mean_extent > cfg.vibrato_wide_extent_cents:
        moments.append(CoachingMoment(
            id="wide_vibrato:0",
            type="wide_vibrato",
            title="Vibrato is quite wide",
            summary=(
                f"Your vibrato spans {mean_extent:.0f} cents peak-to-peak on average — "
                f"consider narrowing the oscillation for a more controlled sound."
            ),
            start_s=vib_notes[0].start_s,
            end_s=vib_notes[-1].end_s,
            score=mean_extent / 200.0,
            note_indices=[n.note_index for n in vib_notes],
            detail={"mean_extent_cents": round(mean_extent, 1)},
        ))
    # Wobble (too slow)
    if mean_rate < cfg.vibrato_min_rate_hz:
        moments.append(CoachingMoment(
            id="vibrato_quality:wobble",
            type="vibrato_quality",
            title="Vibrato rate is slow (wobble)",
            summary=(
                f"Vibrato averages {mean_rate:.1f} Hz — slower than the natural 4–7 Hz range. "
                f"Focus on releasing tension in the throat to speed up the oscillation."
            ),
            start_s=vib_notes[0].start_s,
            end_s=vib_notes[-1].end_s,
            score=0.7,
            note_indices=[n.note_index for n in vib_notes],
            detail={"mean_rate_hz": round(mean_rate, 2)},
        ))
    # Bleat (too fast)
    elif mean_rate > cfg.vibrato_max_rate_hz:
        moments.append(CoachingMoment(
            id="vibrato_quality:bleat",
            type="vibrato_quality",
            title="Vibrato rate is fast (bleat)",
            summary=(
                f"Vibrato averages {mean_rate:.1f} Hz — faster than the comfortable 4–7 Hz range. "
                f"Ease the breath pressure slightly to slow it down."
            ),
            start_s=vib_notes[0].start_s,
            end_s=vib_notes[-1].end_s,
            score=0.65,
            note_indices=[n.note_index for n in vib_notes],
            detail={"mean_rate_hz": round(mean_rate, 2)},
        ))
    # Delayed vibrato (comparative: reference vibrato starts earlier)
    if ref_mean_rate is not None and ref_mean_rate > 0 and len(vib_notes) < len(notes) * 0.5:
        moments.append(CoachingMoment(
            id="delayed_vibrato:0",
            type="delayed_vibrato",
            title="Less vibrato than the original",
            summary=(
                f"The original uses vibrato more consistently — you're either using it "
                f"selectively or it's appearing later in sustained notes than on the reference."
            ),
            start_s=vib_notes[0].start_s,
            end_s=vib_notes[-1].end_s,
            score=0.5,
            note_indices=[n.note_index for n in vib_notes],
            detail={"user_vib_notes": len(vib_notes), "total_notes": len(notes),
                    "feedback_basis_override": "comparative"},
        ))
    # Straight tone control (no vibrato, but stable sustain)
    non_vib = [
        n for n in notes
        if n.vibrato_rate_hz is None
        and n.sustain_pitch_std_cents is not None
        and n.sustain_pitch_std_cents < 10.0
        and n.pct_in_tune is not None and n.pct_in_tune > 0.6
    ]
    if len(non_vib) >= 3 and len(vib_notes) < len(notes) * 0.3:
        ref_uses_vib = any(n.ref_vibrato_rate_hz is not None for n in non_vib)
        basis = "comparative" if ref_uses_vib else "absolute"
        moments.append(CoachingMoment(
            id="straight_tone_control:0",
            type="straight_tone_control",
            title=("Stylistic straight tone" if ref_uses_vib else "Clean straight tone"),
            summary=(
                (f"The original uses vibrato here, but you're delivering a steady straight tone — "
                 f"a clear artistic choice.")
                if ref_uses_vib else
                (f"You're holding {len(non_vib)} notes with a stable, vibrato-free tone — "
                 f"great control.")
            ),
            start_s=non_vib[0].start_s,
            end_s=non_vib[-1].end_s,
            score=float(len(non_vib)) / max(1, len(notes)),
            note_indices=[n.note_index for n in non_vib],
            detail={"count": len(non_vib), "feedback_basis_override": basis},
        ))
    return moments


# ---------------------------------------------------------------------------
# Sprint 2: Continuous-loudness detectors
# ---------------------------------------------------------------------------


def detect_breathy_onset(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Notes with slow/breathy attack onset."""
    cfg = config.adsr
    threshold = cfg.onset_breathy_threshold_db_per_s
    breathy = [
        n for n in notes
        if n.attack_rms_slope_db_per_s is not None
        and n.attack_rms_slope_db_per_s < threshold
        and n.attack_duration_s is not None
        and n.attack_duration_s >= cfg.min_attack_duration_s
        and n.voiced_coverage >= 0.3
    ]
    if not breathy:
        return []
    # Dual-basis: if reference also has slow attacks, it's a stylistic match
    ref_match = sum(
        1 for n in breathy
        if n.ref_attack_rms_slope_db_per_s is not None
        and n.ref_attack_rms_slope_db_per_s < threshold
    )
    ref_ratio = ref_match / len(breathy) if breathy else 0.0
    mean_slope = _mean([n.attack_rms_slope_db_per_s for n in breathy]) or 0.0
    if ref_ratio > 0.5:
        basis = "comparative"
        title = "Matching the breathy onset style"
        summary = (
            f"Your slow attacks ({mean_slope:.1f} dB/s) echo the original's airy onset — "
            f"a deliberate textural choice."
        )
    else:
        basis = "absolute"
        title = "Breathy or slow note onsets"
        summary = (
            f"Attacks on {len(breathy)} note(s) build slowly ({mean_slope:.1f} dB/s) — "
            f"engage your breath support earlier for a cleaner onset."
        )
    return [CoachingMoment(
        id="breathy_onset:0",
        type="breathy_onset",
        title=title,
        summary=summary,
        start_s=breathy[0].start_s,
        end_s=breathy[-1].end_s,
        score=abs(mean_slope - threshold) / 20.0,
        note_indices=[n.note_index for n in breathy],
        detail={"mean_attack_rms_slope": round(mean_slope, 2), "count": len(breathy),
                "feedback_basis_override": basis},
    )]


def detect_clean_onset(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: notes with fast, clean attack onset."""
    cfg = config.adsr
    threshold = cfg.onset_clean_threshold_db_per_s
    clean = [
        n for n in notes
        if n.attack_rms_slope_db_per_s is not None
        and n.attack_rms_slope_db_per_s > threshold
        and n.attack_duration_s is not None
        and n.attack_duration_s >= cfg.min_attack_duration_s
    ]
    if len(clean) < 3:
        return []
    mean_slope = _mean([n.attack_rms_slope_db_per_s for n in clean]) or 0.0
    return [CoachingMoment(
        id="clean_onset:0",
        type="clean_onset",
        title="Crisp, decisive attacks",
        summary=(
            f"You hit {len(clean)} notes cleanly ({mean_slope:.0f} dB/s ramp) — "
            f"no hesitation, confident onset."
        ),
        start_s=clean[0].start_s,
        end_s=clean[-1].end_s,
        score=min(1.0, mean_slope / 50.0),
        note_indices=[n.note_index for n in clean],
        detail={"mean_attack_rms_slope": round(mean_slope, 1), "count": len(clean)},
    )]


def detect_envelope_moments(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Surface sforzando, crescendo, swell, note_crescendo, note_swell moments."""
    out: list[CoachingMoment] = []
    sforzando = [n for n in notes if n.envelope_shape == "sforzando" and n.voiced_coverage >= 0.3]
    if sforzando:
        ref_match = sum(1 for n in sforzando if n.ref_envelope_shape == "sforzando")
        basis = "comparative" if ref_match / len(sforzando) > 0.5 else "absolute"
        m = sforzando[0]
        out.append(CoachingMoment(
            id=f"sforzando_attack:{m.note_index}",
            type="sforzando_attack",
            title="Strong accent on the attack",
            summary=(
                f"You give '{m.lyric_word}' a sharp, accented onset that fades — "
                + ("matching the original's emphasis." if basis == "comparative"
                   else "a bold dynamic choice.")
            ),
            start_s=m.start_s, end_s=m.end_s,
            score=0.7,
            note_indices=[n.note_index for n in sforzando],
            detail={"count": len(sforzando), "feedback_basis_override": basis},
        ))
    crescendo = [
        n for n in notes
        if n.envelope_shape == "crescendo" and n.voiced_coverage >= 0.4
        and n.end_s - n.start_s >= 0.3
    ]
    if crescendo:
        ref_match = sum(1 for n in crescendo if n.ref_envelope_shape == "crescendo")
        basis = "comparative" if ref_match / len(crescendo) > 0.5 else "absolute"
        m = max(crescendo, key=lambda n: n.end_s - n.start_s)
        out.append(CoachingMoment(
            id=f"note_crescendo:{m.note_index}",
            type="note_crescendo",
            title="Building volume through the note",
            summary=(
                f"You grow in volume through '{m.lyric_word}' — "
                + ("echoing the original's shape." if basis == "comparative"
                   else "an expressive dynamic shaping.")
            ),
            start_s=m.start_s, end_s=m.end_s,
            score=0.6,
            note_indices=[m.note_index],
            detail={"feedback_basis_override": basis},
        ))
    swells = [
        n for n in notes
        if n.envelope_shape == "swell" and n.voiced_coverage >= 0.4
        and n.end_s - n.start_s >= 0.4
    ]
    if swells:
        ref_match = sum(1 for n in swells if n.ref_envelope_shape == "swell")
        basis = "comparative" if ref_match / len(swells) > 0.5 else "absolute"
        m = max(swells, key=lambda n: n.end_s - n.start_s)
        out.append(CoachingMoment(
            id=f"note_swell:{m.note_index}",
            type="note_swell",
            title="Swell within the note",
            summary=(
                f"You build then taper on '{m.lyric_word}' — "
                + ("the original does the same." if basis == "comparative"
                   else "a musical, shaped phrase.")
            ),
            start_s=m.start_s, end_s=m.end_s,
            score=0.6,
            note_indices=[m.note_index],
            detail={"feedback_basis_override": basis},
        ))
    return out


def detect_support_fade(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Notes where RMS drops significantly in the sustain window."""
    cfg = config.adsr
    fading = [
        n for n in notes
        if n.sustain_rms_slope_db_per_s is not None
        and n.sustain_rms_slope_db_per_s < cfg.support_fade_slope_db_per_s
        and n.sustain_rms_std_db is not None
        and n.sustain_rms_std_db > cfg.support_fade_std_db
        and n.voiced_coverage >= 0.4
    ]
    if len(fading) < 2:
        return []
    mean_slope = _mean([n.sustain_rms_slope_db_per_s for n in fading]) or 0.0
    return [CoachingMoment(
        id="support_fade:0",
        type="support_fade",
        title="Breath support drops mid-note",
        summary=(
            f"Volume falls {abs(mean_slope):.1f} dB/s through the held portion of "
            f"{len(fading)} note(s) — keep a steady breath column through the full length."
        ),
        start_s=fading[0].start_s,
        end_s=fading[-1].end_s,
        score=abs(mean_slope) / 10.0,
        note_indices=[n.note_index for n in fading],
        detail={"mean_sustain_slope_db_per_s": round(mean_slope, 2), "count": len(fading)},
    )]


def detect_dynamic_sustain(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: long notes with very steady volume during sustain."""
    cfg = config.adsr
    steady = [
        n for n in notes
        if n.sustain_rms_std_db is not None
        and n.sustain_rms_std_db <= cfg.dynamic_sustain_max_std_db
        and n.sustain_rms_slope_db_per_s is not None
        and abs(n.sustain_rms_slope_db_per_s) <= cfg.dynamic_sustain_max_slope_db_per_s
        and (n.end_s - n.start_s) >= 0.4
        and n.voiced_coverage >= 0.5
    ]
    if not steady:
        return []
    best = min(steady, key=lambda n: n.sustain_rms_std_db)
    return [CoachingMoment(
        id=f"dynamic_sustain:{best.note_index}",
        type="dynamic_sustain",
        title="Perfectly steady volume",
        summary=(
            f"'{best.lyric_word}' holds within {best.sustain_rms_std_db:.1f} dB throughout — "
            f"solid breath support."
        ),
        start_s=best.start_s, end_s=best.end_s,
        score=max(0.0, 1.0 - best.sustain_rms_std_db),
        note_indices=[best.note_index],
        detail={"sustain_rms_std_db": round(best.sustain_rms_std_db, 2)},
    )]


def detect_release_cutoff(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Notes that end too abruptly (steep negative release RMS slope)."""
    cutoffs = [
        n for n in notes
        if n.release_rms_slope_db_per_s is not None
        and n.release_rms_slope_db_per_s < -30.0
        and n.voiced_coverage >= 0.4
    ]
    if len(cutoffs) < 2:
        return []
    mean_slope = _mean([n.release_rms_slope_db_per_s for n in cutoffs]) or 0.0
    return [CoachingMoment(
        id="release_cutoff:0",
        type="release_cutoff",
        title="Abrupt note endings",
        summary=(
            f"{len(cutoffs)} note(s) end with a sharp cutoff ({mean_slope:.0f} dB/s) — "
            f"try tapering off more gradually for a smoother release."
        ),
        start_s=cutoffs[0].start_s,
        end_s=cutoffs[-1].end_s,
        score=abs(mean_slope) / 60.0,
        note_indices=[n.note_index for n in cutoffs],
        detail={"mean_release_slope_db_per_s": round(mean_slope, 1), "count": len(cutoffs)},
    )]


# ---------------------------------------------------------------------------
# Sprint 2: Cross-dimensional detectors
# ---------------------------------------------------------------------------


def detect_breath_support_issues(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Co-occurring pitch flatness + sustain fade → breath support diagnosis."""
    cfg = config.adsr
    qualifying: list[NoteMeasurementV2] = []
    for n in notes:
        flat = n.median_cents is not None and n.median_cents < cfg.breath_support_flat_threshold_cents
        fading = (n.sustain_rms_slope_db_per_s is not None
                  and n.sustain_rms_slope_db_per_s < cfg.breath_support_fade_threshold_db_per_s)
        if flat and fading:
            qualifying.append(n)
    if len(qualifying) < cfg.breath_support_min_notes:
        return None
    # Surface the longest consecutive run
    best_run: list[NoteMeasurementV2] = []
    current_run: list[NoteMeasurementV2] = []
    q_set = {n.note_index for n in qualifying}
    for n in notes:
        if n.note_index in q_set:
            current_run.append(n)
            if len(current_run) > len(best_run):
                best_run = list(current_run)
        else:
            current_run = []
    if not best_run:
        best_run = qualifying
    mean_cents = _mean([n.median_cents for n in best_run]) or 0.0
    return CoachingMoment(
        id="breath_support_issue:0",
        type="breath_support_issue",
        title="Breath support dropping here",
        summary=(
            f"This phrase goes {abs(mean_cents):.0f}c flat as your volume fades — "
            f"focus on sustaining breath pressure through the end of each note."
        ),
        start_s=best_run[0].start_s,
        end_s=best_run[-1].end_s,
        score=abs(mean_cents) / 100.0,
        note_indices=[n.note_index for n in best_run],
        detail={"mean_cents": round(mean_cents, 1), "run_length": len(best_run)},
    )


def detect_registration_strain(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> Optional[CoachingMoment]:
    """Sharp pitch + loud + high note → likely registration/chest-push strain."""
    cfg = config.adsr
    strained = [
        n for n in notes
        if n.midi_pitch >= cfg.passaggio_midi
        and n.median_cents is not None
        and n.median_cents > cfg.registration_sharp_threshold_cents
        and n.user_rms_relative_db is not None
        and n.user_rms_relative_db > cfg.registration_loud_threshold_db
    ]
    if not strained:
        return None
    worst = max(strained, key=lambda n: n.median_cents)
    return CoachingMoment(
        id=f"registration_strain:{worst.note_index}",
        type="registration_strain",
        title=f"Pushing on '{worst.lyric_word}'",
        summary=(
            f"This {worst.note_name} sits {worst.median_cents:.0f}c sharp while you're singing loud — "
            f"try easing back the pressure or shifting into a lighter registration."
        ),
        start_s=worst.start_s,
        end_s=worst.end_s,
        score=worst.median_cents / 100.0,
        note_indices=[worst.note_index],
        detail={"median_cents": round(worst.median_cents, 1),
                "user_rms_relative_db": round(worst.user_rms_relative_db, 1),
                "note_name": worst.note_name},
    )


def detect_controlled_crescendo(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Volume growing while pitch accuracy stays high — affirming."""
    cfg = config.adsr
    hcfg = config.highlights
    best: Optional[tuple[float, int, int]] = None
    best_score = -1.0
    for start, end in _windowed_indices(len(notes), hcfg.loudness_dynamic_window_min,
                                         hcfg.loudness_dynamic_window_max):
        window = notes[start:end]
        mean_delta = _mean([n.rms_delta_db for n in window])
        mean_pct = _mean([n.pct_in_tune for n in window])
        if mean_delta is None or mean_pct is None:
            continue
        if (mean_delta > cfg.controlled_crescendo_delta_db
                and mean_pct > cfg.controlled_crescendo_pct_in_tune):
            sc = mean_delta + mean_pct
            if sc > best_score:
                best_score = sc
                best = (mean_delta, start, end)
    if best is None:
        return []
    mean_delta, s, e = best
    window_notes = notes[s:e]
    mean_pct = _mean([n.pct_in_tune for n in window_notes]) or 0.0
    return [CoachingMoment(
        id="controlled_crescendo:0",
        type="controlled_crescendo",
        title="Controlled volume push",
        summary=(
            f"You built {mean_delta:.1f} dB more energy here while staying "
            f"{mean_pct * 100:.0f}% in tune — real dynamic control."
        ),
        start_s=window_notes[0].start_s,
        end_s=window_notes[-1].end_s,
        score=best_score / 3.0,
        note_indices=[n.note_index for n in window_notes],
        detail={"mean_rms_delta_db": round(mean_delta, 2), "mean_pct_in_tune": round(mean_pct, 3)},
    )]


def detect_loud_pitch_instability(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Pitch accuracy drops on loud notes."""
    cfg = config.adsr
    rms_vals = [n.user_rms_db for n in notes if n.user_rms_db is not None]
    if not rms_vals:
        return []
    import statistics
    q_val = statistics.quantiles(rms_vals, n=4)[2]  # 75th percentile
    loud_inaccurate = [
        n for n in notes
        if n.user_rms_db is not None
        and n.user_rms_db >= q_val
        and n.pct_in_tune is not None
        and n.pct_in_tune < cfg.loud_instability_max_pct_in_tune
    ]
    if not loud_inaccurate:
        return []
    mean_pct = _mean([n.pct_in_tune for n in loud_inaccurate]) or 0.0
    return [CoachingMoment(
        id="loud_pitch_instability:0",
        type="loud_pitch_instability",
        title="Pitch slips when you sing loud",
        summary=(
            f"On your {len(loud_inaccurate)} loudest note(s), pitch accuracy drops to "
            f"{mean_pct * 100:.0f}% — try maintaining breath support as you push volume."
        ),
        start_s=loud_inaccurate[0].start_s,
        end_s=loud_inaccurate[-1].end_s,
        score=1.0 - mean_pct,
        note_indices=[n.note_index for n in loud_inaccurate],
        detail={"mean_pct_in_tune": round(mean_pct, 3), "count": len(loud_inaccurate)},
    )]


def detect_soft_passage_control(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: quiet notes with high pitch accuracy."""
    cfg = config.adsr
    rms_vals = [n.user_rms_db for n in notes if n.user_rms_db is not None]
    if not rms_vals:
        return []
    import statistics
    try:
        q_val = statistics.quantiles(rms_vals, n=4)[0]  # 25th percentile
    except statistics.StatisticsError:
        return []
    soft_accurate = [
        n for n in notes
        if n.user_rms_db is not None
        and n.user_rms_db <= q_val
        and n.pct_in_tune is not None
        and n.pct_in_tune > cfg.soft_control_min_pct_in_tune
    ]
    if not soft_accurate:
        return []
    mean_pct = _mean([n.pct_in_tune for n in soft_accurate]) or 0.0
    return [CoachingMoment(
        id="soft_passage_control:0",
        type="soft_passage_control",
        title="Accurate even when singing softly",
        summary=(
            f"You stay {mean_pct * 100:.0f}% in tune across your quietest {len(soft_accurate)} note(s) — "
            f"great control at low dynamic."
        ),
        start_s=soft_accurate[0].start_s,
        end_s=soft_accurate[-1].end_s,
        score=mean_pct,
        note_indices=[n.note_index for n in soft_accurate],
        detail={"mean_pct_in_tune": round(mean_pct, 3), "count": len(soft_accurate)},
    )]


def detect_vibrato_with_support(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: vibrato notes with steady volume support."""
    cfg = config.adsr
    supported = [
        n for n in notes
        if n.vibrato_rate_hz is not None
        and n.sustain_rms_std_db is not None
        and n.sustain_rms_std_db <= cfg.vibrato_with_support_max_rms_std
    ]
    if len(supported) < 2:
        return []
    mean_rate = _mean([n.vibrato_rate_hz for n in supported]) or 0.0
    return [CoachingMoment(
        id="vibrato_with_support:0",
        type="vibrato_with_support",
        title="Vibrato supported by steady breath",
        summary=(
            f"Your vibrato ({mean_rate:.1f} Hz) sits on a steady breath column — "
            f"the two are working together nicely."
        ),
        start_s=supported[0].start_s,
        end_s=supported[-1].end_s,
        score=0.8,
        note_indices=[n.note_index for n in supported],
        detail={"count": len(supported), "mean_rate_hz": round(mean_rate, 2)},
    )]


def detect_scoop_with_fade(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Notes that both scoop into pitch and fade in volume."""
    cfg = config.adsr
    compound = [
        n for n in notes
        if n.scoop_cents is not None and n.scoop_cents > cfg.scoop_min_cents
        and n.sustain_rms_slope_db_per_s is not None
        and n.sustain_rms_slope_db_per_s < cfg.scoop_with_fade_fade_threshold_db_per_s
    ]
    if not compound:
        return []
    mean_scoop = _mean([n.scoop_cents for n in compound]) or 0.0
    return [CoachingMoment(
        id="scoop_with_fade:0",
        type="scoop_with_fade",
        title="Approaching and fading on notes",
        summary=(
            f"{len(compound)} note(s) scoop into pitch ({mean_scoop:.0f}c) and then lose volume — "
            f"a compound onset issue. Focus on direct attack and sustained support together."
        ),
        start_s=compound[0].start_s,
        end_s=compound[-1].end_s,
        score=mean_scoop / 80.0,
        note_indices=[n.note_index for n in compound],
        detail={"mean_scoop_cents": round(mean_scoop, 1), "count": len(compound)},
    )]


def detect_technique_accuracy_tradeoff(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Expression peaks while pitch accuracy drops."""
    cfg = config.adsr
    hcfg = config.highlights
    tech_by_idx = {t.note_index: t for t in techniques}
    best: Optional[tuple[float, int, int]] = None
    best_score = -1.0
    for start, end in _windowed_indices(len(notes), hcfg.window_min, hcfg.window_max):
        window = notes[start:end]
        n_with_tech = sum(
            1 for n in window
            if n.note_index in tech_by_idx
            and len(tech_by_idx[n.note_index].user_techniques) > 0
        )
        density = n_with_tech / max(1, len(window))
        mean_pct = _mean([n.pct_in_tune for n in window])
        if mean_pct is None:
            continue
        if (density > cfg.technique_tradeoff_min_density
                and mean_pct < cfg.technique_tradeoff_max_pct_in_tune):
            sc = density - mean_pct
            if sc > best_score:
                best_score = sc
                best = (mean_pct, start, end)
    if best is None:
        return []
    mean_pct, s, e = best
    window_notes = notes[s:e]
    return [CoachingMoment(
        id="technique_accuracy_tradeoff:0",
        type="technique_accuracy_tradeoff",
        title="Expression at the expense of pitch",
        summary=(
            f"Lots of expression here, but pitch drops to {mean_pct * 100:.0f}% — "
            f"try nailing the notes first, then adding texture."
        ),
        start_s=window_notes[0].start_s,
        end_s=window_notes[-1].end_s,
        score=best_score,
        note_indices=[n.note_index for n in window_notes],
        detail={"mean_pct_in_tune": round(mean_pct, 3)},
    )]


def detect_expressive_stability(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: expression matches reference while pitch accuracy stays high."""
    cfg = config.adsr
    hcfg = config.highlights
    tech_by_idx = {t.note_index: t for t in techniques}
    best: Optional[tuple[float, float, int, int]] = None
    best_score = -1.0
    for start, end in _windowed_indices(len(notes), hcfg.window_min, hcfg.window_max):
        window = notes[start:end]
        n_with_tech = sum(
            1 for n in window
            if n.note_index in tech_by_idx
            and len(tech_by_idx[n.note_index].user_techniques) > 0
        )
        density = n_with_tech / max(1, len(window))
        mean_pct = _mean([n.pct_in_tune for n in window])
        if mean_pct is None:
            continue
        if (density > cfg.expressive_stability_min_density
                and mean_pct > cfg.expressive_stability_min_pct_in_tune):
            sc = density + mean_pct
            if sc > best_score:
                best_score = sc
                best = (density, mean_pct, start, end)
    if best is None:
        return []
    density, mean_pct, s, e = best
    window_notes = notes[s:e]
    return [CoachingMoment(
        id="expressive_stability:0",
        type="expressive_stability",
        title="Expressive and in tune",
        summary=(
            f"You're adding {density * 100:.0f}% technique density here while staying "
            f"{mean_pct * 100:.0f}% in tune — expression without sacrificing pitch."
        ),
        start_s=window_notes[0].start_s,
        end_s=window_notes[-1].end_s,
        score=best_score / 2.0,
        note_indices=[n.note_index for n in window_notes],
        detail={"technique_density": round(density, 3), "mean_pct_in_tune": round(mean_pct, 3)},
    )]


def detect_high_note_control(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: high notes (>= passaggio) with good pitch and clean attack."""
    cfg = config.adsr
    controlled = [
        n for n in notes
        if n.midi_pitch >= cfg.passaggio_midi
        and n.pct_in_tune is not None
        and n.pct_in_tune > cfg.high_note_min_pct_in_tune
        and (n.scoop_cents is None or abs(n.scoop_cents) <= cfg.high_note_max_abs_scoop_cents)
    ]
    if not controlled:
        return []
    best = max(controlled, key=lambda n: n.pct_in_tune)
    return [CoachingMoment(
        id=f"high_note_control:{best.note_index}",
        type="high_note_control",
        title=f"Controlled on the high note",
        summary=(
            f"You hit {best.note_name} ('{best.lyric_word}') cleanly — "
            f"{best.pct_in_tune * 100:.0f}% in tune with a direct attack."
        ),
        start_s=best.start_s,
        end_s=best.end_s,
        score=best.pct_in_tune,
        note_indices=[best.note_index],
        detail={"note_name": best.note_name, "pct_in_tune": round(best.pct_in_tune, 3)},
    )]


# ---------------------------------------------------------------------------
# Sprint 2: Phrase-timing and phrase-pitch detectors
# ---------------------------------------------------------------------------


def detect_phrase_timing_bias(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Windows with consistent directional timing bias (rushed or dragged)."""
    cfg = config.adsr
    hcfg = config.highlights
    out: list[CoachingMoment] = []

    # Track best rushed and dragged separately
    best_rushed: Optional[tuple[float, int, int]] = None
    best_dragged: Optional[tuple[float, int, int]] = None

    for start, end in _windowed_indices(len(notes), hcfg.timing_consistency_window_min,
                                         hcfg.timing_consistency_window_max):
        window = notes[start:end]
        offsets = [n.arrival_offset_ms for n in window if n.arrival_offset_ms is not None]
        if len(offsets) < 3:
            continue
        import statistics
        mean_off = _mean(offsets) or 0.0
        std_off = statistics.pstdev(offsets)
        if std_off >= cfg.phrase_timing_bias_std_ms:
            continue  # too variable — timing_consistency handles this
        if mean_off < cfg.rushed_phrase_mean_ms:
            sc = abs(mean_off)
            if best_rushed is None or sc > best_rushed[0]:
                best_rushed = (sc, start, end)
        elif mean_off > cfg.dragged_phrase_mean_ms:
            sc = mean_off
            if best_dragged is None or sc > best_dragged[0]:
                best_dragged = (sc, start, end)

    if best_rushed:
        sc, s, e = best_rushed
        w = notes[s:e]
        mean_off = _mean([n.arrival_offset_ms for n in w if n.arrival_offset_ms is not None]) or 0.0
        out.append(CoachingMoment(
            id="rushed_phrase:0",
            type="rushed_phrase",
            title="Rushing through this phrase",
            summary=(
                f"You're consistently {abs(mean_off):.0f}ms early across this phrase — "
                f"try relaxing into the beat."
            ),
            start_s=w[0].start_s, end_s=w[-1].end_s,
            score=sc / 100.0,
            note_indices=[n.note_index for n in w],
            detail={"mean_arrival_offset_ms": round(mean_off, 1)},
        ))
    if best_dragged:
        sc, s, e = best_dragged
        w = notes[s:e]
        mean_off = _mean([n.arrival_offset_ms for n in w if n.arrival_offset_ms is not None]) or 0.0
        out.append(CoachingMoment(
            id="dragged_phrase:0",
            type="dragged_phrase",
            title="Dragging behind the beat",
            summary=(
                f"You're consistently {mean_off:.0f}ms late across this phrase — "
                f"push slightly forward in the bar."
            ),
            start_s=w[0].start_s, end_s=w[-1].end_s,
            score=sc / 100.0,
            note_indices=[n.note_index for n in w],
            detail={"mean_arrival_offset_ms": round(mean_off, 1)},
        ))
    return out


def detect_rhythmic_precision(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Affirming: tight timing window."""
    cfg = config.adsr
    hcfg = config.highlights
    best: Optional[tuple[float, int, int]] = None
    best_mean = float("inf")
    for start, end in _windowed_indices(len(notes), hcfg.timing_consistency_window_min,
                                         hcfg.timing_consistency_window_max):
        window = notes[start:end]
        offsets = [abs(n.arrival_offset_ms) for n in window if n.arrival_offset_ms is not None]
        if len(offsets) < 3:
            continue
        mean_abs = _mean(offsets) or 0.0
        if mean_abs < cfg.rhythmic_precision_mean_ms and mean_abs < best_mean:
            best_mean = mean_abs
            best = (mean_abs, start, end)
    if best is None:
        return []
    mean_abs, s, e = best
    w = notes[s:e]
    return [CoachingMoment(
        id="rhythmic_precision:0",
        type="rhythmic_precision",
        title="Locked to the beat",
        summary=(
            f"Entrances here land within {mean_abs:.0f}ms on average — precise timing."
        ),
        start_s=w[0].start_s, end_s=w[-1].end_s,
        score=max(0.0, 1.0 - mean_abs / cfg.rhythmic_precision_mean_ms),
        note_indices=[n.note_index for n in w],
        detail={"mean_abs_offset_ms": round(mean_abs, 1)},
    )]


def detect_phrase_pitch_arc(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Pitch accuracy degrades from first half to second half of a phrase."""
    cfg = config.adsr
    hcfg = config.highlights
    best: Optional[tuple[float, int, int]] = None
    best_drop = 0.0
    for start, end in _windowed_indices(len(notes), hcfg.window_min * 2, hcfg.window_max * 2):
        window = notes[start:end]
        half = len(window) // 2
        first_pct = _mean([n.pct_in_tune for n in window[:half]])
        second_pct = _mean([n.pct_in_tune for n in window[half:]])
        if first_pct is None or second_pct is None:
            continue
        drop = first_pct - second_pct
        if drop > cfg.phrase_arc_drop_threshold and drop > best_drop:
            best_drop = drop
            best = (drop, start, end)
    if best is None:
        return []
    drop, s, e = best
    w = notes[s:e]
    half = len(w) // 2
    first_pct = _mean([n.pct_in_tune for n in w[:half]]) or 0.0
    second_pct = _mean([n.pct_in_tune for n in w[half:]]) or 0.0
    return [CoachingMoment(
        id="phrase_pitch_arc:0",
        type="phrase_pitch_arc",
        title="Pitch fades through the phrase",
        summary=(
            f"You start this phrase at {first_pct * 100:.0f}% in tune but slip to "
            f"{second_pct * 100:.0f}% by the end — sustain breath support all the way through."
        ),
        start_s=w[0].start_s, end_s=w[-1].end_s,
        score=drop,
        note_indices=[n.note_index for n in w],
        detail={"first_half_pct": round(first_pct, 3), "second_half_pct": round(second_pct, 3)},
    )]


# ---------------------------------------------------------------------------
# Sprint 2: Section-level detectors
# ---------------------------------------------------------------------------


def detect_section_improvement(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Same-kind section shows better pct_in_tune on later occurrence."""
    cfg = config.adsr
    from collections import defaultdict
    by_kind: dict[str, list[SectionTrend]] = defaultdict(list)
    for s in sections:
        if s.kind:
            by_kind[s.kind].append(s)
    out: list[CoachingMoment] = []
    for kind, sec_list in by_kind.items():
        if len(sec_list) < 2:
            continue
        sec_list = sorted(sec_list, key=lambda s: s.start_s)
        for i in range(len(sec_list) - 1):
            a, b = sec_list[i], sec_list[i + 1]
            if a.pct_in_tune is None or b.pct_in_tune is None:
                continue
            delta = b.pct_in_tune - a.pct_in_tune
            if delta >= cfg.section_improvement_min_delta:
                note_indices = [n.note_index for n in notes
                                if n.start_s >= b.start_s and n.end_s <= b.end_s]
                out.append(CoachingMoment(
                    id=f"section_improvement:{b.name}",
                    type="section_improvement",
                    scope="section",
                    title=f"Improving in the {kind}",
                    summary=(
                        f"Your later {kind} is {delta * 100:.0f}% more accurate than the first — "
                        f"you're locking in as the song goes on."
                    ),
                    start_s=b.start_s, end_s=b.end_s,
                    score=delta,
                    note_indices=note_indices,
                    section_names=[b.name],
                    detail={"delta_pct": round(delta, 3), "kind": kind},
                ))
            elif -delta >= cfg.section_improvement_min_delta:
                note_indices = [n.note_index for n in notes
                                if n.start_s >= b.start_s and n.end_s <= b.end_s]
                out.append(CoachingMoment(
                    id=f"section_regression:{b.name}",
                    type="section_regression",
                    scope="section",
                    title=f"Dropping off in the later {kind}",
                    summary=(
                        f"Your later {kind} slips {abs(delta) * 100:.0f}% from the first — "
                        f"fatigue or focus: check what changes."
                    ),
                    start_s=b.start_s, end_s=b.end_s,
                    score=abs(delta),
                    note_indices=note_indices,
                    section_names=[b.name],
                    detail={"delta_pct": round(delta, 3), "kind": kind},
                ))
    return out


def detect_section_vibrato_contrast(
    reference: ReferenceAnnotation,
    sections: list[SectionTrend],
    notes: list[NoteMeasurementV2],
    *,
    config: CoachingConfig,
) -> list[CoachingMoment]:
    """Vibrato density differs across section kinds."""
    cfg = config.adsr
    from collections import defaultdict
    by_kind: dict[str, list[SectionTrend]] = defaultdict(list)
    for s in sections:
        if s.kind:
            by_kind[s.kind].append(s)

    # Compute vibrato density per kind from note data
    kind_density: dict[str, float] = {}
    for kind, sec_list in by_kind.items():
        kind_notes = [
            n for n in notes
            if any(s.start_s <= n.start_s < s.end_s for s in sec_list)
        ]
        if not kind_notes:
            continue
        vib_count = sum(1 for n in kind_notes if n.vibrato_rate_hz is not None)
        kind_density[kind] = vib_count / len(kind_notes)

    out: list[CoachingMoment] = []
    kinds = sorted(kind_density.keys(), key=lambda k: kind_density[k])
    if len(kinds) < 2:
        return []
    low_kind, high_kind = kinds[0], kinds[-1]
    gap = kind_density[high_kind] - kind_density[low_kind]
    if gap < cfg.section_vibrato_min_density_gap:
        return []
    out.append(CoachingMoment(
        id="section_vibrato_contrast:0",
        type="section_vibrato_contrast",
        scope="section",
        title=f"More vibrato in the {high_kind}",
        summary=(
            f"You use vibrato on {kind_density[high_kind] * 100:.0f}% of notes in the {high_kind} "
            f"vs {kind_density[low_kind] * 100:.0f}% in the {low_kind} — a deliberate contrast."
        ),
        start_s=sections[0].start_s, end_s=sections[-1].end_s,
        score=gap,
        note_indices=[],
        section_names=list({s.name for s in sections}),
        detail={"high_kind": high_kind, "low_kind": low_kind,
                "high_density": round(kind_density[high_kind], 3),
                "low_density": round(kind_density[low_kind], 3)},
    ))
    return out


# ---------------------------------------------------------------------------
# Top-level selection — diversity-aware round-robin
# ---------------------------------------------------------------------------

MOMENT_CATEGORY: dict[str, str] = {
    # Existing
    "best_pitch_phrase": "pitch",
    "pitch_struggle": "pitch",
    "sharp_flat_note": "pitch",
    "section_strength": "pitch",
    "section_weakness": "pitch",
    "best_overall_section": "pitch",
    "weakest_overall_section": "pitch",
    "expressive_match": "technique",
    "expressive_moment": "technique",
    "missed_expression": "technique",
    "vocal_texture": "technique",
    "late_entrance": "alignment",
    "timing_consistency": "alignment",
    "section_delta": "alignment",
    "fade_within_notes": "dynamics",
    "dynamic_drop": "dynamics",
    "dynamic_surge": "dynamics",
    "section_dynamic_contrast": "dynamics",
    # Sprint 2: continuous pitch
    "scoop_habit": "pitch",
    "pitch_overshoot": "pitch",
    "clean_attack": "pitch",
    "falling_release": "pitch",
    "steady_sustain": "pitch",
    "pitch_instability": "pitch",
    "vibrato_quality": "technique",
    "consistent_vibrato": "technique",
    "wide_vibrato": "technique",
    "delayed_vibrato": "technique",
    "straight_tone_control": "technique",
    # Sprint 2: continuous loudness
    "breathy_onset": "dynamics",
    "clean_onset": "dynamics",
    "sforzando_attack": "dynamics",
    "note_crescendo": "dynamics",
    "note_swell": "dynamics",
    "support_fade": "dynamics",
    "dynamic_sustain": "dynamics",
    "release_cutoff": "dynamics",
    # Sprint 2: cross-dimensional
    "breath_support_issue": "pitch",
    "registration_strain": "pitch",
    "controlled_crescendo": "dynamics",
    "loud_pitch_instability": "pitch",
    "soft_passage_control": "pitch",
    "vibrato_with_support": "technique",
    "scoop_with_fade": "pitch",
    "technique_accuracy_tradeoff": "technique",
    "expressive_stability": "technique",
    "high_note_control": "pitch",
    # Sprint 2: phrase / section
    "rushed_phrase": "alignment",
    "dragged_phrase": "alignment",
    "rhythmic_precision": "alignment",
    "phrase_pitch_arc": "pitch",
    "section_improvement": "pitch",
    "section_regression": "pitch",
    "section_vibrato_contrast": "technique",
}

# "absolute"  = universal technique quality (pitch, timing, dynamics — any vocal coach would flag)
# "comparative" = reference style matching (expressive choices compared against this specific recording)
# Note: dual-basis detectors default to "absolute" here; individual moments can override
# by setting detail["feedback_basis_override"] which is applied in select_highlights().
MOMENT_FEEDBACK_BASIS: dict[str, str] = {
    # Existing
    "best_pitch_phrase":        "absolute",
    "pitch_struggle":           "absolute",
    "sharp_flat_note":          "absolute",
    "late_entrance":            "absolute",
    "timing_consistency":       "absolute",
    "section_delta":            "comparative",
    "fade_within_notes":        "absolute",
    "dynamic_drop":             "absolute",
    "dynamic_surge":            "absolute",
    "section_strength":         "absolute",
    "section_weakness":         "absolute",
    "best_overall_section":     "absolute",
    "weakest_overall_section":  "absolute",
    "section_dynamic_contrast": "absolute",
    "expressive_match":         "comparative",
    "expressive_moment":        "comparative",
    "missed_expression":        "comparative",
    "vocal_texture":            "comparative",
    # Sprint 2: continuous pitch (dual defaults to absolute)
    "scoop_habit":              "absolute",
    "pitch_overshoot":          "absolute",
    "clean_attack":             "absolute",
    "falling_release":          "absolute",
    "steady_sustain":           "absolute",
    "pitch_instability":        "absolute",
    "vibrato_quality":          "absolute",
    "consistent_vibrato":       "absolute",
    "wide_vibrato":             "absolute",
    "delayed_vibrato":          "comparative",
    "straight_tone_control":    "absolute",
    # Sprint 2: continuous loudness
    "breathy_onset":            "absolute",
    "clean_onset":              "absolute",
    "sforzando_attack":         "absolute",
    "note_crescendo":           "absolute",
    "note_swell":               "comparative",
    "support_fade":             "absolute",
    "dynamic_sustain":          "absolute",
    "release_cutoff":           "absolute",
    # Sprint 2: cross-dimensional
    "breath_support_issue":     "absolute",
    "registration_strain":      "absolute",
    "controlled_crescendo":     "comparative",
    "loud_pitch_instability":   "absolute",
    "soft_passage_control":     "absolute",
    "vibrato_with_support":     "absolute",
    "scoop_with_fade":          "absolute",
    "technique_accuracy_tradeoff": "comparative",
    "expressive_stability":     "comparative",
    "high_note_control":        "absolute",
    # Sprint 2: phrase / section
    "rushed_phrase":            "absolute",
    "dragged_phrase":           "absolute",
    "rhythmic_precision":       "absolute",
    "phrase_pitch_arc":         "absolute",
    "section_improvement":      "absolute",
    "section_regression":       "absolute",
    "section_vibrato_contrast": "comparative",
}

_CATEGORY_ORDER = ["pitch", "technique", "alignment", "dynamics"]


# ---------------------------------------------------------------------------
# Confidence evidence scoring
# ---------------------------------------------------------------------------


def _compute_evidence_strength(
    moment: CoachingMoment,
    notes_by_idx: dict[int, NoteMeasurementV2],
    techs_by_idx: dict[int, NoteTechniqueComparison],
) -> float:
    """Return a 0–1 evidence strength score for ``moment``.

    Absolute moments weight user voiced coverage and NanoPitch confidence.
    Comparative moments additionally penalise weak reference voiced coverage
    and incorporate the student's technique certainty.
    """
    note_rows = [notes_by_idx[i] for i in moment.note_indices if i in notes_by_idx]
    if not note_rows:
        # Section-scope moment without resolvable note rows → treat as high confidence.
        return 1.0

    mean_vc = _mean([n.voiced_coverage for n in note_rows]) or 0.0
    mean_mvc = _mean(
        [n.mean_voicing_confidence for n in note_rows if n.mean_voicing_confidence is not None]
    )
    duration_s = sum(n.end_s - n.start_s for n in note_rows)
    duration_factor = min(1.0, duration_s / 4.0)

    if moment.feedback_basis == "comparative":
        # Comparative: confidence is bounded by the weakest side (user or ref).
        mean_ref_vc = _mean(
            [n.ref_voiced_coverage for n in note_rows if n.ref_voiced_coverage is not None]
        )
        dual_voicing = min(mean_vc, mean_ref_vc) if mean_ref_vc is not None else mean_vc

        # Technique certainty from student scores for the primary technique.
        tech_certainty = dual_voicing  # fallback
        if moment.techniques:
            primary_tech = moment.techniques[0]
            scores: list[float] = []
            for i in moment.note_indices:
                tc = techs_by_idx.get(i)
                if tc and tc.user_technique_scores:
                    raw = tc.user_technique_scores.get(primary_tech)
                    if raw is not None:
                        # For missed_expression the user should NOT have the technique;
                        # high student score → less certain the note was truly absent.
                        scores.append(1.0 - raw if moment.type == "missed_expression" else raw)
            if scores:
                tech_certainty = _mean(scores) or dual_voicing

        return min(1.0, 0.5 * dual_voicing + 0.3 * tech_certainty + 0.2 * duration_factor)
    else:
        # Absolute: user-only voicing + NanoPitch confidence + duration.
        voicing_conf = mean_mvc if mean_mvc is not None else mean_vc

        tech_certainty = mean_vc  # fallback
        if moment.techniques:
            primary_tech = moment.techniques[0]
            scores = []
            for i in moment.note_indices:
                tc = techs_by_idx.get(i)
                if tc and tc.user_technique_scores:
                    raw = tc.user_technique_scores.get(primary_tech)
                    if raw is not None:
                        scores.append(raw)
            if scores:
                tech_certainty = _mean(scores) or mean_vc

        return min(1.0, 0.4 * mean_vc + 0.3 * voicing_conf + 0.2 * duration_factor + 0.1 * tech_certainty)


def _tier_from_strength(strength: float, low: float, medium: float) -> str:
    """Map a 0–1 evidence strength value to a confidence tier string."""
    if strength >= medium:
        return "high"
    if strength >= low:
        return "medium"
    return "low"


def _select_diverse(
    moments: list[CoachingMoment],
    *,
    cap: int,
    max_per_type: int,
    max_per_category: int,
    suppress_low: bool = True,
) -> list[CoachingMoment]:
    """Pick moments round-robin across categories for maximum variety.

    Within each category the best-scored candidate is taken first, but the
    algorithm cycles through *all* categories before returning to any one,
    guaranteeing that every category with candidates gets representation
    before any category gets a second slot.

    Low-confidence moments are dropped before the round-robin when
    ``suppress_low`` is True (default).
    """
    if suppress_low:
        moments = [m for m in moments if m.confidence != "low"]

    queues: dict[str, list[CoachingMoment]] = {}
    for m in moments:
        cat = MOMENT_CATEGORY.get(m.type, "other")
        queues.setdefault(cat, []).append(m)
    for cat in queues:
        queues[cat].sort(key=lambda m: m.score, reverse=True)

    cats = [c for c in _CATEGORY_ORDER if c in queues]
    for extra in queues:
        if extra not in cats:
            cats.append(extra)

    chosen: list[CoachingMoment] = []
    type_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}

    progress = True
    while len(chosen) < cap and progress:
        progress = False
        for cat in cats:
            if len(chosen) >= cap:
                break
            if cat_counts.get(cat, 0) >= max_per_category:
                continue
            queue = queues.get(cat, [])
            while queue:
                candidate = queue.pop(0)
                if type_counts.get(candidate.type, 0) >= max_per_type:
                    continue
                if candidate.scope == "local":
                    has_overlap = any(
                        m.type == candidate.type
                        and m.scope == "local"
                        and not (candidate.end_s <= m.start_s or candidate.start_s >= m.end_s)
                        for m in chosen
                    )
                    if has_overlap:
                        continue
                chosen.append(candidate)
                type_counts[candidate.type] = type_counts.get(candidate.type, 0) + 1
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                progress = True
                break

    return chosen


def select_highlights(
    reference: ReferenceAnnotation,
    notes: list[NoteMeasurementV2],
    techniques: list[NoteTechniqueComparison],
    *,
    config: Optional[CoachingConfig] = None,
    sections: Optional[list[SectionTrend]] = None,
    vocal_profile=None,  # Optional[VocalProfile] — imported lazily to avoid circular dep
) -> HighlightsReport:
    """Run every detector, then pick a diverse set via round-robin.

    Categories (pitch, technique, alignment, dynamics) are cycled so that
    every category with candidates gets at least one slot before any
    category gets a second.

    When *vocal_profile* is provided (a ``VocalProfile`` instance), candidate
    scores are multiplied by ``cfg.llm.emphasis_boost`` for detectors in
    ``emphasize_highlights`` and by ``cfg.llm.deemphasis_penalty`` for
    detectors in ``deemphasize_highlights`` before selection.
    """
    cfg = config or CoachingConfig()
    candidates: list[CoachingMoment] = []

    # Pitch phrase highlights
    candidates.extend(detect_best_pitch_phrases(reference, notes, config=cfg))
    candidates.extend(detect_pitch_struggles(reference, notes, config=cfg))

    # Single-note sharp/flat callouts
    candidates.extend(detect_sharp_flat_notes(reference, notes, config=cfg))

    # Entrance timing (multiple notes)
    candidates.extend(detect_entrance_timing_notes(reference, notes, config=cfg))

    # Timing-consistency phrase
    timing = detect_timing_consistency(reference, notes, config=cfg)
    if timing is not None:
        candidates.append(timing)

    # STARS expression detectors (each returns a list, one per technique)
    candidates.extend(detect_expressive_match(reference, notes, techniques, config=cfg))
    candidates.extend(detect_expressive_moment(reference, notes, techniques, config=cfg))
    candidates.extend(detect_missed_expression(reference, notes, techniques, config=cfg))

    # Per-technique vocal texture highlights
    candidates.extend(
        detect_vocal_texture_moments(reference, notes, techniques, config=cfg)
    )

    # Loudness / dynamics (local)
    fade = detect_fade_within_notes(reference, notes, config=cfg)
    if fade is not None:
        candidates.append(fade)
    drop = detect_dynamic_drop(reference, notes, config=cfg)
    if drop is not None:
        candidates.append(drop)
    surge = detect_dynamic_surge(reference, notes, config=cfg)
    if surge is not None:
        candidates.append(surge)

    # Section-scope candidates
    if sections:
        candidates.extend(detect_section_moments(
            reference, sections, notes, techniques, config=cfg,
        ))

    # ── Sprint 2: continuous pitch detectors ─────────────────────────────
    candidates.extend(detect_scoop_habit(reference, notes, config=cfg))
    candidates.extend(detect_pitch_overshoot(reference, notes, config=cfg))
    candidates.extend(detect_clean_attack(reference, notes, config=cfg))
    candidates.extend(detect_falling_release(reference, notes, config=cfg))
    candidates.extend(detect_steady_sustain(reference, notes, config=cfg))
    candidates.extend(detect_pitch_instability(reference, notes, config=cfg))
    candidates.extend(detect_vibrato_quality(reference, notes, config=cfg))

    # ── Sprint 2: continuous loudness detectors ───────────────────────────
    candidates.extend(detect_breathy_onset(reference, notes, config=cfg))
    candidates.extend(detect_clean_onset(reference, notes, config=cfg))
    candidates.extend(detect_envelope_moments(reference, notes, config=cfg))
    candidates.extend(detect_support_fade(reference, notes, config=cfg))
    candidates.extend(detect_dynamic_sustain(reference, notes, config=cfg))
    candidates.extend(detect_release_cutoff(reference, notes, config=cfg))

    # ── Sprint 2: cross-dimensional detectors ────────────────────────────
    breath = detect_breath_support_issues(reference, notes, config=cfg)
    if breath is not None:
        candidates.append(breath)
    strain = detect_registration_strain(reference, notes, config=cfg)
    if strain is not None:
        candidates.append(strain)
    candidates.extend(detect_controlled_crescendo(reference, notes, config=cfg))
    candidates.extend(detect_loud_pitch_instability(reference, notes, config=cfg))
    candidates.extend(detect_soft_passage_control(reference, notes, config=cfg))
    candidates.extend(detect_vibrato_with_support(reference, notes, config=cfg))
    candidates.extend(detect_scoop_with_fade(reference, notes, config=cfg))
    candidates.extend(
        detect_technique_accuracy_tradeoff(reference, notes, techniques, config=cfg)
    )
    candidates.extend(
        detect_expressive_stability(reference, notes, techniques, config=cfg)
    )
    candidates.extend(detect_high_note_control(reference, notes, config=cfg))

    # ── Sprint 2: phrase timing + pitch arc ───────────────────────────────
    candidates.extend(detect_phrase_timing_bias(reference, notes, config=cfg))
    candidates.extend(detect_rhythmic_precision(reference, notes, config=cfg))
    candidates.extend(detect_phrase_pitch_arc(reference, notes, config=cfg))

    # ── Sprint 2: section-level ───────────────────────────────────────────
    if sections:
        candidates.extend(detect_section_improvement(reference, sections, notes, config=cfg))
        candidates.extend(detect_section_vibrato_contrast(reference, sections, notes, config=cfg))

    # Stamp feedback_basis then compute evidence strength + confidence tier.
    # Dual-basis detectors may override via detail["feedback_basis_override"].
    notes_by_idx = {n.note_index: n for n in notes}
    techs_by_idx = {t.note_index: t for t in techniques}
    low_t = cfg.confidence.low_threshold
    med_t = cfg.confidence.medium_threshold
    for m in candidates:
        basis = m.detail.get("feedback_basis_override")
        m.feedback_basis = basis if basis else MOMENT_FEEDBACK_BASIS.get(m.type, "absolute")
        strength = _compute_evidence_strength(m, notes_by_idx, techs_by_idx)
        m.confidence = _tier_from_strength(strength, low_t, med_t)
        m.detail["evidence_strength"] = round(strength, 3)

    # ── Sprint 3: vocal profile emphasis weighting ────────────────────────
    if vocal_profile is not None:
        emphasize = set(getattr(vocal_profile, "emphasize_highlights", []))
        deemphasize = set(getattr(vocal_profile, "deemphasize_highlights", []))
        boost = getattr(cfg.llm, "emphasis_boost", 1.5)
        penalty = getattr(cfg.llm, "deemphasis_penalty", 0.5)
        for m in candidates:
            if m.type in emphasize:
                m.score *= boost
                # Store the override rationale in detail for UI/audit.
                notes_map = getattr(vocal_profile, "highlight_notes", {})
                if m.type in notes_map:
                    m.detail["profile_emphasis_note"] = notes_map[m.type]
            elif m.type in deemphasize:
                m.score *= penalty
                notes_map = getattr(vocal_profile, "highlight_notes", {})
                if m.type in notes_map:
                    m.detail["profile_deemphasis_note"] = notes_map[m.type]

    chosen = _select_diverse(
        candidates,
        cap=cfg.highlights.cap,
        max_per_type=cfg.highlights.max_per_type,
        max_per_category=cfg.highlights.max_per_category,
        suppress_low=cfg.confidence.suppress_low,
    )
    chosen.sort(key=lambda m: m.start_s)
    return HighlightsReport(moments=chosen, cap=cfg.highlights.cap)


__all__ = [
    "MOMENT_CATEGORY",
    "MOMENT_FEEDBACK_BASIS",
    "TECH_HINTS",
    "TECH_LABELS",
    # Existing detectors
    "detect_best_overall_section",
    "detect_best_pitch_phrase",
    "detect_best_pitch_phrases",
    "detect_entrance_timing_notes",
    "detect_expressive_match",
    "detect_expressive_moment",
    "detect_fade_within_notes",
    "detect_late_entrance",
    "detect_missed_expression",
    "detect_pitch_struggle",
    "detect_pitch_struggles",
    "detect_section_moments",
    "detect_section_pitch_deltas",
    "detect_section_strength",
    "detect_section_technique_drops",
    "detect_section_weakness",
    "detect_sharp_flat_notes",
    "detect_timing_consistency",
    "detect_vocal_texture_moments",
    "detect_weakest_overall_section",
    # Sprint 2: continuous pitch
    "detect_scoop_habit",
    "detect_pitch_overshoot",
    "detect_clean_attack",
    "detect_falling_release",
    "detect_steady_sustain",
    "detect_pitch_instability",
    "detect_vibrato_quality",
    # Sprint 2: continuous loudness
    "detect_breathy_onset",
    "detect_clean_onset",
    "detect_envelope_moments",
    "detect_support_fade",
    "detect_dynamic_sustain",
    "detect_release_cutoff",
    # Sprint 2: cross-dimensional
    "detect_breath_support_issues",
    "detect_registration_strain",
    "detect_controlled_crescendo",
    "detect_loud_pitch_instability",
    "detect_soft_passage_control",
    "detect_vibrato_with_support",
    "detect_scoop_with_fade",
    "detect_technique_accuracy_tradeoff",
    "detect_expressive_stability",
    "detect_high_note_control",
    # Sprint 2: phrase / section
    "detect_phrase_timing_bias",
    "detect_rhythmic_precision",
    "detect_phrase_pitch_arc",
    "detect_section_improvement",
    "detect_section_vibrato_contrast",
    "select_highlights",
]
