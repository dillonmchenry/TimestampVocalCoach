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
                title=f"Try more {_label(tech)}",
                summary=(
                    f"The reference leans on {_hint(tech)} on {count} notes here; "
                    "your take stays a bit straighter in tone."
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
                        f"{kind_a} notes — bringing it across helps the "
                        "performance feel cohesive."
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
# Top-level selection — diversity-aware round-robin
# ---------------------------------------------------------------------------

MOMENT_CATEGORY: dict[str, str] = {
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
}

_CATEGORY_ORDER = ["pitch", "technique", "alignment", "dynamics"]


def _select_diverse(
    moments: list[CoachingMoment],
    *,
    cap: int,
    max_per_type: int,
    max_per_category: int,
) -> list[CoachingMoment]:
    """Pick moments round-robin across categories for maximum variety.

    Within each category the best-scored candidate is taken first, but the
    algorithm cycles through *all* categories before returning to any one,
    guaranteeing that every category with candidates gets representation
    before any category gets a second slot.
    """
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
) -> HighlightsReport:
    """Run every detector, then pick a diverse set via round-robin.

    Categories (pitch, technique, alignment, dynamics) are cycled so that
    every category with candidates gets at least one slot before any
    category gets a second.
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

    chosen = _select_diverse(
        candidates,
        cap=cfg.highlights.cap,
        max_per_type=cfg.highlights.max_per_type,
        max_per_category=cfg.highlights.max_per_category,
    )
    chosen.sort(key=lambda m: m.start_s)
    return HighlightsReport(moments=chosen, cap=cfg.highlights.cap)


__all__ = [
    "MOMENT_CATEGORY",
    "TECH_HINTS",
    "TECH_LABELS",
    "detect_best_overall_section",
    "detect_best_pitch_phrase",
    "detect_best_pitch_phrases",
    "detect_entrance_timing_notes",
    "detect_expressive_match",
    "detect_expressive_moment",
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
    "select_highlights",
]
