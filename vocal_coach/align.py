"""Sprint-1 stretch: aggregate one ReferenceNote into a NoteCard.

The full alignment / highlight engine lands in sprint 2. This module exists
only to *prove the pieces compose*: given a single reference note plus the
three measurement tracks (pitch, loudness, STARS), produce one well-typed
``NoteCard`` matching the user's design-doc shape::

    {
      "expected_pitch": { "midi": 69, "name": "A4" },
      "lyric_word": "love",
      "section": "Final Chorus",
      "time": "101.24s\u2013102.08s",
      "measurements": {
        "pitch": "-27 cents, drifting down",
        "arrival": "+130ms late",
        "volume": "fades near end"
      },
      "phonemes": [
        { "L": [] },
        { "AH": ["vibrato", "breathy ending"] },
        { "V": [] }
      ],
      "tags": ["slightly flat", "late arrival", "fading ending"]
    }

All thresholds below are placeholders. Sprint 2 will replace this module with
a config-driven pipeline that scans the entire song at multiple granularities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vocal_coach.schemas import (
    LoudnessTrack,
    NoteCard,
    NoteExpectedPitch,
    NoteMeasurements,
    NotePhonemeAnnotation,
    PitchTrack,
    ReferenceAnnotation,
    ReferenceNote,
    StarsPhoneme,
    StarsTrack,
)


# Tunable thresholds (sprint 2 will move these into a config)
ARRIVAL_VOICING_THR = 0.5     # voicing prob above which we count a frame as voiced
ARRIVAL_LATE_MS = 80.0        # late arrival cutoff
ARRIVAL_EARLY_MS = -80.0      # early arrival cutoff
ARRIVAL_SEARCH_BACK_S = 0.20       # how far before note.start_s to look for an entrance
ARRIVAL_SEARCH_FORWARD_S = 0.50    # how far after note.start_s to look (singers drift late more than early)
ARRIVAL_GAP_TOLERANCE_S = 0.030    # prev-note gap below this -> treat as continuation (slur/legato)
ARRIVAL_UNVOICED_LEAD_S = 0.05     # rising edge requires this much prior unvoiced
ARRIVAL_PITCH_LOCK_CENTS = 50.0    # within this many cents of target = "on pitch"
ARRIVAL_PITCH_LOCK_HOLD_S = 0.05   # must stay within ARRIVAL_PITCH_LOCK_CENTS for this long
PITCH_FLAT_CENTS = -20.0      # below this -> "flat"
PITCH_SHARP_CENTS = 20.0      # above this -> "sharp"
PITCH_DRIFT_CENTS_PER_SEC = 40.0  # |slope| above this -> drift tag
VOLUME_FADE_DB_PER_SEC = -8.0  # slope below this in the second half -> fade


def _midi_to_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    out = np.full_like(hz, np.nan, dtype=np.float64)
    voiced = hz > 0
    out[voiced] = 69.0 + 12.0 * np.log2(hz[voiced] / 440.0)
    return out


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    """Plain least-squares slope; nan-safe (returns 0 on insufficient data)."""
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    tt = t[mask]
    yy = y[mask]
    return float(np.polyfit(tt, yy, 1)[0])


def _section_for_time(reference: ReferenceAnnotation, t: float) -> Optional[str]:
    for sec in reference.sections:
        if sec.start_s <= t < sec.end_s:
            return sec.name
    return reference.sections[-1].name if reference.sections else None


# ---------------------------------------------------------------------------
# Pitch / arrival / volume measurements
# ---------------------------------------------------------------------------


def _measure_pitch(
    note: ReferenceNote,
    times: np.ndarray,
    f0_hz: np.ndarray,
) -> tuple[str, list[str]]:
    inside = (times >= note.start_s) & (times < note.end_s)
    if not inside.any():
        return ("no voiced frames inside note window", ["no signal"])
    midi = _hz_to_midi(f0_hz[inside])
    cents = 100.0 * (midi - note.midi_pitch)  # cents from reference
    valid = np.isfinite(cents)
    if valid.sum() < 2:
        return ("no voiced frames inside note window", ["no signal"])

    median_cents = float(np.nanmedian(cents))
    slope = _slope(times[inside][valid], cents[valid])  # cents/sec

    parts = [f"{median_cents:+.0f} cents"]
    tags: list[str] = []
    if median_cents <= PITCH_FLAT_CENTS - 30:
        tags.append("flat")
    elif median_cents <= PITCH_FLAT_CENTS:
        tags.append("slightly flat")
    elif median_cents >= PITCH_SHARP_CENTS + 30:
        tags.append("sharp")
    elif median_cents >= PITCH_SHARP_CENTS:
        tags.append("slightly sharp")

    if slope <= -PITCH_DRIFT_CENTS_PER_SEC:
        parts.append("drifting down")
        tags.append("drifting down")
    elif slope >= PITCH_DRIFT_CENTS_PER_SEC:
        parts.append("drifting up")
        tags.append("drifting up")

    return (", ".join(parts), tags)


def _format_delta_ms(delta_ms: float) -> str:
    """Render a millisecond offset as e.g. '+130ms late' or '-50ms early'."""
    sign = "+" if delta_ms >= 0 else ""
    if delta_ms > 0:
        suffix = " late"
    elif delta_ms < 0:
        suffix = " early"
    else:
        suffix = " on time"
    return f"{sign}{delta_ms:.0f}ms{suffix}"


def _infer_hop_seconds(times: np.ndarray, default: float = 0.01) -> float:
    """Recover the frame hop from a uniformly-spaced times array."""
    if times.size < 2:
        return default
    return float(times[1] - times[0])


def _arrival_from_voicing_edge(
    note: ReferenceNote,
    times: np.ndarray,
    voicing: np.ndarray,
) -> tuple[str, list[str]]:
    """Onset-mode arrival: find the rising edge of voicing nearest the note onset.

    A rising edge is a voiced frame whose preceding ``ARRIVAL_UNVOICED_LEAD_S``
    of frames were unvoiced. This rejects the degenerate "first voiced frame
    happens to sit at the start of the search window" failure mode that the
    legacy detector exhibited.
    """
    search = (times >= note.start_s - ARRIVAL_SEARCH_BACK_S) & (
        times <= note.start_s + ARRIVAL_SEARCH_FORWARD_S
    )
    if not search.any():
        return ("no voicing trace", [])

    voiced = voicing >= ARRIVAL_VOICING_THR
    hop = _infer_hop_seconds(times)
    lead_frames = max(1, int(round(ARRIVAL_UNVOICED_LEAD_S / hop)))

    search_idx = np.where(search)[0]
    rising_edges: list[int] = []
    for i in search_idx:
        if not voiced[i]:
            continue
        lead_start = max(0, i - lead_frames)
        if lead_start == i:
            continue  # not enough history to confirm a rising edge
        if not voiced[lead_start:i].any():
            rising_edges.append(i)

    if rising_edges:
        # Choose the rising edge closest to the reference onset.
        best = min(rising_edges, key=lambda i: abs(times[i] - note.start_s))
        delta_ms = (float(times[best]) - note.start_s) * 1000.0
        if delta_ms <= ARRIVAL_EARLY_MS:
            tag = "early arrival"
        elif delta_ms >= ARRIVAL_LATE_MS:
            tag = "late arrival"
        else:
            tag = ""
        return (_format_delta_ms(delta_ms), [tag] if tag else [])

    # No clean rising edge inside the window. Two sub-cases:
    if not voiced[search].any():
        return ("no voiced onset detected", ["missed entrance"])
    # Voicing was continuous through the onset region -- the singer never
    # truly re-attacked. Flag this; we can't measure a meaningful delta.
    return ("voicing continuous through onset", ["no clean entrance"])


def _arrival_from_pitch_lock(
    note: ReferenceNote,
    times: np.ndarray,
    f0_hz: np.ndarray,
) -> tuple[str, list[str]]:
    """Continuation-mode arrival: find when F0 locks onto the new target pitch.

    Used when the previous reference note butts directly against this one
    (slur, melisma, or word-boundary legato). Voicing is continuous, so the
    interesting event is the F0 transition, not a voicing onset.

    "Lock" means F0 sits within +/- ``ARRIVAL_PITCH_LOCK_CENTS`` of the target
    for at least ``ARRIVAL_PITCH_LOCK_HOLD_S`` of consecutive frames.
    """
    search = (times >= note.start_s - ARRIVAL_SEARCH_BACK_S) & (
        times <= min(note.start_s + ARRIVAL_SEARCH_FORWARD_S, note.end_s)
    )
    if not search.any():
        return ("no voicing trace", [])

    midi = _hz_to_midi(f0_hz)
    cents = 100.0 * (midi - note.midi_pitch)
    on_pitch = np.isfinite(cents) & (np.abs(cents) <= ARRIVAL_PITCH_LOCK_CENTS)

    hop = _infer_hop_seconds(times)
    hold_frames = max(1, int(round(ARRIVAL_PITCH_LOCK_HOLD_S / hop)))

    search_idx = np.where(search)[0]
    if search_idx.size == 0:
        return ("no voicing trace", [])
    last_idx = int(search_idx[-1])

    for i in search_idx:
        run_end = i + hold_frames
        if run_end > last_idx + 1:
            break  # not enough remaining frames to confirm a sustained lock
        if on_pitch[i:run_end].all():
            delta_ms = (float(times[i]) - note.start_s) * 1000.0
            if delta_ms <= ARRIVAL_EARLY_MS:
                tag = "early pitch arrival"
            elif delta_ms >= ARRIVAL_LATE_MS:
                tag = "late pitch arrival"
            else:
                tag = ""
            return (f"pitch locked {_format_delta_ms(delta_ms)}", [tag] if tag else [])

    return ("did not reach target pitch", ["missed pitch transition"])


def _measure_arrival(
    note: ReferenceNote,
    prev_note: Optional[ReferenceNote],
    times: np.ndarray,
    f0_hz: np.ndarray,
    voicing: np.ndarray,
) -> tuple[str, list[str]]:
    """Dispatch to the appropriate arrival detector for ``note``.

    A note is treated as a *continuation* of the previous note (and measured
    via pitch-lock) when the prior reference note ends within
    ``ARRIVAL_GAP_TOLERANCE_S`` of this note's start. Otherwise it's an
    *onset* (measured via voicing rising-edge), which is the right regime for
    the first note of the song and for any note preceded by a rest.
    """
    is_continuation = (
        prev_note is not None
        and (note.start_s - prev_note.end_s) < ARRIVAL_GAP_TOLERANCE_S
    )
    if is_continuation:
        return _arrival_from_pitch_lock(note, times, f0_hz)
    return _arrival_from_voicing_edge(note, times, voicing)


def _measure_volume(
    note: ReferenceNote,
    times: np.ndarray,
    rms_db: np.ndarray,
) -> tuple[str, list[str]]:
    inside = (times >= note.start_s) & (times < note.end_s)
    if inside.sum() < 4:
        return ("note too short for volume analysis", [])
    t_in = times[inside]
    db_in = rms_db[inside]
    duration = note.end_s - note.start_s

    half_t = note.start_s + duration / 2.0
    second_half = inside & (times >= half_t)
    slope_full = _slope(t_in, db_in)
    slope_tail = _slope(times[second_half], rms_db[second_half]) if second_half.any() else slope_full

    parts: list[str] = []
    tags: list[str] = []
    if slope_tail <= VOLUME_FADE_DB_PER_SEC and (db_in[-1] < db_in[:max(1, len(db_in)//2)].mean() - 3.0):
        parts.append("fades near end")
        tags.append("fading ending")
    elif slope_full >= -VOLUME_FADE_DB_PER_SEC:
        parts.append("steady")
    else:
        parts.append("decreasing")

    return (parts[0], tags)


# ---------------------------------------------------------------------------
# Phoneme/technique annotations from STARS
# ---------------------------------------------------------------------------


# How STARS technique flags map to user-facing tags.
STARS_TECH_LABELS = {
    "vibrato": "vibrato",
    "glissando": "glissando",
    "falsetto": "falsetto",
    "pharyngeal": "pharyngeal resonance",
    "breathe": "breathy",
    "bubble": "bubble",
    "weak": "weak voice",
    "strong": "strong voice",
    "mixed": "mixed voice",
}


_PHONEME_OVERLAP_SLACK_S = 0.02


def _stars_span_for_phoneme(
    phoneme: str,
    note: ReferenceNote,
    stars: StarsTrack,
) -> Optional[StarsPhoneme]:
    """Best STARS span for ``phoneme`` overlapping this note window (for technique tags)."""
    best: Optional[StarsPhoneme] = None
    best_dist = float("inf")
    for ph in stars.phonemes:
        if ph.phoneme != phoneme or ph.phoneme in {"<SP>", "<AP>"}:
            continue
        if ph.end_s <= note.start_s - _PHONEME_OVERLAP_SLACK_S or ph.start_s >= note.end_s + _PHONEME_OVERLAP_SLACK_S:
            continue
        # Prefer the span whose start is closest to this note's onset.
        dist = abs(ph.start_s - note.start_s)
        if dist < best_dist:
            best_dist = dist
            best = ph
    return best


def _phonemes_inside(
    note: ReferenceNote,
    stars: Optional[StarsTrack],
) -> list[NotePhonemeAnnotation]:
    """Return phonemes for this note in reference order.

    Phone membership comes from GTSinger ``ph_start``/``ph_end`` vs note bounds
    (stored on ``note.phonemes`` when the reference was built). STARS word labels
    can disagree with GTSinger on fast passages, so we match STARS spans by
    phoneme symbol + overlap with the note window only to attach technique tags.
    """
    if stars is None:
        return [NotePhonemeAnnotation(phoneme=p, tags=[]) for p in note.phonemes]

    out: list[NotePhonemeAnnotation] = []
    for ph_name in note.phonemes:
        span = _stars_span_for_phoneme(ph_name, note, stars)
        if span is None:
            out.append(NotePhonemeAnnotation(phoneme=ph_name, tags=[]))
            continue
        active = [
            STARS_TECH_LABELS[t]
            for t, v in span.techniques.items()
            if v and t in STARS_TECH_LABELS and t != "mixed"
        ]
        out.append(NotePhonemeAnnotation(phoneme=ph_name, tags=active))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_note(
    note: ReferenceNote,
    *,
    reference: ReferenceAnnotation,
    pitch: PitchTrack,
    loudness: LoudnessTrack,
    stars: Optional[StarsTrack] = None,
) -> NoteCard:
    """Produce a NoteCard for one reference note from the four pipeline tracks."""
    p_times = np.array([f.time for f in pitch.frames])
    p_f0 = np.array([f.f0_hz for f in pitch.frames])
    p_voicing = np.array([f.voicing_confidence for f in pitch.frames])

    l_times = np.array([f.time for f in loudness.frames])
    l_db = np.array([f.rms_db for f in loudness.frames])

    prev_note = (
        reference.notes[note.index - 1] if note.index > 0 and reference.notes else None
    )

    pitch_desc, pitch_tags = _measure_pitch(note, p_times, p_f0)
    arrival_desc, arrival_tags = _measure_arrival(note, prev_note, p_times, p_f0, p_voicing)
    volume_desc, volume_tags = _measure_volume(note, l_times, l_db)
    phonemes = _phonemes_inside(note, stars)

    section = _section_for_time(reference, 0.5 * (note.start_s + note.end_s))
    time_str = f"{note.start_s:.2f}s\u2013{note.end_s:.2f}s"

    tags = list(dict.fromkeys(pitch_tags + arrival_tags + volume_tags))  # dedupe, preserve order

    return NoteCard(
        expected_pitch=NoteExpectedPitch(midi=note.midi_pitch, name=note.note_name),
        lyric_word=note.lyric_word,
        section=section,
        time=time_str,
        measurements=NoteMeasurements(
            pitch=pitch_desc,
            arrival=arrival_desc,
            volume=volume_desc,
        ),
        phonemes=phonemes,
        tags=tags,
    )


def pick_demo_note(
    reference: ReferenceAnnotation,
    *,
    min_duration_s: float = 0.30,
) -> ReferenceNote:
    """Pick a representative reference note for a demo card.

    Strategy: the longest note that's at least ``min_duration_s`` seconds long;
    fallback to whichever is longest if all are shorter.
    """
    if not reference.notes:
        raise ValueError("reference annotation has no notes")
    eligible = [n for n in reference.notes if n.duration_s >= min_duration_s]
    pool = eligible if eligible else reference.notes
    return max(pool, key=lambda n: n.duration_s)
