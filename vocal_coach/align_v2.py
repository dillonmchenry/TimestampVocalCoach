"""Sprint-2 dual-track note alignment.

The Sprint-1 ``align.py`` aggregates one note from one wav. ``align_v2``
operates on a *song* (UltraStar chart + reference vocal artifacts) and a
*performance* (user vocal + NanoPitch + STARS), producing per-note
``NoteMeasurementV2`` rows plus per-note ``NoteTechniqueComparison``s. The
highlight engine (``vocal_coach.highlights``) consumes those.

Pipeline:

    UltraStar note grid
        |
        v
    estimate_global_offset_s(reference, user_pitch) -- align user wav -> song time
        |
        v
    map_user_pitch_to_song_time(user_pitch, offset_s)   (frame-level shift)
    map_user_stars_to_song_time(user_stars, offset_s)
        |
        v
    for each ReferenceNote:
        core_window_s(note)
        measure pitch (median cents, pct_in_tune, drift)
        measure arrival (NanoPitch voicing/pitch-lock vs expected onset)
        compare reference vs user STARS techniques in note window
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from vocal_coach.coaching_config import CoachingConfig
from vocal_coach.schemas import (
    LoudnessTrack,
    NoteMeasurementV2,
    NoteTechniqueComparison,
    PitchTrack,
    ReferenceAnnotation,
    ReferenceNote,
    StarsPhoneme,
    StarsTrack,
)


# Phoneme symbols that should never carry a feature claim.
_NON_LYRIC_PHONES = {"<SP>", "<AP>", "<UNK>", ""}


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


@dataclass
class PitchArrays:
    """NumPy view of a ``PitchTrack`` for vectorized math."""

    times: np.ndarray
    f0_hz: np.ndarray
    voicing: np.ndarray
    hop_seconds: float

    @classmethod
    def from_track(cls, track: PitchTrack) -> "PitchArrays":
        if not track.frames:
            return cls(
                times=np.zeros(0, dtype=np.float64),
                f0_hz=np.zeros(0, dtype=np.float64),
                voicing=np.zeros(0, dtype=np.float64),
                hop_seconds=track.hop_seconds,
            )
        times = np.fromiter((f.time for f in track.frames), dtype=np.float64, count=len(track.frames))
        f0 = np.fromiter((f.f0_hz for f in track.frames), dtype=np.float64, count=len(track.frames))
        v = np.fromiter(
            (f.voicing_confidence for f in track.frames),
            dtype=np.float64,
            count=len(track.frames),
        )
        return cls(times=times, f0_hz=f0, voicing=v, hop_seconds=track.hop_seconds)


@dataclass
class LoudnessArrays:
    """NumPy view of a ``LoudnessTrack`` for per-note windowed measurements.

    ``median_db`` is the track-level loudness reference (median of frames above
    the silence floor).  Subtracting it from any per-note mean gives a
    *relative* loudness value that is comparable across recordings made at
    different mic gains.
    """

    times: np.ndarray   # (T,) seconds
    rms_db: np.ndarray  # (T,) dBFS
    median_db: float    # track-level normalization anchor

    # Threshold below which a frame is considered silence (not included in
    # the median calculation so pockets of silence don't drag the anchor down).
    _SILENCE_FLOOR_DB: float = -60.0

    @classmethod
    def from_track(cls, track: LoudnessTrack) -> "LoudnessArrays":
        if not track.frames:
            return cls(
                times=np.zeros(0, dtype=np.float64),
                rms_db=np.zeros(0, dtype=np.float32),
                median_db=-60.0,
            )
        times = np.fromiter(
            (f.time for f in track.frames), dtype=np.float64, count=len(track.frames)
        )
        rms_db = np.fromiter(
            (f.rms_db for f in track.frames), dtype=np.float32, count=len(track.frames)
        )
        voiced_mask = rms_db > cls._SILENCE_FLOOR_DB
        if voiced_mask.any():
            median_db = float(np.median(rms_db[voiced_mask]))
        else:
            median_db = float(np.median(rms_db))
        return cls(times=times, rms_db=rms_db, median_db=median_db)

    def shift_to_song_time(self, offset_s: float) -> "LoudnessArrays":
        """Return a copy with times shifted by ``-offset_s``."""
        return LoudnessArrays(
            times=self.times - offset_s,
            rms_db=self.rms_db,
            median_db=self.median_db,
        )

    def _window(self, start_s: float, end_s: float) -> np.ndarray:
        mask = (self.times >= start_s) & (self.times < end_s)
        return self.rms_db[mask]

    def mean_rms_db(self, start_s: float, end_s: float) -> Optional[float]:
        frames = self._window(start_s, end_s)
        return float(np.mean(frames)) if frames.size > 0 else None

    def fade_db_per_s(self, start_s: float, end_s: float) -> Optional[float]:
        """Linear RMS slope across the window in dB/s (negative = fading out)."""
        duration = end_s - start_s
        if duration < 0.08:
            return None
        mask = (self.times >= start_s) & (self.times < end_s)
        t = self.times[mask]
        db = self.rms_db[mask].astype(np.float64)
        if t.size < 3:
            return None
        return _slope(t - t[0], db)


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    out = np.full_like(hz, np.nan, dtype=np.float64)
    voiced = hz > 0
    out[voiced] = 69.0 + 12.0 * np.log2(hz[voiced] / 440.0)
    return out


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    return float(np.polyfit(t[mask], y[mask], 1)[0])


# ---------------------------------------------------------------------------
# Global offset (user vocal -> song time)
# ---------------------------------------------------------------------------


def _voiced_mask_song_time(reference: ReferenceAnnotation, times: np.ndarray) -> np.ndarray:
    """Boolean per-frame mask of "song expects voiced singing here"."""
    expected = np.zeros_like(times, dtype=bool)
    for note in reference.notes:
        expected |= (times >= note.start_s) & (times < note.end_s)
    return expected


def estimate_global_offset_s(
    reference: ReferenceAnnotation,
    pitch_user: PitchTrack,
    *,
    config: Optional[CoachingConfig] = None,
) -> float:
    """Estimate ``offset`` such that ``song_time = user_time - offset``.

    We compare the chart's expected voiced regions (per ``ReferenceNote``)
    against the user's NanoPitch voicing track on a discrete offset grid.
    The score is the count of frames where both expectations agree, with a
    small penalty per offset to break ties.
    """
    cfg = (config or CoachingConfig()).global_offset
    user = PitchArrays.from_track(pitch_user)
    if user.times.size == 0:
        return 0.0
    user_voiced = user.voicing >= cfg.voicing_threshold
    if not user_voiced.any():
        return 0.0

    hop = max(user.hop_seconds, 1e-6)
    search_steps = int(round(cfg.search_range_s / cfg.step_s))
    offsets = np.arange(-search_steps, search_steps + 1, dtype=np.float64) * cfg.step_s
    if offsets.size == 0:
        return 0.0

    # Pre-compute per-frame song time for *zero* offset.
    song_times_zero = user.times.copy()

    best_offset = 0.0
    best_score = -np.inf
    min_overlap_frames = int(round(cfg.min_voiced_overlap_s / hop))

    for off in offsets:
        song_times = song_times_zero - off
        expected = _voiced_mask_song_time(reference, song_times)
        agree = expected & user_voiced
        if agree.sum() < min_overlap_frames:
            continue
        # Penalize larger offsets very slightly to prefer the closest match.
        score = float(agree.sum()) - 0.05 * abs(off) / cfg.step_s
        if score > best_score:
            best_score = score
            best_offset = float(off)

    return best_offset


def estimate_octave_shift_semitones(
    reference: ReferenceAnnotation,
    pitch_user: PitchTrack,
    *,
    global_offset_s: float = 0.0,
    config: Optional[CoachingConfig] = None,
    max_octaves: int = 3,
) -> int:
    """Detect an integer-octave (multiple of 12 semitones) shift between
    the user's vocal register and the chart's targets.

    For each ``ReferenceNote``, take the median user MIDI on voiced frames
    inside the (shifted-into-song-time) note window and subtract
    ``note.midi_pitch``. The median of those per-note residuals is rounded
    to the nearest 12 semitones; that becomes the shift we add to every
    chart MIDI before scoring.

    This formulation is correct even when the chart itself is off by an
    octave from the recorded vocal -- whatever the chart says, we transpose
    targets so they land where the user is actually singing, then let the
    per-note octave-fold safety net inside ``_measure_pitch`` clean up
    individual outliers.

    Returns 0 when there's not enough voiced data to decide. The result is
    clamped to +/- ``max_octaves * 12`` so a runaway detection (e.g. a user
    file that's mostly noise) can't silently transpose targets by 5 octaves.
    """
    cfg = (config or CoachingConfig()).pitch
    user = PitchArrays.from_track(pitch_user)
    if user.times.size == 0:
        return 0
    user_voiced = (user.voicing >= cfg.voicing_threshold) & (user.f0_hz > 0)
    if not user_voiced.any():
        return 0
    user_midi = _hz_to_midi(user.f0_hz)

    user_song_times = user.times - global_offset_s
    residuals: list[float] = []
    for note in reference.notes:
        m = (
            (user_song_times >= note.start_s)
            & (user_song_times < note.end_s)
            & user_voiced
        )
        if not m.any():
            continue
        per_note_med = float(np.nanmedian(user_midi[m]))
        residuals.append(per_note_med - note.midi_pitch)
    if not residuals:
        return 0
    median_residual = float(np.median(residuals))
    steps = int(round(median_residual / 12.0))
    steps = max(-max_octaves, min(max_octaves, steps))
    return steps * 12


def shift_pitch_to_song_time(pitch: PitchArrays, offset_s: float) -> PitchArrays:
    """Return a new ``PitchArrays`` whose times are shifted by ``-offset_s``."""
    return PitchArrays(
        times=pitch.times - offset_s,
        f0_hz=pitch.f0_hz,
        voicing=pitch.voicing,
        hop_seconds=pitch.hop_seconds,
    )


def shift_stars_to_song_time(stars: StarsTrack, offset_s: float) -> StarsTrack:
    """Shift every STARS phoneme/note span by ``-offset_s``."""
    new_phonemes = [
        ph.model_copy(
            update={
                "start_s": ph.start_s - offset_s,
                "end_s": ph.end_s - offset_s,
            }
        )
        for ph in stars.phonemes
    ]
    new_notes = [
        n.model_copy(
            update={
                "start_s": n.start_s - offset_s,
                "end_s": n.end_s - offset_s,
            }
        )
        for n in stars.notes
    ]
    return stars.model_copy(update={"phonemes": new_phonemes, "notes": new_notes})


# ---------------------------------------------------------------------------
# Core windows + per-note pitch
# ---------------------------------------------------------------------------


def core_window_s(
    note: ReferenceNote,
    *,
    arrival_offset_ms: Optional[float],
    config: Optional[CoachingConfig] = None,
) -> tuple[float, float]:
    """Trim a note window to the "core" used for pitch scoring.

    The trimmed window is shifted by the user's detected arrival offset so
    that the comparison is performed against where the user *actually*
    landed instead of where the chart said the note should start.
    """
    cfg = (config or CoachingConfig()).core_window
    start = note.start_s + cfg.attack_trim_s
    end = note.end_s - cfg.release_trim_s
    if arrival_offset_ms is not None:
        shift = arrival_offset_ms / 1000.0
        start += shift
        end += shift
    if end - start < cfg.min_core_s:
        center = 0.5 * (note.start_s + note.end_s)
        start = center - cfg.min_core_s / 2.0
        end = center + cfg.min_core_s / 2.0
    return float(start), float(end)


def _measure_pitch(
    note: ReferenceNote,
    pitch: PitchArrays,
    core_start: float,
    core_end: float,
    *,
    config: CoachingConfig,
    octave_shift_semitones: int = 0,
) -> dict:
    """Compute median cents, pct_in_tune, drift, and pitch tags.

    Cents are computed against ``note.midi_pitch + octave_shift_semitones``
    (the shifted target) and then **per-frame octave-folded** into
    ``[-600, +600]`` cents, so a single note that was bumped up or down by
    an octave during the take is still scored as same-pitch-class. The
    octave we folded out is reported as ``note_octave_offset`` for UI
    diagnostics; non-zero values get an ``octave above``/``octave below``
    pitch tag.
    """
    cfg = config.pitch
    inside = (pitch.times >= core_start) & (pitch.times < core_end)
    if not inside.any():
        return {
            "median_cents": None,
            "pct_in_tune": None,
            "drift_cents_per_s": None,
            "voiced_coverage": 0.0,
            "pitch_tags": ["no signal"],
            "note_octave_offset": 0,
        }
    voiced_inside = inside & (pitch.voicing >= cfg.voicing_threshold) & (pitch.f0_hz > 0)
    voiced_coverage = float(voiced_inside.sum() / max(1, inside.sum()))

    if voiced_inside.sum() < 2:
        return {
            "median_cents": None,
            "pct_in_tune": None,
            "drift_cents_per_s": None,
            "voiced_coverage": voiced_coverage,
            "pitch_tags": ["unsung"] if voiced_coverage < cfg.min_voiced_coverage else [],
            "note_octave_offset": 0,
        }

    target_midi = note.midi_pitch + int(octave_shift_semitones)
    midi = _hz_to_midi(pitch.f0_hz[voiced_inside])
    raw_cents = 100.0 * (midi - target_midi)
    folded_cents = ((raw_cents + 600.0) % 1200.0) - 600.0
    median_cents = float(np.nanmedian(folded_cents))
    drift = _slope(pitch.times[voiced_inside], folded_cents)
    pct_in_tune = float(np.mean(np.abs(folded_cents) <= cfg.in_tune_cents))

    # Per-note residual octave: how many octaves were folded out per frame?
    # We take the median across frames so a single noisy frame can't flip
    # the diagnostic tag.
    if raw_cents.size:
        per_frame_octaves = np.round((raw_cents - folded_cents) / 1200.0)
        note_octave = int(np.median(per_frame_octaves))
    else:
        note_octave = 0

    tags: list[str] = []
    if median_cents <= cfg.flat_cents - 30:
        tags.append("flat")
    elif median_cents <= cfg.flat_cents:
        tags.append("slightly flat")
    elif median_cents >= cfg.sharp_cents + 30:
        tags.append("sharp")
    elif median_cents >= cfg.sharp_cents:
        tags.append("slightly sharp")
    if drift <= -cfg.drift_cents_per_s:
        tags.append("drifting down")
    elif drift >= cfg.drift_cents_per_s:
        tags.append("drifting up")
    if voiced_coverage < cfg.min_voiced_coverage:
        tags.append("unsung")
    if note_octave < 0:
        tags.append("octave below")
    elif note_octave > 0:
        tags.append("octave above")

    return {
        "median_cents": median_cents,
        "pct_in_tune": pct_in_tune,
        "drift_cents_per_s": float(drift),
        "voiced_coverage": voiced_coverage,
        "pitch_tags": tags,
        "note_octave_offset": note_octave,
    }


# ---------------------------------------------------------------------------
# Arrival / onset detection
# ---------------------------------------------------------------------------


def _expected_onset_s(
    note: ReferenceNote,
    stars_ref: Optional[StarsTrack],
) -> float:
    """Pick the expected user onset for ``note``.

    Priority order:
      1. First reference STARS phoneme overlapping the note window.
      2. Note start from UltraStar.
    """
    if stars_ref is None:
        return note.start_s
    candidate: Optional[float] = None
    for ph in stars_ref.phonemes:
        if ph.phoneme in _NON_LYRIC_PHONES:
            continue
        if ph.end_s <= note.start_s or ph.start_s >= note.end_s:
            continue
        if candidate is None or ph.start_s < candidate:
            candidate = ph.start_s
    if candidate is None:
        return note.start_s
    return float(candidate)


def _arrival_via_voicing(
    note: ReferenceNote,
    expected_onset: float,
    pitch: PitchArrays,
    *,
    config: CoachingConfig,
) -> Optional[float]:
    """Onset-mode arrival: voicing rising edge near ``expected_onset``."""
    cfg = config.arrival
    voicing_thr = config.pitch.voicing_threshold
    search = (pitch.times >= expected_onset - cfg.search_back_s) & (
        pitch.times <= expected_onset + cfg.search_forward_s
    )
    if not search.any():
        return None

    voiced = pitch.voicing >= voicing_thr
    hop = max(pitch.hop_seconds, 1e-6)
    lead_frames = max(1, int(round(cfg.unvoiced_lead_s / hop)))

    indices = np.where(search)[0]
    rising = []
    for i in indices:
        if not voiced[i]:
            continue
        lead_start = max(0, i - lead_frames)
        if lead_start == i:
            continue
        if not voiced[lead_start:i].any():
            rising.append(i)
    if not rising:
        return None
    best = min(rising, key=lambda i: abs(pitch.times[i] - expected_onset))
    return float(pitch.times[best])


def _arrival_via_pitch_lock(
    note: ReferenceNote,
    expected_onset: float,
    pitch: PitchArrays,
    *,
    config: CoachingConfig,
    octave_shift_semitones: int = 0,
) -> Optional[float]:
    """Continuation-mode arrival: pitch lock to target across a hold.

    ``cents`` is octave-folded so the pitch-lock check is octave-invariant
    around ``note.midi_pitch + octave_shift_semitones``.
    """
    cfg = config.arrival
    voicing_thr = config.pitch.voicing_threshold
    search = (pitch.times >= expected_onset - cfg.search_back_s) & (
        pitch.times <= min(expected_onset + cfg.search_forward_s, note.end_s)
    )
    if not search.any():
        return None
    target_midi = note.midi_pitch + int(octave_shift_semitones)
    midi = _hz_to_midi(pitch.f0_hz)
    raw_cents = 100.0 * (midi - target_midi)
    cents = ((raw_cents + 600.0) % 1200.0) - 600.0
    voiced = (pitch.voicing >= voicing_thr) & np.isfinite(cents)
    on_pitch = voiced & (np.abs(cents) <= cfg.pitch_lock_cents)

    hop = max(pitch.hop_seconds, 1e-6)
    hold = max(1, int(round(cfg.pitch_lock_hold_s / hop)))
    indices = np.where(search)[0]
    if indices.size == 0:
        return None
    last_idx = int(indices[-1])
    for i in indices:
        end = i + hold
        if end > last_idx + 1:
            break
        if on_pitch[i:end].all():
            return float(pitch.times[i])
    return None


def _measure_arrival(
    note: ReferenceNote,
    prev_note: Optional[ReferenceNote],
    expected_onset: float,
    pitch: PitchArrays,
    *,
    config: CoachingConfig,
    octave_shift_semitones: int = 0,
) -> tuple[Optional[float], list[str]]:
    """Return (arrival_offset_ms, tags). ``None`` means we couldn't decide."""
    cfg = config.arrival
    is_continuation = (
        prev_note is not None
        and (note.start_s - prev_note.end_s) < cfg.gap_tolerance_s
    )
    arrival_t: Optional[float]
    if is_continuation:
        arrival_t = _arrival_via_pitch_lock(
            note, expected_onset, pitch, config=config,
            octave_shift_semitones=octave_shift_semitones,
        )
    else:
        arrival_t = _arrival_via_voicing(note, expected_onset, pitch, config=config)

    if arrival_t is None:
        return (None, ["missed entrance"])
    delta_ms = (arrival_t - expected_onset) * 1000.0
    tags: list[str] = []
    if delta_ms <= cfg.early_ms:
        tags.append("early arrival")
    elif delta_ms >= cfg.late_ms:
        tags.append("late arrival")
    return (float(delta_ms), tags)


# ---------------------------------------------------------------------------
# STARS phoneme/technique mapping
# ---------------------------------------------------------------------------


def _phones_overlapping_note(
    note: ReferenceNote,
    stars: Optional[StarsTrack],
    *,
    slack_s: float = 0.02,
) -> list[StarsPhoneme]:
    """Return all STARS phoneme spans (excluding silence) overlapping ``note``."""
    if stars is None:
        return []
    out: list[StarsPhoneme] = []
    for ph in stars.phonemes:
        if ph.phoneme in _NON_LYRIC_PHONES:
            continue
        if ph.end_s <= note.start_s - slack_s or ph.start_s >= note.end_s + slack_s:
            continue
        out.append(ph)
    return out


def _active_techniques(phones: Iterable[StarsPhoneme]) -> set[str]:
    """Aggregate technique flags across all phones in a window."""
    out: set[str] = set()
    for ph in phones:
        for name, value in ph.techniques.items():
            if value:
                out.add(name)
    return out


def map_stars_phones_to_notes(
    reference: ReferenceAnnotation,
    stars: Optional[StarsTrack],
    *,
    slack_s: float = 0.02,
) -> dict[int, list[StarsPhoneme]]:
    """Per-note list of overlapping STARS phoneme spans."""
    return {
        note.index: _phones_overlapping_note(note, stars, slack_s=slack_s)
        for note in reference.notes
    }


def map_user_stars_to_song_time(
    stars_user: Optional[StarsTrack],
    global_offset_s: float,
) -> Optional[StarsTrack]:
    """Shift a user STARS track so its spans live in song time."""
    if stars_user is None:
        return None
    return shift_stars_to_song_time(stars_user, global_offset_s)


def compare_note_techniques(
    note: ReferenceNote,
    *,
    ref_phones: list[StarsPhoneme],
    user_phones: list[StarsPhoneme],
) -> NoteTechniqueComparison:
    """Reference vs user technique sets for ``note``."""
    ref = _active_techniques(ref_phones)
    usr = _active_techniques(user_phones)
    matched = sorted(ref & usr)
    missed = sorted(ref - usr)
    added = sorted(usr - ref)
    return NoteTechniqueComparison(
        note_index=note.index,
        reference_techniques=sorted(ref),
        user_techniques=sorted(usr),
        matched=matched,
        missed=missed,
        user_added=added,
    )


# ---------------------------------------------------------------------------
# Public API: per-note measurement
# ---------------------------------------------------------------------------


def _measure_loudness(
    note: ReferenceNote,
    loudness_user: "LoudnessArrays",
    loudness_ref: Optional["LoudnessArrays"],
) -> dict:
    """Return per-note loudness scalars for ``NoteMeasurementV2``."""
    user_rms = loudness_user.mean_rms_db(note.start_s, note.end_s)
    fade = loudness_user.fade_db_per_s(note.start_s, note.end_s)
    ref_rms = loudness_ref.mean_rms_db(note.start_s, note.end_s) if loudness_ref else None

    # Normalised delta: relative user level vs. relative reference level.
    rms_delta: Optional[float] = None
    if user_rms is not None and ref_rms is not None and loudness_ref is not None:
        rms_delta = (user_rms - loudness_user.median_db) - (ref_rms - loudness_ref.median_db)

    return {
        "user_rms_db": user_rms,
        "ref_rms_db": ref_rms,
        "rms_delta_db": rms_delta,
        "rms_fade_db_per_s": fade,
    }


def measure_note(
    note: ReferenceNote,
    *,
    prev_note: Optional[ReferenceNote],
    pitch_user: PitchArrays,
    stars_ref: Optional[StarsTrack],
    config: CoachingConfig,
    octave_shift_semitones: int = 0,
    loudness_user: Optional["LoudnessArrays"] = None,
    loudness_ref: Optional["LoudnessArrays"] = None,
) -> NoteMeasurementV2:
    """Run all per-note measurements in song time.

    ``octave_shift_semitones`` is the global integer-octave correction
    applied to chart MIDI before pitch and arrival are evaluated.
    """
    expected_onset = _expected_onset_s(note, stars_ref)
    arrival_offset_ms, arrival_tags = _measure_arrival(
        note, prev_note, expected_onset, pitch_user, config=config,
        octave_shift_semitones=octave_shift_semitones,
    )
    cs, ce = core_window_s(note, arrival_offset_ms=arrival_offset_ms, config=config)
    pitch_stats = _measure_pitch(
        note, pitch_user, cs, ce, config=config,
        octave_shift_semitones=octave_shift_semitones,
    )

    loud = (
        _measure_loudness(note, loudness_user, loudness_ref)
        if loudness_user is not None
        else {}
    )

    return NoteMeasurementV2(
        note_index=note.index,
        start_s=note.start_s,
        end_s=note.end_s,
        midi_pitch=note.midi_pitch,
        note_name=note.note_name,
        lyric_word=note.lyric_word,
        voiced_coverage=pitch_stats["voiced_coverage"],
        median_cents=pitch_stats["median_cents"],
        pct_in_tune=pitch_stats["pct_in_tune"],
        drift_cents_per_s=pitch_stats["drift_cents_per_s"],
        arrival_offset_ms=arrival_offset_ms,
        core_start_s=cs,
        core_end_s=ce,
        note_octave_offset=pitch_stats["note_octave_offset"],
        pitch_tags=pitch_stats["pitch_tags"],
        arrival_tags=arrival_tags,
        user_rms_db=loud.get("user_rms_db"),
        ref_rms_db=loud.get("ref_rms_db"),
        rms_delta_db=loud.get("rms_delta_db"),
        rms_fade_db_per_s=loud.get("rms_fade_db_per_s"),
    )


def measure_song(
    reference: ReferenceAnnotation,
    *,
    pitch_user: PitchTrack,
    pitch_ref: Optional[PitchTrack] = None,
    stars_ref: Optional[StarsTrack] = None,
    stars_user: Optional[StarsTrack] = None,
    loudness_user: Optional[LoudnessTrack] = None,
    loudness_ref: Optional[LoudnessTrack] = None,
    config: Optional[CoachingConfig] = None,
    global_offset_s: Optional[float] = None,
    octave_shift_semitones: Optional[int] = None,
) -> tuple[
    list[NoteMeasurementV2],
    list[NoteTechniqueComparison],
    float,
    int,
]:
    """Run the full per-song dual-track measurement.

    Parameters
    ----------
    reference
        Built from the UltraStar chart; supplies note grid + lyrics/phones.
    pitch_user
        NanoPitch on the user vocal (in user time).
    pitch_ref
        Optional NanoPitch on the reference vocal. Currently accepted for
        forward-compat (e.g. future drift / register diagnostics); the
        per-performance octave shift is detected from chart-vs-user.
    stars_ref
        Reference STARS, already in song time.
    stars_user
        User STARS in user time. We shift it into song time using the
        estimated global offset.
    config
        Optional ``CoachingConfig`` override. Defaults are loaded if omitted.
    global_offset_s
        If provided, skip estimation and use this offset directly.
    octave_shift_semitones
        If provided, skip auto-detection and apply this integer-semitone
        offset (must be a multiple of 12) to chart MIDI before scoring.

    Returns
    -------
    (notes, techniques, global_offset_s, octave_shift_semitones)
    """
    cfg = config or CoachingConfig()
    del pitch_ref  # reserved; see docstring
    offset = (
        global_offset_s
        if global_offset_s is not None
        else estimate_global_offset_s(reference, pitch_user, config=cfg)
    )
    shift = (
        int(octave_shift_semitones)
        if octave_shift_semitones is not None
        else estimate_octave_shift_semitones(
            reference,
            pitch_user,
            global_offset_s=offset,
            config=cfg,
        )
    )

    user_arrays = shift_pitch_to_song_time(PitchArrays.from_track(pitch_user), offset)
    user_stars_song = map_user_stars_to_song_time(stars_user, offset)

    # Loudness arrays — user track is shifted to song time; reference is already
    # in song time (it was recorded against the chart directly).
    loud_user: Optional[LoudnessArrays] = (
        LoudnessArrays.from_track(loudness_user).shift_to_song_time(offset)
        if loudness_user is not None
        else None
    )
    loud_ref: Optional[LoudnessArrays] = (
        LoudnessArrays.from_track(loudness_ref) if loudness_ref is not None else None
    )

    ref_per_note = map_stars_phones_to_notes(reference, stars_ref)
    user_per_note = map_stars_phones_to_notes(reference, user_stars_song)

    notes_out: list[NoteMeasurementV2] = []
    techniques: list[NoteTechniqueComparison] = []
    for i, note in enumerate(reference.notes):
        prev = reference.notes[i - 1] if i > 0 else None
        notes_out.append(
            measure_note(
                note,
                prev_note=prev,
                pitch_user=user_arrays,
                stars_ref=stars_ref,
                config=cfg,
                octave_shift_semitones=shift,
                loudness_user=loud_user,
                loudness_ref=loud_ref,
            )
        )
        techniques.append(
            compare_note_techniques(
                note,
                ref_phones=ref_per_note.get(note.index, []),
                user_phones=user_per_note.get(note.index, []),
            )
        )
    return notes_out, techniques, float(offset), int(shift)


__all__ = [
    "LoudnessArrays",
    "PitchArrays",
    "compare_note_techniques",
    "core_window_s",
    "estimate_global_offset_s",
    "estimate_octave_shift_semitones",
    "map_stars_phones_to_notes",
    "map_user_stars_to_song_time",
    "measure_note",
    "measure_song",
    "shift_pitch_to_song_time",
    "shift_stars_to_song_time",
]
