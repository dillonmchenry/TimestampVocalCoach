"""Per-frame loudness extraction.

Sprint 1 emits a frame-aligned loudness track at the same 10 ms hop NanoPitch
uses, so downstream alignment can read pitch and loudness from the same
timeline. We do not compute LUFS or other broadcast measures here; a simple
RMS in dBFS is enough to detect "fades near end" and similar coaching cues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from vocal_coach.schemas import LoudnessFrame, LoudnessTrack


# Default cadence: matches NanoPitch's 10 ms hop at 16 kHz so the loudness
# array can be indexed directly against the pitch array.
DEFAULT_TARGET_SR = 16000
DEFAULT_HOP_SAMPLES = 160  # 10 ms @ 16 kHz
DEFAULT_FRAME_SAMPLES = 400  # 25 ms @ 16 kHz, mirrors mel window
LOUDNESS_DB_FLOOR = -120.0


def compute_loudness(
    wav_path: Path,
    *,
    target_sr: int = DEFAULT_TARGET_SR,
    hop_samples: int = DEFAULT_HOP_SAMPLES,
    frame_samples: int = DEFAULT_FRAME_SAMPLES,
    sample_id: Optional[str] = None,
) -> LoudnessTrack:
    """Compute a per-frame RMS dBFS track for the given wav.

    Parameters
    ----------
    wav_path
        Path to a wav file (any sample rate / channel count).
    target_sr, hop_samples, frame_samples
        Frame cadence. Defaults align with NanoPitch (10 ms hop, 25 ms window
        at 16 kHz).
    sample_id
        Optional id stored on the LoudnessTrack. Defaults to ``wav_path.stem``.
    """
    wav, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    rms = librosa.feature.rms(
        y=wav,
        frame_length=frame_samples,
        hop_length=hop_samples,
        center=False,
    ).squeeze(0)  # (T,)

    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    rms_db = np.maximum(rms_db, LOUDNESS_DB_FLOOR).astype(np.float32)

    hop_seconds = hop_samples / float(target_sr)
    frames = [
        LoudnessFrame(time=float(i) * hop_seconds, rms_db=float(rms_db[i]))
        for i in range(len(rms_db))
    ]

    return LoudnessTrack(
        sample_id=sample_id or wav_path.stem,
        sample_rate=target_sr,
        hop_seconds=hop_seconds,
        frames=frames,
    )


def write_loudness_track(track: LoudnessTrack, out_path: Path) -> None:
    Path(out_path).write_text(track.model_dump_json(indent=2), encoding="utf-8")
