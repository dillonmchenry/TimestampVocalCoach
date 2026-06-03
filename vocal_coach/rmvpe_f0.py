"""Shared RMVPE F0 extractor for the student model pipeline.

Used by both:
  * ``scripts/export_student_dataset.py``  (training corpus generation)
  * ``vocal_coach/student_runner.py``       (inference)

Having a single implementation guarantees that the F0 feature is computed
identically at train and inference time, which is required for the student
model's technique classification to generalise correctly.

RMVPE operates natively at 16 kHz.  Its ``get_pitch`` method handles
resampling from any source sample rate and interpolates the output onto the
desired target frame grid, so the caller just specifies their hop_size and
sample_rate.

Model weights are expected at ``<repo_root>/rmvpe/model.pt`` (same location
that the STARS bilingual config points to via ``pe_ckpt``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STARS_DIR = _REPO_ROOT / "third_party" / "stars"
_DEFAULT_RMVPE_CKPT = _REPO_ROOT / "rmvpe" / "model.pt"

# Internal hop length RMVPE was designed for (samples at 16 kHz = 10 ms).
_RMVPE_NATIVE_HOP = 160


def _ensure_stars_on_path() -> None:
    """Add third_party/stars to sys.path so RMVPE imports resolve."""
    stars_str = str(_STARS_DIR)
    if stars_str not in sys.path:
        sys.path.insert(0, stars_str)


# ---------------------------------------------------------------------------
# Singleton model cache (loaded once per process)
# ---------------------------------------------------------------------------

_rmvpe_cache: dict[str, object] = {}  # key: "device:ckpt_path"


def _load_rmvpe(ckpt_path: Path, device: str):
    """Load (or return cached) RMVPE model."""
    cache_key = f"{device}:{ckpt_path}"
    if cache_key in _rmvpe_cache:
        return _rmvpe_cache[cache_key]

    _ensure_stars_on_path()
    from modules.pe.rmvpe import RMVPE  # type: ignore  (third-party)

    model = RMVPE(str(ckpt_path), hop_length=_RMVPE_NATIVE_HOP, device=device)
    _rmvpe_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_f0_rmvpe(
    wav: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    n_frames: int,
    device: str = "cuda",
    ckpt_path: Optional[Path] = None,
    fmin: float = 50.0,
    fmax: float = 1100.0,
) -> np.ndarray:
    """Return an F0 array of shape ``(n_frames,)`` in Hz, 0 = unvoiced.

    Args:
        wav:         Raw mono waveform as float32 numpy array.
        sample_rate: Sample rate of ``wav`` (e.g. 24000).
        hop_length:  Target hop length in samples at ``sample_rate``.
                     RMVPE resamples its 10 ms grid to this.
        n_frames:    Expected output length — should equal the mel frame count
                     so mel and f0 arrays have the same leading dimension.
        device:      Torch device string.
        ckpt_path:   Override for RMVPE checkpoint path.
        fmin/fmax:   Voiced-pitch range after extraction (Hz).  Notes outside
                     this range are zeroed (treated as unvoiced).
    """
    ckpt = ckpt_path or _DEFAULT_RMVPE_CKPT
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"RMVPE checkpoint not found at {ckpt}. "
            "Place rmvpe/model.pt in the repo root."
        )

    pe = _load_rmvpe(ckpt, device)
    f0, _uv = pe.get_pitch(
        wav,
        sample_rate=sample_rate,
        hop_size=hop_length,
        length=n_frames,
        interp_uv=False,  # keep zeros at unvoiced frames
        fmin=fmin,
        fmax=fmax,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    # Clip to n_frames in case get_pitch returns a marginally longer array.
    return f0[:n_frames]


def extract_f0_rmvpe_from_file(
    wav_path: Path,
    *,
    sample_rate: int,
    hop_length: int,
    n_frames: Optional[int] = None,
    device: str = "cuda",
    ckpt_path: Optional[Path] = None,
) -> np.ndarray:
    """Convenience wrapper: load wav from disk, return F0 on the mel grid.

    ``n_frames`` defaults to what librosa would compute for the given audio
    length; callers that already have the mel array should pass
    ``n_frames=mel.shape[0]`` for guaranteed alignment.
    """
    import librosa  # used only for loading; no pyin

    wav, _sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    if n_frames is None:
        # Estimate frame count the same way librosa.melspectrogram does with
        # center=True: n_frames = ceil(len(wav) / hop_length).
        n_frames = int(np.ceil(len(wav) / hop_length))

    return extract_f0_rmvpe(
        wav,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_frames=n_frames,
        device=device,
        ckpt_path=ckpt_path,
    )
