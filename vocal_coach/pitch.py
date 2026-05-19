"""NanoPitch wrapper: wav -> log-mel -> NanoPitch -> Viterbi -> PitchTrack.

NanoPitch is implemented in a separate repo (default
``C:/Users/dillo/Documents/GitHub/NanoPitch``). The model and decoders live
in ``training/model.py``; this module imports them by appending the training
directory to ``sys.path``.

The mel preprocessing constants are pinned by NanoPitch's C/WASM deployment
(``deployment/wasm/nanopitch.h``):

    sample_rate = 16000
    n_fft       = 512
    hop_length  = 160   (10 ms)
    win_length  = 400   (25 ms)
    n_mels      = 40
    mel_scale   = HTK
    fmin, fmax  = 0, 8000
    log         = natural log of power mel

There is no Python wav->mel function in the NanoPitch repo (training consumes
pre-extracted ``.npz`` mels), so we re-implement the spec above using librosa.
A validation script in ``scripts/validate_pitch.py`` checks that our wav->mel
+ model + decoder pipeline produces F0 matching ``training/evaluate.py`` on a
clip that already has both wav and mel available in ``data/test.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import librosa
import numpy as np
import torch

from vocal_coach.schemas import PitchFrame, PitchTrack


# ---------------------------------------------------------------------------
# Mel preprocessing — must mirror the constants in deployment/wasm/nanopitch.h
# ---------------------------------------------------------------------------

NANOPITCH_SAMPLE_RATE = 16000
NANOPITCH_N_FFT = 512
NANOPITCH_HOP_LENGTH = 160
NANOPITCH_WIN_LENGTH = 400
NANOPITCH_N_MELS = 40
NANOPITCH_FMIN = 0.0
NANOPITCH_FMAX = 8000.0
NANOPITCH_HOP_SECONDS = NANOPITCH_HOP_LENGTH / NANOPITCH_SAMPLE_RATE  # 0.01

# Default location of NanoPitch on disk.  Override with the env var
# NANOPITCH_DIR (project root) or by passing nanopitch_dir explicitly.
DEFAULT_NANOPITCH_DIR = Path("C:/Users/dillo/Documents/GitHub/NanoPitch")
DEFAULT_CHECKPOINT_RELPATH = "training/runs/best_150+late_clean_112gru_model/checkpoints/best.pth"


def _resolve_nanopitch_dir(nanopitch_dir: Optional[Path] = None) -> Path:
    import os

    if nanopitch_dir is not None:
        return Path(nanopitch_dir).resolve()
    env = os.environ.get("NANOPITCH_DIR")
    if env:
        return Path(env).resolve()
    return DEFAULT_NANOPITCH_DIR.resolve()


def _ensure_nanopitch_on_path(nanopitch_dir: Path) -> None:
    """Insert NanoPitch's ``training/`` dir on sys.path so we can import model.py."""
    training_dir = (nanopitch_dir / "training").resolve()
    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"NanoPitch training directory not found at {training_dir}. "
            "Set NANOPITCH_DIR or pass nanopitch_dir=... explicitly."
        )
    p = str(training_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def wav_to_logmel(
    wav: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Convert a (possibly wrong-rate, possibly stereo) waveform into NanoPitch
    log-mel features of shape ``(T, 40)``.

    The mel parameters mirror the C/WASM defaults so the trained weights see
    the same statistics they were trained on.
    """
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    if sr != NANOPITCH_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=NANOPITCH_SAMPLE_RATE)
    wav = wav.astype(np.float32, copy=False)

    # ``norm=None`` is critical: NanoPitch's C/WASM implementation uses
    # unnormalized triangular mel filters (peak = 1.0). Librosa's default
    # ``slaney`` normalization rescales each filter by 2/(f_high - f_low),
    # producing a ~4.5 nat offset in log-mel mean that breaks the VAD head's
    # calibration even though the pitch posteriorgram still tracks correctly.
    mel_power = librosa.feature.melspectrogram(
        y=wav,
        sr=NANOPITCH_SAMPLE_RATE,
        n_fft=NANOPITCH_N_FFT,
        hop_length=NANOPITCH_HOP_LENGTH,
        win_length=NANOPITCH_WIN_LENGTH,
        n_mels=NANOPITCH_N_MELS,
        fmin=NANOPITCH_FMIN,
        fmax=NANOPITCH_FMAX,
        htk=True,
        norm=None,
        power=2.0,
        center=False,
    )  # (n_mels, T)

    log_mel = np.log(mel_power + 1e-10).astype(np.float32)
    return log_mel.T  # (T, n_mels)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_nanopitch(
    checkpoint_path: Path,
    nanopitch_dir: Optional[Path] = None,
    device: str = "cpu",
):
    """Load a NanoPitch model from a .pth checkpoint, mirroring evaluate.py.

    Returns ``(model, viterbi_decode_realtime, viterbi_decode_offline)``.
    """
    nanopitch_dir = _resolve_nanopitch_dir(nanopitch_dir)
    _ensure_nanopitch_on_path(nanopitch_dir)

    # Imported lazily so the module is usable without NanoPitch on the path
    # (e.g. for schema-only test runs).
    from model import NanoPitch, viterbi_decode, viterbi_decode_realtime  # type: ignore

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    kwargs = dict(ckpt.get("model_kwargs", {"cond_size": 64, "gru_size": 96}))
    kwargs.pop("dropout_p", None)  # legacy checkpoints
    model = NanoPitch(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, viterbi_decode_realtime, viterbi_decode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_nanopitch_on_logmel(
    model,
    decoder,
    log_mel: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Forward + decode in one shot.

    Parameters
    ----------
    model
        A loaded NanoPitch.
    decoder
        Either ``viterbi_decode_realtime`` (default for streaming-comparable
        results) or ``viterbi_decode`` (offline, global-optimal).
    log_mel
        ``(T, 40)`` log-mel feature array.

    Returns
    -------
    (f0_hz, voicing_prob)
        Both 1D float32 arrays of length T.
    """
    if log_mel.ndim != 2 or log_mel.shape[-1] != NANOPITCH_N_MELS:
        raise ValueError(
            f"log_mel must be (T, {NANOPITCH_N_MELS}); got shape {log_mel.shape}"
        )
    mel_t = torch.from_numpy(log_mel.astype(np.float32)).unsqueeze(0).to(device)
    vad, pitch, _ = model(mel_t)
    voicing = vad.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32)
    posteriorgram = pitch.squeeze(0).cpu().numpy().astype(np.float32)
    f0 = decoder(posteriorgram).astype(np.float32)
    return f0, voicing


def extract_f0(
    wav_path: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    nanopitch_dir: Optional[Path] = None,
    decoder: Literal["realtime", "offline"] = "realtime",
    device: str = "cpu",
) -> PitchTrack:
    """Run NanoPitch on a wav and return a typed ``PitchTrack``.

    Parameters
    ----------
    wav_path
        Path to a wav (any sample rate / channel count; we resample/downmix).
    checkpoint_path
        Path to a NanoPitch ``.pth``. Defaults to the best-150 run inside
        ``DEFAULT_NANOPITCH_DIR``.
    nanopitch_dir
        Path to the NanoPitch project root (defaults to DEFAULT_NANOPITCH_DIR).
    decoder
        ``"realtime"`` (default) matches the WASM/browser behavior;
        ``"offline"`` uses the globally-optimal Viterbi for ground-truth comparison.
    device
        ``"cpu"``, ``"cuda"``, or ``"cuda:0"`` etc.
    """
    nanopitch_dir = _resolve_nanopitch_dir(nanopitch_dir)
    if checkpoint_path is None:
        checkpoint_path = nanopitch_dir / DEFAULT_CHECKPOINT_RELPATH
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"NanoPitch checkpoint not found at {checkpoint_path}"
        )

    wav, sr = librosa.load(str(wav_path), sr=None, mono=True)
    log_mel = wav_to_logmel(wav, sr)

    model, viterbi_realtime, viterbi_offline = load_nanopitch(
        checkpoint_path, nanopitch_dir=nanopitch_dir, device=device
    )
    chosen = viterbi_realtime if decoder == "realtime" else viterbi_offline
    f0, voicing = run_nanopitch_on_logmel(model, chosen, log_mel, device=device)

    frames = [
        PitchFrame(
            time=float(i) * NANOPITCH_HOP_SECONDS,
            f0_hz=float(f0[i]),
            voicing_confidence=float(voicing[i]),
        )
        for i in range(len(f0))
    ]
    sample_id = wav_path.stem
    return PitchTrack(
        sample_id=sample_id,
        sample_rate=NANOPITCH_SAMPLE_RATE,
        hop_seconds=NANOPITCH_HOP_SECONDS,
        decoder=decoder,
        checkpoint=str(checkpoint_path).replace("\\", "/"),
        frames=frames,
    )


def write_pitch_track(track: PitchTrack, out_path: Path) -> None:
    Path(out_path).write_text(track.model_dump_json(indent=2), encoding="utf-8")
