"""Validate the NanoPitch wrapper against the canonical evaluate.py path.

Our pipeline has two responsibilities:

1. wav -> log-mel (re-implemented in librosa to match the C constants)
2. log-mel -> model -> Viterbi -> f0 (a thin wrapper over NanoPitch's own code)

(2) is the easier one to validate parity-style: feed a pre-extracted clip from
``data/test.npz`` (used by NanoPitch's evaluate.py) directly through both our
wrapper and the model code, and confirm both decoded F0 arrays match.

(1) — the wav->mel re-implementation — has no rigorous parity test because
``data/test.npz`` only contains the *output* mels, not the source wavs. We
report a few sanity numbers (mel mean/std, NaN/inf check, voicing fraction
on the GTSinger sample) so a regression here would be visible.

Usage::

    python scripts/validate_pitch.py
    python scripts/validate_pitch.py --clip-index 7 --device cuda
    python scripts/validate_pitch.py --no-gtsinger-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.pitch import (  # noqa: E402
    DEFAULT_NANOPITCH_DIR,
    DEFAULT_CHECKPOINT_RELPATH,
    NANOPITCH_N_MELS,
    load_nanopitch,
    run_nanopitch_on_logmel,
    wav_to_logmel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nanopitch-dir", type=Path, default=DEFAULT_NANOPITCH_DIR)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override the checkpoint path (defaults to the best-150 run inside --nanopitch-dir)",
    )
    p.add_argument(
        "--clip-index",
        type=int,
        default=0,
        help="Which clip from data/test.npz to use for the parity check (default: 0)",
    )
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    p.add_argument(
        "--gtsinger-wav",
        type=Path,
        default=ROOT / "data/samples/EN-Alto-1__innocence__0000/0000.wav",
        help="Wav whose log-mel + decoded F0 will be sanity-checked",
    )
    p.add_argument(
        "--no-gtsinger-check",
        action="store_true",
        help="Skip the wav->mel sanity step",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for parity comparison",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for parity comparison (Hz)",
    )
    return p.parse_args()


def parity_check(args: argparse.Namespace) -> bool:
    """Confirm wrapper(model+decoder) matches a hand-rolled (model+decoder) call."""
    nanopitch_dir = args.nanopitch_dir.resolve()
    test_npz = nanopitch_dir / "data" / "test.npz"
    if not test_npz.is_file():
        print(f"  [skip] {test_npz} not found; cannot run parity check.")
        return True

    checkpoint = args.checkpoint or (nanopitch_dir / DEFAULT_CHECKPOINT_RELPATH)
    print(f"  Loading model from {checkpoint}")
    model, viterbi_realtime, _ = load_nanopitch(
        checkpoint, nanopitch_dir=nanopitch_dir, device=args.device
    )

    print(f"  Loading {test_npz} (clip {args.clip_index})")
    npz = np.load(str(test_npz))
    clip = npz["clips"][args.clip_index]  # (T, 40)
    if clip.shape[-1] != NANOPITCH_N_MELS:
        print(f"  [fail] clip has {clip.shape[-1]} mel bands, expected {NANOPITCH_N_MELS}")
        return False
    log_mel = clip.astype(np.float32)

    # Path A: through our wrapper
    f0_wrapper, voicing_wrapper = run_nanopitch_on_logmel(
        model, viterbi_realtime, log_mel, device=args.device
    )

    # Path B: hand-rolled (mirrors evaluate.py)
    with torch.no_grad():
        mel_t = torch.from_numpy(log_mel).unsqueeze(0).to(args.device)
        vad, pitch, _ = model(mel_t)
    voicing_ref = vad.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32)
    posteriorgram = pitch.squeeze(0).cpu().numpy().astype(np.float32)
    f0_ref = viterbi_realtime(posteriorgram).astype(np.float32)

    ok_v = np.allclose(voicing_wrapper, voicing_ref, rtol=args.rtol, atol=args.atol)
    ok_f = np.allclose(f0_wrapper, f0_ref, rtol=args.rtol, atol=args.atol)
    print(f"  voicing parity: {'PASS' if ok_v else 'FAIL'} (max abs diff {np.max(np.abs(voicing_wrapper - voicing_ref)):.2e})")
    print(f"  f0 parity     : {'PASS' if ok_f else 'FAIL'} (max abs diff {np.max(np.abs(f0_wrapper - f0_ref)):.2e})")
    return bool(ok_v and ok_f)


def gtsinger_sanity(args: argparse.Namespace) -> bool:
    """Sanity-check wav->mel + voicing on the GTSinger sample."""
    wav_path = args.gtsinger_wav
    if not wav_path.is_file():
        print(f"  [skip] {wav_path} not found; run scripts/download_gtsinger_sample.py first.")
        return True

    import librosa
    wav, sr = librosa.load(str(wav_path), sr=None, mono=True)
    log_mel = wav_to_logmel(wav, sr)

    n_nan = int(np.isnan(log_mel).sum())
    n_inf = int(np.isinf(log_mel).sum())
    mean = float(log_mel.mean())
    std = float(log_mel.std())
    print(f"  wav         : {wav_path.name} ({sr} Hz, {len(wav)/sr:.2f}s)")
    print(f"  log_mel     : shape {log_mel.shape}, mean {mean:.2f}, std {std:.2f}, nan {n_nan}, inf {n_inf}")
    if n_nan or n_inf:
        print("  [fail] non-finite values in log-mel; mel preprocessing is broken.")
        return False

    nanopitch_dir = args.nanopitch_dir.resolve()
    checkpoint = args.checkpoint or (nanopitch_dir / DEFAULT_CHECKPOINT_RELPATH)
    model, viterbi_realtime, _ = load_nanopitch(
        checkpoint, nanopitch_dir=nanopitch_dir, device=args.device
    )
    f0, voicing = run_nanopitch_on_logmel(
        model, viterbi_realtime, log_mel, device=args.device
    )

    voiced = f0 > 0
    voiced_frac = float(voiced.mean())
    if voiced.any():
        f0_voiced = f0[voiced]
        f0_med = float(np.median(f0_voiced))
        f0_min = float(f0_voiced.min())
        f0_max = float(f0_voiced.max())
    else:
        f0_med = f0_min = f0_max = float("nan")
    print(f"  voiced frac : {voiced_frac:.2%}")
    print(f"  f0 (voiced) : median {f0_med:.1f} Hz, range {f0_min:.1f}-{f0_max:.1f} Hz")

    # Loose sanity bounds: an alto vocal should sit roughly within these.
    ok = (
        0.2 < voiced_frac < 0.95
        and 80.0 < f0_med < 800.0
        and not (np.isnan(f0_med) or np.isinf(f0_med))
    )
    print(f"  bounds      : {'PASS' if ok else 'FAIL'} (expected voiced_frac in [0.2, 0.95], f0_med in [80, 800] Hz)")
    return ok


def main() -> int:
    args = parse_args()
    print("[1/2] Parity check (test.npz clip -> wrapper vs hand-rolled)")
    parity_ok = parity_check(args)

    if args.no_gtsinger_check:
        print("\n[2/2] GTSinger wav->mel sanity check skipped.")
        return 0 if parity_ok else 1

    print("\n[2/2] GTSinger wav->mel sanity check")
    sanity_ok = gtsinger_sanity(args)

    print("\nResult:")
    print(f"  parity : {'PASS' if parity_ok else 'FAIL'}")
    print(f"  sanity : {'PASS' if sanity_ok else 'FAIL'}")
    return 0 if (parity_ok and sanity_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
