"""Re-generate f0.npy for every clip in the student corpus using RMVPE.

This script is a surgical patch: it rewrites **only** ``f0.npy`` for each
clip that has a ``mel.npy`` (providing the target frame count).  Everything
else (mel.npy, labels.json, stars.json, manifest.jsonl) is left untouched.

Wav files are resolved by re-discovering clips from the raw data directories,
using the same ``discover_*`` functions as ``export_student_dataset.py``.

After running this you must retrain the student model so it sees RMVPE F0
at both train and inference time.

Usage::

    python scripts/regen_f0.py \\
        --gtsinger-dir  data/raw/GTSinger/English \\
        --nus48e-dir    data/raw/NUS_48E \\
        --songs-dir     data/songs \\
        --corpus-dir    data/student_corpus \\
        [--device cuda] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import discovery helpers from the export script to guarantee wav-path
# resolution is identical to what was used at corpus creation time.
from scripts.export_student_dataset import (  # noqa: E402
    discover_gtsinger_clips,
    discover_nus48e_clips,
    discover_song_clips,
)

import numpy as np  # noqa: E402
import librosa  # noqa: E402
from vocal_coach.rmvpe_f0 import extract_f0_rmvpe  # noqa: E402


# ---------------------------------------------------------------------------
# Constants matching export_student_dataset.py
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
HOP_LENGTH = 128


def _build_wav_lookup(
    gtsinger_dir: Path | None,
    nus48e_dir: Path | None,
    songs_dir: Path | None,
) -> dict[str, Path]:
    """Build {clip_id -> wav_path} by re-discovering clips from raw dirs."""
    lookup: dict[str, Path] = {}

    if gtsinger_dir and gtsinger_dir.is_dir():
        print(f"[regen_f0] Discovering GTSinger clips in {gtsinger_dir} ...")
        for clip in discover_gtsinger_clips(gtsinger_dir):
            lookup[clip.clip_id] = clip.wav_path
        print(f"           → {len(lookup)} GTSinger clips")

    prev = len(lookup)
    if nus48e_dir and nus48e_dir.is_dir():
        print(f"[regen_f0] Discovering NUS-48E clips in {nus48e_dir} ...")
        for clip in discover_nus48e_clips(nus48e_dir):
            lookup[clip.clip_id] = clip.wav_path
        print(f"           → {len(lookup) - prev} NUS-48E clips")

    prev = len(lookup)
    if songs_dir and songs_dir.is_dir():
        print(f"[regen_f0] Discovering song clips in {songs_dir} ...")
        for clip in discover_song_clips(songs_dir):
            lookup[clip.clip_id] = clip.wav_path
        print(f"           → {len(lookup) - prev} song clips")

    return lookup


def regen(
    corpus_dir: Path,
    wav_lookup: dict[str, Path],
    device: str,
    dry_run: bool,
) -> None:
    label_paths = sorted(corpus_dir.rglob("labels.json"))
    total = len(label_paths)
    print(f"\n[regen_f0] {total} clips in corpus. {len(wav_lookup)} wav paths resolved.")

    skipped_no_wav = 0
    skipped_no_mel = 0
    errors = 0
    done = 0
    t0 = time.monotonic()

    for i, label_path in enumerate(label_paths):
        clip_dir = label_path.parent
        mel_path = clip_dir / "mel.npy"
        f0_path = clip_dir / "f0.npy"

        if not mel_path.exists():
            skipped_no_mel += 1
            continue

        # Derive clip_id from directory structure: source/clip_id
        clip_id = clip_dir.name
        wav_path = wav_lookup.get(clip_id)
        if wav_path is None or not wav_path.is_file():
            skipped_no_wav += 1
            if skipped_no_wav <= 5 or (i + 1) % 500 == 0:
                print(f"  [skip no wav] {clip_id}")
            continue

        if dry_run:
            done += 1
            continue

        try:
            mel = np.load(mel_path)
            n_frames = mel.shape[0]

            wav, _sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
            f0 = extract_f0_rmvpe(
                wav,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                n_frames=n_frames,
                device=device,
            )
            np.save(f0_path, f0)
            done += 1
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {clip_id}: {exc}")

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (done + errors) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{total}] done={done} skip_wav={skipped_no_wav} "
                f"skip_mel={skipped_no_mel} err={errors} "
                f"speed={rate:.1f} clip/s ETA={eta/60:.1f}min"
            )

    elapsed = time.monotonic() - t0
    verb = "Would write" if dry_run else "Wrote"
    print(
        f"\n[regen_f0] {verb} {done} f0.npy in {elapsed/60:.1f} min. "
        f"skip_no_wav={skipped_no_wav}, skip_no_mel={skipped_no_mel}, errors={errors}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "data" / "student_corpus",
        help="Root of the student corpus.",
    )
    p.add_argument(
        "--gtsinger-dir",
        type=Path,
        default=None,
        help="GTSinger English directory (same as used for export).",
    )
    p.add_argument(
        "--nus48e-dir",
        type=Path,
        default=None,
        help="NUS-48E root directory.",
    )
    p.add_argument(
        "--songs-dir",
        type=Path,
        default=ROOT / "data" / "songs",
        help="UltraStar song bundles root.",
    )
    p.add_argument("--device", default="cuda", help="Torch device for RMVPE.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count resolvable clips; do not write any files.",
    )
    args = p.parse_args()

    if not args.corpus_dir.is_dir():
        sys.exit(f"[regen_f0] corpus-dir not found: {args.corpus_dir}")

    wav_lookup = _build_wav_lookup(args.gtsinger_dir, args.nus48e_dir, args.songs_dir)
    regen(args.corpus_dir, wav_lookup, args.device, args.dry_run)


if __name__ == "__main__":
    main()
