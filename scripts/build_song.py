"""Run the reference-side measurement pipeline for a Sprint-2 song bundle.

Given ``data/songs/<song_id>/`` produced by ``import_ultrastar.py``, this
runs (and caches under ``reference/``):

    1. NanoPitch on the reference vocal -> reference/pitch.json
    2. STARS on the reference vocal     -> reference/stars.json (+ stars_out/)
    3. Per-frame RMS                    -> reference/loudness.json

It then updates ``manifest.json`` with the precomputed paths so the FastAPI
layer and ``analyze_performance.py`` don't have to recompute these on every
upload.

Usage::

    python scripts/build_song.py data/songs/losing-my-religion
    python scripts/build_song.py data/songs/losing-my-religion --skip-stars
    python scripts/build_song.py data/songs/losing-my-religion --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.loudness import compute_loudness, write_loudness_track  # noqa: E402
from vocal_coach.pitch import extract_f0, write_pitch_track  # noqa: E402
from vocal_coach.reference import load_reference  # noqa: E402
from vocal_coach.song import load_manifest, write_manifest  # noqa: E402
from vocal_coach.stars_runner import (  # noqa: E402
    DEFAULT_STARS_DIR,
    STARS_PROFILE_FULL,
    STARS_PROFILES,
    run_stars_with_profile,
    write_stars_track,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("song_dir", type=Path)
    p.add_argument(
        "--device",
        default="cuda",
        help="NanoPitch torch device (default: cuda; CPU fallback is automatic)",
    )
    p.add_argument(
        "--skip-pitch",
        action="store_true",
        help="Reuse existing reference/pitch.json instead of re-running NanoPitch",
    )
    p.add_argument(
        "--skip-stars",
        action="store_true",
        help="Reuse existing reference/stars.json (or skip entirely if absent)",
    )
    p.add_argument(
        "--skip-loudness",
        action="store_true",
        help="Reuse existing reference/loudness.json instead of recomputing",
    )
    p.add_argument(
        "--stars-dir",
        type=Path,
        default=DEFAULT_STARS_DIR,
        help="Path to the cloned STARS repo (default: third_party/stars)",
    )
    p.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="Value for CUDA_VISIBLE_DEVICES when invoking STARS",
    )
    p.add_argument(
        "--stars-profile",
        choices=list(STARS_PROFILES),
        default=STARS_PROFILE_FULL,
        help=(
            "Which STARS implementation to use for the reference vocal. "
            "Default 'full' (teacher) preserves the rich style/mood needed "
            "for downstream coaching. 'fast' uses the distilled student."
        ),
    )
    p.add_argument(
        "--student-dir",
        type=Path,
        default=None,
        help="Override the student checkpoint directory (defaults to ./stars_student).",
    )
    return p.parse_args()


def _resolve_torch_device(requested: str) -> str:
    if requested.lower() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return requested
    except ImportError:
        pass
    print("[build_song] CUDA unavailable; falling back to cpu for NanoPitch.")
    return "cpu"


def main() -> int:
    args = parse_args()
    song_dir = args.song_dir.resolve()
    if not song_dir.is_dir():
        print(f"ERROR: {song_dir} is not a directory", file=sys.stderr)
        return 2
    if not (song_dir / "manifest.json").is_file():
        print(
            f"ERROR: {song_dir / 'manifest.json'} missing. "
            "Run scripts/import_ultrastar.py first.",
            file=sys.stderr,
        )
        return 2

    manifest = load_manifest(song_dir)
    reference = load_reference(song_dir)
    ref_audio = song_dir / manifest.reference_vocal_path
    if not ref_audio.is_file():
        print(f"ERROR: reference vocal not found at {ref_audio}", file=sys.stderr)
        return 2

    sample_id = reference.sample_id
    print(f"[build_song] song_id    : {manifest.song_id}")
    print(f"[build_song] vocal      : {ref_audio}")
    print(f"[build_song] duration   : {manifest.duration_s:.2f}s")

    ref_subdir = song_dir / "reference"
    ref_subdir.mkdir(parents=True, exist_ok=True)

    pitch_json = ref_subdir / "pitch.json"
    if args.skip_pitch and pitch_json.is_file():
        print(f"[build_song] pitch      : reusing {pitch_json}")
    else:
        device = _resolve_torch_device(args.device)
        print(f"[build_song] pitch      : running NanoPitch on {device}")
        pitch = extract_f0(ref_audio, device=device)
        pitch.sample_id = sample_id
        write_pitch_track(pitch, pitch_json)
        print(f"[build_song] pitch      : wrote {pitch_json} ({len(pitch.frames)} frames)")

    loud_json = ref_subdir / "loudness.json"
    if args.skip_loudness and loud_json.is_file():
        print(f"[build_song] loudness   : reusing {loud_json}")
    else:
        loudness = compute_loudness(ref_audio, sample_id=sample_id)
        write_loudness_track(loudness, loud_json)
        print(
            f"[build_song] loudness   : wrote {loud_json} ({len(loudness.frames)} frames)"
        )

    stars_json = ref_subdir / "stars.json"
    stars_meta_path = song_dir / "stars_metadata.json"
    if args.skip_stars:
        if stars_json.is_file():
            print(f"[build_song] stars      : reusing {stars_json}")
        else:
            print("[build_song] stars      : skipped (no existing stars.json)")
    elif not stars_meta_path.is_file():
        print(
            f"[build_song] stars      : skipped ({stars_meta_path} missing; "
            "rerun scripts/import_ultrastar.py)"
        )
    else:
        stars_save_dir = ref_subdir / "stars_out"
        print(
            f"[build_song] stars      : profile={args.stars_profile} -> {stars_save_dir}"
        )
        stars = run_stars_with_profile(
            profile=args.stars_profile,
            metadata_path=stars_meta_path,
            save_dir=stars_save_dir,
            sample_id=sample_id,
            stars_dir=args.stars_dir,
            cuda_visible_devices=args.cuda_visible_devices,
            student_dir=args.student_dir,
        )
        write_stars_track(stars, stars_json)
        print(
            f"[build_song] stars      : wrote {stars_json} "
            f"({len(stars.phonemes)} phonemes, {len(stars.notes)} notes)"
        )

    manifest.reference_pitch_path = (
        f"reference/{pitch_json.name}" if pitch_json.is_file() else None
    )
    manifest.reference_stars_path = (
        f"reference/{stars_json.name}" if stars_json.is_file() else None
    )
    manifest.reference_loudness_path = (
        f"reference/{loud_json.name}" if loud_json.is_file() else None
    )
    write_manifest(song_dir, manifest)
    print(f"[build_song] manifest   : updated {song_dir / 'manifest.json'}")
    print()
    print("Reference pipeline done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
