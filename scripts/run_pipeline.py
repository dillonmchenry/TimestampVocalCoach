"""End-to-end Sprint 1 driver.

Given a sample directory (the layout produced by ``scripts/download_gtsinger_sample.py``
+ ``scripts/build_reference.py``), runs:

    1. NanoPitch       -> pitch.json
    2. STARS inference -> stars.json (and an internal stars_out/ scratch dir)
    3. RMS loudness    -> loudness.json
    4. Joins all three with the reference annotation -> timeline.json

No alignment/highlight logic is performed in sprint 1.

Usage::

    python scripts/run_pipeline.py data/samples/EN-Alto-1__innocence__0000
    python scripts/run_pipeline.py data/samples/EN-Alto-1__innocence__0000 --skip-stars
    python scripts/run_pipeline.py data/samples/EN-Alto-1__innocence__0000 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.loudness import compute_loudness, write_loudness_track  # noqa: E402
from vocal_coach.pitch import extract_f0, write_pitch_track  # noqa: E402
from vocal_coach.reference import load_reference  # noqa: E402
from vocal_coach.schemas import Timeline  # noqa: E402
from vocal_coach.stars_runner import (  # noqa: E402
    DEFAULT_STARS_DIR,
    parse_stars_output,
    run_stars,
    write_stars_track,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample_dir", type=Path)
    p.add_argument(
        "--device", default="cuda",
        help="NanoPitch torch device (default: cuda; falls back to cpu automatically if no GPU)"
    )
    p.add_argument(
        "--skip-pitch", action="store_true",
        help="Reuse existing pitch.json instead of re-running NanoPitch",
    )
    p.add_argument(
        "--skip-stars", action="store_true",
        help="Reuse existing stars.json (or skip entirely if absent)",
    )
    p.add_argument(
        "--skip-loudness", action="store_true",
        help="Reuse existing loudness.json instead of recomputing",
    )
    p.add_argument(
        "--stars-dir", type=Path, default=DEFAULT_STARS_DIR,
        help="Path to the cloned STARS repo (default: third_party/stars)",
    )
    p.add_argument(
        "--cuda-visible-devices", default="0",
        help="Value for CUDA_VISIBLE_DEVICES when invoking STARS (default: 0)",
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
    print("[pipeline] CUDA unavailable; falling back to cpu for NanoPitch.")
    return "cpu"


def main() -> int:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    if not sample_dir.is_dir():
        print(f"ERROR: {sample_dir} is not a directory", file=sys.stderr)
        return 2

    ref_path = sample_dir / "reference_annotation.json"
    if not ref_path.is_file():
        print(
            f"ERROR: {ref_path} missing. Run scripts/build_reference.py first.",
            file=sys.stderr,
        )
        return 2

    reference = load_reference(sample_dir)
    wav_path = sample_dir / reference.audio_path
    sample_id = reference.sample_id
    print(f"[pipeline] sample_id  : {sample_id}")
    print(f"[pipeline] wav        : {wav_path}")
    print(f"[pipeline] duration   : {reference.duration_s:.2f}s")

    pitch_json = sample_dir / "pitch.json"
    if args.skip_pitch and pitch_json.is_file():
        print(f"[pipeline] pitch      : reusing {pitch_json}")
        from vocal_coach.schemas import PitchTrack
        pitch = PitchTrack.model_validate_json(pitch_json.read_text(encoding="utf-8"))
    else:
        device = _resolve_torch_device(args.device)
        print(f"[pipeline] pitch      : running NanoPitch on {device}")
        pitch = extract_f0(wav_path, device=device)
        # Override sample_id so it matches the rest of the timeline.
        pitch.sample_id = sample_id
        write_pitch_track(pitch, pitch_json)
        print(f"[pipeline] pitch      : wrote {pitch_json} ({len(pitch.frames)} frames)")

    loud_json = sample_dir / "loudness.json"
    if args.skip_loudness and loud_json.is_file():
        print(f"[pipeline] loudness   : reusing {loud_json}")
        from vocal_coach.schemas import LoudnessTrack
        loudness = LoudnessTrack.model_validate_json(loud_json.read_text(encoding="utf-8"))
    else:
        loudness = compute_loudness(wav_path, sample_id=sample_id)
        write_loudness_track(loudness, loud_json)
        print(f"[pipeline] loudness   : wrote {loud_json} ({len(loudness.frames)} frames)")

    stars_json = sample_dir / "stars.json"
    stars = None
    if args.skip_stars:
        if stars_json.is_file():
            from vocal_coach.schemas import StarsTrack
            stars = StarsTrack.model_validate_json(stars_json.read_text(encoding="utf-8"))
            print(f"[pipeline] stars      : reusing {stars_json}")
        else:
            print("[pipeline] stars      : skipped (no existing stars.json)")
    else:
        stars_meta_path = sample_dir / "stars_metadata.json"
        if not stars_meta_path.is_file():
            print(
                f"[pipeline] stars      : skipped ({stars_meta_path} missing; "
                "rerun build_reference.py to produce it)"
            )
        else:
            stars_save_dir = sample_dir / "stars_out"
            print(f"[pipeline] stars      : running STARS subprocess -> {stars_save_dir}")
            stars = run_stars(
                metadata_path=stars_meta_path,
                save_dir=stars_save_dir,
                sample_id=sample_id,
                stars_dir=args.stars_dir,
                cuda_visible_devices=args.cuda_visible_devices,
            )
            write_stars_track(stars, stars_json)
            print(f"[pipeline] stars      : wrote {stars_json} "
                  f"({len(stars.phonemes)} phonemes, {len(stars.notes)} notes)")

    timeline = Timeline(
        sample_id=sample_id,
        sample_dir=str(sample_dir).replace("\\", "/"),
        duration_s=reference.duration_s,
        reference=reference,
        pitch=pitch,
        stars=stars,
        loudness=loudness,
    )
    timeline_path = sample_dir / "timeline.json"
    timeline_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    print(f"[pipeline] timeline   : wrote {timeline_path}")
    print()
    print("Sprint 1 pipeline done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
