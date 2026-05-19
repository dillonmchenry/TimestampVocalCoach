"""Download exactly one GTSinger segment to ``data/samples/<sample_id>/``.

GTSinger as a whole is large (~80 GB across 9 languages). For sprint 1 we only
need one English segment, so this script targets a *single segment folder* and
fetches its four constituent files (.wav, .json, .TextGrid, .musicxml) via the
HuggingFace Hub HTTP API instead of cloning the dataset repo.

Default selection: ``English/EN-Alto-1/Breathy/innocence/Control_Group/0000``
(a short segment with sane note durations; sleepyhead/0000 has a known
2.8 s mis-label on the word "the"). Override with ``--segment-path``.

Usage::

    python scripts/download_gtsinger_sample.py
    python scripts/download_gtsinger_sample.py --segment-path English/EN-Tenor-1/Vibrato/treasure/Control_Group/0003
    python scripts/download_gtsinger_sample.py --output data/samples/my_sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_REPO_ID = "GTSinger/GTSinger"
DEFAULT_SEGMENT_PATH = "English/EN-Alto-1/Breathy/innocence/Control_Group/0000"
EXPECTED_EXTENSIONS = (".wav", ".json", ".TextGrid", ".musicxml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--segment-path",
        default=DEFAULT_SEGMENT_PATH,
        help=(
            "Path of the segment *without* extension, relative to the dataset root. "
            f"Default: {DEFAULT_SEGMENT_PATH}"
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        help="Local directory to put the four files. Default: data/samples/<sample_id>/",
    )
    p.add_argument(
        "--sample-id",
        default=None,
        help="Override the directory name under data/samples/. "
             "Default: derived from the segment path.",
    )
    p.add_argument(
        "--repo-id",
        default=DATASET_REPO_ID,
        help=f"HuggingFace dataset repo id (default: {DATASET_REPO_ID})",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Dataset revision/branch (default: main)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Missing dependency: huggingface_hub.\n"
            "Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    seg_no_ext = args.segment_path.rstrip("/")
    if any(seg_no_ext.endswith(ext) for ext in EXPECTED_EXTENSIONS):
        print(
            f"--segment-path should be the file stem WITHOUT extension; got {seg_no_ext}",
            file=sys.stderr,
        )
        return 2

    if args.sample_id is None:
        # English/EN-Alto-1/Breathy/innocence/Control_Group/0000 ->
        #   EN-Alto-1__innocence__0000
        parts = seg_no_ext.split("/")
        if len(parts) >= 5:
            sample_id = f"{parts[1]}__{parts[3].replace(' ', '_')}__{parts[-1]}"
        else:
            sample_id = parts[-1]
    else:
        sample_id = args.sample_id

    workspace_root = Path(__file__).resolve().parents[1]
    if args.output is None:
        output_dir = workspace_root / "data" / "samples" / sample_id
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GTSinger segment '{seg_no_ext}' -> {output_dir}")
    for ext in EXPECTED_EXTENSIONS:
        remote_path = f"{seg_no_ext}{ext}"
        try:
            local = hf_hub_download(
                repo_id=args.repo_id,
                filename=remote_path,
                repo_type="dataset",
                revision=args.revision,
                local_dir=str(output_dir),
            )
        except Exception as exc:
            print(f"  [warn] failed to download {remote_path}: {exc}", file=sys.stderr)
            continue
        print(f"  ok  {remote_path}")

        # hf_hub_download nests the file under its remote path inside local_dir.
        # Flatten to <output_dir>/<basename> so build_reference.py can find it.
        nested = Path(local)
        flat = output_dir / nested.name
        if nested.resolve() != flat.resolve():
            flat.write_bytes(nested.read_bytes())

    # Clean up the per-fetch nested directories (e.g. ``English/EN-Alto-1/...``
    # and the ``.cache`` lock dir) that hf_hub_download leaves behind.
    import shutil as _shutil
    for stray in output_dir.iterdir():
        if stray.is_dir() and stray.name not in {""}:  # always remove subdirs
            _shutil.rmtree(stray, ignore_errors=True)

    # Sanity: do we have a wav and a json?
    wavs = list(output_dir.glob("*.wav"))
    jsons = [p for p in output_dir.glob("*.json")
             if p.name not in {"reference_annotation.json", "stars_metadata.json"}]
    if not wavs or not jsons:
        print(
            f"ERROR: expected at least one .wav and one annotation .json in {output_dir}; "
            f"found wavs={[w.name for w in wavs]}, jsons={[j.name for j in jsons]}",
            file=sys.stderr,
        )
        return 3

    print(f"\nSample ready at {output_dir}")
    print(f"  sample_id: {sample_id}")
    print(f"  wav:       {wavs[0].name}")
    print(f"  ann:       {jsons[0].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
