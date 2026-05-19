"""Build reference_annotation.json + stars_metadata.json from a GTSinger sample.

Run *after* ``scripts/download_gtsinger_sample.py``. Given the directory the
downloader produced (containing ``0000.wav`` + ``0000.json`` + others), this
script reads GTSinger's per-segment annotation and writes:

    <sample_dir>/reference_annotation.json   # ReferenceAnnotation schema
    <sample_dir>/stars_metadata.json         # what STARS inference consumes

Usage::

    python scripts/build_reference.py data/samples/EN-Alto-1__innocence__0000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script without `pip install -e .` by injecting the repo
# root onto sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.reference import (  # noqa: E402  (sys.path adjustment above)
    build_reference_annotation,
    write_reference_artifacts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample_dir", type=Path, help="Path to the GTSinger sample directory")
    p.add_argument("--sample-id", default=None, help="Override the sample id (default: derived from dir name)")
    p.add_argument("--language", default="English", help="Language tag (default: English)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sample_dir.is_dir():
        print(f"ERROR: {args.sample_dir} is not a directory", file=sys.stderr)
        return 2

    annotation = build_reference_annotation(
        args.sample_dir,
        sample_id=args.sample_id,
        language=args.language,
    )
    ref_path, stars_meta_path = write_reference_artifacts(args.sample_dir, annotation)

    print(f"Wrote {ref_path}")
    print(f"  sample_id : {annotation.sample_id}")
    print(f"  duration  : {annotation.duration_s:.2f}s")
    print(f"  words     : {len(annotation.words)}")
    print(f"  phones    : {len(annotation.phones)}")
    print(f"  notes     : {len(annotation.notes)}")
    print(f"Wrote {stars_meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
