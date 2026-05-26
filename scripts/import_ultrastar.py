"""Import an UltraStar song bundle into ``data/songs/<song_id>/``.

This converts an UltraStar ``.txt`` chart + isolated reference vocal +
optional instrumental into the Sprint-2 song layout::

    data/songs/<song_id>/
        manifest.json
        <chart>.txt
        <reference vocal>.{wav,mp3,...}
        <instrumental>.{wav,mp3,...}     (optional)
        reference_annotation.json
        stars_metadata.json

Usage::

    python scripts/import_ultrastar.py data/Losing-My-Religion --song-id losing-my-religion
    python scripts/import_ultrastar.py data/songs/losing-my-religion
    python scripts/import_ultrastar.py data/Losing-My-Religion --midi-offset -12

The importer auto-detects the chart file (single ``.txt``), the reference
vocal (file with ``vocal`` in the name and no ``instrumental`` substring),
and the instrumental (file with ``instrumental`` in the name). Override any
of those with the matching ``--*`` flag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.song import import_ultrastar_song  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "song_dir",
        type=Path,
        help="Directory holding the UltraStar .txt chart and audio assets",
    )
    p.add_argument(
        "--song-id",
        default=None,
        help="Override the song slug (default: derived from #TITLE)",
    )
    p.add_argument(
        "--chart",
        default=None,
        help="Filename of the UltraStar .txt chart (default: auto-detect)",
    )
    p.add_argument(
        "--reference-vocal",
        default=None,
        help="Filename of the isolated reference vocal (default: auto-detect)",
    )
    p.add_argument(
        "--instrumental",
        default=None,
        help="Filename of the instrumental backing track (default: auto-detect)",
    )
    p.add_argument(
        "--midi-offset",
        type=int,
        default=0,
        help=(
            "Constant offset added to UltraStar pitch when converting to MIDI. "
            "Use this to fix charts that are an octave off the actual recording "
            "(e.g. -12 to drop one octave)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    song_dir = args.song_dir.resolve()
    if not song_dir.is_dir():
        print(f"ERROR: {song_dir} is not a directory", file=sys.stderr)
        return 2

    annotation, manifest = import_ultrastar_song(
        song_dir=song_dir,
        chart_filename=args.chart,
        reference_vocal=args.reference_vocal,
        instrumental=args.instrumental,
        song_id=args.song_id,
        midi_offset=args.midi_offset,
    )

    print(f"Imported UltraStar song to {song_dir}")
    print(f"  song_id     : {manifest.song_id}")
    print(f"  title       : {manifest.title}")
    print(f"  artist      : {manifest.artist}")
    print(f"  language    : {manifest.language}")
    print(f"  duration    : {manifest.duration_s:.2f}s")
    print(f"  notes       : {len(annotation.notes)}")
    print(f"  words       : {len(annotation.words)}")
    print(f"  phones      : {len(annotation.phones)}")
    print(f"  sections    : {len(annotation.sections)}")
    print(f"  bpm         : {manifest.ultrastar.bpm}")
    print(f"  gap_ms      : {manifest.ultrastar.gap_ms}")
    print(f"  midi_offset : {manifest.ultrastar.midi_offset}")
    print(f"  chart       : {manifest.chart_path}")
    print(f"  vocal       : {manifest.reference_vocal_path}")
    print(f"  instrumental: {manifest.instrumental_path or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
