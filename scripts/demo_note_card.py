"""Print note-level coaching cards for a sample.

This is the sprint-1 *stretch* deliverable: it proves that the four pipeline
tracks (reference, pitch, stars, loudness) compose into the user's target
JSON shape. By default the first five reference notes are aggregated and
printed as one JSON array. Sprint 2 will scan all notes and all granularities.

Usage::

    python scripts/demo_note_card.py data/samples/EN-Alto-1__innocence__0000
    python scripts/demo_note_card.py data/samples/EN-Alto-1__innocence__0000 --count 5
    python scripts/demo_note_card.py data/samples/EN-Alto-1__innocence__0000 --note-index 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.align import aggregate_note  # noqa: E402
from vocal_coach.reference import load_reference  # noqa: E402
from vocal_coach.schemas import NoteCard, LoudnessTrack, PitchTrack, ReferenceNote, StarsTrack, Timeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample_dir", type=Path)
    p.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of notes from the start of the reference (default: 5)",
    )
    p.add_argument(
        "--note-index",
        type=int,
        default=None,
        help="If set, emit only this note (overrides --count)",
    )
    return p.parse_args()


def _build_card(
    reference,
    pitch: PitchTrack,
    loudness: LoudnessTrack,
    stars: StarsTrack | None,
    note_index: int,
) -> tuple[ReferenceNote, NoteCard]:
    note = reference.notes[note_index]
    card = aggregate_note(
        note,
        reference=reference,
        pitch=pitch,
        loudness=loudness,
        stars=stars,
    )
    return note, card


def main() -> int:
    args = parse_args()
    timeline_path = args.sample_dir / "timeline.json"
    if not timeline_path.is_file():
        print(
            f"ERROR: {timeline_path} missing. Run scripts/run_pipeline.py first.",
            file=sys.stderr,
        )
        return 2

    timeline = Timeline.model_validate_json(timeline_path.read_text(encoding="utf-8"))
    reference = load_reference(args.sample_dir)
    n_notes = len(reference.notes)
    if n_notes == 0:
        print("ERROR: reference has no notes", file=sys.stderr)
        return 2

    if args.note_index is not None:
        indices = [args.note_index]
        if not (0 <= args.note_index < n_notes):
            print(
                f"ERROR: --note-index out of range (have {n_notes} notes)",
                file=sys.stderr,
            )
            return 2
    else:
        if args.count < 1:
            print("ERROR: --count must be at least 1", file=sys.stderr)
            return 2
        indices = list(range(min(args.count, n_notes)))

    cards: list[NoteCard] = []
    for idx in indices:
        note, card = _build_card(
            reference, timeline.pitch, timeline.loudness, timeline.stars, idx
        )
        cards.append(card)

        out_path = args.sample_dir / f"note_card_{note.index:03d}.json"
        out_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")

        print(
            f"Note {note.index}: {note.note_name} on {note.lyric_word!r}  "
            f"[{note.start_s:.2f}, {note.end_s:.2f}]  ({note.duration_s:.2f}s)  "
            f"-> {out_path.name}"
        )

    if len(cards) > 1:
        combined_path = args.sample_dir / f"note_cards_first{len(cards)}.json"
        combined_path.write_text(
            json.dumps([c.model_dump() for c in cards], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"Wrote combined {combined_path}")
        print()
        print(json.dumps([c.model_dump() for c in cards], indent=2, ensure_ascii=False))
    else:
        print()
        print(json.dumps(cards[0].model_dump(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
