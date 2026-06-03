"""Build a `ReferenceAnnotation` from a single GTSinger sample.

GTSinger ships per-segment annotations as four files in one folder:

    0000.wav        # 48 kHz mono vocal
    0000.json       # rich per-word annotation (notes, phonemes, techniques)
    0000.TextGrid   # Praat alignment (we don't need it; .json subsumes it)
    0000.musicxml   # score (we don't need it for sprint 1)

The .json structure (one entry per *word*, including " " silences) looks like:

    [
      {
        "word": "the",
        "start_time": 0.78, "end_time": 3.62,
        "ph": ["DH", "AH0"],
        "ph_start": [0.78, 0.82],
        "ph_end":   [0.82, 3.62],
        "note":     [60],
        "note_start":[0.78], "note_end":[3.62], "note_dur":[2.82051],
        "vibrato": ["0", "0"], ...
        "singing_method": "pop", "pace": "moderate",
        "range": "medium", "emotion": "happy"
      },
      ...
    ]

We project that into our `ReferenceAnnotation` schema, which is also the
source from which `stars_metadata.json` is derived.

Notes on phoneme/word handling:
- Silences in GTSinger are encoded as a word of " " with a phoneme of " ".
  STARS's bilingual phone-set uses "<SP>" for silence, so we remap " " -> "<SP>".
- A single word can have multiple notes (a slur). We emit one ReferenceNote
  per entry of `note[]`, attaching only phonemes whose `ph_start`/`ph_end`
  overlap that note's `note_start`/`note_end`.
- We skip rest entries (note == 0) when building ReferenceNote, since they
  are silences, not sung notes. The phonemes/words are still kept in the
  flattened ph/word lists for STARS alignment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import soundfile as sf

from vocal_coach.schemas import (
    ReferenceAnnotation,
    ReferenceNote,
    ReferenceSection,
    StarsMetadataEntry,
)


# Used to remap GTSinger's " " phoneme into STARS's "<SP>" silence token.
GTSINGER_SILENCE_PH = " "
STARS_SILENCE_PH = "<SP>"


def _midi_to_name(midi: int) -> str:
    """Convert a MIDI number to a pitch-class+octave name, e.g. 60 -> C4."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_class = names[midi % 12]
    octave = midi // 12 - 1
    return f"{pitch_class}{octave}"


def _wav_metadata(wav_path: Path) -> tuple[int, float]:
    """Return (sample_rate, duration_seconds) without loading the whole file."""
    info = sf.info(str(wav_path))
    return info.samplerate, info.frames / float(info.samplerate)


def phonemes_overlapping_interval(
    phones: list[str],
    ph_starts: list[float],
    ph_ends: list[float],
    interval_start: float,
    interval_end: float,
    *,
    slack_s: float = 0.0,
) -> list[str]:
    """Return phoneme symbols whose ``[ph_start, ph_end]`` overlaps ``[interval_start, interval_end]``."""
    out: list[str] = []
    for ph, p_start, p_end in zip(phones, ph_starts, ph_ends):
        if float(p_end) <= interval_start - slack_s or float(p_start) >= interval_end + slack_s:
            continue
        out.append(ph)
    return out


def load_gtsinger_segment(json_path: Path) -> list[dict]:
    """Load a GTSinger per-segment annotation JSON file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_reference_annotation(
    sample_dir: Path,
    sample_id: Optional[str] = None,
    annotation_filename: Optional[str] = None,
    wav_filename: Optional[str] = None,
    language: str = "English",
) -> ReferenceAnnotation:
    """Build a ReferenceAnnotation from a GTSinger sample directory.

    Parameters
    ----------
    sample_dir
        Directory containing the GTSinger segment files.
    sample_id
        Logical name for this sample. Defaults to the directory's path stem
        joined with the segment number (e.g. ``EN-Alto-1__innocence__0000``).
    annotation_filename, wav_filename
        Optional explicit basenames. If omitted, we auto-detect by looking
        for a single ``*.json`` (excluding ``reference_annotation.json`` and
        ``stars_metadata.json``) and a single ``*.wav`` in ``sample_dir``.
    language
        Tag stored on the ReferenceAnnotation; STARS will infer its own.
    """
    sample_dir = Path(sample_dir)

    if wav_filename is None:
        wav_candidates = list(sample_dir.glob("*.wav"))
        if len(wav_candidates) != 1:
            raise ValueError(
                f"Expected exactly one .wav in {sample_dir}, "
                f"found {len(wav_candidates)}: {[c.name for c in wav_candidates]}"
            )
        wav_path = wav_candidates[0]
    else:
        wav_path = sample_dir / wav_filename

    if annotation_filename is None:
        # Prefer <wav_stem>.json (GTSinger layout) over globbing pipeline artifacts.
        stem_json = sample_dir / f"{wav_path.stem}.json"
        if stem_json.is_file():
            annotation_path = stem_json
        else:
            pipeline_jsons = {
                "reference_annotation.json",
                "stars_metadata.json",
                "pitch.json",
                "stars.json",
                "loudness.json",
                "timeline.json",
            }
            pipeline_jsons |= {p.name for p in sample_dir.glob("note_card_*.json")}
            if (sample_dir / "note_cards_first5.json").is_file():
                pipeline_jsons.add("note_cards_first5.json")
            candidates = [
                p for p in sample_dir.glob("*.json") if p.name not in pipeline_jsons
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one GTSinger annotation .json in {sample_dir}, "
                    f"found {len(candidates)}: {[c.name for c in candidates]}"
                )
            annotation_path = candidates[0]
    else:
        annotation_path = sample_dir / annotation_filename

    if sample_id is None:
        sample_id = f"{sample_dir.name}__{annotation_path.stem}"

    sample_rate, duration_s = _wav_metadata(wav_path)
    entries = load_gtsinger_segment(annotation_path)

    words: list[str] = []
    phones: list[str] = []
    ph2word: list[int] = []
    word_durs: list[float] = []
    ph_durs: list[float] = []
    notes: list[ReferenceNote] = []

    for word_idx, entry in enumerate(entries):
        word = entry["word"]
        words.append(word)
        word_durs.append(float(entry["end_time"]) - float(entry["start_time"]))

        # Phonemes for this word (with " " -> "<SP>" remap)
        word_phones = [
            STARS_SILENCE_PH if ph == GTSINGER_SILENCE_PH else ph
            for ph in entry["ph"]
        ]
        word_ph_durs = [
            float(end) - float(start)
            for start, end in zip(entry["ph_start"], entry["ph_end"])
        ]
        for ph, ph_dur in zip(word_phones, word_ph_durs):
            phones.append(ph)
            ph_durs.append(ph_dur)
            ph2word.append(word_idx)

        # Notes — skip "note == 0" entries (those are rests).
        for n_idx, midi in enumerate(entry["note"]):
            midi_int = int(midi)
            if midi_int == 0:
                continue
            note_start = float(entry["note_start"][n_idx])
            note_end = float(entry["note_end"][n_idx])
            note_phones = phonemes_overlapping_interval(
                word_phones,
                entry["ph_start"],
                entry["ph_end"],
                note_start,
                note_end,
            )
            notes.append(
                ReferenceNote(
                    index=len(notes),
                    start_s=note_start,
                    end_s=note_end,
                    midi_pitch=midi_int,
                    note_name=_midi_to_name(midi_int),
                    lyric_word=word,
                    word_index=word_idx,
                    phonemes=note_phones,
                )
            )

    # Sprint 1 has no real section labels; expose one "Full" section so the
    # downstream highlight engine has a non-empty list to iterate over.
    sections = [ReferenceSection(name="Full", start_s=0.0, end_s=duration_s)]

    return ReferenceAnnotation(
        sample_id=sample_id,
        audio_path=wav_path.name,
        sample_rate=sample_rate,
        duration_s=duration_s,
        language=language,
        sections=sections,
        words=words,
        phones=phones,
        ph2word=ph2word,
        word_durs_s=word_durs,
        ph_durs_s=ph_durs,
        notes=notes,
    )


def write_reference_artifacts(
    sample_dir: Path,
    annotation: ReferenceAnnotation,
    *,
    indent: int = 2,
) -> tuple[Path, Path]:
    """Write reference_annotation.json and stars_metadata.json into sample_dir."""
    sample_dir = Path(sample_dir)

    ref_path = sample_dir / "reference_annotation.json"
    ref_path.write_text(annotation.model_dump_json(indent=indent), encoding="utf-8")

    stars_entry: StarsMetadataEntry = annotation.to_stars_metadata(
        wav_fn=str((sample_dir / annotation.audio_path).resolve()).replace("\\", "/"),
    )
    stars_meta_path = sample_dir / "stars_metadata.json"
    stars_meta_path.write_text(
        json.dumps([stars_entry.model_dump(exclude_none=True)], indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )

    return ref_path, stars_meta_path


def load_reference(sample_dir: Path) -> ReferenceAnnotation:
    """Load a previously-written reference_annotation.json.

    Sprint 3: if a ``sections.yaml`` sidecar lives next to the annotation,
    its sections override whatever was baked into the JSON (so users can
    refine section labels without re-running ``import_ultrastar.py``).
    """
    sample_dir = Path(sample_dir)
    annotation = ReferenceAnnotation.model_validate_json(
        (sample_dir / "reference_annotation.json").read_text(encoding="utf-8")
    )
    # Lazy import to avoid circular dependency (song.py -> reference.py).
    try:
        from vocal_coach.song import apply_sections_sidecar  # noqa: WPS433
    except Exception:
        return annotation
    return apply_sections_sidecar(sample_dir, annotation)
