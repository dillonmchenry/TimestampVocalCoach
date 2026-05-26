"""UltraStar (.txt) chart parser.

The format we target is described at
https://github.com/UltraStar-Deluxe/format/blob/main/The%20UltraStar%20File%20Format%20(v1).md

A chart is a flat ASCII text file with header lines like::

    #TITLE:Losing My Religion
    #ARTIST:R.E.M.
    #LANGUAGE:English
    #BPM:200
    #GAP:14000
    #MP3:R.E.M. - Losing My Religion.mp3

followed by note lines and phrase boundaries::

    : 32 9 21  life,
    * 9 20 23 Oh...
    F 50 4 19 ah        ; freestyle (rare)
    R 50 4 -2 mm        ; rap (rare)
    - 69                ; phrase break
    E                   ; end of song

Each note line is::

    note-type start-beat duration-beats pitch text

Pitch is in half-steps relative to ``C4`` (UltraStar pitch ``0`` == MIDI 60).
Time is in *beats* relative to ``#GAP``: each beat is 1/4 of a quarter-note at
the song's tempo, so::

    seconds_per_beat = 60 / (BPM * 4)
    t_seconds        = gap_ms / 1000 + beat * seconds_per_beat

Note types we honor:

    ``:``  regular sung note
    ``*``  golden / emphasis note (treated like a regular note)
    ``F``  freestyle note (no pitch scoring; we still keep it for lyric timing)
    ``R``  rap note (lyrics only; pitch typically ``-2`` and unscoreable)

Phrase boundary lines (``-``) are converted into ``ReferenceSection`` entries
so the highlight engine can iterate per-phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# UltraStar pitch ``0`` == MIDI 60 (C4) per the v1 spec.
ULTRASTAR_BASE_MIDI = 60

NOTE_TYPE_REGULAR = ":"
NOTE_TYPE_GOLDEN = "*"
NOTE_TYPE_FREESTYLE = "F"
NOTE_TYPE_RAP = "R"
NOTE_TYPE_RAP_GOLDEN = "G"
PHRASE_BREAK = "-"
END_OF_SONG = "E"


SCOREABLE_NOTE_TYPES = {NOTE_TYPE_REGULAR, NOTE_TYPE_GOLDEN}


@dataclass
class UltraStarNote:
    """One note line in the chart, projected into seconds."""

    index: int
    note_type: str
    start_beat: int
    duration_beats: int
    ultrastar_pitch: int
    text: str
    start_s: float
    end_s: float
    midi_pitch: int
    is_golden: bool
    is_freestyle: bool
    is_rap: bool
    starts_new_word: bool = False
    """True if this note begins a new word (leading space in text or first
    note after a phrase break)."""


@dataclass
class UltraStarPhraseBreak:
    """One phrase boundary marker (``- 69``)."""

    index: int
    end_beat: int
    end_s: float


@dataclass
class UltraStarChart:
    """Parsed UltraStar chart."""

    title: str
    artist: str
    language: str
    bpm: float
    gap_ms: float
    audio_ref: Optional[str]
    cover: Optional[str]
    background: Optional[str]
    edition: Optional[str]
    notes: list[UltraStarNote]
    phrase_breaks: list[UltraStarPhraseBreak]

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / (self.bpm * 4.0)

    @property
    def end_beat(self) -> int:
        if not self.notes:
            return 0
        last = self.notes[-1]
        return last.start_beat + last.duration_beats


def _parse_header_value(value: str) -> str:
    return value.strip()


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _maybe_strip_bom(line: str) -> str:
    if line.startswith("\ufeff"):
        return line[1:]
    return line


def parse_ultrastar(
    chart_path: Path,
    *,
    encoding: str = "utf-8",
    midi_offset: int = 0,
) -> UltraStarChart:
    """Parse an UltraStar ``.txt`` chart from ``chart_path``.

    ``midi_offset`` is added to every ``midi_pitch`` so per-song corrections
    (e.g. octave shifts vs the actual reference recording) can be applied
    without editing the chart itself.
    """
    chart_path = Path(chart_path)
    raw = chart_path.read_text(encoding=encoding, errors="replace")
    lines = raw.splitlines()

    title = ""
    artist = ""
    language = "English"
    bpm: Optional[float] = None
    gap_ms: float = 0.0
    audio_ref: Optional[str] = None
    cover: Optional[str] = None
    background: Optional[str] = None
    edition: Optional[str] = None

    notes: list[UltraStarNote] = []
    phrase_breaks: list[UltraStarPhraseBreak] = []

    body_started = False
    seconds_per_beat: Optional[float] = None
    pending_phrase_break = False

    for raw_line in lines:
        line = _maybe_strip_bom(raw_line).rstrip("\r\n")
        if not line:
            continue

        # --- header tags --------------------------------------------------
        if line.startswith("#"):
            if ":" not in line:
                continue
            tag, _, value = line.partition(":")
            tag = tag[1:].strip().upper()
            value = _parse_header_value(value)
            if tag == "TITLE":
                title = value
            elif tag == "ARTIST":
                artist = value
            elif tag == "LANGUAGE":
                language = value
            elif tag == "BPM":
                bpm = _to_float(value)
            elif tag == "GAP":
                gap_ms = _to_float(value)
            elif tag in ("MP3", "AUDIO"):
                audio_ref = value
            elif tag == "COVER":
                cover = value
            elif tag == "BACKGROUND":
                background = value
            elif tag == "EDITION":
                edition = value
            continue

        if bpm is None:
            raise ValueError(
                f"UltraStar chart {chart_path} has note lines before #BPM was set"
            )
        if seconds_per_beat is None:
            seconds_per_beat = 60.0 / (bpm * 4.0)

        # --- end-of-song marker -------------------------------------------
        first = line[0]
        if first == END_OF_SONG and (len(line) == 1 or line[1].isspace()):
            break

        # --- phrase break (``- 69`` or ``- 69 71`` for line offsets) ------
        if first == PHRASE_BREAK:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                end_beat = int(tokens[1])
            except ValueError:
                continue
            phrase_breaks.append(
                UltraStarPhraseBreak(
                    index=len(phrase_breaks),
                    end_beat=end_beat,
                    end_s=gap_ms / 1000.0 + end_beat * seconds_per_beat,
                )
            )
            pending_phrase_break = True
            continue

        # --- note line ----------------------------------------------------
        if first not in {
            NOTE_TYPE_REGULAR,
            NOTE_TYPE_GOLDEN,
            NOTE_TYPE_FREESTYLE,
            NOTE_TYPE_RAP,
            NOTE_TYPE_RAP_GOLDEN,
        }:
            continue

        # ``: 32 9 21  life,`` → 4 numeric tokens + free-text lyric.
        # We must preserve the leading whitespace of the lyric: in UltraStar,
        # a leading space marks a NEW word, no leading space CONTINUES the
        # previous word (so ``: 58 2 23 big`` + ``: 61 6 21 ger`` -> ``bigger``).
        # ``str.split`` collapses runs of whitespace, so we tokenize manually.
        head, _, tail = line.partition(" ")  # strip the note-type
        if not tail:
            continue
        # Tail is the rest of the line. Skip ONE separating space, then the
        # numeric tokens are space-delimited, then the lyric is everything
        # after the next whitespace boundary -- preserving leading spaces.
        idx = 0
        while idx < len(tail) and tail[idx] == " ":
            idx += 1
        numeric_parts: list[str] = []
        cursor = idx
        for _ in range(3):
            j = cursor
            while j < len(tail) and tail[j] != " ":
                j += 1
            if j == cursor:
                numeric_parts = []
                break
            numeric_parts.append(tail[cursor:j])
            # Advance past exactly one space; second+ spaces are kept on the
            # next token (and end up as the lyric's leading whitespace).
            cursor = j + 1 if j < len(tail) else j
        if len(numeric_parts) != 3:
            continue
        try:
            start_beat = int(numeric_parts[0])
            duration_beats = int(numeric_parts[1])
            pitch_raw = int(numeric_parts[2])
        except ValueError:
            continue
        text = tail[cursor:] if cursor < len(tail) else ""

        body_started = True
        start_s = gap_ms / 1000.0 + start_beat * seconds_per_beat
        end_s = start_s + duration_beats * seconds_per_beat
        midi = ULTRASTAR_BASE_MIDI + pitch_raw + midi_offset

        starts_new_word = (
            len(notes) == 0
            or pending_phrase_break
            or text.startswith(" ")
        )
        pending_phrase_break = False

        notes.append(
            UltraStarNote(
                index=len(notes),
                note_type=first,
                start_beat=start_beat,
                duration_beats=duration_beats,
                ultrastar_pitch=pitch_raw,
                text=text,
                start_s=start_s,
                end_s=end_s,
                midi_pitch=midi,
                is_golden=(first in {NOTE_TYPE_GOLDEN, NOTE_TYPE_RAP_GOLDEN}),
                is_freestyle=(first == NOTE_TYPE_FREESTYLE),
                is_rap=(first in {NOTE_TYPE_RAP, NOTE_TYPE_RAP_GOLDEN}),
                starts_new_word=starts_new_word,
            )
        )

    if bpm is None:
        raise ValueError(f"UltraStar chart {chart_path} is missing #BPM")
    if not body_started:
        raise ValueError(f"UltraStar chart {chart_path} has no note lines")

    return UltraStarChart(
        title=title,
        artist=artist,
        language=language,
        bpm=bpm,
        gap_ms=gap_ms,
        audio_ref=audio_ref,
        cover=cover,
        background=background,
        edition=edition,
        notes=notes,
        phrase_breaks=phrase_breaks,
    )


# ---------------------------------------------------------------------------
# Lyric reconstruction
# ---------------------------------------------------------------------------


def words_from_notes(notes: list[UltraStarNote]) -> list[tuple[str, list[int]]]:
    """Group consecutive note-syllables into words.

    UltraStar splits lyrics into syllables. We rely on
    ``UltraStarNote.starts_new_word`` (set during parsing) which is True when:

    * the syllable text begins with whitespace (``: 32 9 21 _life,``),
    * the syllable is the first note of the song,
    * the syllable is the first note after a phrase break (``- 69``).

    Trailing punctuation and tildes (``~``) used for held syllables are
    stripped for the visible word form.

    Returns
    -------
    list of (word_text, [note_indices])
        ``word_text`` is the cleaned word; ``note_indices`` are the indices
        into ``notes`` belonging to that word in chart order.
    """
    words: list[tuple[str, list[int]]] = []
    current_text = ""
    current_indices: list[int] = []
    for note in notes:
        cleaned = note.text.lstrip()
        if note.starts_new_word and current_indices:
            words.append((current_text, current_indices))
            current_text = ""
            current_indices = []
        current_text += cleaned
        current_indices.append(note.index)
    if current_indices:
        words.append((current_text, current_indices))
    cleaned_words: list[tuple[str, list[int]]] = []
    for text, idxs in words:
        bare = text.replace("~", "").strip()
        cleaned_words.append((bare or text, idxs))
    return cleaned_words
