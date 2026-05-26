"""UltraStar parser unit tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

from vocal_coach.ultrastar import (
    parse_ultrastar,
    words_from_notes,
)


def _write_chart(tmp_path: Path, body: str) -> Path:
    chart = tmp_path / "song.txt"
    chart.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return chart


def test_parse_simple_chart(tmp_path: Path) -> None:
    chart_path = _write_chart(
        tmp_path,
        """
        #TITLE:Test
        #ARTIST:Tester
        #BPM:60
        #GAP:0
        : 0 4 0 Hel
        : 4 4 2 lo
        - 8
        : 8 4 4  world
        E
        """,
    )
    chart = parse_ultrastar(chart_path)
    assert chart.title == "Test"
    assert chart.bpm == 60
    assert chart.gap_ms == 0
    # BPM=60 means seconds_per_beat = 60/(60*4) = 0.25s
    assert chart.seconds_per_beat == 0.25
    assert len(chart.notes) == 3
    assert chart.notes[0].start_s == 0.0
    assert chart.notes[0].end_s == 1.0   # 4 beats * 0.25s
    assert chart.notes[1].start_s == 1.0
    assert chart.notes[2].start_s == 2.0
    # MIDI mapping: pitch 0 -> 60, pitch 4 -> 64
    assert chart.notes[0].midi_pitch == 60
    assert chart.notes[2].midi_pitch == 64
    assert len(chart.phrase_breaks) == 1


def test_words_from_notes_groups_by_leading_space(tmp_path: Path) -> None:
    chart_path = _write_chart(
        tmp_path,
        """
        #BPM:60
        #GAP:0
        : 0 2 0 big
        : 2 2 0 ger
        : 4 2 0  than
        E
        """,
    )
    chart = parse_ultrastar(chart_path)
    words = words_from_notes(chart.notes)
    assert [w[0] for w in words] == ["bigger", "than"]
    # "bigger" spans the first two notes (indices 0, 1)
    assert words[0][1] == [0, 1]
    assert words[1][1] == [2]


def test_phrase_break_starts_new_word(tmp_path: Path) -> None:
    chart_path = _write_chart(
        tmp_path,
        """
        #BPM:60
        #GAP:0
        : 0 2 0 first
        - 2
        : 4 2 0 second
        E
        """,
    )
    chart = parse_ultrastar(chart_path)
    words = words_from_notes(chart.notes)
    assert [w[0] for w in words] == ["first", "second"]


def test_midi_offset(tmp_path: Path) -> None:
    chart_path = _write_chart(
        tmp_path,
        """
        #BPM:60
        #GAP:0
        : 0 2 24 hi
        E
        """,
    )
    chart = parse_ultrastar(chart_path, midi_offset=-12)
    # 60 + 24 - 12 = 72
    assert chart.notes[0].midi_pitch == 72


def test_golden_and_freestyle(tmp_path: Path) -> None:
    chart_path = _write_chart(
        tmp_path,
        """
        #BPM:60
        #GAP:0
        * 0 2 0 Gold
        F 4 2 0 free
        : 8 2 0 normal
        E
        """,
    )
    chart = parse_ultrastar(chart_path)
    assert chart.notes[0].is_golden
    assert chart.notes[1].is_freestyle
    assert not chart.notes[2].is_golden
