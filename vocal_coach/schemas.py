"""Pydantic schemas for every JSON artifact in the sprint-1 pipeline.

The pipeline produces five JSON files per sample, plus one envelope:

    reference_annotation.json   -> ReferenceAnnotation
    stars_metadata.json         -> StarsMetadataEntry (list of)
    pitch.json                  -> PitchTrack
    stars.json                  -> StarsTrack
    loudness.json               -> LoudnessTrack
    timeline.json               -> Timeline (joins all of the above)

Sprint 2 adds NoteCard via vocal_coach/align.py; the schema for it lives at the
bottom of this file so sprint 1 already has a contract to aim at when the
stretch goal kicks in.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reference annotation (output of scripts/build_reference.py)
# ---------------------------------------------------------------------------


class ReferenceNote(BaseModel):
    """One note in the reference song (the *expected* performance)."""

    index: int = Field(..., description="Zero-based index into ReferenceAnnotation.notes")
    start_s: float = Field(..., description="Note onset in seconds")
    end_s: float = Field(..., description="Note offset in seconds")
    midi_pitch: int = Field(..., description="MIDI pitch number (60 = C4)")
    note_name: str = Field(..., description='Pitch-class name with octave, e.g. "A4"')
    lyric_word: str = Field(..., description="The word this note is sung on")
    word_index: int = Field(..., description="Index into ReferenceAnnotation.words")
    phonemes: list[str] = Field(
        default_factory=list,
        description="Phonemes that fall inside this note (slur-aware)",
    )

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class ReferenceSection(BaseModel):
    """A high-level song section, e.g. 'Verse 1' or 'Final Chorus'."""
    name: str
    start_s: float
    end_s: float


class ReferenceAnnotation(BaseModel):
    """Everything the system needs to know about the *reference* song.

    This file is the scaffold against which the user performance is measured.
    It also doubles as the source-of-truth from which `stars_metadata.json` is
    derived (see `to_stars_metadata`).
    """

    sample_id: str
    audio_path: str = Field(..., description="Path to the reference wav (relative to the sample dir)")
    sample_rate: int = Field(..., description="Sample rate of audio_path")
    duration_s: float
    language: str = Field("English", description='Vocabulary language tag, e.g. "English"')
    sections: list[ReferenceSection] = Field(default_factory=list)
    words: list[str] = Field(default_factory=list, description="Sung words, in order")
    phones: list[str] = Field(default_factory=list, description="Phonemes, in order")
    ph2word: list[int] = Field(
        default_factory=list,
        description="For each phoneme i, the index of the word it belongs to",
    )
    word_durs_s: Optional[list[float]] = Field(
        None, description="Optional pre-known word durations (seconds)"
    )
    ph_durs_s: Optional[list[float]] = Field(
        None, description="Optional pre-known phoneme durations (seconds)"
    )
    notes: list[ReferenceNote] = Field(default_factory=list)

    def to_stars_metadata(self, wav_fn: str) -> "StarsMetadataEntry":
        """Project a ReferenceAnnotation into the metadata shape STARS expects.

        STARS's inference reads a JSON list of dicts with the keys below.
        """
        return StarsMetadataEntry(
            item_name=self.sample_id,
            wav_fn=wav_fn,
            word=list(self.words),
            ph=list(self.phones),
            ph2words=list(self.ph2word),
            ph_durs=self.ph_durs_s,
            word_durs=self.word_durs_s,
        )


class StarsMetadataEntry(BaseModel):
    """One entry of `metadata.json` consumed by `inference/stars.py`.

    Note STARS uses the field name `ph2words` (with an 's') and `word`
    (singular) for the word list — we mirror that exactly here.
    """

    item_name: str
    wav_fn: str
    word: list[str]
    ph: list[str]
    ph2words: list[int]
    # Optional: if present, STARS skips its own duration prediction.
    ph_durs: Optional[list[float]] = None
    word_durs: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# NanoPitch output  (vocal_coach.pitch.extract_f0)
# ---------------------------------------------------------------------------


class PitchFrame(BaseModel):
    time: float = Field(..., description="Frame center time in seconds")
    f0_hz: float = Field(..., description="Decoded F0 in Hz; 0.0 means unvoiced")
    voicing_confidence: float = Field(
        ..., description="VAD head probability that the frame is voiced"
    )


class PitchTrack(BaseModel):
    sample_id: str
    sample_rate: int = Field(16000, description="Sample rate the model was fed")
    hop_seconds: float = Field(0.01, description="Time between frames")
    decoder: str = Field("realtime", description='"realtime" (greedy) or "offline" (Viterbi)')
    checkpoint: str = Field(..., description="Path to the NanoPitch checkpoint used")
    frames: list[PitchFrame]


# ---------------------------------------------------------------------------
# STARS output  (vocal_coach.stars_runner.run_stars)
# ---------------------------------------------------------------------------

# Names from third_party/stars/inference/stars.py — kept in this exact order.
STARS_TECH_NAMES: list[str] = [
    "bubble",
    "breathe",
    "pharyngeal",
    "vibrato",
    "glissando",
    "mixed",
    "falsetto",
    "weak",
    "strong",
]


class StarsPhoneme(BaseModel):
    """One predicted phoneme span with its 0/1 technique flags."""

    index: int = Field(..., description="Zero-based index into StarsTrack.phonemes")
    phoneme: str
    word: str = Field(..., description="The word this phoneme belongs to ('<SP>' for silence)")
    word_index: int = Field(..., description="-1 for silence/<SP>")
    start_s: float
    end_s: float
    techniques: dict[str, int] = Field(
        default_factory=dict,
        description="Map of technique name -> 0/1 (see STARS_TECH_NAMES for keys)",
    )


class StarsNote(BaseModel):
    """One predicted note span (STARS's own MIDI-style transcription)."""

    index: int
    start_s: float
    end_s: float
    midi_pitch: int


class StarsStyle(BaseModel):
    """Global per-sample style classification from STARS."""

    language: str
    gender: str
    emotion: str
    method: str
    pace: str
    range: str
    technique_group: str


class StarsTrack(BaseModel):
    sample_id: str
    sample_rate: int = Field(24000, description="Sample rate STARS was run at")
    hop_seconds: float = Field(
        128.0 / 24000.0, description="STARS's mel hop, used for span-> time conversion"
    )
    style: StarsStyle
    phonemes: list[StarsPhoneme]
    notes: list[StarsNote]


# ---------------------------------------------------------------------------
# Loudness output  (vocal_coach.loudness.compute_loudness)
# ---------------------------------------------------------------------------


class LoudnessFrame(BaseModel):
    time: float
    rms_db: float = Field(..., description="20*log10(RMS); -inf clamped to -120")


class LoudnessTrack(BaseModel):
    sample_id: str
    sample_rate: int
    hop_seconds: float = 0.01
    frames: list[LoudnessFrame]


# ---------------------------------------------------------------------------
# Timeline envelope (one file per sample, joins all four tracks)
# ---------------------------------------------------------------------------


class Timeline(BaseModel):
    sample_id: str
    sample_dir: str
    duration_s: float
    reference: ReferenceAnnotation
    pitch: PitchTrack
    stars: Optional[StarsTrack] = Field(
        None, description="None if STARS inference was skipped or failed"
    )
    loudness: LoudnessTrack


# ---------------------------------------------------------------------------
# Sprint 2 / stretch — note-level coaching card
# ---------------------------------------------------------------------------


class NoteMeasurements(BaseModel):
    """Free-text human-readable measurements; the structured numbers live next door."""

    pitch: str = Field(..., description='e.g. "-27 cents, drifting down"')
    arrival: str = Field(..., description='e.g. "+130ms late"')
    volume: str = Field(..., description='e.g. "fades near end"')


class NotePhonemeAnnotation(BaseModel):
    phoneme: str
    tags: list[str] = Field(
        default_factory=list,
        description='STARS-derived tags, e.g. ["vibrato", "breathy ending"]',
    )


class NoteExpectedPitch(BaseModel):
    """Target pitch the singer is expected to hit on this note."""

    midi: int = Field(..., description="MIDI pitch number (60 = C4)")
    name: str = Field(..., description='Pitch-class name with octave, e.g. "A4"')


class NoteCard(BaseModel):
    """Note-level coaching card for one reference note."""

    expected_pitch: NoteExpectedPitch
    lyric_word: str = Field(..., description="Lyric word sung on this note")
    section: Optional[str] = None
    time: str = Field(..., description='Formatted range, e.g. "101.24s\u2013102.08s"')
    measurements: NoteMeasurements
    phonemes: list[NotePhonemeAnnotation]
    tags: list[str] = Field(default_factory=list)
