"""Pydantic schemas for every JSON artifact in the pipeline.

Sprint 1 produces five JSON files per sample, plus one envelope:

    reference_annotation.json   -> ReferenceAnnotation
    stars_metadata.json         -> StarsMetadataEntry (list of)
    pitch.json                  -> PitchTrack
    stars.json                  -> StarsTrack
    loudness.json               -> LoudnessTrack
    timeline.json               -> Timeline (joins all of the above)

Sprint 2 adds a song-centric layout under ``data/songs/<song_id>/`` with:

    manifest.json               -> SongManifest (UltraStar bundle metadata)
    reference_annotation.json   -> ReferenceAnnotation (built from UltraStar chart)
    reference/pitch.json        -> PitchTrack (NanoPitch on reference vocal)
    reference/stars.json        -> StarsTrack (STARS on reference vocal)
    reference/loudness.json     -> LoudnessTrack
    performances/<perf_id>/
        pitch.json              -> user PitchTrack
        stars.json              -> user StarsTrack
        analysis.json           -> PerformanceAnalysis (note-level + highlights)

The Sprint 1 ``NoteCard`` and the new Sprint 2 ``PerformanceAnalysis`` share
the same ``ReferenceAnnotation`` substrate and live alongside it in this file.
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


# ---------------------------------------------------------------------------
# Sprint 2 — UltraStar song bundle
# ---------------------------------------------------------------------------


class UltraStarMetadata(BaseModel):
    """Header metadata parsed from an UltraStar ``.txt`` chart.

    UltraStar timing is encoded as ``beat`` units relative to ``#GAP``:

        seconds_per_beat = 60 / (BPM * 4)   # UltraStar BPM is 1/4-note divisions
        t_seconds        = gap_ms / 1000 + beat * seconds_per_beat

    Pitch is encoded as half-steps relative to C4 (UltraStar pitch ``0`` ==
    MIDI 60). We expose ``midi_offset`` here so a per-song fix-up (e.g. when
    the chart was authored an octave off the actual recording) can be applied
    without reparsing.
    """

    bpm: float = Field(..., description='UltraStar #BPM (1/4-note divisions per minute)')
    gap_ms: float = Field(..., description='UltraStar #GAP in milliseconds')
    audio_ref: Optional[str] = Field(
        None, description='Original #MP3 / #AUDIO field from the chart, if present'
    )
    cover: Optional[str] = None
    background: Optional[str] = None
    edition: Optional[str] = None
    midi_offset: int = Field(
        0,
        description=(
            "Constant added to UltraStar pitch when converting to MIDI; default "
            "mapping is midi = 60 + ultrastar_pitch + midi_offset."
        ),
    )

    @property
    def seconds_per_beat(self) -> float:
        # UltraStar treats "BPM" as quarter-beats per minute, so each chart
        # beat is 1/4 of a quarter-note at the song's tempo.
        return 60.0 / (self.bpm * 4.0)


class SongManifest(BaseModel):
    """Top-level manifest for one playable song under ``data/songs/<song_id>/``.

    Paths are stored relative to the song directory so the bundle is portable.
    """

    song_id: str
    title: str
    artist: str
    language: str = "English"
    source_format: str = Field("ultrastar", description='Currently only "ultrastar"')
    chart_path: str = Field(..., description='Relative path to the UltraStar .txt chart')
    reference_vocal_path: str = Field(
        ..., description='Relative path to the isolated reference vocal'
    )
    instrumental_path: Optional[str] = Field(
        None, description='Relative path to the instrumental backing track'
    )
    duration_s: float
    ultrastar: UltraStarMetadata
    reference_pitch_path: Optional[str] = Field(
        None, description='Relative path to precomputed reference/pitch.json'
    )
    reference_stars_path: Optional[str] = Field(
        None, description='Relative path to precomputed reference/stars.json'
    )
    reference_loudness_path: Optional[str] = Field(
        None, description='Relative path to precomputed reference/loudness.json'
    )


# ---------------------------------------------------------------------------
# Sprint 2 — performance analysis (dual-track + highlights)
# ---------------------------------------------------------------------------


class NoteMeasurementV2(BaseModel):
    """Numeric per-note measurements computed by ``align_v2``.

    All quantities live in *song time* (the user vocal has already been shifted
    by the estimated global offset before measurement).
    """

    note_index: int
    start_s: float = Field(..., description="UltraStar note start in song time")
    end_s: float = Field(..., description="UltraStar note end in song time")
    midi_pitch: int
    note_name: str
    lyric_word: str

    voiced_coverage: float = Field(
        0.0,
        description='Fraction of frames inside the note window with voicing >= threshold',
    )
    median_cents: Optional[float] = Field(
        None, description='Median cents from target MIDI across voiced frames'
    )
    pct_in_tune: Optional[float] = Field(
        None,
        description='Fraction of voiced frames inside the core window within the in-tune cents window',
    )
    drift_cents_per_s: Optional[float] = Field(
        None, description='Linear cents/s slope across the note window (drift)'
    )
    arrival_offset_ms: Optional[float] = Field(
        None,
        description='User arrival vs expected onset; positive = late, negative = early',
    )
    core_start_s: Optional[float] = None
    core_end_s: Optional[float] = None

    note_octave_offset: int = Field(
        0,
        description=(
            'Per-note residual octave error after the global octave shift has '
            'been applied: 0 = same octave as the (shifted) target, -1 = the '
            'user sang this note one octave lower than the rest of their take, '
            '+1 = one octave higher.'
        ),
    )

    pitch_tags: list[str] = Field(default_factory=list)
    arrival_tags: list[str] = Field(default_factory=list)


class NoteTechniqueComparison(BaseModel):
    """Reference vs. user STARS technique sets for one note window."""

    note_index: int
    reference_techniques: list[str] = Field(default_factory=list)
    user_techniques: list[str] = Field(default_factory=list)
    matched: list[str] = Field(
        default_factory=list,
        description='Techniques present in both reference and user',
    )
    missed: list[str] = Field(
        default_factory=list,
        description='Reference techniques the user did not produce',
    )
    user_added: list[str] = Field(
        default_factory=list,
        description='Techniques the user added that the reference does not have',
    )


class CoachingMoment(BaseModel):
    """One ranked highlight surfaced to the user."""

    id: str
    type: str = Field(
        ...,
        description=(
            'Category: "best_pitch_phrase", "pitch_struggle", "expressive_match", '
            '"expressive_moment", "missed_expression", "late_entrance", ...'
        ),
    )
    title: str
    summary: str
    start_s: float
    end_s: float
    score: float = Field(..., description='Higher = more salient (relative within type)')
    note_indices: list[int] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    detail: dict = Field(
        default_factory=dict,
        description='Free-form structured data for UI tooltips (cents, pct_in_tune, etc.)',
    )


class HighlightsReport(BaseModel):
    """Bundle of coaching moments emitted for one performance analysis."""

    moments: list[CoachingMoment] = Field(default_factory=list)
    cap: int = Field(5, description='Maximum number of moments selected for display')


class PerformanceAnalysis(BaseModel):
    """Full analysis output for one user performance against a song.

    Sources are referenced by relative path so the JSON stays small; the
    measurement summary lives in ``notes`` and the user-facing output lives
    in ``highlights``.
    """

    song_id: str
    perf_id: str
    reference_sample_id: str
    duration_s: float
    global_offset_s: float = Field(
        0.0,
        description='song_time = user_time - global_offset_s',
    )
    octave_shift_semitones: int = Field(
        0,
        description=(
            'Integer-semitone (multiple of 12) offset added to every chart '
            'MIDI before scoring this user. Negative = user is singing in a '
            'lower register than the chart; positive = higher. Auto-detected '
            'by align_v2.estimate_octave_shift_semitones unless overridden.'
        ),
    )
    pitch_user_path: str
    pitch_ref_path: Optional[str] = None
    stars_user_path: Optional[str] = None
    stars_ref_path: Optional[str] = None
    notes: list[NoteMeasurementV2] = Field(default_factory=list)
    techniques: list[NoteTechniqueComparison] = Field(default_factory=list)
    highlights: HighlightsReport = Field(default_factory=HighlightsReport)
    analysis_version: str = Field("v2", description='Schema version tag for migrations')
