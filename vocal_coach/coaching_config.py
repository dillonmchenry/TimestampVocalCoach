"""Tunable thresholds for the Sprint-2 alignment + highlight engine.

Sprint 1's ``align.py`` had numbers hard-coded at module top. Sprint 2
moves those into a YAML file (``config/coaching.yaml``) so a future tuning
spike can iterate on a real recording without touching code. We still
ship a frozen Python ``CoachingConfig`` default so the system works
out-of-the-box if no YAML file is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PitchConfig:
    """Pitch-window measurements applied per ``ReferenceNote``."""

    in_tune_cents: float = 50.0
    """Voiced frames within +/- this many cents count as 'in tune'."""

    flat_cents: float = -20.0
    sharp_cents: float = 20.0
    drift_cents_per_s: float = 40.0
    voicing_threshold: float = 0.5

    min_voiced_coverage: float = 0.20
    """Below this voiced coverage we don't claim a real attempt at the note."""


@dataclass
class ArrivalConfig:
    """Arrival/onset detection thresholds."""

    late_ms: float = 50.0
    early_ms: float = -50.0
    search_back_s: float = 0.20
    search_forward_s: float = 0.50
    gap_tolerance_s: float = 0.030
    unvoiced_lead_s: float = 0.05
    pitch_lock_cents: float = 50.0
    pitch_lock_hold_s: float = 0.05


@dataclass
class CoreWindowConfig:
    """Core-window trimming applied before scoring pitch."""

    attack_trim_s: float = 0.05
    """Seconds shaved off the head of each note window before scoring."""

    release_trim_s: float = 0.05
    """Seconds shaved off the tail."""

    min_core_s: float = 0.05
    """Floor: never trim a window below this duration."""


@dataclass
class GlobalOffsetConfig:
    """Global-offset estimation between user vocal and song timeline."""

    search_range_s: float = 1.5
    """Search +/- this many seconds for the best offset."""

    step_s: float = 0.020
    """Resolution of the offset grid search."""

    voicing_threshold: float = 0.5
    """User pitch frame is voiced if voicing_confidence >= this."""

    min_voiced_overlap_s: float = 1.0
    """Reject offsets that yield less than this much voiced overlap."""


@dataclass
class HighlightsConfig:
    """Highlight-engine thresholds."""

    cap: int = 15
    """Maximum highlights returned per performance."""

    window_min: int = 4
    window_max: int = 8
    """Rolling-phrase window size (in notes) for STARS / expression highlights."""

    pitch_window_min: int = 8
    pitch_window_max: int = 16
    """Rolling-phrase window size (in notes) for pitch best/struggle highlights."""

    pitch_phrases_per_type: int = 2
    """How many non-overlapping pitch phrase highlights to surface per type
    (``best_pitch_phrase`` and ``pitch_struggle``)."""

    max_per_type: int = 2
    """Maximum highlights of any single type in the final capped list."""

    max_per_category: int = 5
    """Maximum highlights from any single feedback category (pitch, technique,
    alignment, dynamics) in the final list.  Enforced via round-robin so every
    category with candidates gets representation."""

    best_phrase_min_pct_in_tune: float = 0.40
    pitch_struggle_max_pct_in_tune: float = 0.55
    expressive_techniques: tuple[str, ...] = (
        "vibrato",
        "glissando",
        "falsetto",
        "breathe",
        "pharyngeal",
        "mixed",
    )
    """STARS techniques surfaced as 'expression'."""

    # ------------------------------------------------------------------
    # Sprint 3: section-level detectors
    # ------------------------------------------------------------------

    section_max_moments: int = 5
    """Maximum section-scope moments admitted in the final cap."""

    section_min_notes: int = 3
    """Section must contain at least this many scoreable notes to qualify."""

    section_strength_min_pct_in_tune: float = 0.45
    """Section pct_in_tune at or above this is surfaced as a strength."""

    section_weakness_max_pct_in_tune: float = 0.55
    """Section pct_in_tune at or below this is surfaced as a weakness."""

    section_delta_min_pct_in_tune: float = 0.15
    """Minimum pct_in_tune gap between two section kinds to surface a delta."""

    section_delta_min_cents: float = 25.0
    """Minimum |median_cents| gap between two section kinds to surface a delta."""

    section_kind_pairs: tuple[tuple[str, str], ...] = (
        ("verse", "chorus"),
        ("verse", "refrain"),
        ("pre_chorus", "chorus"),
    )
    """Cross-kind pairs to compare. Each is (kind_a, kind_b); ``a`` is the
    contrast group (often the weaker side) and ``b`` is the reference group."""

    section_technique_min_density_gap: float = 0.25
    """Min user-density gap between two section kinds for a technique-drop highlight."""

    # ------------------------------------------------------------------
    # Loudness / dynamics detectors
    # ------------------------------------------------------------------

    loudness_fade_threshold_db_per_s: float = -5.0
    """rms_fade_db_per_s at or below this flags a note as fading out.
    -5 dB/s means the note loses ~5 dB of level per second — clearly audible."""

    loudness_fade_min_voiced_coverage: float = 0.30
    """Only flag fade-out on notes where the user actually sang (voiced_coverage
    above this threshold), so silent notes don't generate spurious warnings."""

    loudness_fade_min_window_notes: int = 2
    """A fade-out phrase highlight requires at least this many fading notes
    in a row before it's surfaced."""

    loudness_dynamic_delta_db: float = 2.0
    """Minimum normalised rms_delta_db (absolute value) for a dynamic_drop or
    dynamic_surge phrase highlight.  2 dB is clearly perceptible."""

    loudness_dynamic_window_min: int = 4
    loudness_dynamic_window_max: int = 10
    """Phrase-window sizes (in notes) for the dynamic drop/surge detectors."""

    section_dynamic_contrast_min_db: float = 1.5
    """Minimum dB gap in mean_rms_db between chorus and verse sections for a
    section_dynamic_contrast highlight.  Values below this are considered
    "flat" dynamics — not worth surfacing."""

    # ------------------------------------------------------------------
    # Single-note pitch callouts
    # ------------------------------------------------------------------

    sharp_flat_note_min_cents: float = 25.0
    """Individual notes at or beyond this |median_cents| are surfaced as
    sharp/flat callouts."""

    sharp_flat_note_max: int = 3
    """Maximum sharp/flat single-note callouts emitted."""

    # ------------------------------------------------------------------
    # Multiple entrance timing callouts
    # ------------------------------------------------------------------

    entrance_timing_max: int = 4
    """How many individual late/early entrance notes to surface."""

    # ------------------------------------------------------------------
    # Timing-consistency window detector
    # ------------------------------------------------------------------

    timing_consistency_window_min: int = 4
    timing_consistency_window_max: int = 8
    timing_consistency_mean_ms: float = 40.0
    """Mean |arrival_offset_ms| above this in a window triggers a
    timing_consistency moment."""

    # ------------------------------------------------------------------
    # Per-technique vocal texture highlights
    # ------------------------------------------------------------------

    vocal_texture_techniques: tuple[str, ...] = (
        "breathe",
        "vibrato",
        "falsetto",
        "strong",
    )
    """Techniques eligible for individual vocal-texture callouts.
    These produce short, per-technique local moments like 'Breathy colour
    on these notes'."""

    vocal_texture_min_notes: int = 1
    """Minimum consecutive notes with a technique to surface a texture highlight."""

    # ------------------------------------------------------------------
    # Best / weakest overall section (blended)
    # ------------------------------------------------------------------

    section_best_overall_min_notes: int = 4
    """Section needs at least this many notes for a blended best/weakest pick."""


@dataclass
class CoachingConfig:
    pitch: PitchConfig = field(default_factory=PitchConfig)
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    core_window: CoreWindowConfig = field(default_factory=CoreWindowConfig)
    global_offset: GlobalOffsetConfig = field(default_factory=GlobalOffsetConfig)
    highlights: HighlightsConfig = field(default_factory=HighlightsConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "CoachingConfig":
        """Load thresholds from YAML; fall back to defaults if file is missing."""
        if config_path is None:
            return cls()
        path = Path(config_path)
        if not path.is_file():
            return cls()
        try:
            import yaml  # type: ignore
        except Exception:
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls()
        for section_name, section_value in data.items():
            section = getattr(cfg, section_name, None)
            if section is None or not isinstance(section_value, dict):
                continue
            for k, v in section_value.items():
                if hasattr(section, k):
                    setattr(section, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG_RELPATH = "config/coaching.yaml"
