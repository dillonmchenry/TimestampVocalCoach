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

    late_ms: float = 80.0
    early_ms: float = -80.0
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

    cap: int = 5
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

    best_phrase_min_pct_in_tune: float = 0.55
    pitch_struggle_max_pct_in_tune: float = 0.40
    expressive_techniques: tuple[str, ...] = (
        "vibrato",
        "glissando",
        "falsetto",
        "breathe",
        "pharyngeal",
        "mixed",
    )
    """STARS techniques surfaced as 'expression'."""


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
