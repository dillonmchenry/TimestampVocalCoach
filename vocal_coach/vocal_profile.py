"""Sprint-3 vocal profile generation.

A ``VocalProfile`` captures the artistic and stylistic context of one song.
It is generated once during ``scripts/build_song.py`` by calling GPT-4o-mini
with the song's title, artist, and language, plus the full list of available
highlight detector types.

The profile drives two downstream uses:

1. **Highlight emphasis weighting** in ``select_highlights()`` — detectors in
   ``emphasize_highlights`` receive a score boost; those in ``deemphasize_highlights``
   receive a penalty, so genre-appropriate highlights surface more often.

2. **Phase B prompt injection** — ``coaching_context`` is inserted into the card-
   rewriting and performance-summary prompts to give the LLM stylistic grounding.

Usage::

    from vocal_coach.vocal_profile import generate_vocal_profile
    from vocal_coach.llm import LLMClient
    from vocal_coach.coaching_config import CoachingConfig

    cfg = CoachingConfig.load(...)
    client = LLMClient.from_config(cfg.llm)
    profile = generate_vocal_profile(manifest, client)
    if profile is None:
        # LLM unavailable — skip; highlight selection will use default weights
        ...
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The full list of detector types the LLM can reference by name.
# Keep in sync with MOMENT_CATEGORY in highlights.py.
# ---------------------------------------------------------------------------

_DETECTOR_DESCRIPTIONS: dict[str, str] = {
    # Pitch
    "best_pitch_phrase": "A phrase window where the singer was most in-tune (affirming)",
    "pitch_struggle": "A phrase window where the singer was most out-of-tune",
    "sharp_flat_note": "An individual note that was notably sharp or flat",
    "scoop_habit": "A pattern of scooping into notes from below (portamento from below target pitch)",
    "pitch_overshoot": "A pattern of approaching notes from above before settling",
    "clean_attack": "A run of notes attacked directly on pitch without sliding (affirming)",
    "falling_release": "Notes where pitch falls steeply during the release phase",
    "steady_sustain": "Long notes held with very stable pitch (affirming)",
    "pitch_instability": "Long notes with erratic, non-vibrato pitch wobble",
    "breath_support_issue": "Phrases where pitch flatness and volume fade occur simultaneously, indicating breath support collapse",
    "registration_strain": "High notes sung with chest-voice tension causing sharpness and excess volume",
    "loud_pitch_instability": "Louder notes where pitch accuracy drops compared to quieter notes",
    "soft_passage_control": "Quiet notes sung with accurate pitch (affirming)",
    "high_note_control": "High notes (above passaggio) sung accurately and cleanly (affirming)",
    "scoop_with_fade": "Notes that both scoop into pitch and fade in volume simultaneously",
    "phrase_pitch_arc": "A phrase where pitch accuracy degrades from start to end",
    "section_strength": "A section with strong overall pitch accuracy (affirming)",
    "section_weakness": "A section with weak overall pitch accuracy",
    "best_overall_section": "The section with the strongest blend of pitch, timing, and technique (affirming)",
    "weakest_overall_section": "The section with the weakest blend of pitch, timing, and technique",
    "section_improvement": "A repeated section that was more accurate on its second occurrence (affirming)",
    "section_regression": "A repeated section that was less accurate on its second occurrence",
    # Technique
    "expressive_match": "A phrase where the singer matched the reference artist's vocal techniques (vibrato, falsetto, etc.) (affirming)",
    "expressive_moment": "A phrase where the singer added their own expressive techniques not in the reference (affirming)",
    "missed_expression": "A phrase where the reference used techniques the singer did not match",
    "vocal_texture": "A callout for a specific vocal texture (breathy, strong, etc.) used in a phrase",
    "vibrato_quality": "Observation about vibrato rate and extent compared to healthy ranges",
    "consistent_vibrato": "Vibrato that is stable and consistent in rate across notes (affirming)",
    "wide_vibrato": "Vibrato extent exceeding 120 cents, making pitch centre hard to perceive",
    "delayed_vibrato": "Vibrato that only appears in the second half of sustained notes",
    "straight_tone_control": "Intentional use of straight tone (no vibrato) with steady pitch (affirming/comparative)",
    "vibrato_with_support": "Vibrato combined with steady breath support (affirming)",
    "technique_accuracy_tradeoff": "Sections where high expression matching coincides with lower pitch accuracy",
    "expressive_stability": "Sections where both technique matching and pitch accuracy are strong (affirming)",
    "section_vibrato_contrast": "Vibrato usage that differs systematically across section types",
    # Alignment
    "late_entrance": "Individual notes entered significantly late or early",
    "timing_consistency": "A phrase where all notes were consistently late or early (systemic drift)",
    "section_delta": "Systematic pitch or timing differences between section types",
    "rushed_phrase": "A phrase where all notes arrived consistently early (rushing)",
    "dragged_phrase": "A phrase where all notes arrived consistently late (dragging)",
    "rhythmic_precision": "A phrase with very tight timing across all notes (affirming)",
    # Dynamics
    "fade_within_notes": "A phrase where volume decreases progressively across consecutive notes",
    "dynamic_drop": "A phrase where the singer is notably quieter than the reference",
    "dynamic_surge": "A phrase where the singer is notably louder than the reference",
    "section_dynamic_contrast": "Whether the song's sections (verse vs chorus) have appropriate volume contrast",
    "breathy_onset": "Notes where volume builds slowly at onset, producing a breathy start",
    "clean_onset": "Notes where volume rises rapidly at onset, producing a clean, firm start (affirming)",
    "sforzando_attack": "Notes with a sharp accent-then-sustain dynamic shape",
    "note_crescendo": "Notes where volume builds continuously from onset to release",
    "note_swell": "Notes with a soft-loud-soft (swell) dynamic shape",
    "support_fade": "Notes where volume drops mid-sustain due to breath support collapse",
    "dynamic_sustain": "Notes with very stable, even volume through the sustain (affirming)",
    "release_cutoff": "Notes that end very abruptly rather than tapering",
    "controlled_crescendo": "A phrase that builds in volume while maintaining pitch accuracy (affirming)",
}


# ---------------------------------------------------------------------------
# System prompt for vocal profile generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a vocal pedagogy and music knowledge expert. You will be given a song title, \
artist name, and language, and you will generate a structured vocal coaching profile \
that describes the stylistic and technical context for coaching a singer learning this song.

You must return valid JSON matching the schema below exactly.

SCHEMA:
{
  "genre_tags": ["string", ...],          // 2-5 genre labels
  "vocal_style": "string",                // 2-4 sentences describing the artist's vocal approach
  "key_techniques": ["string", ...],      // 3-6 specific vocal techniques the artist uses
  "emphasize_highlights": ["string", ...], // detector type names to boost (see list below)
  "deemphasize_highlights": ["string", ...], // detector type names to penalise
  "highlight_notes": {"detector_type": "reason", ...}, // rationale for each override
  "coaching_context": "string"            // 2-3 sentences for coaches to keep in mind
}

IMPORTANT RULES:
- emphasize_highlights and deemphasize_highlights must ONLY contain detector type names \
from the provided list. Do not invent new names.
- highlight_notes must provide an entry for every detector in emphasize_highlights and \
deemphasize_highlights.
- Aim for 2-5 emphasize entries and 2-5 deemphasize entries. Leave the lists empty if the \
song has very generic vocal demands.
- coaching_context should be practical guidance for interpreting the measurements, not \
a general description of the song.
"""


def _build_user_prompt(title: str, artist: str, language: str) -> str:
    detector_lines = "\n".join(
        f"  {name}: {desc}" for name, desc in _DETECTOR_DESCRIPTIONS.items()
    )
    return (
        f"Song: {title}\n"
        f"Artist: {artist}\n"
        f"Language: {language}\n\n"
        f"Available detector types (use these exact strings in emphasize_highlights "
        f"and deemphasize_highlights):\n"
        f"{detector_lines}\n\n"
        f"Generate the vocal profile JSON."
    )


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------


def generate_vocal_profile(
    manifest,  # SongManifest — imported lazily to avoid circular import
    llm,       # LLMClient or None
) -> "Optional[VocalProfile]":
    """Generate and return a ``VocalProfile`` for *manifest*.

    Returns ``None`` if *llm* is ``None`` or the API call fails, so callers
    can skip gracefully without breaking the pipeline.
    """
    from vocal_coach.schemas import VocalProfile  # lazy to avoid circular import

    if llm is None:
        logger.info("[vocal_profile] LLM client not available; skipping profile generation")
        return None

    title = getattr(manifest, "title", "") or ""
    artist = getattr(manifest, "artist", "") or ""
    language = getattr(manifest, "language", "English") or "English"
    song_id = getattr(manifest, "song_id", "") or ""

    if not title and not artist:
        logger.warning("[vocal_profile] manifest has no title or artist; skipping")
        return None

    logger.info("[vocal_profile] generating for '%s' by '%s'", title, artist)

    system = _SYSTEM_PROMPT
    user = _build_user_prompt(title, artist, language)

    raw = llm.chat_json(system=system, user=user, schema=_VocalProfileRaw)
    if raw is None:
        logger.warning("[vocal_profile] LLM call failed or returned None")
        return None

    # Validate that detector names are real.
    valid_detectors = set(_DETECTOR_DESCRIPTIONS.keys())
    emphasize = [d for d in raw.emphasize_highlights if d in valid_detectors]
    deemphasize = [d for d in raw.deemphasize_highlights if d in valid_detectors]
    dropped = (
        set(raw.emphasize_highlights) | set(raw.deemphasize_highlights)
    ) - valid_detectors
    if dropped:
        logger.warning(
            "[vocal_profile] LLM returned unknown detector names (dropped): %s", dropped
        )

    # Only keep notes for detectors that survived validation.
    kept_detectors = set(emphasize) | set(deemphasize)
    highlight_notes = {k: v for k, v in raw.highlight_notes.items() if k in kept_detectors}

    profile = VocalProfile(
        song_id=song_id,
        genre_tags=raw.genre_tags,
        vocal_style=raw.vocal_style,
        key_techniques=raw.key_techniques,
        emphasize_highlights=emphasize,
        deemphasize_highlights=deemphasize,
        highlight_notes=highlight_notes,
        coaching_context=raw.coaching_context,
        model_used=llm.model,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "[vocal_profile] done — %d emphasize, %d deemphasize detectors",
        len(emphasize),
        len(deemphasize),
    )
    return profile


# ---------------------------------------------------------------------------
# Pydantic schema for the raw LLM JSON response (lenient, then we validate)
# ---------------------------------------------------------------------------


from pydantic import BaseModel, Field  # noqa: E402


class _VocalProfileRaw(BaseModel):
    """Loose schema that accepts the raw LLM output before validation."""

    model_config = {"extra": "ignore"}

    genre_tags: list[str] = Field(default_factory=list)
    vocal_style: str = ""
    key_techniques: list[str] = Field(default_factory=list)
    emphasize_highlights: list[str] = Field(default_factory=list)
    deemphasize_highlights: list[str] = Field(default_factory=list)
    highlight_notes: dict[str, str] = Field(default_factory=dict)
    coaching_context: str = ""


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_vocal_profile(profile_path) -> "Optional[VocalProfile]":
    """Load a ``VocalProfile`` from *profile_path*; return ``None`` if missing."""
    from pathlib import Path
    from vocal_coach.schemas import VocalProfile

    path = Path(profile_path)
    if not path.is_file():
        return None
    try:
        return VocalProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[vocal_profile] failed to load %s: %s", path, exc)
        return None


__all__ = ["generate_vocal_profile", "load_vocal_profile", "_DETECTOR_DESCRIPTIONS"]
