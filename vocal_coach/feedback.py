"""Sprint-3 Phase B: LLM-powered feedback generation.

Two public functions are exposed:

``rewrite_card_summaries``
    Batch-rewrites the ``summary`` field of every ``CoachingMoment`` using a
    single GPT-4o-mini call grounded by detector-keyed RAG playbooks and the
    vocal profile context.  Falls back to the original deterministic summary
    if the LLM is unavailable or fails.

``generate_performance_summary``
    Synthesises all highlights, overview stats, and section trends into a
    structured ``PerformanceSummary`` (narrative + takeaways + trends) using
    a single GPT-4o-mini call grounded by RAG pedagogy passages.  Returns
    ``None`` on failure so the pipeline never breaks.

Both calls are evidence-constrained: the prompts contain only measurements
that the pipeline actually computed and instruct the model not to invent
observations.  Every rewriting decision is traceable via
``detail["deterministic_summary"]`` and ``detail["llm_evidence"]``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal Pydantic response schemas for LLM structured output
# ---------------------------------------------------------------------------


class _CardSummary(BaseModel):
    """Single rewritten card in the batch response."""

    model_config = {"extra": "ignore"}

    id: str
    summary: str
    practice_tip: Optional[str] = None


class _CardSummaryBatch(BaseModel):
    """Full batch response: one entry per coaching card."""

    model_config = {"extra": "ignore"}

    cards: list[_CardSummary]


class _PerformanceSummaryRaw(BaseModel):
    """Raw LLM response schema for the performance summary call."""

    model_config = {"extra": "ignore"}

    narrative: str = ""
    takeaways: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CARD_SYSTEM_BASE = """\
You are a vocal coach giving specific, comparative feedback on a singing performance.
For each coaching card, write a single sentence that describes what the measurements show — \
the observation only. Do not include coaching suggestions, instructions, or exercises in the summary. \
Those belong in the practice_tip field, not here.

RULES:
- One sentence maximum per card. Shorter is better.
- Only describe what the measurements show. Do not invent observations.
- Do NOT mention the associated lyric — it is already displayed separately on the card.
- Do NOT include any suggestion, instruction, drill, or fix in the summary.
- Avoid vague praise ("great job", "nice work").
- COMPARATIVE FRAMING (most important rule): Frame ALL feedback relative to the original \
artist's recording. Whether the card is affirming or corrective, always say what the artist \
does in the same passage and how the user's performance compares. Never describe a vocal \
moment without referencing whether the artist does the same thing. Use the artist's name \
(from SONG CONTEXT) rather than "the reference."
- ACCESSIBLE LANGUAGE: Do NOT use technical jargon or raw numbers. Translate measurements:
  • "slightly flat" or "slightly sharp" instead of any number of cents
  • "a beat early" or "a beat late" instead of milliseconds
  • "a blend of chest and head voice" instead of "mixed chest/head voice"
  • Describe vibrato as "a gentle wavering in pitch" on first mention if clarity helps
  • Never quote raw values (dB, Hz, ratios, percentages)
- When a playbook passage is provided, let its coaching register guide your tone.

PRACTICE TIP ENHANCEMENT (optional):
If a "practice_tip" is provided in the card evidence, you may optionally produce an enhanced
version that references specific lyrics or notes from this moment when doing so makes the
exercise more concrete (e.g. "Try sustaining the 'blue' vowel shape on a single pitch for
four beats"). If the tip is already sufficiently specific, or the lyric context adds nothing
meaningful, return it unchanged. If no practice_tip is provided, omit the field entirely.
Do NOT suggest sustaining a note if the note is short or fast — the tip must suit the context.

RESPONSE FORMAT:
Return valid JSON with a single key "cards" containing an array. Each element must have:
  "id": the exact id string from the input card,
  "summary": the single-sentence observational summary (no suggestions),
  "practice_tip": (optional) the practice tip, enhanced with lyric context if useful.

Only include cards that were in the input. Do not add cards or omit cards.\
"""

_SUMMARY_SYSTEM_BASE = """\
You are a vocal coach writing a performance summary after reviewing a complete singing performance.
You have access to every coaching highlight, overall statistics, section-by-section measurements, \
and context about the song and original artist.

YOUR TASKS:
  (1) Write a 3-4 sentence narrative assessment. Lead with how the overall performance \
compares to the original artist's approach — reference the artist by name, their signature \
vocal choices on this song, and where the user's take matched or diverged. Ground every \
observation in the comparison, not in abstract technique.
  (2) List 2-4 actionable takeaways — specific, prioritised directional observations \
about what to focus on next. Each takeaway should name a section or pattern and compare \
it to the artist's approach (e.g. "In the chorus your voice lost power mid-phrase where \
[Artist] sustains fully — that's the biggest gap to close").
  (3) List 1-3 trends you observe by looking across ALL the highlights together — \
patterns that span multiple cards or sections (e.g. "Pitch accuracy drops in every chorus \
compared to the verses", "Your expressiveness grows as the song progresses — \
the later sections sound more like the artist than the opening").

RULES:
- Only cite observations present in the data. Do not invent.
- Write like a coach talking to a singer, not an engineer reading a report.
- ACCESSIBLE LANGUAGE — never quote raw numbers. Translate every measurement:
  • Instead of "arriving 489ms late" say "consistently late on your entrances"
  • Instead of "-28 dB/s fade" say "your voice loses power quickly through the phrase"
  • Instead of "25 cents flat" say "running a touch flat"
  • Use magnitude words ("very", "slightly", "consistently", "noticeably")
  • Do NOT say "cents", "dB", "Hz", "milliseconds", or any ratio/percentage
  • Do NOT say "mixed chest/head voice" — say "a blend of chest and head voice"
- Reference the original artist by name (not "the reference") throughout.
- TIMING: Only mention timing if it correlates with another issue (e.g. rushing a phrase \
that compresses expressiveness, or late entrances that suggest breath control difficulty). \
Do not mention timing as a standalone observation.
- Takeaways should be directional observations, not prescriptive drills.
- Trends must be genuine cross-highlight patterns, not a restatement of a single card.

RESPONSE FORMAT:
Return valid JSON with keys:
  "narrative": string (3-4 sentences),
  "takeaways": array of strings (2-4 items),
  "trends": array of strings (1-3 items).\
"""


# ---------------------------------------------------------------------------
# Evidence extraction helpers
# ---------------------------------------------------------------------------

# Keys from CoachingMoment.detail that are useful signal for the LLM.
# Keys that are only for UI/audit (evidence_strength, profile_*_note, llm_evidence)
# are excluded to keep the prompt compact.
_DETAIL_KEYS_INCLUDE = {
    "mean_pct_in_tune",
    "median_cents",
    "direction",
    "note_name",
    "section",
    "window_size",
    "count",
    "notes_in_window",
    "pct_in_tune",
    "cents",
    "rate_hz",
    "extent_cents",
    "technique",
    "user_note_count",
    "ref_note_count",
    "mean_arrival_offset_ms",
    "rms_fade_db_per_s",
    "rms_delta_db",
    "onset_slope_db_per_s",
    "support_fade_db_per_s",
    "scoop_cents",
    "pitch_std_cents",
    "kind_a",
    "kind_b",
    "delta_pct_in_tune",
    "delta_cents",
    "delta_rms_db",
    "user_density",
    "ref_density",
    "pct_in_tune_a",
    "pct_in_tune_b",
}


def _extract_measurements(detail: dict) -> dict:
    """Filter detail dict to only the measurement keys useful to the LLM."""
    return {k: v for k, v in detail.items() if k in _DETAIL_KEYS_INCLUDE}


def _lyric_for_moment(moment, reference) -> str:
    """Extract a short lyric snippet from the reference annotation."""
    if reference is None or not moment.note_indices:
        return ""
    ref_notes = getattr(reference, "notes", [])
    seen: set[int] = set()
    words: list[str] = []
    for idx in moment.note_indices[:12]:
        if idx >= len(ref_notes):
            continue
        note = ref_notes[idx]
        word_idx = getattr(note, "word_index", None)
        if word_idx is not None and word_idx in seen:
            continue
        word = (getattr(note, "lyric_word", "") or "").strip()
        if word:
            words.append(word)
            if word_idx is not None:
                seen.add(word_idx)
        if len(words) >= 8:
            break
    return " ".join(words)


def _build_card_evidence(moment, reference, playbook_passage: str) -> dict:
    """Assemble the evidence dict for one CoachingMoment."""
    ev: dict = {
        "id": moment.id,
        "type": moment.type,
        "feedback_basis": moment.feedback_basis,
        "confidence": moment.confidence,
        "scope": moment.scope,
        "title": moment.title,
        "section": moment.detail.get("section") or (
            ", ".join(moment.section_names) if moment.section_names else None
        ),
        "lyric_context": _lyric_for_moment(moment, reference) or None,
        "measurements": _extract_measurements(moment.detail),
        "playbook": playbook_passage or None,
    }
    if moment.practice_tip:
        ev["practice_tip"] = moment.practice_tip
    return ev


# ---------------------------------------------------------------------------
# Coaching card summary rewriting
# ---------------------------------------------------------------------------


def rewrite_card_summaries(
    moments: list,
    *,
    llm,        # LLMClient | None
    rag,        # RAGStore | None
    vocal_profile=None,   # VocalProfile | None
    reference=None,       # ReferenceAnnotation | None
    song_title: str = "",
    artist: str = "",
    max_tokens: int = 2048,
) -> list:
    """Rewrite CoachingMoment.summary fields using a batched LLM call.

    Returns the same ``moments`` list (modified in place) with:
    - ``summary`` replaced by LLM-generated coaching prose
    - ``detail["deterministic_summary"]`` set to the original template value
    - ``detail["llm_evidence"]`` set to the evidence dict sent to the LLM

    Falls back to the original summaries if ``llm`` is ``None`` or the call
    fails, so the pipeline never breaks.
    """
    if llm is None:
        logger.info("[feedback] LLM not available — skipping card summary rewriting")
        return moments
    if not moments:
        return moments

    # Build evidence objects (one per moment), retrieving playbook passage per type.
    evidence_list: list[dict] = []
    for m in moments:
        passage = ""
        if rag is not None:
            passages = rag.query_detector(m.type, k=1)
            passage = passages[0] if passages else ""
        ev = _build_card_evidence(m, reference, passage)
        evidence_list.append(ev)

    # Build the system prompt with song context and optional stylistic context.
    system = _CARD_SYSTEM_BASE
    song_ctx_parts: list[str] = []
    if song_title or artist:
        label = f'"{song_title}"' if song_title else "this song"
        if artist:
            label += f" by {artist}"
        song_ctx_parts.append(f"Song: {label}.")
    if vocal_profile:
        vs = getattr(vocal_profile, "vocal_style", "")
        if vs:
            song_ctx_parts.append(vs)
        cc = getattr(vocal_profile, "coaching_context", "")
        if cc:
            song_ctx_parts.append(cc)
    if song_ctx_parts:
        system += "\n\nSONG CONTEXT:\n" + " ".join(song_ctx_parts)

    # Process cards in chunks so each LLM call stays well within token limits.
    # ~5 cards per call keeps each response comfortably under 1 000 output tokens.
    CHUNK_SIZE = 5
    chunks = [
        evidence_list[i : i + CHUNK_SIZE]
        for i in range(0, len(evidence_list), CHUNK_SIZE)
    ]

    rewritten: dict[str, str] = {}
    rewritten_tips: dict[str, str] = {}
    orig_max = llm.max_tokens
    llm.max_tokens = max_tokens
    try:
        for chunk_idx, chunk in enumerate(chunks):
            user = json.dumps({"cards": chunk}, ensure_ascii=False, indent=2)
            result = llm.chat_json(system=system, user=user, schema=_CardSummaryBatch)
            if result is None:
                logger.warning(
                    "[feedback] card rewriting failed for chunk %d/%d — those cards keep deterministic summaries",
                    chunk_idx + 1, len(chunks),
                )
                continue
            for c in result.cards:
                if c.summary.strip():
                    rewritten[c.id] = c.summary
                if c.practice_tip and c.practice_tip.strip():
                    rewritten_tips[c.id] = c.practice_tip.strip()
            logger.debug(
                "[feedback] chunk %d/%d: %d/%d cards rewritten",
                chunk_idx + 1, len(chunks), len(result.cards), len(chunk),
            )
    finally:
        llm.max_tokens = orig_max

    matched = 0
    for m, ev in zip(moments, evidence_list):
        new_summary = rewritten.get(m.id, "")
        if new_summary:
            m.detail["deterministic_summary"] = m.summary
            m.detail["llm_evidence"] = ev
            m.summary = new_summary
            matched += 1
        else:
            logger.debug("[feedback] no LLM summary for card %s — keeping deterministic", m.id)
        # Apply LLM-enhanced practice tip (falls back to the static one already set).
        enhanced_tip = rewritten_tips.get(m.id, "")
        if enhanced_tip:
            m.practice_tip = enhanced_tip

    logger.info("[feedback] card rewriting: %d/%d cards updated", matched, len(moments))
    return moments


# ---------------------------------------------------------------------------
# Performance summary generation
# ---------------------------------------------------------------------------

# Maps highlight types to broad topic tags for RAG pedagogy retrieval.
_TYPE_TO_TOPICS: dict[str, list[str]] = {
    "best_pitch_phrase": ["pitch", "pitch_accuracy"],
    "pitch_struggle": ["pitch", "pitch_accuracy"],
    "sharp_flat_note": ["pitch", "pitch_accuracy"],
    "scoop_habit": ["pitch", "phrasing_timing"],
    "pitch_overshoot": ["pitch"],
    "clean_attack": ["pitch"],
    "falling_release": ["pitch", "breath_support"],
    "steady_sustain": ["pitch"],
    "pitch_instability": ["pitch", "vibrato"],
    "breath_support_issue": ["breath_support", "pitch"],
    "registration_strain": ["registration", "pitch"],
    "loud_pitch_instability": ["pitch", "dynamics_expression"],
    "soft_passage_control": ["pitch", "dynamics_expression"],
    "high_note_control": ["registration", "pitch"],
    "scoop_with_fade": ["pitch", "breath_support"],
    "phrase_pitch_arc": ["pitch", "breath_support"],
    "vibrato_quality": ["vibrato"],
    "consistent_vibrato": ["vibrato"],
    "wide_vibrato": ["vibrato"],
    "delayed_vibrato": ["vibrato"],
    "straight_tone_control": ["vibrato"],
    "expressive_match": ["style_and_genre"],
    "expressive_moment": ["style_and_genre"],
    "missed_expression": ["style_and_genre"],
    "vocal_texture": ["style_and_genre"],
    "vibrato_with_support": ["vibrato", "breath_support"],
    "technique_accuracy_tradeoff": ["pitch", "vibrato"],
    "expressive_stability": ["vibrato", "style_and_genre"],
    "section_vibrato_contrast": ["vibrato"],
    "late_entrance": ["phrasing_timing"],
    "timing_consistency": ["phrasing_timing"],
    "rushed_phrase": ["phrasing_timing"],
    "dragged_phrase": ["phrasing_timing"],
    "rhythmic_precision": ["phrasing_timing"],
    "section_delta": ["phrasing_timing"],
    "fade_within_notes": ["breath_support", "dynamics_expression"],
    "dynamic_drop": ["dynamics_expression"],
    "dynamic_surge": ["dynamics_expression"],
    "section_dynamic_contrast": ["dynamics_expression"],
    "breathy_onset": ["dynamics_expression", "breath_support"],
    "clean_onset": ["dynamics_expression"],
    "sforzando_attack": ["dynamics_expression"],
    "note_crescendo": ["dynamics_expression"],
    "note_swell": ["dynamics_expression"],
    "support_fade": ["breath_support", "dynamics_expression"],
    "dynamic_sustain": ["dynamics_expression"],
    "release_cutoff": ["dynamics_expression"],
    "controlled_crescendo": ["dynamics_expression", "breath_support"],
}


def _topics_from_moments(moments: list) -> list[str]:
    """Derive RAG topic tags from the types present in the highlight set."""
    seen: set[str] = set()
    tags: list[str] = []
    for m in moments:
        for tag in _TYPE_TO_TOPICS.get(m.type, []):
            if tag not in seen:
                tags.append(tag)
                seen.add(tag)
    return tags


def _section_summary_rows(sections: list) -> list[dict]:
    """Compact per-section dict for the summary prompt."""
    rows = []
    for s in sections:
        row: dict = {"name": s.name}
        if s.kind:
            row["kind"] = s.kind
        if s.pct_in_tune is not None:
            row["pct_in_tune"] = round(s.pct_in_tune, 3)
        if s.median_cents is not None:
            row["median_cents"] = round(s.median_cents, 1)
        if s.arrival_offset_ms_mean is not None:
            row["arrival_offset_ms_mean"] = round(s.arrival_offset_ms_mean, 1)
        if s.mean_rms_db is not None:
            row["mean_rms_db"] = round(s.mean_rms_db, 1)
        if s.technique_density:
            # Only include techniques with non-trivial density
            row["technique_density"] = {
                k: round(v, 3) for k, v in s.technique_density.items() if v > 0.05
            }
        rows.append(row)
    return rows


def _highlight_summary_rows(moments: list) -> list[dict]:
    """Compact per-highlight dict for the summary prompt, ordered by time."""
    rows = []
    for m in sorted(moments, key=lambda x: x.start_s):
        row: dict = {
            "type": m.type,
            "scope": m.scope,
            "title": m.title,
            "summary": m.summary,   # the (already-rewritten) summary
            "confidence": m.confidence,
        }
        section = m.detail.get("section") or (
            ", ".join(m.section_names) if m.section_names else None
        )
        if section:
            row["section"] = section
        # Include key measurements if present.
        measurements = _extract_measurements(m.detail)
        if measurements:
            row["measurements"] = measurements
        rows.append(row)
    return rows


def _overview_dict(overview) -> dict:
    """Compact overview dict for the summary prompt."""
    if overview is None:
        return {}
    d: dict = {}
    for field in (
        "mimic_score", "pct_in_tune", "median_cents", "voiced_coverage",
        "arrival_offset_ms_mean", "expressive_density", "technique_match_rate",
        "note_count", "strongest_section",
    ):
        val = getattr(overview, field, None)
        if val is not None:
            d[field] = round(val, 3) if isinstance(val, float) else val
    return d


def generate_performance_summary(
    moments: list,
    overview,             # PerformanceOverview | None
    sections: list,
    *,
    llm,                  # LLMClient | None
    rag,                  # RAGStore | None
    vocal_profile=None,   # VocalProfile | None
    song_title: str = "",
    artist: str = "",
    max_tokens: int = 1024,
) -> "Optional[PerformanceSummary]":
    """Generate a structured performance summary via LLM.

    Returns a ``PerformanceSummary`` with narrative, takeaways, and observed
    trends, or ``None`` if the LLM is unavailable or the call fails.
    """
    from vocal_coach.schemas import PerformanceSummary   # lazy to avoid circular

    if llm is None:
        logger.info("[feedback] LLM not available — skipping performance summary")
        return None

    # RAG: retrieve pedagogy passages relevant to the highlight types present.
    pedagogy_context = ""
    if rag is not None:
        topics = _topics_from_moments(moments)
        passages = rag.query_topics(topics, k=5)
        if passages:
            pedagogy_context = "\n\n---\n\n".join(passages[:5])

    # Build the evidence payload.
    payload: dict = {
        "overview": _overview_dict(overview),
        "sections": _section_summary_rows(sections),
        "highlights": _highlight_summary_rows(moments),
    }
    if pedagogy_context:
        payload["pedagogy_context"] = pedagogy_context

    # System prompt with song context and stylistic grounding.
    system = _SUMMARY_SYSTEM_BASE
    song_ctx_parts: list[str] = []
    if song_title or artist:
        label = f'"{song_title}"' if song_title else "this song"
        if artist:
            label += f" by {artist}"
        song_ctx_parts.append(f"Song: {label}.")
    if vocal_profile:
        genres = getattr(vocal_profile, "genre_tags", [])
        if genres:
            song_ctx_parts.append(f"Genre: {', '.join(genres)}.")
        vs = getattr(vocal_profile, "vocal_style", "")
        if vs:
            song_ctx_parts.append(vs)
        cc = getattr(vocal_profile, "coaching_context", "")
        if cc:
            song_ctx_parts.append(cc)
    if song_ctx_parts:
        system += "\n\nSONG CONTEXT:\n" + " ".join(song_ctx_parts)

    user = json.dumps(payload, ensure_ascii=False, indent=2)

    # Override max_tokens for this call.
    orig_max = llm.max_tokens
    llm.max_tokens = max_tokens
    try:
        raw = llm.chat_json(system=system, user=user, schema=_PerformanceSummaryRaw)
    finally:
        llm.max_tokens = orig_max

    if raw is None:
        logger.warning("[feedback] performance summary LLM call failed")
        return None

    if not raw.narrative.strip():
        logger.warning("[feedback] performance summary returned empty narrative")
        return None

    summary = PerformanceSummary(
        narrative=raw.narrative,
        takeaways=raw.takeaways,
        trends=raw.trends,
        model_used=llm.model,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "[feedback] summary: %d takeaways, %d trends",
        len(summary.takeaways),
        len(summary.trends),
    )
    return summary


# ---------------------------------------------------------------------------
# Section story LLM rewriting
# ---------------------------------------------------------------------------

_STORY_SYSTEM_BASE = """\
You are a vocal coach writing a concise section-level assessment after reviewing a \
singing performance.
For each section of the song you will receive aggregate metrics and a list of specific \
coaching observations from that section.

YOUR TASK:
Write 2-3 sentences that describe what happened in this section of the performance, \
comparing the user's delivery to the original artist. Use the specific coaching \
observations to make the story concrete — you may reference particular findings \
(e.g. "the high notes went flat", "you dropped the vibrato on the held notes") but \
synthesise them into a section-level narrative rather than repeating card summaries.

RULES:
- Ground every observation in how the original artist performs this section. \
Always frame feedback as comparison to the artist, not generic advice.
- Use plain, accessible language. Do NOT use technical terms: \
  • "slightly flat" or "slightly sharp" instead of "X cents"  
  • "a beat early" or "a beat late" instead of milliseconds  
  • "a blend of chest and head voice" instead of "mixed chest/head voice"  
  • "a gentle wavering in pitch" instead of "vibrato" (unless already on screen)
- Do NOT use raw numbers (cents, dB, Hz, milliseconds, ratios).
- Keep it conversational — like a coach talking to a singer between takes, not a report.
- Do NOT repeat card summaries word-for-word.
- Prioritise the most impactful observations; omit minor details.

RESPONSE FORMAT:
Return valid JSON with a single key "stories" containing an array. Each element must have:
  "section_name": the exact section_name from the input,
  "narrative": the 2-3 sentence story.\
"""


class _StorySummary(BaseModel):
    """Single rewritten story in the batch response."""

    model_config = {"extra": "ignore"}

    section_name: str
    narrative: str


class _StorySummaryBatch(BaseModel):
    """Full batch response: one entry per section story."""

    model_config = {"extra": "ignore"}

    stories: list[_StorySummary]


def rewrite_section_stories(
    stories: list,
    moments: list,
    *,
    llm,
    rag=None,
    vocal_profile=None,
    song_title: str = "",
    artist: str = "",
    max_tokens: int = 1024,
) -> list:
    """Rewrite SectionStory.narrative fields using a batched LLM call.

    Each story is enriched with the selected ``CoachingMoment`` entries whose
    time span overlaps the section, giving the LLM specific findings to
    reference rather than only aggregate metrics.

    Falls back to ``deterministic_summary`` when the LLM is unavailable or
    the call fails, so the pipeline never breaks.
    """
    if llm is None:
        logger.info("[feedback] LLM not available — skipping section story rewriting")
        return stories
    if not stories:
        return stories

    def _moments_for_section(story) -> list[dict]:
        """Return compact moment dicts whose time span overlaps a story section."""
        out = []
        for m in moments:
            # Overlap: moment starts before section ends AND ends after section starts.
            if m.end_s > story.start_s and m.start_s < story.end_s:
                row: dict = {
                    "type": m.type,
                    "title": m.title,
                    "summary": m.summary,
                    "feedback_basis": m.feedback_basis,
                    "confidence": m.confidence,
                }
                meas = _extract_measurements(m.detail)
                if meas:
                    row["measurements"] = meas
                out.append(row)
        return out

    # Build evidence payload for each story.
    evidence_list: list[dict] = []
    for story in stories:
        ev: dict = {
            "section_name": story.section_name,
            "section_kind": story.section_kind,
            "deterministic_summary": story.deterministic_summary,
            "dimensions": story.dimensions,
            "coaching_observations": _moments_for_section(story),
        }
        evidence_list.append(ev)

    # Build system prompt with song context.
    system = _STORY_SYSTEM_BASE
    song_ctx_parts: list[str] = []
    if song_title or artist:
        label = f'"{song_title}"' if song_title else "this song"
        if artist:
            label += f" by {artist}"
        song_ctx_parts.append(f"Song: {label}.")
    if vocal_profile:
        vs = getattr(vocal_profile, "vocal_style", "")
        if vs:
            song_ctx_parts.append(vs)
        cc = getattr(vocal_profile, "coaching_context", "")
        if cc:
            song_ctx_parts.append(cc)
    if song_ctx_parts:
        system += "\n\nSONG CONTEXT:\n" + " ".join(song_ctx_parts)

    orig_max = llm.max_tokens
    llm.max_tokens = max_tokens
    rewritten: dict[str, str] = {}
    try:
        user = json.dumps({"stories": evidence_list}, ensure_ascii=False, indent=2)
        result = llm.chat_json(system=system, user=user, schema=_StorySummaryBatch)
        if result is None:
            logger.warning("[feedback] section story rewriting failed — keeping deterministic summaries")
        else:
            for s in result.stories:
                if s.narrative.strip():
                    rewritten[s.section_name] = s.narrative
            logger.info("[feedback] section story rewriting: %d/%d stories updated", len(rewritten), len(stories))
    finally:
        llm.max_tokens = orig_max

    for story in stories:
        narrative = rewritten.get(story.section_name, "")
        if narrative:
            story.narrative = narrative

    return stories


__all__ = ["rewrite_card_summaries", "rewrite_section_stories", "generate_performance_summary"]
