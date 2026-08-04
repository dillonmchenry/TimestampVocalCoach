---
name: Sprint 3 Phase A Foundation
overview: "Build the foundational infrastructure for Sprint 3's LLM-powered coaching: a shared OpenAI client, a ChromaDB-backed RAG database populated with vocal pedagogy knowledge and per-highlight playbooks, and a vocal profile generator that characterizes each song's style expectations using GPT-4o-mini during reference preprocessing."
todos:
  - id: llm-infra
    content: Create vocal_coach/llm.py (OpenAI wrapper), LLMConfig dataclass, coaching.yaml llm section, add deps to requirements.txt
    status: completed
  - id: rag-store
    content: Create vocal_coach/rag.py (ChromaDB wrapper with build + query), scripts/build_rag.py, data/rag/ directory structure
    status: completed
  - id: playbooks
    content: Author detector-keyed YAML playbooks (~60-100 entries across pitch, technique, alignment, dynamics, cross-dimensional, section) with pedagogy, coaching voice, few-shot summaries, and practice tips under data/rag/playbooks/
    status: completed
  - id: pedagogy-sources
    content: Curate and add vocal pedagogy source documents under data/rag/sources/ (open-access papers, public-domain texts)
    status: completed
  - id: vocal-profile
    content: Create vocal_coach/vocal_profile.py, VocalProfile schema, prompt engineering, integrate into build_song.py
    status: completed
  - id: manifest-update
    content: Add vocal_profile_path to SongManifest schema and wire through web API manifest endpoint
    status: completed
  - id: genre-thresholds
    content: (Optional) Use vocal profile highlight_emphasis to weight highlight scores in select_highlights()
    status: completed
  - id: tests
    content: Unit tests for llm.py (mocked), rag.py (integration), vocal_profile.py (mocked), updated build_song.py
    status: completed
isProject: false
---

# Sprint 3 Phase A: Foundation (RAG + Vocal Profile + LLM Infrastructure)

Sprint 3 introduces LLM-powered coaching feedback. Phase A builds the three foundational pieces that Phase B (coaching card rewriting + performance summary) will consume: a shared LLM client, a retrieval-augmented generation database, and per-song vocal profiles.

---

## 1. LLM Client Infrastructure

Create a thin OpenAI wrapper at [`vocal_coach/llm.py`](vocal_coach/llm.py) that the rest of the codebase imports. This centralizes model selection, retry logic, and structured-output parsing.

**New file**: `vocal_coach/llm.py`
- `LLMClient` class wrapping `openai.OpenAI` (sync client -- async not needed since analysis already runs inline)
- `chat_json()` method that sends a system + user prompt pair and parses the response into a Pydantic model via `response_format={"type": "json_object"}`
- `chat_text()` method for free-text completions (used by the performance summary in Phase B)
- Graceful fallback: if the OpenAI key is missing or the call fails, return `None` so the pipeline still produces deterministic-only output
- Environment variable: `OPENAI_API_KEY` (loaded from env or a `.env` file via `python-dotenv`)

**Config additions**:
- Add `LLMConfig` dataclass to [`coaching_config.py`](vocal_coach/coaching_config.py):

```python
@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_tokens: int = 1024
    enabled: bool = True  # kill-switch for LLM features
```

- Add `llm:` section to [`config/coaching.yaml`](config/coaching.yaml)

**Dependencies** (add to [`requirements.txt`](requirements.txt)):
- `openai>=1.30`
- `python-dotenv>=1.0`

---

## 2. RAG Database (ChromaDB + Vocal Pedagogy Playbooks)

Build a local ChromaDB vector store containing vocal coaching knowledge that the LLM retrieves from when generating feedback. The database has two collections:

### 2a. Collection: `pedagogy`
General vocal pedagogy knowledge chunked from primary sources. Each document is a ~300-500 token passage with metadata tags for topic area (breath support, vibrato, pitch accuracy, dynamics, phrasing, registration, etc.).

**Legal sourcing approach**: Use openly licensed or public-domain pedagogical content:
- Open-access vocal science papers (e.g., from NCBI/PubMed, Journal of Voice open-access articles)
- Public-domain vocal pedagogy textbooks (pre-1929 works)
- The project's own coaching knowledge distilled into structured markdown documents under `data/rag/sources/`

### 2b. Collection: `playbooks` (detector-keyed)

Per the earlier LLM architecture discussion, the playbooks should be **keyed to individual detector/highlight types** (not just categories). Target ~60-100 entries, one per detector type. Each entry contains:
- What this highlight type means in vocal pedagogy terms
- Common root causes
- Coaching language patterns (how a human coach would phrase this feedback -- direct, technically precise, no hedging)
- 1-2 **few-shot examples** of ideal card summary text for this detector type (used by Phase B to ground the LLM's prose register)
- Practice suggestions

This three-layer approach (register-engineered system prompt + detector-keyed RAG retrieval + few-shot examples) was identified in the earlier architecture discussion as the solution to the "encouraging but generic" LLM prose problem.

Playbooks are organized as YAML files under `data/rag/playbooks/`, grouped by category for authoring convenience but with per-detector entries:

```yaml
# data/rag/playbooks/pitch.yaml
entries:
  - detector: scoop_habit
    pedagogy: "Scooping (portamento from below) often indicates..."
    coaching_voice: "Lead with what the singer did, then why, then what to try."
    few_shot_summaries:
      - "You glide up into 4 notes in the verse by about 30 cents..."
      - "The original artist scoops into these same notes..."
    practice_tip: "Isolate the note and attack it from a half-step above..."
  - detector: pitch_instability
    ...
```

```
data/rag/
  playbooks/
    pitch.yaml              # scoop_habit, pitch_overshoot, sharp_flat_note, pitch_instability, etc.
    technique.yaml          # vibrato_quality, vocal_texture, expressive_match, etc.
    alignment.yaml          # rushed_phrase, late_entrance, timing_consistency, etc.
    dynamics.yaml           # support_fade, dynamic_drop, breathy_onset, etc.
    cross_dimensional.yaml  # breath_support_issue, registration_strain, etc.
    section.yaml            # section_improvement, section_vibrato_contrast, etc.
  sources/
    ... pedagogy markdown files ...
```

**New file**: `vocal_coach/rag.py`
- `RAGStore` class wrapping `chromadb.PersistentClient`
- `build(sources_dir, playbooks_dir)` class method that chunks, embeds, and persists
- `query(highlight_type, k=3)` returns top-k relevant passages for a given highlight type
- `query_topics(topics, k=5)` returns passages matching a list of topic tags (for the performance summary)
- Embedding: use OpenAI's `text-embedding-3-small` (cheap, good enough for this corpus size)

**New script**: `scripts/build_rag.py`
- Reads all files from `data/rag/sources/` and `data/rag/playbooks/`
- Chunks them (playbooks stay as whole entries; pedagogy files get split)
- Upserts into ChromaDB at `data/rag/chroma_db/`
- Idempotent (safe to re-run)

**Dependencies** (add to `requirements.txt`):
- `chromadb>=0.5`

---

## 3. Vocal Profile Generation

A per-song vocal profile characterizes the artistic and stylistic context of the reference track. It is generated once during `build_song.py` and stored as a JSON sidecar.

**New file**: `vocal_coach/vocal_profile.py`
- `generate_vocal_profile(manifest: SongManifest, llm: LLMClient) -> VocalProfile`
- Prompt design: the prompt includes the song title, artist, and language, plus the **full list of all ~54 detector type names** grouped by category with a one-line description of each. The LLM produces structured JSON describing:
  - **genre_tags**: e.g. `["pop-rock", "soft-rock", "ballad"]`
  - **vocal_style**: free-text description of the artist's vocal approach for this song
  - **key_techniques**: list of techniques the artist is known for in this context (e.g. `["vibrato", "falsetto", "dynamic contrast"]`)
  - **emphasize_highlights**: detector types especially relevant for this song/artist/genre (score boost during selection). E.g. `["vibrato_quality", "dynamic_sustain", "steady_sustain"]`
  - **deemphasize_highlights**: detector types less relevant or stylistically inappropriate to flag (score penalty). E.g. `["scoop_habit", "straight_tone_control"]`
  - **highlight_notes**: per-type rationale explaining why each override exists. E.g. `{"scoop_habit": "Scooping is a signature technique of McCartney's ballad phrasing -- not a flaw to correct"}`
  - **coaching_context**: a 2-3 sentence coaching framing that Phase B can inject into feedback prompts (e.g. "This Beatles ballad emphasizes smooth legato phrasing and controlled dynamics. The vocal line is melodically simple but demands precise pitch and emotional delivery.")
- Prompt includes detector types from the `MOMENT_CATEGORY` dict in `highlights.py` so the LLM selects from valid names; any returned type not in the valid set is silently dropped

**New schema** in [`schemas.py`](vocal_coach/schemas.py):

```python
class VocalProfile(BaseModel):
    song_id: str
    genre_tags: list[str] = Field(default_factory=list)
    vocal_style: str = ""
    key_techniques: list[str] = Field(default_factory=list)
    emphasize_highlights: list[str] = Field(
        default_factory=list,
        description="Detector types especially relevant for this song/genre (score boost).",
    )
    deemphasize_highlights: list[str] = Field(
        default_factory=list,
        description="Detector types less relevant or stylistically inappropriate (score penalty).",
    )
    highlight_notes: dict[str, str] = Field(
        default_factory=dict,
        description="Per-type rationale for emphasis/de-emphasis overrides.",
    )
    coaching_context: str = ""
    model_used: str = ""
    generated_at: Optional[str] = None
```

**Pipeline integration** in [`scripts/build_song.py`](scripts/build_song.py):
- After the existing reference pipeline steps (pitch, loudness, STARS), add a new step:
  1. Load or initialize the LLM client
  2. Call `generate_vocal_profile(manifest, llm)`
  3. Write `vocal_profile.json` to `data/songs/<song_id>/`
  4. Update `SongManifest` with `vocal_profile_path: Optional[str]`
- Skippable via `--skip-vocal-profile` flag (mirrors existing `--skip-stars` pattern)
- If `OPENAI_API_KEY` is not set, skip gracefully with a log message

**Web API**: Add `vocal_profile_path` to the manifest response so the UI can display genre/style info.

---

## 4. (Optional) Genre-Based Threshold Adjustment

If the vocal profile identifies highlight types that are more or less relevant for the genre (e.g., `scoop_habit` is a stylistic feature of the artist, not a flaw), the emphasis/de-emphasis lists are used to scale candidate scores before the `_select_diverse()` round-robin in [`highlights.py`](vocal_coach/highlights.py).

Implementation:
- `select_highlights()` gains an optional `vocal_profile: Optional[VocalProfile]` parameter
- After scoring but before selection, multiply each candidate's `score` by a configurable boost (e.g., 1.5x) if its `type` is in `emphasize_highlights`, or a penalty (e.g., 0.5x) if in `deemphasize_highlights`
- Boost/penalty multipliers are configurable in `coaching.yaml` under `highlights:` (e.g., `emphasis_boost: 1.5`, `deemphasis_penalty: 0.5`)
- Types not in either list keep the default multiplier of 1.0
- The `highlight_notes` rationale for each override can optionally be injected into the card's `detail` dict for UI display (e.g., explaining why a scoop was not flagged)
- This is lightweight enough to include in Phase A, but can be deferred to Phase B if time is tight

---

## 5. Files Changed Summary

| File | Change |
|------|--------|
| `vocal_coach/llm.py` | **New**: OpenAI client wrapper |
| `vocal_coach/rag.py` | **New**: ChromaDB RAG store |
| `vocal_coach/vocal_profile.py` | **New**: Vocal profile generation |
| `scripts/build_rag.py` | **New**: RAG database build script |
| `data/rag/playbooks/*.yaml` | **New**: Per-category coaching playbooks |
| `data/rag/sources/*.md` | **New**: Vocal pedagogy knowledge docs |
| `vocal_coach/schemas.py` | Add `VocalProfile` model, add `vocal_profile_path` to `SongManifest` |
| `vocal_coach/coaching_config.py` | Add `LLMConfig` dataclass |
| `config/coaching.yaml` | Add `llm:` section |
| `scripts/build_song.py` | Add vocal profile generation step |
| `requirements.txt` | Add `openai`, `python-dotenv`, `chromadb` |
| `vocal_coach/highlights.py` | (Optional) Accept `vocal_profile` for emphasis weighting |

---

## 6. Testing Strategy

- **`llm.py`**: Unit test with mocked OpenAI responses; verify graceful fallback when key is missing
- **`rag.py`**: Integration test that builds a small test collection and queries it; verify chunking and retrieval relevance
- **`vocal_profile.py`**: Unit test with mocked LLM output; verify Pydantic parsing and schema compliance
- **`build_song.py`**: End-to-end test that runs the updated script on the existing "yesterday" song bundle with a mocked LLM client
- **Playbooks**: Validate YAML structure with a simple schema check script

---

## 7. Phase B Preview (separate plan)

Phase B will consume all of Phase A's outputs. Key design decisions from the earlier architecture discussion:

- **Batched LLM calls**: One call for all card summaries (not per-card), one separate call for the performance narrative. This keeps cost at ~$0.01-0.05 per performance and latency manageable.
- **Evidence-constrained prompts**: The LLM receives a structured evidence schema per card (type, measurements, lyric, confidence, feedback_basis, section) and is instructed: "Do not invent any information -- only describe what the measurements show."
- **Fallback to f-string templates**: If the OpenAI call fails or the key is missing, the existing deterministic title/summary strings are preserved. The pipeline never breaks due to LLM unavailability.
- **Audit logging**: Both the structured evidence input and the LLM output are stored in `analysis.json` so every coaching card's provenance is traceable.

Phase B deliverables:
- **Coaching card rewriting**: After `select_highlights()`, pass all `CoachingMoment` objects + vocal profile + RAG-retrieved playbook content through GPT-4o-mini in a single batched call. Title stays deterministic (informed by highlight type); `summary` body is LLM-generated. Original f-string summary preserved in `detail["deterministic_summary"]` as fallback.
- **Performance summary**: After all highlights + overview stats are computed, generate a 3-4 sentence natural-language performance narrative with actionable next steps, grounded by RAG-retrieved pedagogy. Every claim must trace to structured measurements passed in the prompt.
- **Schema updates**: New `performance_summary` field on `PerformanceAnalysis`; `detail["deterministic_summary"]` on `CoachingMoment` for audit trail.
- **UI updates**: Display the performance summary above coaching cards; styled LLM feedback card summaries.
- **(Stretch) Reflective questioning**: Optional `reflection_prompt` field per card -- "Try this" practice-oriented prompts generated alongside the card summary (Approach F from earlier discussion).
