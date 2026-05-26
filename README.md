# Timestamp Vocal Coach

An end-to-end **timestamped vocal coach** prototype for karaoke-style performances. The system separates **ML measurement** (pitch, phoneme techniques, loudness) from **deterministic coaching logic** (note-level comparisons, highlight selection, feedback templates).

The current checkpoint is **Sprint 2**: a full-song karaoke pipeline around an UltraStar bundle (**R.E.M. — *Losing My Religion***), with NanoPitch + STARS run on both reference and user vocals, auto-detected octave transposition for singers in a different register than the chart, a configurable highlight engine, and a FastAPI + Wavesurfer demo UI. Sprint 1 (`vocal_coach/align.py`, `scripts/build_reference.py`, GTSinger sample) is preserved below for historical context.

---

## Sprint 1: What was accomplished

### Reference from GTSinger (annotated example)

Sprint 1 uses a single **English GTSinger** segment (default: `English/EN-Alto-1/Breathy/innocence/Control_Group/0000`) as a stand-in for a future karaoke reference track.


| Source                            | Role                                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `0000.wav`                        | Isolated vocal recording                                                                                                                    |
| `0000.json`                       | Word-centric annotation: lyrics, phonemes (`ph` + `ph_start` / `ph_end`), note events (`note` + `note_start` / `note_end`), technique flags |
| `0000.musicxml` / `0000.TextGrid` | Present but not consumed in Sprint 1                                                                                                        |


`scripts/build_reference.py` projects GTSinger into our schema:

- `**reference_annotation.json`**: flat `words` / `phones` / `ph2word`, plus a `notes[]` list (each note has `start_s`, `end_s`, `midi_pitch`, `lyric_word`, and **phonemes assigned by time overlap** with GTSinger phone boundaries).
- `**stars_metadata.json`**: the word/phone list (and optional durations) STARS inference expects.

For the demo we run **the same wav** as both “reference” and “performance” (self-reference smoke test). Real karaoke in Sprint 2 will use **reference vocal + user vocal** separately.

### NanoPitch (offline): continuous pitch

**NanoPitch** ([separate repo](https://github.com/smuleinc/NanoPitch)) estimates **F0 and voicing** every 10 ms.


| Output field         | Meaning                                      |
| -------------------- | -------------------------------------------- |
| `time`               | Frame center (seconds)                       |
| `f0_hz`              | Decoded fundamental frequency (0 = unvoiced) |
| `voicing_confidence` | VAD probability that the frame is voiced     |

Sprint 1 uses this for:

- **Pitch deviation**: median cents vs target MIDI inside each reference note window.
- **Arrival detection**: voicing rising-edge or pitch-lock near note onset (see below).

### STARS (offline): phonemes and vocal techniques

**STARS** runs as a subprocess on the same wav with lyrics + phones from the reference (`stars_metadata.json`). It does **not** train new weights in this repo.


| Output (in `stars.json`)                                   | How it was used in sprint 1                                                                               |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Per-phoneme `start_s` / `end_s`                            | Timing cross-check only; **phone membership on cards comes from GTSinger overlap**, not STARS word labels |
| Technique flags (vibrato, breathy, glissando, falsetto, …) | **Yes**, attached to phonemes on note cards                                                               |


STARS is treated as a **phoneme-level feature extractor** in Sprint 1. Sprint 2 will lean on STARS to **timestamp phones from lyrics + G2P** on full karaoke material where hand labels do not exist.

### Naive note alignment (pitch, arrival, volume)

For each **reference note** (window `[start_s, end_s)` and `target_midi` from GTSinger), `vocal_coach/align.py` compares NanoPitch inside that window:

**Pitch**

- Convert each voiced frame’s `f0_hz` to MIDI, then **cents vs `note.midi_pitch`**.
- Report **median cents** (flat/sharp tags) and optional **drift** (slope of cents over time).

**Arrival** (two regimes)


| Regime                    | When                                                        | How measured                                                                                                                    |
| ------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Onset / re-attack**     | Previous note ends ≥ ~30 ms before this one (or first note) | **Voicing rising edge**: first frame where voicing crosses threshold after a short unvoiced lead-in, compared to `note.start_s` |
| **Continuation / legato** | Previous note butts against this one (slur, melisma)        | **Pitch lock**: first frame where F0 stays within ±50 cents of target for ~50 ms, compared to `note.start_s`                    |


**Volume**

- RMS slope over the note window (emphasis on second half for “fade near end”).

These rules are **naive placeholders**. Self-reference on GTSinger often shows non-zero pitch/arrival because labels, NanoPitch, and our definitions measure different events (e.g. chart note start vs acoustic pitch lock). Sprint 2 will anchor expected times to **karaoke MIDI** and **STARS phone onsets**, and add **% in tune** over aligned core windows.

### Note-level aggregate objects (coaching cards)

`aggregate_note()` merges the four tracks into one `**NoteCard`** per reference note:

```json
{
  "expected_pitch": { "midi": 61, "name": "C#4" },
  "lyric_word": "waking",
  "section": "Full",
  "time": "1.35s–1.63s",
  "measurements": {
    "pitch": "+6 cents, drifting down",
    "arrival": "+40ms late",
    "volume": "fades near end"
  },
  "phonemes": [
    { "phoneme": "W", "tags": [] },
    { "phoneme": "EY1", "tags": [] },
    { "phoneme": "K", "tags": [] }
  ],
  "tags": ["drifting down", "fading ending"]
}
```

- **Phonemes on the card** = phones from GTSinger that overlap the note interval; **tags** from STARS when a matching symbol overlaps the note window.
- `**scripts/demo_note_card.py`** emits the first five notes plus `note_cards_first5.json`.
- `**timeline.json`** = full join of reference + `pitch.json` + `stars.json` + `loudness.json` (substrate for Sprint 2 scans).

`**notebooks/sprint1_demo.ipynb`** visualizes reference notes, NanoPitch F0, loudness, STARS phoneme bands, and cents deviation over time.

### Sprint 1 diagram

```
GTSinger 0000.json  ──►  reference_annotation.json  (note windows + target MIDI)
        │
        ├──► stars_metadata.json  ──►  STARS  ──►  stars.json (techniques + style)
        │
0000.wav ──┼──►  NanoPitch  ──►  pitch.json (F0 + voicing)
        │
        └──►  librosa RMS  ──►  loudness.json

reference + pitch + stars + loudness  ──►  timeline.json
        │
        └──►  align.aggregate_note  ──►  note_card_*.json
```
---

## Sprint 2: Full-song karaoke pipeline

Sprint 2 graduates from the GTSinger self-reference to an end-to-end karaoke
demo around the **Losing My Religion** UltraStar bundle. The new layer adds
a song-centric data layout, a UltraStar-driven note grid, dual-track
NanoPitch + STARS measurement (reference *and* user vocals), and a small
FastAPI + Wavesurfer UI for picking a song, dropping in a recording, and
seeing highlight cards on the waveform.

### Song bundle layout

```
data/songs/<song_id>/
  manifest.json                # SongManifest (UltraStar metadata, paths)
  song.txt                     # UltraStar chart
  reference_vocal.mp3
  instrumental.mp3             # optional, used by the UI for context
  reference_annotation.json    # ReferenceAnnotation built from chart + G2P
  stars_metadata.json          # word/phone list shared with STARS
  reference/
    pitch.json                 # NanoPitch on reference vocal (precomputed)
    stars.json                 # STARS on reference vocal (precomputed)
    loudness.json              # per-frame RMS
  performances/<perf_id>/
    performance.<wav|mp3|...>
    pitch.json                 # NanoPitch on user vocal
    stars.json                 # STARS on user vocal
    stars_metadata.json        # same lyrics; wav_fn = user audio
    analysis.json              # PerformanceAnalysis (per-note + highlights)
```

`reference_annotation.json` is built from the UltraStar chart by
`vocal_coach/song.py`: `#BPM` + `#GAP` + per-syllable beat triplets become
`ReferenceNote` entries (one note per scoreable syllable), syllable text is
re-grouped into words, and English lyrics are run through `g2p_en` to
populate the phone list STARS expects.

### Dual-track measurement

Per song (one-time, slow):

- **NanoPitch** on the reference vocal → `reference/pitch.json`
- **STARS** on the reference vocal → `reference/stars.json` (phone timings + technique flags)

Per user upload (per-performance, fast-ish):

- **NanoPitch** on the user vocal → `pitch.json`
- **STARS** on the user vocal → `stars.json`
- **Global offset** between the user vocal and the song timeline
  (`vocal_coach.align_v2.estimate_global_offset_s`) — coarse search of
  voiced overlap between the user pitch track and the chart's expected
  voiced regions, no DTW.

### Note alignment v2 (`vocal_coach/align_v2.py`)

For every UltraStar note we now compute a typed `NoteMeasurementV2`:

| Field                | How it's measured |
|----------------------|-------------------|
| `median_cents`       | Median of `100 · (midi_user – midi_target)` across voiced frames in the core window. |
| `pct_in_tune`        | Fraction of voiced frames in the core window within ±50 cents (tunable). |
| `drift_cents_per_s`  | Linear cents/s slope across the core window. |
| `arrival_offset_ms`  | Voicing rising-edge (or pitch-lock for legato) vs the expected onset; expected onset is the first reference-STARS phone overlapping the note, falling back to the chart's note start. |
| `core_start_s`/`core_end_s` | Note window trimmed by attack/release and shifted by the detected user arrival. |

`NoteTechniqueComparison` rows align reference STARS techniques and
user STARS techniques on the same note window (matched / missed / user-added).

All thresholds live in [`config/coaching.yaml`](config/coaching.yaml) and
are loaded into `vocal_coach.coaching_config.CoachingConfig`.

### Highlights engine (`vocal_coach/highlights.py`)

Six deterministic detectors scan rolling note windows and emit
`CoachingMoment` entries; the top-level `select_highlights` ranks by score,
caps at the configured total, and dedupes overlap of the same type:

| Detector              | Source signal                                                |
|-----------------------|--------------------------------------------------------------|
| `best_pitch_phrase`   | Highest mean `pct_in_tune` in a window (multi-phrase)        |
| `pitch_struggle`      | Lowest mean `pct_in_tune` in a window (multi-phrase)         |
| `late_entrance`       | Largest \|`arrival_offset_ms`\| past the late/early threshold |
| `expressive_match`    | Most user STARS techniques **matching** the reference        |
| `expressive_moment`   | Most user STARS expressive techniques (matched + added)      |
| `missed_expression`   | Reference STARS technique the user didn't reproduce          |

Pitch detectors run on **wider, configurable windows** (default 8–16 notes)
and can surface multiple non-overlapping phrases per type via
`pitch_phrases_per_type`. STARS / expression detectors keep the original
4–8-note windows so each technique highlight stays tight to a phrase. All
windowing knobs live under `highlights:` in
[`config/coaching.yaml`](config/coaching.yaml):

| Setting                       | Purpose                                                  |
|-------------------------------|----------------------------------------------------------|
| `pitch_window_min/max`        | Note-count bounds for pitch highlight windows            |
| `pitch_phrases_per_type`      | How many non-overlapping pitch phrases per type          |
| `window_min/max`              | Note-count bounds for STARS / expression highlights      |
| `max_per_type`                | Cap on highlights of any one type in the final list      |
| `best_phrase_min_pct_in_tune` | Qualifying floor for a "best phrase" callout             |
| `pitch_struggle_max_pct_in_tune` | Qualifying ceiling for a "struggle" callout           |

Technique callouts use **user-friendly copy** in titles and summaries
(`TECH_LABELS` + `TECH_HINTS` in `highlights.py`); STARS keys like
`pharyngeal` surface as "deep, resonant tone" rather than raw jargon.

### Auto octave-shift (singer in a different register)

`vocal_coach.align_v2.estimate_octave_shift_semitones` detects a single
integer-octave (multiple of 12 semitones) transposition between the user's
vocal register and the UltraStar chart by taking the median per-note
residual of (user MIDI − chart MIDI) and rounding to the nearest octave.
That shift is then added to every chart target before pitch and arrival
are scored, so a user singing an octave (or three) below the chart still
gets in-tune frames credited correctly. Per-frame cents are
**octave-folded** into `[-600, +600]` as a safety net, so a single
mid-take octave jump is scored as the same pitch class and tagged as
`octave above` / `octave below` on that note's `note_octave_offset`.
The detected shift is persisted on `PerformanceAnalysis.octave_shift_semitones`
and rendered as an auto-transpose badge in the UI.

### FastAPI + Wavesurfer demo (`web/`)

`web/api/main.py` exposes the song list, manifest, audio streams, and a
single `POST /api/songs/<song_id>/analyze` that runs the full
`measure_song` + `select_highlights` pipeline on an uploaded performance
and returns a `PerformanceAnalysis`. The static frontend in
`web/static/` lets you pick a song, drag-and-drop a performance, see the
waveform with colored highlight regions, and click any highlight card to
jump the playhead.

```powershell
uvicorn web.api.main:app --reload
# then open http://127.0.0.1:8000
```

### Sprint 2 demo path (Losing My Religion)

```powershell
# 1. Normalize the bundle into the song layout
python scripts/import_ultrastar.py data/songs/losing-my-religion `
    --reference-vocal reference_vocal.mp3 `
    --instrumental instrumental.mp3

# 2. Precompute reference NanoPitch + STARS + loudness (slow; once per song)
python scripts/build_song.py data/songs/losing-my-religion

# 3. Analyze a performance from the CLI
python scripts/analyze_performance.py data/songs/losing-my-religion `
    "data/Losing-My-Religion/Losing My Religion_user_vocals.wav" `
    --perf-id take-1

# 4. Or run the same flow through the UI
uvicorn web.api.main:app --reload
```

Pass `--midi-offset -12` to `import_ultrastar.py` if the chart's pitch
column is one octave above the actual recording (UltraStar charts often
encode pitches near `+24` to mean a melody an octave lower). In most
cases this is no longer required — `analyze_performance.py` now
auto-detects an integer-octave shift between the user vocal and the
chart targets and applies it before scoring (see "Auto octave-shift"
above).

---

## Sprint 3: Goals

Sprint 3 builds on the Sprint 2 single-song pipeline to make the coach
**broader (more songs)**, **lighter (faster STARS)**, **smarter (trend
detection across whole sections)**, and **more readable (overview +
cards UI)**.

### 1. Multiple reference songs for selection

The song bundle layout (`data/songs/<song_id>/...`) and the FastAPI
`/api/songs` endpoint already support multiple songs; Sprint 3 lights
the rest of the path up:

- A second (and third) UltraStar import beyond *Losing My Religion*,
  with `build_song.py` precomputing reference NanoPitch + STARS for
  each.
- Song-picker UX polish in `web/static/`: artist / language / duration
  badges, search, and remembering the last selection.
- Per-song coaching overrides (e.g. a `coaching_overrides.yaml` inside
  the song bundle) so genre-specific thresholds — talk-sung pop vs
  belting power ballad — can ship per song without globally retuning
  `config/coaching.yaml`.
- A small **karaoke catalog** doc covering UltraStar source legality,
  stem separation tooling, and the steps to add a new song end-to-end.

### 2. Train a STARS feature-extraction student model

The Sprint 2 [STARS lite spike](docs/sprint2_stars_lite.md) concluded
that the current bilingual checkpoint is **fine for offline batch but
heavy for interactive upload** (~90 s per ~4-minute vocal on a single
GPU). Sprint 3 takes the distillation path:

- **Teacher**: today's `stars_bilingual` Conformer (RMVPE F0 +
  phoneme/technique heads).
- **Student**: a 2-layer Conformer (or comparable) trained to mimic
  STARS technique flags + phone timings on the songs we already have
  reference STARS for. The labels are STARS's own outputs — no human
  re-annotation required.
- **Targets**: ≥5× wall-clock speedup on a single GPU, ≤1 GB
  combined weight footprint, technique-flag agreement with the teacher
  within an acceptance band on a held-out set.
- **Integration**: `vocal_coach/stars_runner.py` already accepts
  `extra_args`; add `--stars-profile fast|full` to
  `analyze_performance.py` and the FastAPI endpoint so the UI can opt
  into the student model.

This unblocks **real-time-ish feedback** in the demo without changing
the highlight engine.

### 3. Trend-detecting highlight engine

Today every highlight is a **local phrase**. Sprint 3 layers a
**section-level pass** on top so the engine can reason across the
whole song:

- New detectors that scan UltraStar sections / verses / choruses
  instead of rolling 4–16-note windows (e.g. "you're consistently
  flat in the first chorus but in tune in the second").
- **Trend stats** per `ReferenceSection`: mean / variance of
  `median_cents`, voiced coverage, `pct_in_tune`, arrival bias,
  technique densities — all surfaced as a new
  `SectionTrend` data model.
- Cross-section deltas: "verses are 30 cents flatter than choruses",
  "the bridge is your strongest section by `pct_in_tune`", "you keep
  vibrato in the chorus but drop it in the verse".
- Trend moments rank against existing local highlights in
  `select_highlights` so the final list mixes per-phrase callouts with
  song-wide patterns.

### 4. Overview + cards feedback UI

Sprint 2's UI is a list of moments on a waveform. Sprint 3 splits
feedback into two passes:

- **Overview**: at the top of the results panel, a small **stat
  block** — overall `pct_in_tune`, average median cents, detected
  octave shift, voiced coverage, expressive-technique density,
  pitch trend per section — rendered as compact tiles.
- **Cards**: the existing highlight list becomes a row of richer
  **coaching cards** with title, friendly summary, key numeric stat,
  the lyric snippet, and a play-from-here action. The card-row scrolls
  horizontally on small screens.
- **Section ribbon** on the waveform highlights verses / choruses /
  bridges (taken from `ReferenceAnnotation.sections`) so the user can
  jump to any section.
- API: persist these into `PerformanceAnalysis` (new `overview` and
  `sections` fields) so the FastAPI app can return everything in one
  response.

### Sprint 3 success criteria

| Theme | Target |
|-------|--------|
| Songs | ≥3 imported songs selectable in the UI |
| STARS student | ≥5× faster than `stars_bilingual` with documented agreement on technique flags |
| Trend engine | ≥1 section-level highlight type shipped + per-section stats in `analysis.json` |
| UI | Overview stat block + redesigned highlight cards + section ribbon in `web/static/` |

---

## Repo layout

```
TimestampVocalCoach/
├── rmvpe/                              # RMVPE weights (gitignored; setup links into STARS)
├── stars_chinese_english_bilingual/    # STARS bilingual ckpt (gitignored)
├── third_party/stars/                  # clone gwx314/STARS + setup_stars_runtime.py
├── config/coaching.yaml                # Sprint 2 thresholds
├── vocal_coach/
│   ├── schemas.py
│   ├── reference.py                    # GTSinger -> ReferenceAnnotation (Sprint 1)
│   ├── ultrastar.py                    # UltraStar .txt parser (Sprint 2)
│   ├── song.py                         # UltraStar -> ReferenceAnnotation + manifest
│   ├── pitch.py                        # NanoPitch wrapper
│   ├── stars_runner.py                 # STARS subprocess wrapper
│   ├── loudness.py                     # RMS / dBFS
│   ├── align.py                        # Sprint 1 single-note NoteCard
│   ├── align_v2.py                     # Sprint 2 dual-track measurements
│   ├── highlights.py                   # Sprint 2 coaching-moment detectors
│   └── coaching_config.py              # Sprint 2 threshold dataclass
├── web/
│   ├── api/main.py                     # FastAPI app
│   └── static/{index.html,app.js,style.css}
├── scripts/
│   ├── download_gtsinger_sample.py
│   ├── build_reference.py
│   ├── setup_stars_runtime.py
│   ├── validate_pitch.py
│   ├── run_pipeline.py                 # Sprint 1 driver
│   ├── demo_note_card.py               # Sprint 1 demo
│   ├── import_ultrastar.py             # Sprint 2: chart + audio -> song bundle
│   ├── build_song.py                   # Sprint 2: precompute reference tracks
│   └── analyze_performance.py          # Sprint 2: user vocal -> analysis.json
├── notebooks/
│   ├── sprint1_demo.ipynb
│   └── sprint2_dual_track.ipynb
├── docs/sprint2_stars_lite.md
├── data/
│   ├── samples/<sample_id>/            # gitignored, Sprint 1 GTSinger fixtures
│   └── songs/<song_id>/                # Sprint 2 song bundles
└── requirements.txt
```

---

## Setup

**Prerequisites**

- **NanoPitch** cloned locally (default: `../NanoPitch`). Override with `NANOPITCH_DIR` or `--nanopitch-dir`.
- **Python 3.10+**, PyTorch, CUDA recommended for STARS + NanoPitch.
- Model weights: place under `rmvpe/` and `stars_chinese_english_bilingual/`, then run setup (not committed to git).

```powershell
python -m pip install -r requirements.txt
python -m pip install tensorboard mir_eval pyloudnorm scikit-image g2p_en `
                      einops praat-parselmouth torchmetrics pyworld webrtcvad-wheels

git clone https://github.com/gwx314/STARS.git third_party/stars
python scripts/setup_stars_runtime.py
```

---

## End-to-end demo

```powershell
python scripts/download_gtsinger_sample.py
python scripts/build_reference.py data/samples/EN-Alto-1__innocence__0000
python scripts/run_pipeline.py data/samples/EN-Alto-1__innocence__0000
python scripts/demo_note_card.py data/samples/EN-Alto-1__innocence__0000
# notebooks/sprint1_demo.ipynb: Run All
```

`run_pipeline.py` writes `pitch.json`, `stars.json`, `loudness.json`, and `timeline.json` under the sample directory. Re-run `build_reference.py` after changing GTSinger parsing without re-running STARS if you only need updated note/phone assignments on cards (`demo_note_card.py` reloads `reference_annotation.json` directly).

---

## Validation

```powershell
python scripts/validate_pitch.py --device cuda
```

1. **Parity**: bit-identical F0/voicing vs NanoPitch’s `evaluate.py` on a fixed test mel.
2. **Sanity**: reasonable log-mel stats and voiced F0 range on the sample wav.

---

