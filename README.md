# SecondPass

An interpretable **timestamped vocal coach** for karaoke-style performances. The system separates **ML measurement** (pitch, phoneme techniques, loudness) from **deterministic coaching logic** (note-level comparisons, section trends, highlight selection, feedback templates). Upload or record a take against an UltraStar chart; SecondPass returns timestamped coaching cards you can jump to on the waveform.

---

## How it works

```
UltraStar chart + reference vocal
        │
        ├──► reference_annotation.json   (note grid + G2P phones)
        ├──► NanoPitch  ──► reference/pitch.json
        ├──► STARS      ──► reference/stars.json
        └──► RMS        ──► reference/loudness.json

User performance
        │
        ├──► NanoPitch  ──► pitch.json
        ├──► STARS (full or fast student) ──► stars.json
        └──► align_v2 + trends + highlights
                    │
                    └──► PerformanceAnalysis
                         (per-note stats, section trends, overview, coaching moments)
```

| Layer | Role |
| ----- | ---- |
| **NanoPitch** | Continuous F0 and voicing every 10 ms ([smulelabs/NanoPitch](https://github.com/smulelabs/NanoPitch)) |
| **STARS** | Phoneme timings and vocal-technique flags (vibrato, breathy, glissando, falsetto, …) |
| **STARS student (`fast`)** | Distilled in-process model for quicker interactive feedback; falls back to full STARS if the checkpoint is missing |
| **`align_v2`** | Per-note pitch, timing, dynamics vs the chart; auto-detected octave transposition |
| **`trends` + `highlights`** | Section-level stats and ranked coaching moments |
| **`overview`** | Song-wide summary tiles (pitch accuracy, mimic score, strongest/weakest sections) |

Thresholds and highlight caps live in [`config/coaching.yaml`](config/coaching.yaml) and load via `vocal_coach.coaching_config.CoachingConfig`.

---

## Song bundles

Each song lives under `data/songs/<song_id>/`:

```
data/songs/<song_id>/
  manifest.json                # title, artist, paths, precomputed reference tracks
  song.txt                     # UltraStar chart
  reference_vocal.{wav,mp3,...}
  instrumental.{wav,mp3,...}     # optional; used by the web UI for karaoke mode
  reference_annotation.json    # note windows + lyrics + G2P phones (from chart)
  stars_metadata.json          # word/phone list STARS expects
  reference/
    pitch.json                 # NanoPitch on reference vocal (precomputed)
    stars.json                 # STARS on reference vocal (precomputed)
    loudness.json              # per-frame RMS
  performances/<perf_id>/
    performance.<wav|mp3|...>
    pitch.json
    stars.json
    loudness.json
    analysis.json              # full PerformanceAnalysis
```

`vocal_coach/song.py` builds `reference_annotation.json` from the UltraStar chart: `#BPM` + `#GAP` + syllable beat triplets become scorable notes, syllables are grouped into words, and English lyrics run through `g2p_en` for the phone list STARS needs.

Bundled demo songs include *Losing My Religion*, *Yesterday*, *Hot n Cold*, *Million Reasons*, *When September Ends*, and *No Scrubs* (UltraStar imports with precomputed reference tracks where checked in).

---

## Measurement

### Reference (one-time per song)

```powershell
python scripts/import_ultrastar.py <ultrastar-folder> --song-id <song_id>
python scripts/build_song.py data/songs/<song_id>
```

`build_song.py` runs NanoPitch, full STARS, and loudness on the reference vocal and updates `manifest.json` with cached paths.

### User performance

Per upload, the pipeline:

1. **NanoPitch** on the user vocal → `pitch.json`
2. **STARS** on the user vocal → `stars.json` (`--stars-profile full|fast`)
3. **Global offset** — coarse alignment between the user pitch track and chart voiced regions (`align_v2.estimate_global_offset_s`)
4. **Per-note measurements** — `NoteMeasurementV2` for every chart note
5. **Section trends**, **highlights**, and **overview** → `analysis.json`

```powershell
python scripts/analyze_performance.py data/songs/losing-my-religion path/to/user.wav --stars-profile fast
```

---

## Per-note analysis (`align_v2`)

For each UltraStar note, `vocal_coach/align_v2.py` produces a typed measurement:

| Field | Meaning |
| ----- | ------- |
| `median_cents` | Median cents vs chart MIDI in the core window |
| `pct_in_tune` | Fraction of voiced frames within ±50 cents (configurable) |
| `drift_cents_per_s` | Linear pitch drift across the core window |
| `arrival_offset_ms` | Voicing onset (or pitch-lock for legato) vs expected phone/chart onset |
| `core_start_s` / `core_end_s` | Note window trimmed for attack/release and shifted by detected arrival |

`NoteTechniqueComparison` rows align reference and user STARS techniques on the same note (matched / missed / user-added).

### Octave transposition

Singers often perform in a different register than the chart. `estimate_octave_shift_semitones` detects a single integer-octave (multiple of 12 semitones) offset from median user-vs-chart MIDI residuals, applies it to all targets before scoring, and octave-folds per-frame cents into `[-600, +600]` as a safety net. The detected shift is stored on `PerformanceAnalysis.octave_shift_semitones` and shown in the UI.

---

## Coaching moments (`highlights`)

Deterministic detectors scan note windows and song sections, then `select_highlights` ranks, caps, and dedupes results.

**Phrase-level** (rolling note windows):

| Detector | Signal |
| -------- | ------ |
| `best_pitch_phrase` / `pitch_struggle` | Mean `pct_in_tune` over wide configurable windows |
| `sharp_flat_note` | Single notes notably sharp or flat |
| `late_entrance` / `timing_consistency` | Arrival offset vs chart |
| `expressive_match` / `expressive_moment` / `missed_expression` | STARS technique alignment |
| `vocal_texture` | Per-technique callouts (breathy, vibrato, …) |
| `fade_within_notes` / `dynamic_drop` / `dynamic_surge` | Loudness vs reference |

**Section-level** (verses, choruses, bridges from the chart):

| Detector | Signal |
| -------- | ------ |
| `section_strength` / `section_weakness` | Best/worst section by pitch |
| `best_overall_section` / `weakest_overall_section` | Blended pitch + expression + timing |
| `section_delta` | Cross-section pitch/cents/technique contrasts |
| `section_dynamic_contrast` | Verse vs chorus volume contrast |

Technique callouts use friendly copy (`TECH_LABELS` / `TECH_HINTS` in `highlights.py`) so keys like `pharyngeal` read as “deep, resonant tone” in the UI.

Window sizes, caps, and qualifying floors are under `highlights:` in [`config/coaching.yaml`](config/coaching.yaml).

---

## Overview and section trends

`compute_section_trends` aggregates per-section pitch, timing, technique density, and dynamics. `compute_overview` distills the full take into a `PerformanceOverview`: overall `pct_in_tune`, median cents, voiced coverage, detected octave shift, expressive-technique density, strongest/weakest sections, and a **mimic score** (0–100) blending pitch accuracy, technique match rate, and arrival consistency.

The web UI renders overview stat tiles, a horizontal coaching-card row, section ribbons on the waveform, and filterable highlight categories.

---

## Web demo

`web/api/main.py` serves the static frontend and JSON API:

| Endpoint | Purpose |
| -------- | ------- |
| `GET /api/songs` | List available songs |
| `GET /api/songs/{id}/manifest` | Song metadata |
| `GET /api/songs/{id}/audio/instrumental` | Instrumental stream (karaoke mode) |
| `GET /api/songs/{id}/audio/reference` | Reference vocal stream |
| `POST /api/songs/{id}/analyze` | Upload audio; run full pipeline |
| `GET .../performances/{id}/analysis` | Cached `PerformanceAnalysis` |

The UI supports **upload** and **sing-along (karaoke)** modes, optional **fast feedback** (STARS student model), song badges, overview tiles, coaching cards, and waveform seek-from-card. See [Run locally](#run-locally) for install and `uvicorn` startup.

---

## Repo layout

```
SecondPass/
├── nanopitch/                          # vendored NanoPitch runtime (model.py + best.pth)
├── third_party/stars/                  # vendored STARS source (phone set + inference code)
├── data/
│   ├── songs/<song_id>/                # karaoke song bundles (audio + precomputed refs)
│   ├── student_v6/                     # distilled STARS student checkpoint (~7 MB)
│   └── rag/                            # ChromaDB index + playbooks + pedagogy sources
├── vocal_coach/
│   ├── schemas.py
│   ├── ultrastar.py                    # UltraStar .txt parser
│   ├── song.py                         # UltraStar -> song bundle + manifest
│   ├── pitch.py                        # NanoPitch wrapper
│   ├── stars_runner.py                 # STARS subprocess + profile dispatch
│   ├── student_runner.py               # in-process student model (fast profile)
│   ├── loudness.py                     # RMS / dBFS
│   ├── align_v2.py                     # dual-track per-note measurements
│   ├── trends.py                       # section-level aggregates
│   ├── overview.py                     # song-wide summary stats
│   ├── highlights.py                   # coaching-moment detectors
│   ├── llm.py                          # OpenAI client (graceful no-op without key)
│   └── coaching_config.py
├── web/
│   ├── api/main.py
│   └── static/{index.html,app.js,style.css}
├── scripts/
│   ├── setup.py                        # one-shot install + wiring
│   ├── doctor.py                       # preflight check
│   ├── import_ultrastar.py
│   ├── build_song.py
│   ├── analyze_performance.py
│   └── ...
├── config/coaching.yaml                # coaching thresholds
├── .env.example                        # copy to .env; add OPENAI_API_KEY (optional)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Run locally

Python **3.10+** is required. A CUDA GPU is recommended (the student model and NanoPitch run
on CPU too, just slower — expect 3–5× longer analysis on a full-length song).

No sibling repositories are needed. `nanopitch/` and `third_party/stars/` are vendored in the
repo. All bundled songs ship with precomputed `reference/pitch.json` and `reference/stars.json`
so the web demo works on first launch with no GPU preprocessing step.

### Option A — Docker (fastest)

```bash
# GPU (requires NVIDIA container toolkit on the host):
docker compose --profile gpu up

# CPU-only (slower, no special drivers):
docker compose --profile cpu up
```

Or run the pre-built image directly (no clone required):

```bash
docker run --gpus all -p 8000:8000 ghcr.io/dillonmchenry/secondpass:latest
# CPU:
docker run -p 8000:8000 -e SECONDPASS_DEVICE=cpu ghcr.io/dillonmchenry/secondpass:latest
```

Open **http://localhost:8000**.

### Option B — Source install

```bash
git clone https://github.com/dillonmchenry/TimestampVocalCoach.git
cd TimestampVocalCoach

# Create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Install everything (PyTorch + deps + STARS extras + NLTK corpora + phone-set wiring):
python scripts/setup.py

# Verify the environment (shows [OK] / [WARN] / [FAIL] per check):
python scripts/doctor.py

# Start the server:
python -m uvicorn web.api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**, pick a song, upload `yesterday_user_vocal.wav` (included at the
repo root), and click Analyze.

### LLM coaching (optional)

Copy `.env.example` to `.env` and add your OpenAI API key.  Without it the full measurement and
coaching pipeline still runs; only the LLM-written summaries and vocal-profile generation are
skipped.  `doctor.py` will show a `[WARN]` rather than a `[FAIL]` if the key is absent.

### Analyze from the CLI

```bash
python scripts/analyze_performance.py data/songs/losing-my-religion path/to/your_take.wav --stars-profile fast
```

`--stars-profile fast` uses the bundled student model (no extra downloads).
`--stars-profile full` uses the teacher STARS model (slower, highest fidelity) and requires
downloading the 700 MB bilingual checkpoint and `rmvpe/model.pt` from
[verstar/STARS on Hugging Face](https://huggingface.co/verstar/STARS) into the repo root.

### Add a new song

1. Obtain an UltraStar bundle (chart `.txt`, reference vocal, optional instrumental).
2. `python scripts/import_ultrastar.py <folder> --song-id <id>`
3. `python scripts/build_song.py data/songs/<id>` (GPU recommended; several minutes for STARS on a full track)
4. Restart or refresh the web app — the new song appears in the picker.

---
