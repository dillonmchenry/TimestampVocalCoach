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
| **NanoPitch** | Continuous F0 and voicing every 10 ms ([separate repo](https://github.com/smuleinc/NanoPitch)) |
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
├── rmvpe/                              # RMVPE weights (gitignored; setup links into STARS)
├── stars_chinese_english_bilingual/    # STARS bilingual ckpt (gitignored)
├── stars_student/                      # distilled student ckpt (gitignored; optional)
├── third_party/stars/                  # clone gwx314/STARS + setup_stars_runtime.py
├── config/coaching.yaml                # coaching thresholds
├── config/local_demo.yaml.example      # optional demo replay config
├── vocal_coach/
│   ├── schemas.py
│   ├── reference.py                    # GTSinger -> ReferenceAnnotation (dev samples)
│   ├── ultrastar.py                    # UltraStar .txt parser
│   ├── song.py                         # UltraStar -> song bundle + manifest
│   ├── pitch.py                        # NanoPitch wrapper
│   ├── stars_runner.py                 # STARS subprocess + profile dispatch
│   ├── student_runner.py               # in-process student model
│   ├── loudness.py                     # RMS / dBFS
│   ├── align.py                        # legacy single-note alignment (GTSinger)
│   ├── align_v2.py                     # dual-track per-note measurements
│   ├── trends.py                       # section-level aggregates
│   ├── overview.py                     # song-wide summary stats
│   ├── highlights.py                   # coaching-moment detectors
│   └── coaching_config.py
├── web/
│   ├── api/main.py
│   └── static/{index.html,app.js,style.css}
├── scripts/
│   ├── import_ultrastar.py
│   ├── build_song.py
│   ├── analyze_performance.py
│   ├── setup_stars_runtime.py
│   ├── build_reference.py              # GTSinger sample reference (dev)
│   ├── run_pipeline.py
│   └── ...
├── notebooks/
│   ├── sprint1_demo.ipynb
│   └── sprint2_dual_track.ipynb
├── docs/sprint2_stars_lite.md          # STARS student distillation notes
├── data/
│   ├── samples/<sample_id>/            # GTSinger dev fixtures (gitignored)
│   └── songs/<song_id>/                # karaoke song bundles
└── requirements.txt
```

---

## Run locally

You need this repo plus two sibling projects and their checkpoints. Python **3.10+** and a **CUDA** GPU are strongly recommended for STARS and NanoPitch (CPU works but is slow).

### External repos

| Repo | Clone into | Used for |
| ---- | ---------- | -------- |
| [smuleinc/NanoPitch](https://github.com/smuleinc/NanoPitch) | Sibling of this repo, e.g. `../NanoPitch` | F0 + voicing (`pitch.json`) |
| [gwx314/STARS](https://github.com/gwx314/STARS) | `third_party/stars` inside this repo | Phoneme timings + technique flags (`stars.json`) |

SecondPass does **not** vendor NanoPitch; it imports `training/model.py` from your clone. STARS is cloned under `third_party/stars` and wired to local weight files by `scripts/setup_stars_runtime.py`.

### Model weights (gitignored)

Download from [verstar/STARS on Hugging Face](https://huggingface.co/verstar/STARS) and place:

```
SecondPass/
  rmvpe/model.pt
  stars_chinese_english_bilingual/model_ckpt_steps_300000.ckpt
```

NanoPitch needs its training checkpoint inside the NanoPitch clone (default path used by this repo):

```
NanoPitch/
  training/runs/best_150+late_clean_112gru_model/checkpoints/best.pth
```

Optional — **fast feedback** in the UI uses a distilled student under `stars_student/` (see [`docs/sprint2_stars_lite.md`](docs/sprint2_stars_lite.md)). If missing, the app falls back to full STARS.

Bundled songs under `data/songs/` already include precomputed `reference/pitch.json` and `reference/stars.json`, so you can run the web demo without re-running `build_song.py` on first launch.

### Setup commands

From a shell in the SecondPass repo root:

```powershell
# 1. Clone dependencies (adjust paths if you keep repos elsewhere)
git clone https://github.com/smuleinc/NanoPitch.git ../NanoPitch
git clone https://github.com/gwx314/STARS.git third_party/stars

# 2. Point SecondPass at your NanoPitch clone (skip if it lives at ../NanoPitch)
$env:NANOPITCH_DIR = "C:\path\to\NanoPitch"

# 3. Install Python deps
python -m pip install -r requirements.txt
python -m pip install tensorboard mir_eval pyloudnorm scikit-image g2p_en `
                      einops praat-parselmouth torchmetrics pyworld webrtcvad-wheels

# 4. After weights are in rmvpe/ and stars_chinese_english_bilingual/, link them for STARS
python scripts/setup_stars_runtime.py

# 5. Start the demo (bundled songs are ready to analyze)
python -m uvicorn web.api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**, pick a song (yesterday_user_vocal.wav in top  of repo), upload a vocal or use sing-along mode, and analyze.

For a GPU-free demo that replays a cached analysis, copy `config/local_demo.yaml.example` to `config/local_demo.yaml` and tune `source_song_id` / `source_perf_id`.

### Analyze from the CLI

```powershell
python scripts/analyze_performance.py data/songs/losing-my-religion path\to\your_take.wav --stars-profile fast
```

Use `--stars-profile full` for teacher STARS (slower, highest fidelity).

### Add a new song

1. Obtain an UltraStar bundle (chart `.txt`, reference vocal, optional instrumental).
2. `python scripts/import_ultrastar.py <folder> --song-id <id>`
3. `python scripts/build_song.py data/songs/<id>` (GPU; several minutes for STARS on a full track)
4. Restart or refresh the web app — the new song appears in the picker.

---
