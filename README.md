# Timestamp Vocal Coach

An end-to-end **timestamped vocal coach** prototype for karaoke-style performances. The system separates **ML measurement** (pitch, phoneme techniques, loudness) from **deterministic coaching logic** (note-level comparisons, highlight selection, feedback templates).

This repository is the Sprint 1 wiring checkpoint: one annotated GTSinger clip flows through NanoPitch and STARS, joins into a shared timeline, and can be summarized as **note-level coaching cards** that prove the data model works.

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

## Sprint 2: Goals

### Full annotated karaoke song (not GTSinger)

- Build **reference material** from a real track: separated **reference vocal**, **lyrics**, **karaoke MIDI** (note on/off + pitch), optional section map.
- Bootstrap phone alignment with **STARS** (lyrics + G2P → `word` / `ph` / `ph2words`; STARS predicts timings on the reference vocal) instead of hand-labeling every phoneme like GTSinger.
- Keep **chart MIDI** as the source of truth for **note targets and note windows**; use STARS for **syllable timing** and techniques.

### Dual-track pitch comparison

- Run **NanoPitch on reference vocal and user karaoke vocal**.
- Per note: compare continuous F0 to **target MIDI** from the chart; add `**pct_in_tune`** and **median cents** on **time-aligned core windows** (trim attacks / shift by detected arrival).

### Note alignment from phoneme timings

- **Expected lyric onset** per note = first STARS (or MFA) phone overlapping that note window.
- Map STARS phones → notes by **interval overlap** with reference note windows; use phones for tags and timing anchors.

### Explore distilled / lighter STARS and other heads

- Evaluate a **smaller or distilled STARS** variant for faster iteration.
- Use **global style head** (emotion, pace, technique group) for section-level coaching (“you sounded happiest in the chorus”).
- Revisit STARS **note transcription** only for sanity checks (extra/missed notes vs MIDI), not as the main coaching grid.

### Deterministic highlight engine

- Scan **all notes**, then **rolling windows** (4–8 notes), **sections**, and **whole song**.
- Rank by severity + musical salience; emit **coaching moments** (timestamped cards / snippets), not only a single demo note.
- Config-driven thresholds (move magic numbers out of `align.py`).

---

## Repo layout

```
TimestampVocalCoach/
├── rmvpe/                              # RMVPE weights (gitignored; setup links into STARS)
├── stars_chinese_english_bilingual/    # STARS bilingual ckpt (gitignored)
├── third_party/stars/                  # clone gwx314/STARS + setup_stars_runtime.py
├── vocal_coach/
│   ├── schemas.py
│   ├── reference.py
│   ├── pitch.py
│   ├── stars_runner.py
│   ├── loudness.py
│   └── align.py
├── scripts/
│   ├── download_gtsinger_sample.py
│   ├── build_reference.py
│   ├── setup_stars_runtime.py
│   ├── validate_pitch.py
│   ├── run_pipeline.py
│   └── demo_note_card.py
├── notebooks/sprint1_demo.ipynb
├── data/samples/<sample_id>/           # gitignored, regenerable
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

