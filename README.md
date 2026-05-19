# Vocal Coach — Sprint 1

End-to-end Python pipeline that runs **NanoPitch** (continuous F0 / VAD) and
**STARS** (phoneme alignment + vocal-technique annotation) over a single
GTSinger sample, joining the outputs with a deterministic loudness track and
the song's reference annotation into one `timeline.json` per sample.

Sprint 1 is the *wiring sprint*: it produces the substrate that sprint 2 will
turn into per-note coaching cards via reference-window aggregation, and that
sprint 3 will use to drive the highlight engine and feedback templates.

## Repo layout

```
STARS/
├── rmvpe/                              # RMVPE pitch checkpoint (consumed by STARS)
├── stars_chinese/                      # STARS Chinese checkpoint (unused in sprint 1)
├── stars_chinese_english_bilingual/    # STARS bilingual checkpoint (used)
├── third_party/stars/                  # cloned gwx314/STARS (set up via setup_stars_runtime.py)
├── vocal_coach/
│   ├── schemas.py                      # Pydantic models for every JSON artifact
│   ├── reference.py                    # GTSinger sample -> reference_annotation.json
│   ├── pitch.py                        # NanoPitch wrapper -> pitch.json
│   ├── stars_runner.py                 # STARS subprocess wrapper -> stars.json
│   ├── loudness.py                     # RMS dBFS -> loudness.json
│   └── align.py                        # (stretch) aggregate one ReferenceNote into a NoteCard
├── scripts/
│   ├── download_gtsinger_sample.py     # one-segment HF download
│   ├── build_reference.py              # GTSinger -> reference_annotation.json + stars_metadata.json
│   ├── setup_stars_runtime.py          # hardlink RMVPE + STARS ckpts under third_party/stars
│   ├── validate_pitch.py               # NanoPitch parity + wav->mel sanity
│   ├── run_pipeline.py                 # end-to-end driver
│   └── demo_note_card.py               # stretch: print one NoteCard
├── notebooks/
│   └── sprint1_demo.ipynb              # piano-roll + F0 + STARS phoneme bands + cents track
├── data/samples/<sample_id>/           # gitignored
│   ├── 0000.wav                        # GTSinger source audio
│   ├── 0000.json                       # GTSinger raw annotation
│   ├── reference_annotation.json
│   ├── stars_metadata.json
│   ├── pitch.json
│   ├── stars.json
│   ├── loudness.json
│   ├── timeline.json
│   ├── stars_out/                      # STARS subprocess scratch (textgrid, midi, output.json)
│   └── note_card_*.json                # produced by demo_note_card.py
└── requirements.txt
```

## Setup

You'll need:

* **NanoPitch** cloned somewhere — by default we look at
  `C:/Users/dillo/Documents/GitHub/NanoPitch`. Override with the
  `NANOPITCH_DIR` env var or the `--nanopitch-dir` CLI flag.
* **Python 3.10+** with PyTorch and CUDA. STARS upstream targets 3.10+CUDA 12.4;
  it works on PyTorch 2.6 / Python 3.13 with the wheels listed below.
* The STARS bilingual checkpoint already lives in
  `stars_chinese_english_bilingual/` (LFS).

```powershell
# 1. Install Python deps. Both vocal_coach and STARS are installed into the
#    same env. `webrtcvad-wheels` is the prebuilt Windows replacement for
#    STARS's `webrtcvad` requirement (same import name).
python -m pip install -r requirements.txt
python -m pip install tensorboard mir_eval pyloudnorm scikit-image g2p_en `
                      einops praat-parselmouth torchmetrics pyworld webrtcvad-wheels

# 2. Clone STARS into third_party/ (one-time):
git clone https://github.com/gwx314/STARS.git third_party/stars

# 3. Stage RMVPE + STARS bilingual checkpoints + bilingual phone_set.json
#    where STARS's config expects them (one-time):
python scripts/setup_stars_runtime.py
```

## End-to-end demo run

```powershell
# 1. Download one English GTSinger segment (~1.3 MB wav + 30 KB annotations).
python scripts/download_gtsinger_sample.py
# Default segment: English/EN-Alto-1/Breathy/innocence/Control_Group/0000

# 2. Project GTSinger's annotation into our schema and STARS's metadata format.
python scripts/build_reference.py data/samples/EN-Alto-1__innocence__0000

# 3. Run the full pipeline (NanoPitch + STARS + loudness -> timeline.json).
python scripts/run_pipeline.py data/samples/EN-Alto-1__innocence__0000

# 4. (Stretch) Emit note-level coaching cards (first five notes).
python scripts/demo_note_card.py data/samples/EN-Alto-1__innocence__0000

# 5. Open the demo notebook and hit "Run All".
#    notebooks/sprint1_demo.ipynb
```

## Validation

`scripts/validate_pitch.py` runs two sanity checks before you trust the
pipeline on a new wav:

1. **Parity check.** Loads a clip from NanoPitch's pre-extracted
   `data/test.npz` and pushes it through both our `run_nanopitch_on_logmel`
   wrapper and a hand-rolled `model(...) + viterbi_decode_realtime` call.
   Both must produce bit-identical F0 + voicing.

2. **wav→mel sanity.** Runs the full wav→mel→model→Viterbi path on the
   GTSinger sample and asserts the log-mel array has reasonable mean/std,
   no NaN/Inf, and the median voiced F0 sits in `[80, 800] Hz`.

If you regress the wav→mel preprocessing (e.g. flip the librosa filterbank
back to the default `slaney` normalization), step 2 will pass but the VAD
head will collapse to ~0; if you regress the model loading, step 1 will fail
loudly with a non-zero `max abs diff`.

```powershell
python scripts/validate_pitch.py --device cuda
```

## Sprint-2 hooks

Sprint 1 deliberately stops short of any alignment math. The places that
expect new code in sprint 2 are:

* `vocal_coach/align.py` — currently aggregates *one* note via
  `aggregate_note(...)`. Sprint 2 turns this into a song-wide scan with the
  reference-window aggregation strategy, plus rolling-window/section/song
  passes.
* `vocal_coach/schemas.py::ReferenceSection` — sprint 1 emits a single
  "Full" section because GTSinger segments don't carry section labels.
  Sprint 2 will load real section annotations (manually authored or imported
  from MusicXML).
* `vocal_coach.stars_runner` — STARS's note transcription head
  (`StarsTrack.notes`) is captured but unused. Sprint 2 will use these for
  extra-note detection and karaoke-sync sanity checks.

## Out of scope for sprint 1

* Highlight detection engine (sprint 2).
* Feedback text templates / LLM polish (sprint 3).
* Real karaoke recordings (only GTSinger samples are wired up).
* Manual section / phrase annotation tooling.
