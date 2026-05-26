# Sprint 2 — STARS lite spike

The Sprint 2 demo path runs STARS on **both** the reference vocal (once per
song, at build time) and the user vocal (once per upload). The bilingual
checkpoint we ship (`stars_bilingual/model_ckpt_steps_300000.ckpt`, ~700 MB
together with the RMVPE pitch encoder) is fine for offline batch runs but
slow enough on first invocation that an "explore lighter STARS" parallel
spike is worth doing while the rest of the demo gets polished.

This doc summarizes (a) the configs that already exist in `third_party/stars`,
(b) what's *not* in the repo (no distilled checkpoints), and (c) a
recommendation table for Sprint 2 + future-sprint work.

## What ships in `third_party/stars`

| Config                                     | Phone set        | Languages         | Checkpoint                                                       | Notes |
|--------------------------------------------|------------------|-------------------|------------------------------------------------------------------|-------|
| `configs/stars_bilingual.yaml`             | bilingual (ZH+EN) | Chinese + English | `checkpoints/stars_bilingual/model_ckpt_steps_300000.ckpt`       | Default. Used by Sprint 1 + Sprint 2. |
| `configs/stars_chinese.yaml`               | Chinese          | Chinese           | `checkpoints/stars_chinese/model_ckpt_steps_200000.ckpt`         | Mandarin-only. Smaller phone vocabulary; otherwise the same architecture. |
| `configs/base.yaml`                        | shared base       | n/a              | n/a                                                               | Common defaults (sample rate, hop size, model size). |

Architecture (from `configs/base.yaml`):

- 24 kHz audio, 128-sample hop (~5.33 ms per frame)
- Conformer backbone, hidden size **256**, kernel **9**
- 2 conformer layers, U-Net `2-2-2-2` updown rates, `1-1-1-1` channel multiples
- RMVPE for F0 conditioning (`checkpoints/rmvpe/model.pt`, ~190 MB)

There is **no** distilled / quantized / pruned checkpoint in-repo today, and
no ONNX export script. So "lite STARS" in Sprint 2 is a question of
configuration knobs, not a different model family.

## What's measurable in Sprint 2

The `vocal_coach.stars_runner` wrapper makes it cheap to swap configs and
checkpoints. To benchmark, run something like:

```powershell
python scripts/build_song.py data/songs/losing-my-religion --skip-loudness --skip-pitch
```

with the default bilingual config, and again after editing `stars_runner.py`'s
`DEFAULT_CKPT_RELPATH` / `DEFAULT_CONFIG_RELPATH` to point at the Chinese
config + checkpoint. Capture:

- **Cold-start latency** (first run, weights load + cuda warmup)
- **Steady-state latency** (second run with cached weights)
- **VRAM peak** (`nvidia-smi` while running)
- **Phoneme accuracy on English lyrics** (manually spot-check 5 phrases)

We have not finished collecting these numbers yet because both checkpoints
have been excluded from the working tree (see the project root `git status`):

```
D rmvpe/model.pt
D stars_chinese/model_ckpt_steps_200000.ckpt
D stars_chinese_english_bilingual/model_ckpt_steps_300000.ckpt
```

The benchmark should be re-run once the checkpoints are restored from
HuggingFace `verstar/STARS`.

## Recommendations table

| Path | Effort | Risk | Expected upside | Recommendation |
|------|--------|------|-----------------|----------------|
| Bilingual @ FP16 inference | tiny | low (some accuracy delta on edge cases) | ~2x faster, ~50% VRAM | Try first — likely free win for the demo. |
| Bilingual w/ shorter `--max_tokens` per chunk | tiny | low (segmentation overhead) | Smoother memory profile on long songs | Enable for >2-min songs. Already supported via `extra_args` in `run_stars`. |
| Chinese-only checkpoint on English | small | medium-high (English phone-set mismatches lyrics) | ~30% smaller model, faster | Only for non-English songs. **Not recommended for the REM demo.** |
| ONNX export of the Conformer encoder | medium | medium (export complexity) | CPU-friendly inference, removes CUDA dep | Pursue in Sprint 3 if WASM/CPU deployment becomes a goal. |
| Phone-timing-only head (drop technique heads) | medium | medium-high (loses the technique-driven highlights) | ~2x faster | **Avoid** — STARS technique flags are central to the project's value. |
| Knowledge distillation to a 2-layer student | large | high (training infra required) | 5–10x faster, smaller, but quality loss | Defer to a dedicated future sprint. |

## Sprint 2 deliverables

1. `vocal_coach.stars_runner` exposes `extra_args`, so we can plumb any STARS
   CLI flag from `analyze_performance.py` and `build_song.py` without code
   changes.
2. The reference STARS run is **already** precomputed once per song
   (`build_song.py` -> `data/songs/<id>/reference/stars.json`). Only the
   user vocal pays the STARS cost per upload, which keeps demo turnaround
   reasonable on a single-GPU dev box.
3. The fallback path in `web/api/main.py` swallows STARS failures and writes
   `stars_error.txt` so the demo still produces pitch-only highlights when
   the GPU is unavailable.

Future-sprint follow-up:

- Add an `--stars-profile fast|full` flag to `analyze_performance.py` once
  numbers from the Chinese-only spike are in.
- Instrument `stars_runner.run_stars` with a wall-clock log line so we can
  drop a small benchmark script under `scripts/bench_stars.py`.
