# third_party

## STARS (`stars/`)

This folder contains a **vendored copy** of [gwx314/STARS](https://github.com/gwx314/STARS) so
`vocal_coach.rmvpe_f0` can import `modules.pe.rmvpe`.

If you re-clone STARS locally, Git will **not** add the files to TimestampVocalCoach while
`stars/.git` exists (nested repository). Either:

1. Clone and remove the inner git metadata:
   ```powershell
   git clone https://github.com/gwx314/STARS.git stars
   Remove-Item -Recurse -Force stars\.git
   ```
2. Or download a source archive and extract into `stars/` without a `.git` directory.

**RMVPE weights** live at repo root `rmvpe/model.pt` (gitignored — too large for GitHub).
Download from [verstar/STARS on HuggingFace](https://huggingface.co/verstar/STARS) into `rmvpe/`,
then run `python scripts/setup_stars_runtime.py` if you need symlinks under `stars/checkpoints/`.

Teacher STARS `.ckpt` files stay in `stars_chinese_english_bilingual/` (also gitignored) for full
inference; the student demo only needs `rmvpe/model.pt` + `data/student_v6/` inside this repo.
