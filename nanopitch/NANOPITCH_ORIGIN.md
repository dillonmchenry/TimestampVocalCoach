# NanoPitch — vendored runtime

This directory contains two files from the [NanoPitch](https://github.com/smulelabs/NanoPitch)
project published by Smule Labs, vendored here so SecondPass can be cloned and run without a
sibling repository.

| File | Source path in upstream repo |
|------|------------------------------|
| `training/model.py` | `training/model.py` |
| `training/runs/best_150+late_clean_112gru_model/checkpoints/best.pth` | excluded from upstream by `.gitignore`; this copy matches the checkpoint submitted to the NanoPitch leaderboard as `submissions/dillon-best/weights.pth` (SHA256 `DF02C1C99FF5BF923FFCBFBDED6029BB3ABB420CB228AEDEAFDAFE3BAAF7DA7E`) |

**Upstream repo:** https://github.com/smulelabs/NanoPitch  
**Upstream commit (main) at time of vendoring:** `f01b2115b0ebe1370b7dd4f2138d00adbe902643`  
**License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
See [`LICENSE`](https://github.com/smulelabs/NanoPitch/blob/main/LICENSE) in the upstream repo.

No changes were made to `model.py`.

To update to a newer NanoPitch release, replace `model.py` and `best.pth` and update the
commit SHA above.  Alternatively, point SecondPass at a full NanoPitch clone by setting
the `NANOPITCH_DIR` environment variable.
