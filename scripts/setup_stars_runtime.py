"""One-shot setup so ``third_party/stars`` can find its checkpoints and phone-set.

The cloned STARS repo expects the following relative to its own working dir:

    checkpoints/rmvpe/model.pt
    checkpoints/stars_bilingual/model_ckpt_steps_300000.ckpt
    data/processed/bilingual/phone_set.json

We already have those artifacts elsewhere in this repo (the original HF clone
of ``verstar/STARS`` plus the bundled ``chinese_and_english_phone_set.json``).
This script wires them in by creating hard links (so we don't double the
~700 MB of model weights) and copying the small JSON.

Re-running the script is safe; it only creates files that don't already exist.

Usage::

    python scripts/setup_stars_runtime.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARS_DIR = ROOT / "third_party" / "stars"

LINKS = [
    # (source under ROOT, destination under STARS_DIR)
    ("rmvpe/model.pt", "checkpoints/rmvpe/model.pt"),
    (
        "stars_chinese_english_bilingual/model_ckpt_steps_300000.ckpt",
        "checkpoints/stars_bilingual/model_ckpt_steps_300000.ckpt",
    ),
]

# phone_set.json comes straight from the cloned STARS repo, just at a different
# relative path than the inference dataset class wants.
COPIES = [
    ("third_party/stars/chinese_and_english_phone_set.json",
     "third_party/stars/data/processed/bilingual/phone_set.json"),
]


def stage_link(src: Path, dst: Path) -> str:
    if dst.exists():
        return "skip (exists)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        # Cross-volume or permission failure -> fall back to copy.
        shutil.copy2(src, dst)
        return "copy"


def stage_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "skip (exists)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    if not STARS_DIR.is_dir():
        print(f"ERROR: third_party/stars not found ({STARS_DIR}).", file=sys.stderr)
        print("Run: git clone https://github.com/gwx314/STARS.git third_party/stars",
              file=sys.stderr)
        return 1

    failures = 0
    print("Staging STARS runtime files:")
    for src_rel, dst_rel in LINKS:
        src = ROOT / src_rel
        dst = STARS_DIR / dst_rel
        if not src.is_file():
            print(f"  [fail] missing source {src}")
            failures += 1
            continue
        action = stage_link(src, dst)
        print(f"  {action:9s} {dst}")

    for src_rel, dst_rel in COPIES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if not src.is_file():
            print(f"  [fail] missing source {src}")
            failures += 1
            continue
        action = stage_copy(src, dst)
        print(f"  {action:9s} {dst}")

    if failures:
        print(f"\n{failures} step(s) failed. STARS will not be runnable until fixed.")
        return 2
    print("\nSTARS runtime ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
