#!/usr/bin/env python3
"""SecondPass environment preflight check.

Run this after setup.py to verify everything is in place before starting the server:

    python scripts/doctor.py

Prints one line per check with [OK], [WARN], or [FAIL] and a concrete fix for
each failure.  Exits with code 0 if there are no FAILs (WARNs are acceptable).
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PASS = "\033[92m[OK]  \033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def warn(msg: str, fix: str = "") -> None:
    warnings.append(msg)
    print(f"  {WARN} {msg}")
    if fix:
        print(f"         Fix: {fix}")


def fail(msg: str, fix: str = "") -> None:
    failures.append(msg)
    print(f"  {FAIL} {msg}")
    if fix:
        print(f"         Fix: {fix}")


# ---------------------------------------------------------------------------
# Python version
# ---------------------------------------------------------------------------
print("\n=== Python ===")
v = sys.version_info
if v >= (3, 10):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(
        f"Python {v.major}.{v.minor}.{v.micro} — 3.10+ required",
        "Install Python 3.10 or later from https://www.python.org/downloads/",
    )

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
print("\n=== PyTorch / CUDA ===")
try:
    import torch  # type: ignore
    ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        ok(f"CUDA available — {dev}")
    else:
        warn(
            "CUDA not available — analysis will run on CPU (slower)",
            "Install CUDA drivers and the CUDA build of PyTorch. "
            "See: https://pytorch.org/get-started/locally/",
        )
except ImportError:
    fail(
        "torch not installed",
        "Run: python scripts/setup.py",
    )

# ---------------------------------------------------------------------------
# Core packages
# ---------------------------------------------------------------------------
print("\n=== Core packages ===")
required_pkgs = [
    "librosa", "soundfile", "numpy", "pydantic", "yaml",
    "fastapi", "uvicorn", "multipart", "tqdm",
    "g2p_en", "einops", "parselmouth",
]
pkg_display = {
    "yaml": "pyyaml",
    "multipart": "python-multipart",
    "parselmouth": "praat-parselmouth",
}
for mod in required_pkgs:
    display = pkg_display.get(mod, mod)
    try:
        importlib.import_module(mod)
        ok(display)
    except ImportError:
        fail(
            f"{display} not installed",
            f"Run: pip install {display}",
        )

# Optional packages
optional_pkgs = [("openai", "openai"), ("chromadb", "chromadb"), ("dotenv", "python-dotenv")]
for mod, display in optional_pkgs:
    try:
        importlib.import_module(mod)
        ok(f"{display} (optional)")
    except ImportError:
        warn(
            f"{display} not installed — LLM coaching features unavailable",
            f"Run: pip install {display}",
        )

# ---------------------------------------------------------------------------
# ffmpeg (needed for MP3 reference vocals)
# ---------------------------------------------------------------------------
print("\n=== System tools ===")
if shutil.which("ffmpeg"):
    ok("ffmpeg on PATH")
else:
    fail(
        "ffmpeg not found on PATH",
        "Install ffmpeg: https://ffmpeg.org/download.html "
        "(Windows: winget install ffmpeg  |  Linux: apt install ffmpeg  |  macOS: brew install ffmpeg)",
    )

# ---------------------------------------------------------------------------
# NanoPitch
# ---------------------------------------------------------------------------
print("\n=== NanoPitch ===")
nanopitch_dir_env = os.environ.get("NANOPITCH_DIR")
nanopitch_dir = Path(nanopitch_dir_env) if nanopitch_dir_env else REPO_ROOT / "nanopitch"
checkpoint_rel = "training/runs/best_150+late_clean_112gru_model/checkpoints/best.pth"

model_py = nanopitch_dir / "training" / "model.py"
ckpt = nanopitch_dir / checkpoint_rel

if model_py.is_file():
    ok(f"model.py at {model_py.relative_to(REPO_ROOT)}")
else:
    fail(
        f"NanoPitch model.py not found at {model_py}",
        "Run: git checkout nanopitch/training/model.py\n"
        "         Or re-clone the repo (nanopitch/ is now tracked in git).",
    )

if ckpt.is_file():
    size_mb = ckpt.stat().st_size / 1_048_576
    ok(f"best.pth ({size_mb:.1f} MB) at {ckpt.relative_to(REPO_ROOT)}")
else:
    fail(
        f"NanoPitch checkpoint not found at {ckpt}",
        f"Run: git checkout {ckpt.relative_to(REPO_ROOT).as_posix()}",
    )

# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------
print("\n=== STARS student model ===")
student_dir = REPO_ROOT / "data" / "student_v6"
student_ckpt = student_dir / "best.pt"
phone_vocab = student_dir / "phone_vocab.json"

if student_ckpt.is_file():
    size_mb = student_ckpt.stat().st_size / 1_048_576
    ok(f"best.pt ({size_mb:.1f} MB)")
else:
    fail(
        f"Student checkpoint not found at {student_ckpt}",
        "Run: git checkout data/student_v6/",
    )

if phone_vocab.is_file():
    ok("phone_vocab.json")
else:
    fail(
        f"phone_vocab.json not found at {phone_vocab}",
        "Run: git checkout data/student_v6/phone_vocab.json",
    )

# ---------------------------------------------------------------------------
# STARS third-party source
# ---------------------------------------------------------------------------
print("\n=== STARS source (third_party/stars) ===")
stars_dir = REPO_ROOT / "third_party" / "stars"
stars_phset_src = stars_dir / "chinese_and_english_phone_set.json"
stars_phset_wired = stars_dir / "data" / "processed" / "bilingual" / "phone_set.json"

if stars_phset_src.is_file():
    ok("chinese_and_english_phone_set.json")
else:
    fail(
        "STARS phone set JSON not found",
        "Run: git checkout third_party/stars/",
    )

if stars_phset_wired.is_file():
    ok("phone_set.json wired for STARS loader")
else:
    fail(
        "STARS phone set not wired",
        "Run: python scripts/setup.py  (step 5 wires it)",
    )

# ---------------------------------------------------------------------------
# RAG store
# ---------------------------------------------------------------------------
print("\n=== RAG store ===")
rag_db = REPO_ROOT / "data" / "rag" / "chroma_db"
if rag_db.is_dir() and any(rag_db.iterdir()):
    ok(f"ChromaDB at {rag_db.relative_to(REPO_ROOT)}")
else:
    warn(
        "RAG ChromaDB not found — LLM coaching will use playbooks only",
        "Run: python scripts/build_rag.py  (requires OPENAI_API_KEY)",
    )

# ---------------------------------------------------------------------------
# Song bundles
# ---------------------------------------------------------------------------
print("\n=== Song bundles ===")
songs_dir = REPO_ROOT / "data" / "songs"
if songs_dir.is_dir():
    songs = [d for d in songs_dir.iterdir() if d.is_dir() and (d / "manifest.json").is_file()]
    snippet_count = sum(1 for s in songs if s.name.endswith("-snippet"))
    full_count = len(songs) - snippet_count
    ok(f"{full_count} full songs, {snippet_count} snippets in data/songs/")
else:
    fail(
        "data/songs/ not found",
        "Run: git checkout data/songs/",
    )

# ---------------------------------------------------------------------------
# OPENAI_API_KEY
# ---------------------------------------------------------------------------
print("\n=== Environment variables ===")
api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if api_key:
    ok("OPENAI_API_KEY is set")
else:
    # Try loading from .env
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val:
                    api_key = val
                    break
    if api_key:
        ok("OPENAI_API_KEY found in .env")
    else:
        warn(
            "OPENAI_API_KEY not set — LLM features skipped at runtime",
            "Copy .env.example to .env and add your key to enable LLM coaching.",
        )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
if failures:
    print(f"  {len(failures)} FAIL(s), {len(warnings)} WARN(s)")
    print("  Fix the failures above, then re-run this script.")
    print("=" * 60)
    sys.exit(1)
elif warnings:
    print(f"  0 FAILs, {len(warnings)} WARN(s) — good to go (see warnings above)")
    print("  Start the server:")
    print("    python -m uvicorn web.api.main:app --port 8000")
    print("=" * 60)
else:
    print("  All checks passed!")
    print("  Start the server:")
    print("    python -m uvicorn web.api.main:app --port 8000")
    print("=" * 60)
