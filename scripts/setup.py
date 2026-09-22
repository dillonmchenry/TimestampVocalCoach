#!/usr/bin/env python3
"""SecondPass one-shot setup script.

Run this once after cloning (inside your venv):

    python scripts/setup.py

What it does
------------
1. Installs PyTorch with the right CUDA extras (or CPU if no GPU detected).
2. Installs requirements.txt.
3. Installs third_party/stars/requirements.txt (pyworld, webrtcvad, etc.).
   On Windows, substitutes webrtcvad-wheels for webrtcvad (pre-built C ext).
4. Pre-downloads the NLTK corpora that g2p_en needs.
5. Wires the STARS phone set into the path the STARS loader expects.
6. Prints instructions for running doctor.py to verify everything.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], **kwargs) -> int:
    print(f"\n[setup] {' '.join(str(a) for a in args)}")
    result = subprocess.run(args, **kwargs)
    return result.returncode


def pip(*args: str) -> int:
    return run([sys.executable, "-m", "pip", "install", *args])


def main() -> None:
    is_windows = platform.system() == "Windows"

    # ------------------------------------------------------------------
    # 1. PyTorch — install with CUDA if a CUDA-capable GPU is detectable
    # ------------------------------------------------------------------
    # We probe for nvcc or nvidia-smi rather than importing torch (which
    # may not be installed yet) to decide the index URL.
    has_cuda = (
        shutil.which("nvidia-smi") is not None
        or shutil.which("nvcc") is not None
        or os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
    )

    torch_version = "2.6.0"
    torchaudio_version = "2.6.0"

    if has_cuda:
        print(f"\n[setup] CUDA-capable GPU detected — installing PyTorch {torch_version}+cu124")
        code = pip(
            f"torch=={torch_version}",
            f"torchaudio=={torchaudio_version}",
            "--index-url", "https://download.pytorch.org/whl/cu124",
        )
    else:
        print(f"\n[setup] No GPU detected — installing CPU-only PyTorch {torch_version}")
        code = pip(
            f"torch=={torch_version}",
            f"torchaudio=={torchaudio_version}",
        )

    if code != 0:
        print(
            "\n[setup] ERROR: PyTorch install failed.\n"
            "  Visit https://pytorch.org/get-started/locally/ to pick the right command.\n"
            "  Then re-run this script."
        )
        sys.exit(code)

    # ------------------------------------------------------------------
    # 2. Main requirements
    # ------------------------------------------------------------------
    req = REPO_ROOT / "requirements.txt"
    # torch/torchaudio already installed — skip the version requirement
    # to avoid a redundant re-download.
    code = pip("--no-deps" if False else "-r", str(req))
    if code != 0:
        sys.exit(code)

    # ------------------------------------------------------------------
    # 3. STARS extras (third_party/stars/requirements.txt)
    # ------------------------------------------------------------------
    stars_req = REPO_ROOT / "third_party" / "stars" / "requirements.txt"
    if not stars_req.is_file():
        print(f"\n[setup] WARN: {stars_req} not found — skipping STARS extras")
    else:
        # Read the file and swap webrtcvad for webrtcvad-wheels on Windows
        lines = stars_req.read_text(encoding="utf-8").splitlines()
        pkgs: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_windows and line.lower().startswith("webrtcvad"):
                line = "webrtcvad-wheels"
            pkgs.append(line)

        code = pip(*pkgs)
        if code != 0:
            sys.exit(code)

    # ------------------------------------------------------------------
    # 4. NLTK corpora for g2p_en
    # ------------------------------------------------------------------
    print("\n[setup] Downloading NLTK corpora for g2p_en ...")
    nltk_code = subprocess.run(
        [
            sys.executable, "-c",
            "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); "
            "nltk.download('cmudict', quiet=True); "
            "print('[setup] NLTK corpora OK')",
        ]
    ).returncode
    if nltk_code != 0:
        print("[setup] WARN: NLTK download failed — g2p_en may not work correctly.")

    # ------------------------------------------------------------------
    # 5. Wire STARS phone set
    # ------------------------------------------------------------------
    stars_dir = REPO_ROOT / "third_party" / "stars"
    src = stars_dir / "chinese_and_english_phone_set.json"
    dest_dir = stars_dir / "data" / "processed" / "bilingual"
    dest = dest_dir / "phone_set.json"

    if src.is_file():
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            import shutil as _shutil
            _shutil.copy2(src, dest)
            print(f"[setup] Wired phone set: {dest}")
        else:
            print(f"[setup] Phone set already wired: {dest}")
    else:
        print(f"[setup] WARN: {src} not found — STARS phone set not wired.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Verify your environment:")
    print("    python scripts/doctor.py")
    print("  Start the server:")
    print("    python -m uvicorn web.api.main:app --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
