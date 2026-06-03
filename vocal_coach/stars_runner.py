"""STARS subprocess wrapper.

The cloned STARS repo (``third_party/stars``) ships a CLI inference entrypoint
at ``inference/stars.py`` that expects:

  * an ``--ckpt`` (model weights)
  * a ``--config`` yaml
  * a ``--phset`` json
  * a ``--metadata`` json (list of items with ``item_name``, ``wav_fn``,
    ``word``, ``ph``, ``ph2words``)
  * an ``-o`` save dir

It writes ``output.json`` (per-item phonemes, words, notes, technique
sequences, global style) and per-item TextGrid + MIDI files into the save dir.

We shell out (rather than import) for two reasons:

1. STARS hard-codes a CUDA device with ``torch.device(f"cuda:{rank}")``;
   importing the module risks pulling its argparse, multiprocessing and CUDA
   initialization into our parent process.
2. The STARS package uses ``import data_gen.*`` and ``import tasks.*`` which
   only resolve when its repo dir is the working directory (``PYTHONPATH=.``).

Our wrapper builds the metadata, invokes STARS as a subprocess, then parses
``output.json`` into the typed ``StarsTrack`` schema.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from vocal_coach.schemas import (
    STARS_TECH_NAMES,
    StarsMetadataEntry,
    StarsNote,
    StarsPhoneme,
    StarsStyle,
    StarsTrack,
)


# --- Paths ----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STARS_DIR = REPO_ROOT / "third_party" / "stars"

DEFAULT_CKPT_RELPATH = "checkpoints/stars_bilingual/model_ckpt_steps_300000.ckpt"
DEFAULT_CONFIG_RELPATH = "configs/stars_bilingual.yaml"
DEFAULT_PHSET_RELPATH = "chinese_and_english_phone_set.json"

# STARS bilingual config: 24 kHz audio, 128-sample hop -> ~5.33 ms / frame.
STARS_BILINGUAL_SAMPLE_RATE = 24000
STARS_BILINGUAL_HOP = 128
STARS_BILINGUAL_HOP_SECONDS = STARS_BILINGUAL_HOP / STARS_BILINGUAL_SAMPLE_RATE


# --- Public API -----------------------------------------------------------


def write_stars_metadata(
    sample_dir: Path,
    entries: list[StarsMetadataEntry],
    *,
    filename: str = "stars_metadata.json",
) -> Path:
    """Write a STARS metadata.json into ``sample_dir`` and return its path."""
    out = Path(sample_dir) / filename
    out.write_text(
        json.dumps(
            [e.model_dump(exclude_none=True) for e in entries],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out


def run_stars_inference(
    metadata_path: Path,
    *,
    save_dir: Path,
    stars_dir: Path = DEFAULT_STARS_DIR,
    ckpt_relpath: str = DEFAULT_CKPT_RELPATH,
    config_relpath: str = DEFAULT_CONFIG_RELPATH,
    phset_relpath: str = DEFAULT_PHSET_RELPATH,
    cuda_visible_devices: str = "0",
    extra_args: Optional[list[str]] = None,
    python_executable: Optional[str] = None,
) -> Path:
    """Invoke ``inference/stars.py`` as a subprocess. Returns the path to output.json.

    The STARS CLI expects to be run with the cloned repo as CWD because its
    config files reference paths like ``checkpoints/...`` and ``data/...``
    relative to that directory.
    """
    stars_dir = Path(stars_dir).resolve()
    metadata_path = Path(metadata_path).resolve()
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not stars_dir.is_dir():
        raise FileNotFoundError(f"STARS directory not found: {stars_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    for rel in (ckpt_relpath, config_relpath, phset_relpath):
        if not (stars_dir / rel).is_file():
            raise FileNotFoundError(
                f"Required STARS file missing: {stars_dir / rel}\n"
                "Run `python scripts/setup_stars_runtime.py` first."
            )

    cmd = [
        python_executable or sys.executable,
        "inference/stars.py",
        "--ckpt", ckpt_relpath,
        "--config", config_relpath,
        "--phset", phset_relpath,
        "--metadata", str(metadata_path),
        "-o", str(save_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PYTHONPATH"] = str(stars_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"[stars] running: {' '.join(cmd)}")
    print(f"[stars] cwd    : {stars_dir}")
    print(f"[stars] CUDA   : {cuda_visible_devices}")

    proc = subprocess.run(
        cmd,
        cwd=str(stars_dir),
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"STARS inference failed with exit code {proc.returncode}. "
            f"Inspect the logs above and the contents of {save_dir} for partial outputs."
        )

    output_json = save_dir / "output.json"
    if not output_json.is_file():
        raise FileNotFoundError(
            f"STARS reported success but {output_json} is missing."
        )
    return output_json


def parse_stars_output(
    output_json: Path,
    *,
    sample_id: str,
    item_name: Optional[str] = None,
) -> StarsTrack:
    """Parse a STARS ``output.json`` (one item) into a typed ``StarsTrack``."""
    output_json = Path(output_json)
    data = json.loads(output_json.read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"STARS output {output_json} is empty.")
    if item_name is not None:
        matches = [item for item in data if item.get("item_name") == item_name]
        if not matches:
            raise ValueError(
                f"item_name='{item_name}' not found in {output_json}; "
                f"available: {[d.get('item_name') for d in data]}"
            )
        item = matches[0]
    else:
        item = data[0]

    ph_list: list[str] = item["ph_list"]
    word_list: list[str] = item["word_list"]
    ph_durs: list[float] = item.get("ph_durs", [])
    word_durs: list[float] = item.get("word_durs", [])
    note_list: list[int] = item.get("note_list", [])
    note_durs: list[float] = item.get("note_durs", [])
    style_dict: dict = item.get("style", {})

    # Build word-level intervals so we can assign each phoneme to its word
    # based on its mid-time. STARS emits one entry of word_durs per *word*
    # (NOT per phoneme), so we walk word_durs and ph_durs separately.
    word_starts: list[float] = []
    cur_w = 0.0
    for d in word_durs:
        word_starts.append(cur_w)
        cur_w += float(d)
    word_ends = word_starts[1:] + [cur_w]

    SILENCE_WORDS = {"<SP>", "<AP>"}

    def _word_index_for_time(t: float) -> int:
        """Return the index of the lyric word covering time t, or -1 if it
        falls inside a <SP>/<AP> span. Uses linear scan (word counts are tiny)."""
        for j, (ws, we) in enumerate(zip(word_starts, word_ends)):
            if ws <= t < we or (j == len(word_starts) - 1 and t == we):
                if j < len(word_list) and word_list[j] in SILENCE_WORDS:
                    return -1
                return j
        return -1

    phonemes: list[StarsPhoneme] = []
    cur = 0.0
    for i, ph in enumerate(ph_list):
        dur = float(ph_durs[i]) if i < len(ph_durs) else 0.0
        start = cur
        end = cur + dur
        cur = end
        mid = 0.5 * (start + end)
        w_idx = _word_index_for_time(mid)
        word = word_list[w_idx] if 0 <= w_idx < len(word_list) else (
            ph if ph in SILENCE_WORDS else ""
        )
        techniques: dict[str, int] = {}
        for tech_name in STARS_TECH_NAMES:
            seq = item.get(f"{tech_name}_tech", [])
            techniques[tech_name] = int(seq[i]) if i < len(seq) else 0
        phonemes.append(
            StarsPhoneme(
                index=i,
                phoneme=ph,
                word=word,
                word_index=w_idx,
                start_s=start,
                end_s=end,
                techniques=techniques,
            )
        )

    # Notes -- sequential durations -> absolute (start, end).
    notes: list[StarsNote] = []
    cur_note = 0.0
    for i, midi in enumerate(note_list):
        dur = float(note_durs[i]) if i < len(note_durs) else 0.0
        start = cur_note
        end = cur_note + dur
        cur_note = end
        notes.append(
            StarsNote(
                index=i,
                start_s=start,
                end_s=end,
                midi_pitch=int(midi),
            )
        )

    style = StarsStyle(
        language=style_dict.get("language", "unknown"),
        gender=style_dict.get("gender", "unknown"),
        emotion=style_dict.get("emotion", "unknown"),
        method=style_dict.get("method", "unknown"),
        pace=style_dict.get("pace", "unknown"),
        range=style_dict.get("range", "unknown"),
        technique_group=style_dict.get("technique_group", "unknown"),
    )

    return StarsTrack(
        sample_id=sample_id,
        sample_rate=STARS_BILINGUAL_SAMPLE_RATE,
        hop_seconds=STARS_BILINGUAL_HOP_SECONDS,
        style=style,
        phonemes=phonemes,
        notes=notes,
    )


def run_stars(
    *,
    metadata_path: Path,
    save_dir: Path,
    sample_id: str,
    item_name: Optional[str] = None,
    stars_dir: Path = DEFAULT_STARS_DIR,
    cuda_visible_devices: str = "0",
    extra_args: Optional[list[str]] = None,
    keep_save_dir: bool = True,
) -> StarsTrack:
    """End-to-end: invoke STARS and parse its output.json into a StarsTrack."""
    output_json = run_stars_inference(
        metadata_path,
        save_dir=save_dir,
        stars_dir=stars_dir,
        cuda_visible_devices=cuda_visible_devices,
        extra_args=extra_args,
    )
    track = parse_stars_output(output_json, sample_id=sample_id, item_name=item_name)
    if not keep_save_dir:
        shutil.rmtree(save_dir, ignore_errors=True)
    return track


# ---------------------------------------------------------------------------
# Sprint-3 profile dispatch (teacher vs distilled student)
# ---------------------------------------------------------------------------


STARS_PROFILE_FULL = "full"
STARS_PROFILE_FAST = "fast"
STARS_PROFILES = (STARS_PROFILE_FULL, STARS_PROFILE_FAST)


def run_stars_with_profile(
    *,
    profile: str,
    metadata_path: Path,
    save_dir: Path,
    sample_id: str,
    item_name: Optional[str] = None,
    stars_dir: Path = DEFAULT_STARS_DIR,
    cuda_visible_devices: str = "0",
    extra_args: Optional[list[str]] = None,
    keep_save_dir: bool = True,
    student_dir: Optional[Path] = None,
    student_device: Optional[str] = None,
    fallback_to_full: bool = True,
) -> StarsTrack:
    """Dispatch between the teacher subprocess and the distilled student.

    ``profile="full"`` (default) runs the original ``run_stars`` subprocess.
    ``profile="fast"`` loads ``stars_student/`` in-process and returns a
    ``StarsTrack`` matching the same schema. When ``fallback_to_full`` is True
    and the student checkpoint is missing we silently fall back to ``full``
    so the demo keeps working before the student has been trained.
    """
    profile = (profile or STARS_PROFILE_FULL).lower()
    if profile not in STARS_PROFILES:
        raise ValueError(
            f"Unknown stars_profile {profile!r}; expected one of {STARS_PROFILES}"
        )

    if profile == STARS_PROFILE_FAST:
        # Local import keeps the heavyweight student deps optional for callers
        # that always use the full profile.
        try:
            from vocal_coach.student_runner import (  # noqa: WPS433
                DEFAULT_STUDENT_DIR,
                run_student,
            )
        except Exception as exc:
            if not fallback_to_full:
                raise
            print(
                f"[stars] student_runner unavailable ({exc}); falling back to full STARS",
            )
            profile = STARS_PROFILE_FULL
        else:
            sdir = Path(student_dir) if student_dir is not None else DEFAULT_STUDENT_DIR
            try:
                return run_student(
                    metadata_path=metadata_path,
                    sample_id=sample_id,
                    item_name=item_name,
                    student_dir=sdir,
                    device=student_device or ("cuda" if cuda_visible_devices else "cpu"),
                )
            except FileNotFoundError as exc:
                if not fallback_to_full:
                    raise
                print(
                    f"[stars] student checkpoint missing ({exc}); "
                    "falling back to full STARS subprocess"
                )
                profile = STARS_PROFILE_FULL

    return run_stars(
        metadata_path=metadata_path,
        save_dir=save_dir,
        sample_id=sample_id,
        item_name=item_name,
        stars_dir=stars_dir,
        cuda_visible_devices=cuda_visible_devices,
        extra_args=extra_args,
        keep_save_dir=keep_save_dir,
    )


def write_stars_track(track: StarsTrack, out_path: Path) -> None:
    Path(out_path).write_text(track.model_dump_json(indent=2), encoding="utf-8")
