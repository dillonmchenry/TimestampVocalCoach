"""Analyze one user vocal recording against a built song.

Inputs:

    data/songs/<song_id>/             (built by import_ultrastar.py + build_song.py)
    <perf_wav>                         (a user vocal recording, .wav/.mp3/...)

Outputs (under ``data/songs/<song_id>/performances/<perf_id>/``):

    performance.<ext>          (copied or moved next to outputs)
    pitch.json
    stars_metadata.json
    stars.json                 (unless --skip-user-stars)
    analysis.json              (PerformanceAnalysis: per-note + highlights)

Usage::

    python scripts/analyze_performance.py data/songs/losing-my-religion \
        "data/songs/losing-my-religion/performances/take-1/performance.wav"

    python scripts/analyze_performance.py data/songs/losing-my-religion \
        "user.wav" --perf-id take-2 --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.align_v2 import measure_song  # noqa: E402
from vocal_coach.coaching_config import CoachingConfig, DEFAULT_CONFIG_RELPATH  # noqa: E402
from vocal_coach.highlights import select_highlights  # noqa: E402
from vocal_coach.loudness import compute_loudness, write_loudness_track  # noqa: E402
from vocal_coach.pitch import extract_f0, write_pitch_track  # noqa: E402
from vocal_coach.reference import load_reference  # noqa: E402
from vocal_coach.schemas import (  # noqa: E402
    PerformanceAnalysis,
    PitchTrack,
    StarsMetadataEntry,
    StarsTrack,
)
from vocal_coach.song import load_manifest  # noqa: E402
from vocal_coach.stars_runner import (  # noqa: E402
    DEFAULT_STARS_DIR,
    run_stars,
    write_stars_track,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("song_dir", type=Path)
    p.add_argument(
        "user_wav",
        type=Path,
        help="Path to the user vocal recording (wav/mp3/flac/ogg/m4a)",
    )
    p.add_argument(
        "--perf-id",
        default=None,
        help="Performance id (default: short uuid). Used as subdir name.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="NanoPitch torch device (default: cuda; CPU fallback automatic)",
    )
    p.add_argument(
        "--skip-user-stars",
        action="store_true",
        help="Skip user STARS (debug only; expressive highlights become limited)",
    )
    p.add_argument(
        "--skip-user-pitch",
        action="store_true",
        help="Reuse existing performance pitch.json instead of re-running NanoPitch",
    )
    p.add_argument(
        "--stars-dir",
        type=Path,
        default=DEFAULT_STARS_DIR,
        help="Path to the cloned STARS repo (default: third_party/stars)",
    )
    p.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="Value for CUDA_VISIBLE_DEVICES when invoking STARS",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a coaching.yaml override. Defaults to config/coaching.yaml.",
    )
    return p.parse_args()


def _resolve_torch_device(requested: str) -> str:
    if requested.lower() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return requested
    except ImportError:
        pass
    print("[analyze] CUDA unavailable; falling back to cpu for NanoPitch.")
    return "cpu"


def _ensure_perf_audio(user_wav: Path, perf_dir: Path) -> Path:
    """Copy ``user_wav`` into ``perf_dir`` if it isn't already there."""
    user_wav = user_wav.resolve()
    perf_dir.mkdir(parents=True, exist_ok=True)
    target = perf_dir / f"performance{user_wav.suffix.lower()}"
    if user_wav != target.resolve():
        shutil.copy2(user_wav, target)
    return target


def _write_user_stars_metadata(perf_dir: Path, song_dir: Path, user_audio: Path, song_id: str) -> Path:
    """Build performance-side stars_metadata.json that points at the user wav.

    Reuses the song's reference word/phone/ph2word lists since we want STARS
    to align the same lyrics on the user recording.
    """
    song_meta_path = song_dir / "stars_metadata.json"
    if not song_meta_path.is_file():
        raise FileNotFoundError(
            f"{song_meta_path} missing; run scripts/import_ultrastar.py first."
        )
    raw = json.loads(song_meta_path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"{song_meta_path} is empty")
    entry = raw[0]
    user_entry = StarsMetadataEntry(
        item_name=f"{song_id}__perf",
        wav_fn=str(user_audio.resolve()).replace("\\", "/"),
        word=entry["word"],
        ph=entry["ph"],
        ph2words=entry["ph2words"],
        ph_durs=entry.get("ph_durs"),
        word_durs=entry.get("word_durs"),
    )
    out = perf_dir / "stars_metadata.json"
    out.write_text(
        json.dumps([user_entry.model_dump(exclude_none=True)], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def _maybe_load(path: Path, model):
    if path.is_file():
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    return None


def main() -> int:
    args = parse_args()
    song_dir = args.song_dir.resolve()
    if not (song_dir / "manifest.json").is_file():
        print(
            f"ERROR: {song_dir / 'manifest.json'} missing. Run import_ultrastar.py first.",
            file=sys.stderr,
        )
        return 2
    if not args.user_wav.is_file():
        print(f"ERROR: {args.user_wav} not found", file=sys.stderr)
        return 2

    manifest = load_manifest(song_dir)
    reference = load_reference(song_dir)
    perf_id = args.perf_id or uuid.uuid4().hex[:8]
    perf_dir = song_dir / "performances" / perf_id

    user_audio = _ensure_perf_audio(args.user_wav, perf_dir)
    print(f"[analyze] song_id     : {manifest.song_id}")
    print(f"[analyze] perf_id     : {perf_id}")
    print(f"[analyze] user vocal  : {user_audio}")

    # 1. NanoPitch on user vocal
    pitch_user_path = perf_dir / "pitch.json"
    if args.skip_user_pitch and pitch_user_path.is_file():
        print(f"[analyze] pitch       : reusing {pitch_user_path}")
        pitch_user = PitchTrack.model_validate_json(
            pitch_user_path.read_text(encoding="utf-8")
        )
    else:
        device = _resolve_torch_device(args.device)
        print(f"[analyze] pitch       : running NanoPitch on {device}")
        pitch_user = extract_f0(user_audio, device=device)
        pitch_user.sample_id = f"{manifest.song_id}__{perf_id}"
        write_pitch_track(pitch_user, pitch_user_path)
        print(
            f"[analyze] pitch       : wrote {pitch_user_path} "
            f"({len(pitch_user.frames)} frames)"
        )

    # 2. STARS on user vocal
    stars_user: StarsTrack | None = None
    stars_user_path = perf_dir / "stars.json"
    if args.skip_user_stars:
        if stars_user_path.is_file():
            stars_user = StarsTrack.model_validate_json(
                stars_user_path.read_text(encoding="utf-8")
            )
            print(f"[analyze] stars       : reusing {stars_user_path}")
        else:
            print("[analyze] stars       : skipped (no existing user stars.json)")
    else:
        meta_path = _write_user_stars_metadata(perf_dir, song_dir, user_audio, manifest.song_id)
        save_dir = perf_dir / "stars_out"
        print(f"[analyze] stars       : running STARS subprocess -> {save_dir}")
        stars_user = run_stars(
            metadata_path=meta_path,
            save_dir=save_dir,
            sample_id=f"{manifest.song_id}__{perf_id}",
            stars_dir=args.stars_dir,
            cuda_visible_devices=args.cuda_visible_devices,
        )
        write_stars_track(stars_user, stars_user_path)
        print(
            f"[analyze] stars       : wrote {stars_user_path} "
            f"({len(stars_user.phonemes)} phonemes, {len(stars_user.notes)} notes)"
        )

    # 3. Load reference STARS (built by build_song.py)
    stars_ref_path = song_dir / (manifest.reference_stars_path or "reference/stars.json")
    stars_ref = _maybe_load(stars_ref_path, StarsTrack)
    if stars_ref is None:
        print(f"[analyze] ref stars   : not found at {stars_ref_path}; expressive highlights will be limited")

    # 4. Reference pitch — drives the octave-shift detector when present.
    pitch_ref_path = song_dir / (manifest.reference_pitch_path or "reference/pitch.json")
    pitch_ref = _maybe_load(pitch_ref_path, PitchTrack)

    # 5. Run align_v2 + highlights
    config_path = args.config or (ROOT / DEFAULT_CONFIG_RELPATH)
    coaching_cfg = CoachingConfig.load(config_path)
    print(f"[analyze] config      : {config_path}")

    print("[analyze] aligning    : measure_song + select_highlights")
    notes, techniques, offset, octave_shift = measure_song(
        reference,
        pitch_user=pitch_user,
        pitch_ref=pitch_ref,
        stars_ref=stars_ref,
        stars_user=stars_user,
        config=coaching_cfg,
    )
    highlights = select_highlights(reference, notes, techniques, config=coaching_cfg)

    analysis = PerformanceAnalysis(
        song_id=manifest.song_id,
        perf_id=perf_id,
        reference_sample_id=reference.sample_id,
        duration_s=manifest.duration_s,
        global_offset_s=offset,
        octave_shift_semitones=octave_shift,
        pitch_user_path=str(pitch_user_path.relative_to(song_dir)).replace("\\", "/"),
        pitch_ref_path=(
            str(pitch_ref_path.relative_to(song_dir)).replace("\\", "/")
            if pitch_ref_path.is_file()
            else None
        ),
        stars_user_path=(
            str(stars_user_path.relative_to(song_dir)).replace("\\", "/")
            if stars_user_path.is_file()
            else None
        ),
        stars_ref_path=(
            str(stars_ref_path.relative_to(song_dir)).replace("\\", "/")
            if stars_ref_path.is_file()
            else None
        ),
        notes=notes,
        techniques=techniques,
        highlights=highlights,
    )

    # 6. Optional: per-frame loudness on the user vocal (used by the UI)
    loud_path = perf_dir / "loudness.json"
    try:
        loudness = compute_loudness(user_audio, sample_id=f"{manifest.song_id}__{perf_id}")
        write_loudness_track(loudness, loud_path)
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f"[analyze] loudness    : failed ({exc})")

    analysis_path = perf_dir / "analysis.json"
    analysis_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    print(f"[analyze] analysis    : wrote {analysis_path}")
    print(f"[analyze] global_offset_s     = {offset:+.3f}")
    print(f"[analyze] octave_shift_semis  = {octave_shift:+d}")
    print(f"[analyze] highlights ({len(highlights.moments)}):")
    for m in highlights.moments:
        print(
            f"  - [{m.type:>20}] {m.start_s:6.2f}s-{m.end_s:6.2f}s  "
            f"score={m.score:.2f}  {m.title}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
