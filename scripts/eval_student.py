"""Evaluate the distilled student against teacher labels on held-out clips.

Outputs:
  - per-technique F1 and accuracy vs the teacher's flags
  - mean phoneme-boundary error (frames) on a small subset
  - wall-clock speedup over the teacher subprocess on the same clips
  - PASS/FAIL against the README's acceptance band:
        speedup >= 5x          (interactive target)
        avg tech F1 >= 0.65    (per the plan's acceptance band; tune in YAML)

Usage::

    python scripts/eval_student.py
    python scripts/eval_student.py --corpus data/student_corpus \\
        --student-dir stars_student \\
        --max-clips 20

The script exits with code 78 (EX_CONFIG) if either the corpus or the student
checkpoint is missing (see Sprint 3 plan's 'Data checkpoints — user action
required' section).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vocal_coach.schemas import (  # noqa: E402
    STARS_TECH_NAMES,
    StarsMetadataEntry,
    StarsPhoneme,
    StarsStyle,
    StarsTrack,
)
from vocal_coach.stars_runner import (  # noqa: E402
    DEFAULT_STARS_DIR,
    STARS_BILINGUAL_HOP_SECONDS,
    STARS_BILINGUAL_SAMPLE_RATE,
    run_stars,
    write_stars_metadata,
)
from vocal_coach.student.align import BLANK_INDEX, viterbi_align_phones  # noqa: E402
from vocal_coach.student.model import STUDENT_TECH_NAMES  # noqa: E402
from vocal_coach.student_runner import (  # noqa: E402
    DEFAULT_STUDENT_DIR,
    SILENCE_TOKENS,
    _compute_features,
    load_student,
    run_student,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    return precision, recall, f1, accuracy


def _per_technique_metrics(
    teacher: list[dict],
    student: list[dict],
) -> dict[str, dict[str, float]]:
    """Compare per-phoneme technique flag dicts between teacher and student.

    Both lists must be aligned phoneme-to-phoneme (same length, same order).
    """
    n = min(len(teacher), len(student))
    out: dict[str, dict[str, float]] = {}
    for name in STARS_TECH_NAMES:
        y_true = np.array([int(teacher[i].get(name, 0)) for i in range(n)], dtype=np.int64)
        y_pred = np.array([int(student[i].get(name, 0)) for i in range(n)], dtype=np.int64)
        p, r, f1, acc = _binary_f1(y_true, y_pred)
        out[name] = {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "support": int((y_true == 1).sum())}
    macro_f1 = float(np.mean([v["f1"] for v in out.values()]))
    macro_acc = float(np.mean([v["accuracy"] for v in out.values()]))
    out["__macro__"] = {"f1": macro_f1, "accuracy": macro_acc}
    return out


def _boundary_mae(teacher: StarsTrack, student: StarsTrack) -> float:
    """Mean absolute start-time error per phoneme (in seconds)."""
    n = min(len(teacher.phonemes), len(student.phonemes))
    if n == 0:
        return float("nan")
    errs = [
        abs(teacher.phonemes[i].start_s - student.phonemes[i].start_s)
        for i in range(n)
    ]
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Direct inference from pre-computed features (no wav required)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _run_student_from_features(
    mel: np.ndarray,
    f0: np.ndarray,
    phones: list[str],
    model,
    phone_vocab: list[str],
    device: str,
    hop_seconds: float = STARS_BILINGUAL_HOP_SECONDS,
) -> StarsTrack:
    """Run the student model on pre-computed mel+f0 features.

    Used during eval so we test the exact same feature representation the
    model was trained on, without needing the original audio file.
    """
    phone_to_id = {p: i for i, p in enumerate(phone_vocab)}
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
    f0_t = torch.from_numpy(f0).unsqueeze(0).to(device)

    from vocal_coach.student_runner import TECH_THRESHOLDS

    out = model(mel_t, f0_t, mask=None)
    phoneme_logits = out.phoneme_logits.squeeze(0).detach().cpu().numpy()
    h = out.h  # (1, T, d_model) — kept on device for phoneme-level technique prediction

    use_phoneme_level = getattr(model.config, "phoneme_level_tech", False)
    if not use_phoneme_level:
        _tech_probs_np = torch.sigmoid(out.technique_logits.squeeze(0)).detach().cpu().numpy()

    target_phones = [p for p in phones if p not in SILENCE_TOKENS]
    target_ids = [phone_to_id.get(p, phone_to_id.get("<UNK>", 0)) for p in target_phones]
    aligned_spans = viterbi_align_phones(
        phoneme_logits,
        target_phone_ids=target_ids,
        allow_blank=True,
        blank_index=BLANK_INDEX,
    )

    # Stitch silence tokens back in.
    spans_with_silence: list[tuple[str, tuple[int, int]]] = []
    nonsil_idx = 0
    prev_end = 0
    for ph in phones:
        if ph in SILENCE_TOKENS:
            spans_with_silence.append((ph, (prev_end, prev_end)))
        else:
            if nonsil_idx < len(aligned_spans):
                start, end = aligned_spans[nonsil_idx]
            else:
                start, end = (prev_end, prev_end)
            spans_with_silence.append((ph, (start, end)))
            prev_end = end
            nonsil_idx += 1

    T = mel.shape[0]
    phonemes: list[StarsPhoneme] = []
    for i, (ph, (start_f, end_f)) in enumerate(spans_with_silence):
        start_s = float(start_f * hop_seconds)
        end_s = float(end_f * hop_seconds)
        techniques: dict[str, int] = {}
        if end_f > start_f and end_f <= T:
            if use_phoneme_level:
                ph_h = h[0, start_f:end_f].mean(dim=0, keepdim=True)  # (1, d)
                ph_logit = model.technique_head(ph_h).squeeze(0)       # (K,)
                avg_arr = torch.sigmoid(ph_logit).detach().cpu().numpy()
            else:
                avg_arr = _tech_probs_np[start_f:end_f].mean(axis=0)
        else:
            avg_arr = np.zeros(len(STUDENT_TECH_NAMES), dtype=np.float32)
        for k, name in enumerate(STUDENT_TECH_NAMES):
            thresh = TECH_THRESHOLDS.get(name, 0.5)
            techniques[name] = int(avg_arr[k] >= thresh) if k < len(avg_arr) else 0
        phonemes.append(
            StarsPhoneme(
                index=i,
                phoneme=ph,
                word=ph,
                word_index=-1,
                start_s=start_s,
                end_s=end_s,
                techniques=techniques,
            )
        )

    style = StarsStyle(
        language="unknown", gender="unknown", emotion="unknown",
        method="unknown", pace="unknown", range="unknown",
        technique_group="unknown",
    )
    return StarsTrack(
        sample_id="eval",
        sample_rate=STARS_BILINGUAL_SAMPLE_RATE,
        hop_seconds=hop_seconds,
        style=style,
        phonemes=phonemes,
        notes=[],
    )


# ---------------------------------------------------------------------------
# Speedup timing using UltraStar songs (have known wav + phone sequences)
# ---------------------------------------------------------------------------


def _measure_speedup(
    songs_dir: Path,
    student_dir: Path,
    stars_dir: Path,
    device: str,
    n_songs: int = 2,
) -> Optional[float]:
    """Time student vs teacher on UltraStar reference vocals.

    Returns speedup ratio (teacher_s / student_s) or None if no songs found.
    """
    songs_dir = Path(songs_dir)
    if not songs_dir.is_dir():
        return None

    from vocal_coach.song import load_manifest

    candidates = []
    for song_dir in sorted(songs_dir.iterdir()):
        meta_path = song_dir / "stars_metadata.json"
        manifest_path = song_dir / "manifest.json"
        if not meta_path.is_file() or not manifest_path.is_file():
            continue
        try:
            manifest = load_manifest(song_dir)
            wav = (song_dir / manifest.reference_vocal_path).resolve()
            if wav.is_file():
                candidates.append((song_dir, wav, meta_path))
        except Exception:
            continue

    if not candidates:
        return None

    candidates = candidates[:n_songs]
    total_teacher = 0.0
    total_student = 0.0

    model, phone_vocab, resolved_device = load_student(student_dir, device=device)

    for song_dir, wav_path, meta_path in candidates:
        print(f"[eval] timing on {song_dir.name} ...")

        # Student: feature extraction + model forward + Viterbi
        t0 = time.perf_counter()
        try:
            mel_np, f0_np = _compute_features(wav_path)
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            phones = raw[0].get("ph", []) if raw else []
            _run_student_from_features(mel_np, f0_np, phones, model, phone_vocab, resolved_device)
            total_student += time.perf_counter() - t0
        except Exception as exc:
            print(f"[eval]   student timing failed: {exc}")
            continue

        # Teacher: full STARS subprocess
        t1 = time.perf_counter()
        try:
            save_dir = song_dir / "_teacher_timing"
            run_stars(
                metadata_path=meta_path,
                save_dir=save_dir,
                sample_id=raw[0].get("item_name", "clip"),
                stars_dir=stars_dir,
            )
            total_teacher += time.perf_counter() - t1
            # Clean up temp dir
            import shutil
            shutil.rmtree(save_dir, ignore_errors=True)
        except Exception as exc:
            print(f"[eval]   teacher timing failed: {exc}")
            continue

    if total_student > 0 and total_teacher > 0:
        return total_teacher / total_student
    return None


# ---------------------------------------------------------------------------
# Held-out loader
# ---------------------------------------------------------------------------


@dataclass
class HeldOutClip:
    clip_id: str
    feature_dir: Path
    duration_s: float


def _load_held_out(corpus_root: Path, max_clips: Optional[int] = None) -> list[HeldOutClip]:
    manifest = corpus_root / "manifest.jsonl"
    if not manifest.is_file():
        return []
    clips: list[HeldOutClip] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("split") != "held_out":
            continue
        clips.append(
            HeldOutClip(
                clip_id=rec["clip_id"],
                feature_dir=Path(rec["feature_dir"]),
                duration_s=float(rec.get("duration_s", 0.0)),
            )
        )
    if max_clips is not None:
        clips = clips[:max_clips]
    return clips


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=ROOT / "data" / "student_corpus")
    p.add_argument("--student-dir", type=Path, default=DEFAULT_STUDENT_DIR)
    p.add_argument("--stars-dir", type=Path, default=DEFAULT_STARS_DIR)
    p.add_argument("--songs-dir", type=Path, default=ROOT / "data" / "songs",
                   help="UltraStar song dir used for head-to-head speedup timing.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-clips", type=int, default=10)
    p.add_argument("--time-teacher", action="store_true",
                   help="Run head-to-head speedup measurement against the teacher on UltraStar songs.")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional path to dump the full metrics JSON.")
    p.add_argument("--min-speedup", type=float, default=5.0)
    p.add_argument("--min-macro-f1", type=float, default=0.65)
    return p.parse_args()


def _data_checkpoint_or_exit(corpus: Path, student_dir: Path) -> None:
    missing = []
    if not (corpus / "manifest.jsonl").is_file():
        missing.append(f"student corpus at {corpus} (run scripts/export_student_dataset.py)")
    try:
        load_student(student_dir, device="cpu")
    except FileNotFoundError as exc:
        missing.append(f"student checkpoint: {exc}")
    if missing:
        bullet = "\n  - "
        print(
            "[eval] DATA CHECKPOINT — eval prerequisites not found:\n  - "
            + bullet.join(missing),
            file=sys.stderr,
        )
        sys.exit(78)


def main() -> int:
    args = parse_args()
    _data_checkpoint_or_exit(args.corpus, args.student_dir)

    clips = _load_held_out(args.corpus, max_clips=args.max_clips)
    if not clips:
        print("[eval] no held_out clips in manifest.jsonl; nothing to evaluate", file=sys.stderr)
        return 78
    print(f"[eval] evaluating {len(clips)} held-out clips (using pre-computed features)")

    model, phone_vocab, resolved_device = load_student(args.student_dir, device=args.device)
    print(f"[eval] student loaded on {resolved_device}")

    per_clip_metrics: list[dict] = []

    for clip in clips:
        clip_dir = args.corpus / clip.feature_dir
        teacher_path = clip_dir / "stars.json"
        mel_path = clip_dir / "mel.npy"
        f0_path = clip_dir / "f0.npy"
        labels_path = clip_dir / "labels.json"

        if not teacher_path.is_file():
            print(f"[eval] missing stars.json for {clip.clip_id}; skipping")
            continue
        if not mel_path.is_file() or not f0_path.is_file():
            print(f"[eval] missing mel/f0 for {clip.clip_id}; skipping")
            continue
        if not labels_path.is_file():
            print(f"[eval] missing labels.json for {clip.clip_id}; skipping")
            continue

        teacher = StarsTrack.model_validate_json(teacher_path.read_text(encoding="utf-8"))
        mel = np.load(str(mel_path))
        f0 = np.load(str(f0_path))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        phones = labels.get("phones", [])

        if not phones:
            print(f"[eval] empty phone list for {clip.clip_id}; skipping")
            continue

        try:
            student_track = _run_student_from_features(
                mel, f0, phones, model, phone_vocab, resolved_device
            )
        except Exception as exc:
            print(f"[eval] student inference failed for {clip.clip_id}: {exc}")
            continue

        teacher_flags = [ph.techniques for ph in teacher.phonemes]
        student_flags = [ph.techniques for ph in student_track.phonemes]
        boundary_mae_s = _boundary_mae(teacher, student_track)

        clip_metrics = _per_technique_metrics(teacher_flags, student_flags)
        clip_metrics["__boundary_mae_s__"] = boundary_mae_s
        clip_metrics["__clip_id__"] = clip.clip_id
        clip_metrics["__duration_s__"] = clip.duration_s
        per_clip_metrics.append(clip_metrics)

    if not per_clip_metrics:
        print("[eval] no clips were successfully evaluated", file=sys.stderr)
        return 1

    # Aggregate over clips.
    agg: dict[str, dict[str, float]] = {}
    for name in STARS_TECH_NAMES + ["__macro__"]:
        f1s = [m[name]["f1"] for m in per_clip_metrics if name in m]
        accs = [m[name]["accuracy"] for m in per_clip_metrics if name in m]
        supports = [m[name].get("support", 0) for m in per_clip_metrics if name in m and name != "__macro__"]
        agg[name] = {
            "f1": float(np.mean(f1s)) if f1s else 0.0,
            "accuracy": float(np.mean(accs)) if accs else 0.0,
            "total_support": int(sum(supports)) if supports else 0,
        }
    boundary_maes = [
        m["__boundary_mae_s__"] for m in per_clip_metrics
        if isinstance(m.get("__boundary_mae_s__"), float) and not np.isnan(m["__boundary_mae_s__"])
    ]
    boundary_mae_s = float(np.mean(boundary_maes)) if boundary_maes else float("nan")

    # Speedup measurement using UltraStar songs.
    speedup: Optional[float] = None
    total_teacher_secs = 0.0
    total_student_secs = 0.0
    if args.time_teacher:
        print("[eval] measuring wall-clock speedup on UltraStar reference vocals ...")
        speedup = _measure_speedup(
            songs_dir=args.songs_dir,
            student_dir=args.student_dir,
            stars_dir=args.stars_dir,
            device=args.device,
        )

    # Report.
    print()
    print(f"[eval] held-out clips        : {len(per_clip_metrics)}")
    if not np.isnan(boundary_mae_s):
        print(f"[eval] boundary MAE          : {boundary_mae_s:.3f}s")
    else:
        print(f"[eval] boundary MAE          : n/a")
    print(f"[eval] technique F1 / ACC (macro):")
    macro = agg["__macro__"]
    print(f"           macro: F1={macro['f1']:.3f}  ACC={macro['accuracy']:.3f}")
    for name in STARS_TECH_NAMES:
        m = agg[name]
        sup = m.get("total_support", 0)
        print(f"       {name:>10}: F1={m['f1']:.3f}  ACC={m['accuracy']:.3f}  support={sup}")
    if speedup is not None:
        print(f"[eval] wall-clock speedup vs teacher: {speedup:.2f}x")

    pass_macro = macro["f1"] >= args.min_macro_f1
    pass_speedup = speedup is None or speedup >= args.min_speedup
    print()
    print(f"[eval] PASS macro-F1 >= {args.min_macro_f1}? {pass_macro}")
    if speedup is not None:
        print(f"[eval] PASS speedup  >= {args.min_speedup}x? {pass_speedup}")

    if args.report is not None:
        report = {
            "aggregate": agg,
            "boundary_mae_s": boundary_mae_s,
            "speedup": speedup,
            "per_clip": per_clip_metrics,
        }
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[eval] wrote report -> {args.report}")

    return 0 if (pass_macro and pass_speedup) else 1


if __name__ == "__main__":
    sys.exit(main())
