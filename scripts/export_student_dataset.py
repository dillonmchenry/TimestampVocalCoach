"""Export a teacher-labeled corpus for distillation of the STARS student.

Inputs (any subset):

    1. GTSinger English (https://huggingface.co/datasets/AaronZ345/GTSinger).
       Each clip lives in a directory with ``<id>.wav`` + ``<id>.json``.
       Labels are built DIRECTLY from GTSinger's own JSON annotations —
       no STARS inference is required for these clips. This gives ground-truth
       phoneme boundaries (ph_start/ph_end) and per-phoneme technique flags
       (mix, falsetto, breathy, pharyngeal, glissando, vibrato), which are
       mapped to the student's 9-way taxonomy.

    2. NUS-48E (https://smsl.comp.nus.edu.sg/NUS48E/). Each singer has a
       directory of ``.wav`` files with companion ``.txt``/``.lab`` phoneme
       alignment files (seconds format). Labels are derived from STARS teacher
       inference (no GTSinger ground truth available).

    3. UltraStar song bundles already under ``data/songs/<song_id>/``.
       Labels are derived from STARS teacher inference.

GTSinger discovery uses STRATIFIED sampling by ``(singer, technique)`` bucket,
ensuring all singers and all technique classes are represented before any
bucket is exhausted. With 3 English singers × 5 techniques = 15 buckets,
``--max-per-bucket 135`` yields ≈ 2 025 GTSinger clips.

For every clip we persist ``data/student_corpus/<source>/<clip_id>/``::

        mel.npy           (T, 80)  log-mel features
        f0.npy            (T,)     F0 in Hz (0 = unvoiced)
        labels.json       {phones, ph_start_frames, boundaries, techniques,
                           hop_seconds}
        stars.json        StarsTrack JSON (from STARS or reconstructed from
                           GTSinger annotations for reference / eval use)

and a root ``manifest.jsonl``.

Usage::

    python scripts/export_student_dataset.py \\
        --gtsinger-dir  data/raw/GTSinger/English \\
        --nus48e-dir    data/raw/NUS_48E \\
        --songs-dir     data/songs \\
        --output-dir    data/student_corpus \\
        --max-per-bucket 135 \\
        --held-out-frac  0.10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vocal_coach.schemas import (  # noqa: E402
    STARS_TECH_NAMES,
    StarsMetadataEntry,
    StarsPhoneme,
    StarsStyle,
    StarsTrack,
)
from vocal_coach.song import load_manifest  # noqa: E402
from vocal_coach.stars_runner import (  # noqa: E402
    DEFAULT_STARS_DIR,
    STARS_BILINGUAL_HOP_SECONDS,
    STARS_BILINGUAL_SAMPLE_RATE,
    parse_stars_output,
    run_stars_inference,
    write_stars_metadata,
)

try:
    from vocal_coach.student.model import STUDENT_TECH_NAMES
except ImportError:
    STUDENT_TECH_NAMES = [
        "bubble", "breathe", "pharyngeal", "vibrato",
        "glissando", "mixed", "falsetto", "weak", "strong",
    ]

# GTSinger per-phoneme technique field → student taxonomy name.
# GTSinger has 6 technique fields; bubble/weak/strong have no equivalent
# and are left as 0 for GTSinger clips.
GTSINGER_TO_STUDENT: dict[str, str] = {
    "mix": "mixed",
    "falsetto": "falsetto",
    "breathy": "breathe",
    "pharyngeal": "pharyngeal",
    "glissando": "glissando",
    "vibrato": "vibrato",
}

SILENCE_TOKENS = {"<SP>", "<AP>", " "}

# STARS bilingual model: hop_size=256 @ 24000 Hz → ~10.67 ms per frame.
# Any phoneme shorter than this gets start_frame == end_frame in align_ph(),
# raising a BinarizationError.  We clamp ph_durs to this minimum.
_STARS_HOP_S = 256.0 / 24000.0  # ≈ 0.01067 s
_MIN_PH_DUR_S = _STARS_HOP_S * 1.5  # 1.5 frames gives safe rounding margin

# ARPAbet vowel nuclei that require a stress digit in the STARS bilingual
# phone set (EY → EY1, AH → AH1, …).  Consonants have no stress digit.
_ARPA_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
}


def _normalize_arpa_stress(ph: str) -> str:
    """Add default primary-stress digit to bare ARPAbet vowels.

    NUS-48E alignment files use bare vowels (AH, EY, …) but the STARS
    bilingual phone set expects stress-marked vowels (AH1, EY1, …).
    Consonants already match (B, N, D, … need no digit).
    """
    if ph and not ph[-1].isdigit() and ph in _ARPA_VOWELS:
        return ph + "1"
    return ph


# ---------------------------------------------------------------------------
# Clip spec
# ---------------------------------------------------------------------------


@dataclass
class ClipSpec:
    """One unit of work for the export pipeline."""

    clip_id: str
    source: str
    wav_path: Path
    words: list[str]
    phones: list[str]
    ph2word: list[int]
    word_durs: Optional[list[float]] = None
    ph_durs: Optional[list[float]] = None
    # When set, labels come directly from the source annotation (GTSinger)
    # rather than from STARS inference.  Format:
    #   {phones, ph_start_s, ph_end_s, techniques: {student_name: [0/1, ...]}}
    direct_labels: Optional[dict] = None


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "clip"


# ---------------------------------------------------------------------------
# GTSinger discovery — stratified by (singer, technique)
# ---------------------------------------------------------------------------


def _gtsinger_bucket(rel: Path) -> tuple[str, str]:
    """Return (singer, technique) from a GTSinger relative wav path.

    Path layout: ``EN-Alto-1/Vibrato/edelweiss/Vibrato_Group/0000.wav``
    """
    parts = rel.parts
    singer = parts[0] if len(parts) > 0 else "unknown"
    technique = parts[1] if len(parts) > 1 else "unknown"
    return singer, technique


def _parse_gtsinger_direct_labels(entries: list[dict]) -> dict:
    """Build direct label dict from a GTSinger JSON annotation array.

    Returns::
        {
            "phones":     list[str],  # full sequence incl. <SP>/<AP>
            "ph_start_s": list[float],
            "ph_end_s":   list[float],
            "techniques": {student_name: list[int]},  # per-phoneme 0/1
        }
    """
    phones: list[str] = []
    ph_start_s: list[float] = []
    ph_end_s: list[float] = []
    tech: dict[str, list[int]] = {name: [] for name in STUDENT_TECH_NAMES}

    for entry in entries:
        ph_list = entry.get("ph", [])
        starts = entry.get("ph_start", [])
        ends = entry.get("ph_end", [])

        for i, ph in enumerate(ph_list):
            # Normalise silence / breath tokens
            if ph in SILENCE_TOKENS or ph in ("<SP>", "<AP>"):
                ph_out = "<SP>"
            else:
                ph_out = ph
            phones.append(ph_out)

            try:
                ph_start_s.append(float(starts[i]))
                ph_end_s.append(float(ends[i]))
            except (IndexError, TypeError, ValueError):
                ph_start_s.append(0.0)
                ph_end_s.append(0.0)

            # Map GTSinger fields to student taxonomy.
            mapped: dict[str, int] = {}
            for gts_key, student_name in GTSINGER_TO_STUDENT.items():
                seq = entry.get(gts_key, [])
                if i < len(seq):
                    val = seq[i]
                    mapped[student_name] = 0 if (val == "0" or val == 0) else 1
                else:
                    mapped[student_name] = 0

            for name in STUDENT_TECH_NAMES:
                tech[name].append(mapped.get(name, 0))

    return {
        "phones": phones,
        "ph_start_s": ph_start_s,
        "ph_end_s": ph_end_s,
        "techniques": tech,
    }


def discover_gtsinger_clips(
    gtsinger_dir: Path,
    max_per_bucket: Optional[int] = None,
    seed: int = 42,
) -> list[ClipSpec]:
    """Walk a GTSinger language directory with stratified sampling.

    Clips are grouped by ``(singer, technique)`` bucket.  Within each bucket
    they are shuffled (seeded) before capping, so the cap doesn't always
    pick the same songs.  Pass ``max_per_bucket=None`` to take everything.
    """
    gtsinger_dir = Path(gtsinger_dir).resolve()
    if not gtsinger_dir.is_dir():
        return []

    rng = random.Random(seed)

    # Group wav paths by bucket before capping.
    buckets: dict[tuple[str, str], list[Path]] = {}
    for wav_path in sorted(gtsinger_dir.rglob("*.wav")):
        json_path = wav_path.with_suffix(".json")
        if not json_path.is_file():
            continue
        rel = wav_path.relative_to(gtsinger_dir)
        bucket = _gtsinger_bucket(rel)
        buckets.setdefault(bucket, []).append(wav_path)

    print(f"[export] GTSinger buckets found: {len(buckets)}")
    for (singer, tech), paths in sorted(buckets.items()):
        cap_info = f" (cap {max_per_bucket})" if max_per_bucket else ""
        print(f"  {singer}/{tech}: {len(paths)} clips{cap_info}")

    out: list[ClipSpec] = []
    for (singer, technique), wav_paths in sorted(buckets.items()):
        rng.shuffle(wav_paths)
        if max_per_bucket is not None:
            wav_paths = wav_paths[:max_per_bucket]

        for wav_path in wav_paths:
            json_path = wav_path.with_suffix(".json")
            try:
                entries = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(entries, list):
                continue

            # Build phone sequence (for StarsMetadataEntry / STARS compat).
            words: list[str] = []
            phones: list[str] = []
            ph2word: list[int] = []
            word_durs: list[float] = []
            ph_durs: list[float] = []

            for word_idx, entry in enumerate(entries):
                word = entry.get("word", "")
                word_out = "<SP>" if word in SILENCE_TOKENS else word
                words.append(word_out)
                try:
                    word_durs.append(
                        float(entry["end_time"]) - float(entry["start_time"])
                    )
                except (KeyError, TypeError, ValueError):
                    word_durs.append(0.0)

                ph_list = entry.get("ph", [])
                ph_starts = entry.get("ph_start", [])
                ph_ends = entry.get("ph_end", [])
                for i, ph in enumerate(ph_list):
                    ph_out = "<SP>" if ph in SILENCE_TOKENS else ph
                    phones.append(ph_out)
                    ph2word.append(word_idx)
                    try:
                        ph_durs.append(float(ph_ends[i]) - float(ph_starts[i]))
                    except (IndexError, TypeError, ValueError):
                        ph_durs.append(0.0)

            if not phones:
                continue

            # Parse ground-truth labels directly from JSON.
            direct = _parse_gtsinger_direct_labels(entries)

            rel = wav_path.relative_to(gtsinger_dir).with_suffix("")
            clip_id = _slugify(f"gtsinger-{rel.as_posix()}")
            out.append(
                ClipSpec(
                    clip_id=clip_id,
                    source="gtsinger",
                    wav_path=wav_path,
                    words=words,
                    phones=phones,
                    ph2word=ph2word,
                    word_durs=word_durs,
                    ph_durs=ph_durs,
                    direct_labels=direct,
                )
            )
    return out


# ---------------------------------------------------------------------------
# NUS-48E discovery (STARS-labeled)
# ---------------------------------------------------------------------------


def discover_nus48e_clips(nus48e_dir: Path) -> list[ClipSpec]:
    """Walk a NUS-48E directory looking for ``.wav`` + alignment file pairs.

    NUS-48E ships phoneme alignments as either ``.lab`` (HTK 100ns units) or
    ``.txt`` (seconds) files.  Both share the three-column layout::

        start  end  phoneme

    We map sil/sp/pau -> <SP> to match STARS's silence token.
    """
    nus48e_dir = Path(nus48e_dir).resolve()
    if not nus48e_dir.is_dir():
        return []

    out: list[ClipSpec] = []
    for wav_path in sorted(nus48e_dir.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        lab_path = wav_path.with_suffix(".lab")
        if txt_path.is_file():
            align_path = txt_path
        elif lab_path.is_file():
            align_path = lab_path
        else:
            continue

        phones: list[str] = []
        ph_durs: list[float] = []
        try:
            for line in align_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    s = float(parts[0])
                    e = float(parts[1])
                except ValueError:
                    continue
                # HTK times in 100ns; convert to seconds.
                if s > 1e6 or e > 1e6:
                    s /= 1e7
                    e /= 1e7
                ph = parts[2].strip().upper()
                if ph in {"SIL", "SP", "PAU", ""}:
                    ph = "<SP>"
                else:
                    ph = _normalize_arpa_stress(ph)
                phones.append(ph)
                ph_durs.append(max(_MIN_PH_DUR_S, e - s))
        except Exception:
            continue
        if not phones:
            continue

        # Build word segmentation from silence boundaries.
        # Each run of non-silence phones becomes one word; each <SP> token
        # is its own single-phone word.  Without this, all phones would map
        # to one giant "word" and the Viterbi DP matrix would be O(T × 2K)
        # — hundreds of MB for long NUS-48E sing clips.
        words: list[str] = []
        ph2word: list[int] = []
        word_durs: list[float] = []
        i = 0
        while i < len(phones):
            if phones[i] == "<SP>":
                ph2word.append(len(words))
                words.append("<SP>")
                word_durs.append(ph_durs[i])
                i += 1
            else:
                j = i
                while j < len(phones) and phones[j] != "<SP>":
                    j += 1
                word_idx = len(words)
                for k in range(i, j):
                    ph2word.append(word_idx)
                words.append(f"w{word_idx}")
                word_durs.append(sum(ph_durs[i:j]))
                i = j

        rel = wav_path.relative_to(nus48e_dir).with_suffix("")
        clip_id = _slugify(f"nus48e-{rel.as_posix()}")
        out.append(
            ClipSpec(
                clip_id=clip_id,
                source="nus48e",
                wav_path=wav_path,
                words=words,
                phones=phones,
                ph2word=ph2word,
                word_durs=word_durs,
                ph_durs=ph_durs,
            )
        )
    return out


# ---------------------------------------------------------------------------
# UltraStar song discovery (STARS-labeled)
# ---------------------------------------------------------------------------


def discover_song_clips(songs_dir: Path) -> list[ClipSpec]:
    """Walk ``data/songs/<song_id>/`` and reuse each bundle's stars_metadata.json."""
    songs_dir = Path(songs_dir).resolve()
    if not songs_dir.is_dir():
        return []

    out: list[ClipSpec] = []
    for song_dir in sorted(songs_dir.iterdir()):
        if not song_dir.is_dir():
            continue
        meta_path = song_dir / "stars_metadata.json"
        manifest_path = song_dir / "manifest.json"
        if not meta_path.is_file() or not manifest_path.is_file():
            continue
        try:
            manifest = load_manifest(song_dir)
        except Exception:
            continue
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not raw:
            continue
        entry = raw[0]
        wav_path = (song_dir / manifest.reference_vocal_path).resolve()
        if not wav_path.is_file():
            continue
        raw_ph_durs = entry.get("ph_durs") or []
        clamped_ph_durs = [max(_MIN_PH_DUR_S, d) for d in raw_ph_durs]
        raw_word_durs = entry.get("word_durs") or []
        # word_durs must also stay consistent: clamp each to at least _MIN_PH_DUR_S
        clamped_word_durs = [max(_MIN_PH_DUR_S, d) for d in raw_word_durs]
        out.append(
            ClipSpec(
                clip_id=_slugify(f"song-{manifest.song_id}"),
                source="ultrastar",
                wav_path=wav_path,
                words=list(entry.get("word", [])),
                phones=list(entry.get("ph", [])),
                ph2word=list(entry.get("ph2words", [])),
                word_durs=clamped_word_durs,
                ph_durs=clamped_ph_durs,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Feature extraction (mel + F0 at STARS's frame rate)
# ---------------------------------------------------------------------------

from vocal_coach.rmvpe_f0 import extract_f0_rmvpe  # noqa: E402


def _compute_mel_and_f0(
    wav_path: Path,
    *,
    sample_rate: int = STARS_BILINGUAL_SAMPLE_RATE,
    hop_length: int = 128,
    n_mels: int = 80,
    n_fft: int = 512,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mel: (T, n_mels) float32 log-mel, f0: (T,) float32 Hz).

    F0 is extracted with RMVPE (GPU-accelerated neural pitch estimator) so
    that the training feature matches exactly what ``student_runner`` uses at
    inference time.  RMVPE replaces the previous ``librosa.pyin`` call which
    was accurate but extremely slow (CPU-only, O(n × candidates × states)).
    """
    import librosa

    wav, _sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    mel_power = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sample_rate / 2.0,
        power=2.0,
        center=True,
    )
    log_mel = np.log(mel_power + 1e-10).astype(np.float32).T  # (T, n_mels)
    T = log_mel.shape[0]

    f0 = extract_f0_rmvpe(
        wav,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_frames=T,
        device=device,
    )

    return log_mel, f0


# ---------------------------------------------------------------------------
# Label serialization — GTSinger direct path
# ---------------------------------------------------------------------------


def _gtsinger_direct_to_frame_labels(
    direct: dict,
    num_frames: int,
    hop_seconds: float = STARS_BILINGUAL_HOP_SECONDS,
) -> dict:
    """Convert per-phoneme GTSinger labels to the student's frame-level format.

    ``direct`` is the dict returned by ``_parse_gtsinger_direct_labels``.
    Returns the same structure as ``_stars_track_to_labels`` so the training
    script can consume both without modification.
    """
    phones = direct["phones"]
    ph_start_s = direct["ph_start_s"]
    techniques = direct["techniques"]  # {student_name: [per-phoneme 0/1]}

    ph_start_frames = [
        max(0, min(num_frames - 1, int(round(s / hop_seconds))))
        for s in ph_start_s
    ]

    boundaries = [0] * num_frames
    for f in ph_start_frames:
        if 0 <= f < num_frames:
            boundaries[f] = 1

    return {
        "phones": phones,
        "ph_start_frames": ph_start_frames,
        "boundaries": boundaries,
        "techniques": techniques,
        "hop_seconds": hop_seconds,
    }


def _gtsinger_direct_to_stars_track(
    clip: ClipSpec,
    hop_seconds: float = STARS_BILINGUAL_HOP_SECONDS,
) -> StarsTrack:
    """Reconstruct a StarsTrack from GTSinger direct annotations.

    Used to write ``stars.json`` for reference / eval parity.  Phoneme
    boundaries come from GTSinger's JSON; techniques from ground-truth flags.
    """
    direct = clip.direct_labels
    assert direct is not None

    phones = direct["phones"]
    ph_start_s = direct["ph_start_s"]
    ph_end_s = direct["ph_end_s"]
    techniques = direct["techniques"]  # {student_name: [per-phoneme 0/1]}

    phonemes: list[StarsPhoneme] = []
    ph2word = clip.ph2word
    word_list = clip.words
    for i, ph in enumerate(phones):
        w_idx = ph2word[i] if i < len(ph2word) else -1
        word = word_list[w_idx] if 0 <= w_idx < len(word_list) else ph
        tech_dict: dict[str, int] = {
            name: (techniques[name][i] if i < len(techniques.get(name, [])) else 0)
            for name in STARS_TECH_NAMES
        }
        phonemes.append(
            StarsPhoneme(
                index=i,
                phoneme=ph,
                word=word,
                word_index=w_idx,
                start_s=ph_start_s[i] if i < len(ph_start_s) else 0.0,
                end_s=ph_end_s[i] if i < len(direct["ph_end_s"]) else 0.0,
                techniques=tech_dict,
            )
        )

    style = StarsStyle(
        language="english",
        gender="unknown",
        emotion="unknown",
        method="unknown",
        pace="unknown",
        range="unknown",
        technique_group="unknown",
    )
    return StarsTrack(
        sample_id=clip.clip_id,
        sample_rate=STARS_BILINGUAL_SAMPLE_RATE,
        hop_seconds=hop_seconds,
        style=style,
        phonemes=phonemes,
        notes=[],
    )


# ---------------------------------------------------------------------------
# Label serialization — STARS path
# ---------------------------------------------------------------------------


def _stars_track_to_labels(
    track: StarsTrack,
    *,
    hop_seconds: float = STARS_BILINGUAL_HOP_SECONDS,
    num_frames: int,
) -> dict:
    """Convert a parsed StarsTrack into the student's label dict."""
    phones: list[str] = []
    techniques: dict[str, list[int]] = {name: [] for name in STUDENT_TECH_NAMES}
    ph_start_frames: list[int] = []
    for ph in track.phonemes:
        phones.append(ph.phoneme)
        f = int(round(ph.start_s / hop_seconds))
        ph_start_frames.append(max(0, min(num_frames - 1, f)))
        for name in STUDENT_TECH_NAMES:
            techniques[name].append(int(ph.techniques.get(name, 0)))
    boundaries = [0] * num_frames
    for f in ph_start_frames:
        if 0 <= f < num_frames:
            boundaries[f] = 1

    return {
        "phones": phones,
        "ph_start_frames": ph_start_frames,
        "boundaries": boundaries,
        "techniques": techniques,
        "hop_seconds": hop_seconds,
    }


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def _print_data_checkpoint(missing: list[str]) -> None:
    bullet = "\n  - "
    print(
        "[export] DATA CHECKPOINT — required inputs not found:\n  - "
        + bullet.join(missing)
        + "\n\nPer the Sprint 3 plan's 'Data checkpoints — user action required'\n"
        "section, this script pauses until at least one source is supplied:\n"
        "  - GTSinger English: https://huggingface.co/datasets/AaronZ345/GTSinger\n"
        "  - NUS-48E:           https://smsl.comp.nus.edu.sg/NUS48E/\n"
        "  - UltraStar songs:   import via scripts/import_ultrastar.py first\n",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gtsinger-dir", type=Path, default=None)
    p.add_argument("--nus48e-dir", type=Path, default=None)
    p.add_argument(
        "--songs-dir",
        type=Path,
        default=ROOT / "data" / "songs",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "student_corpus",
    )
    p.add_argument(
        "--max-per-bucket",
        type=int,
        default=135,
        help="Max GTSinger clips per (singer, technique) bucket. "
             "15 buckets × 135 ≈ 2 025 GTSinger clips. Set 0 for no cap.",
    )
    p.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional hard cap on total clips after stratification (all sources).",
    )
    p.add_argument("--held-out-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stars-dir", type=Path, default=DEFAULT_STARS_DIR)
    p.add_argument("--cuda-visible-devices", default="0")
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device for RMVPE F0 extraction (default: cuda).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover clips and print counts; do not run STARS or write features.",
    )
    p.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip mel/F0 computation (only write labels.json + stars.json).",
    )
    p.add_argument(
        "--append-manifest",
        action="store_true",
        help="Load existing manifest.jsonl from --output-dir and prepend those "
             "records before writing the new ones. Use when re-running only the "
             "STARS portion after GTSinger clips are already exported.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Discover clips per source ------------------------------------------
    discovered: list[ClipSpec] = []
    missing: list[str] = []

    max_per_bucket = args.max_per_bucket if args.max_per_bucket and args.max_per_bucket > 0 else None

    if args.gtsinger_dir is not None:
        gts = discover_gtsinger_clips(
            args.gtsinger_dir,
            max_per_bucket=max_per_bucket,
            seed=args.seed,
        )
        if not gts:
            missing.append(f"GTSinger English at {args.gtsinger_dir} (no clips found)")
        discovered.extend(gts)
    if args.nus48e_dir is not None:
        nus = discover_nus48e_clips(args.nus48e_dir)
        if not nus:
            missing.append(f"NUS-48E at {args.nus48e_dir} (no clips found)")
        discovered.extend(nus)
    songs = discover_song_clips(args.songs_dir)
    if not songs:
        missing.append(f"UltraStar songs at {args.songs_dir} (no bundles found)")
    discovered.extend(songs)

    print(f"[export] discovered {len(discovered)} clips:")
    by_source: dict[str, int] = {}
    for clip in discovered:
        by_source[clip.source] = by_source.get(clip.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  - {source:>10}: {count} clips")

    if not discovered:
        _print_data_checkpoint(missing)
        return 78

    if args.max_clips is not None and args.max_clips < len(discovered):
        print(f"[export] capping to first {args.max_clips} clips (--max-clips)")
        discovered = discovered[: args.max_clips]

    if args.dry_run:
        print("[export] --dry-run: skipping STARS + feature extraction")
        return 0

    # 2. Split into two paths: GTSinger-direct vs STARS-labeled ---------------
    direct_clips = [c for c in discovered if c.direct_labels is not None]
    stars_clips  = [c for c in discovered if c.direct_labels is None]
    print(f"[export] GTSinger direct-annotation clips : {len(direct_clips)}")
    print(f"[export] STARS-inference clips (NUS+songs): {len(stars_clips)}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest_records: list[dict] = []

    # When --append-manifest is set, seed manifest_records with existing entries
    # so that clips already exported (e.g. GTSinger) are preserved.
    if getattr(args, "append_manifest", False):
        existing_manifest = output_root / "manifest.jsonl"
        if existing_manifest.is_file():
            for line in existing_manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    manifest_records.append(json.loads(line))
            print(f"[export] loaded {len(manifest_records)} existing manifest records")

    # 3a. GTSinger direct path — no STARS needed ----------------------------
    print(f"[export] processing {len(direct_clips)} GTSinger direct clips ...")
    for i, clip in enumerate(direct_clips, 1):
        if i % 100 == 0 or i == len(direct_clips):
            print(f"[export]   {i}/{len(direct_clips)} ...")

        clip_out = output_root / clip.source / clip.clip_id
        clip_out.mkdir(parents=True, exist_ok=True)

        if args.skip_features:
            ph_end_s = clip.direct_labels["ph_end_s"]
            last_end = max(ph_end_s) if ph_end_s else 0.0
            num_frames = max(1, int(round(last_end / STARS_BILINGUAL_HOP_SECONDS)))
        else:
            try:
                mel, f0 = _compute_mel_and_f0(clip.wav_path, device=args.device)
            except Exception as exc:
                print(f"[export] WARNING: mel/F0 failed for {clip.clip_id}: {exc}")
                continue
            np.save(clip_out / "mel.npy", mel)
            np.save(clip_out / "f0.npy", f0)
            num_frames = mel.shape[0]

        labels = _gtsinger_direct_to_frame_labels(
            clip.direct_labels, num_frames=num_frames
        )
        (clip_out / "labels.json").write_text(
            json.dumps(labels), encoding="utf-8"
        )

        # Reconstruct a StarsTrack for eval parity and write stars.json.
        track = _gtsinger_direct_to_stars_track(clip)
        (clip_out / "stars.json").write_text(
            track.model_dump_json(indent=2), encoding="utf-8"
        )

        split = "held_out" if rng.random() < args.held_out_frac else "train"
        duration_s = float(num_frames * STARS_BILINGUAL_HOP_SECONDS)
        manifest_records.append(
            {
                "clip_id": clip.clip_id,
                "source": clip.source,
                "label_source": "gtsinger_direct",
                "split": split,
                "duration_s": duration_s,
                "num_phones": len(labels["phones"]),
                "num_frames": int(num_frames),
                "feature_dir": str(
                    (clip_out).relative_to(output_root)
                ).replace("\\", "/"),
            }
        )

    # 3b. STARS path — batch NUS-48E + UltraStar ---------------------------
    if stars_clips:
        stars_work_dir = output_root / "_stars_work"
        stars_work_dir.mkdir(parents=True, exist_ok=True)

        entries: list[StarsMetadataEntry] = []
        for clip in stars_clips:
            entries.append(
                StarsMetadataEntry(
                    item_name=clip.clip_id,
                    wav_fn=str(clip.wav_path).replace("\\", "/"),
                    word=clip.words,
                    ph=clip.phones,
                    ph2words=clip.ph2word,
                    ph_durs=clip.ph_durs,
                    word_durs=clip.word_durs,
                )
            )
        meta_path = write_stars_metadata(stars_work_dir, entries)
        print(f"[export] wrote STARS metadata: {meta_path} ({len(entries)} items)")
        print(f"[export] running STARS subprocess ...")
        output_json = run_stars_inference(
            meta_path,
            save_dir=stars_work_dir,
            stars_dir=args.stars_dir,
            cuda_visible_devices=args.cuda_visible_devices,
            # ds_workers=0 forces single-process data loading; the default of 1
            # spawns a worker subprocess that fails to load CUDA DLLs on Windows
            # when the paging file is under pressure (WinError 1455).
            extra_args=["--ds_workers", "0"],
        )
        raw_items = json.loads(output_json.read_text(encoding="utf-8"))
        items_by_name = {item.get("item_name"): item for item in raw_items}
        print(f"[export] STARS returned {len(items_by_name)} items")

        for clip in stars_clips:
            item = items_by_name.get(clip.clip_id)
            if item is None:
                print(f"[export] WARNING: STARS missing {clip.clip_id}; skipping")
                continue

            tmp_out = stars_work_dir / f"_{clip.clip_id}.json"
            tmp_out.write_text(json.dumps([item]), encoding="utf-8")
            try:
                track = parse_stars_output(tmp_out, sample_id=clip.clip_id)
            except Exception as exc:
                print(f"[export] WARNING: parse failed for {clip.clip_id}: {exc}")
                tmp_out.unlink(missing_ok=True)
                continue
            tmp_out.unlink(missing_ok=True)

            clip_out = output_root / clip.source / clip.clip_id
            clip_out.mkdir(parents=True, exist_ok=True)

            if args.skip_features:
                num_frames = max(
                    int(ph.end_s / STARS_BILINGUAL_HOP_SECONDS)
                    for ph in track.phonemes
                ) if track.phonemes else 0
            else:
                try:
                    mel, f0 = _compute_mel_and_f0(clip.wav_path, device=args.device)
                except Exception as exc:
                    print(f"[export] WARNING: mel/F0 failed for {clip.clip_id}: {exc}")
                    continue
                np.save(clip_out / "mel.npy", mel)
                np.save(clip_out / "f0.npy", f0)
                num_frames = mel.shape[0]

            labels = _stars_track_to_labels(track, num_frames=num_frames)
            (clip_out / "labels.json").write_text(
                json.dumps(labels), encoding="utf-8"
            )
            (clip_out / "stars.json").write_text(
                track.model_dump_json(indent=2), encoding="utf-8"
            )

            split = "held_out" if rng.random() < args.held_out_frac else "train"
            duration_s = float(num_frames * STARS_BILINGUAL_HOP_SECONDS)
            manifest_records.append(
                {
                    "clip_id": clip.clip_id,
                    "source": clip.source,
                    "label_source": "stars",
                    "split": split,
                    "duration_s": duration_s,
                    "num_phones": len(track.phonemes),
                    "num_frames": int(num_frames),
                    "feature_dir": str(
                        clip_out.relative_to(output_root)
                    ).replace("\\", "/"),
                }
            )

    # 4. Write manifest.jsonl -----------------------------------------------
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fp:
        for rec in manifest_records:
            fp.write(json.dumps(rec) + "\n")
    print(f"[export] wrote {len(manifest_records)} records to {manifest_path}")

    train = sum(1 for r in manifest_records if r["split"] == "train")
    held  = sum(1 for r in manifest_records if r["split"] == "held_out")
    direct_count = sum(1 for r in manifest_records if r["label_source"] == "gtsinger_direct")
    stars_count  = sum(1 for r in manifest_records if r["label_source"] == "stars")
    print(f"[export] split: train={train} held_out={held}")
    print(f"[export] label source: gtsinger_direct={direct_count} stars={stars_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
