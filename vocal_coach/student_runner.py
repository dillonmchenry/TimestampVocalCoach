"""In-process inference wrapper for the distilled STARS student.

Mirrors the public surface of ``vocal_coach.stars_runner.run_stars`` so it can
plug into the existing pipeline behind a ``stars_profile`` switch:

    profile=full   ->  run_stars (subprocess; teacher checkpoint, slow but rich)
    profile=fast   ->  run_student (in-process; student checkpoint, ~5x faster)

The student outputs a ``StarsTrack`` with phoneme spans + 9-way technique flags
matching the exact schema the rest of the pipeline already consumes
(``compare_note_techniques`` in ``vocal_coach.align_v2`` etc.). The student
does NOT produce ``StarsNote`` transcriptions or ``StarsStyle`` — those are
unused for user recordings in the current pipeline, so we fill them with
neutral placeholders.

The student expects:

    * mel : (T, 80) log-mel at 24 kHz / 128-hop  (same as the export script)
    * f0  : (T,)    F0 Hz (0 = unvoiced)

We forced-align the *known* target phoneme sequence (passed as
``ph`` in the same metadata.json shape STARS consumed) through the
student's frame logits via the CTC Viterbi DP in ``vocal_coach.student.align``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vocal_coach.schemas import (
    StarsMetadataEntry,
    StarsPhoneme,
    StarsStyle,
    StarsTrack,
)
from vocal_coach.stars_runner import (
    STARS_BILINGUAL_HOP_SECONDS,
    STARS_BILINGUAL_SAMPLE_RATE,
)
from vocal_coach.student.align import BLANK_INDEX, viterbi_align_phones
from vocal_coach.student.model import STUDENT_TECH_NAMES, StudentSTARS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT_DIR = REPO_ROOT / "data" / "student_v6"  # retrained on RMVPE F0
DEFAULT_STUDENT_CKPT_CANDIDATES = ("best.pt", "final.pt", "model_ckpt_steps_latest.pt")

SILENCE_TOKENS = {"<SP>", "<AP>", "<UNK>"}

# Per-technique classification thresholds calibrated on the v6 held-out set
# (230 clips from 2126-clip corpus).  Model trained on RMVPE F0.
# Derived from scripts/threshold_sweep.py to maximise per-class F1.
# Macro-F1 at these thresholds: 0.541  (vs. 0.520 at flat 0.50).
# Re-run threshold_sweep.py after any retraining.
TECH_THRESHOLDS: dict[str, float] = {
    "bubble":     0.05,  # rare class; pos_mean=0.015, low bar needed
    "breathe":    0.80,
    "pharyngeal": 0.90,
    "vibrato":    0.05,  # rare class; pos_mean=0.199
    "glissando":  0.65,
    "mixed":      0.15,
    "falsetto":   0.40,
    "weak":       0.90,
    "strong":     0.15,
}


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


_LOADED: dict[str, tuple[StudentSTARS, list[str], str]] = {}


def _resolve_ckpt(student_dir: Path) -> Path:
    student_dir = Path(student_dir)
    for name in DEFAULT_STUDENT_CKPT_CANDIDATES:
        path = student_dir / name
        if path.is_file():
            return path
    matches = sorted(student_dir.glob("model_ckpt_steps_*.pt"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(
        f"No student checkpoint found in {student_dir}. Train one with "
        "`python scripts/train_student_stars.py`."
    )


def _resolve_device(req: str) -> str:
    if req.lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return req


def load_student(
    student_dir: Path = DEFAULT_STUDENT_DIR,
    device: str = "cuda",
) -> tuple[StudentSTARS, list[str], str]:
    """Load (or return cached) student model + phone vocab + device."""
    ckpt_path = _resolve_ckpt(student_dir)
    cache_key = f"{ckpt_path}:{device}"
    if cache_key in _LOADED:
        return _LOADED[cache_key]

    resolved_device = _resolve_device(device)
    model, phone_vocab = StudentSTARS.load_checkpoint(ckpt_path, map_location=resolved_device)
    model.to(resolved_device).eval()
    _LOADED[cache_key] = (model, phone_vocab, resolved_device)
    return model, phone_vocab, resolved_device


# ---------------------------------------------------------------------------
# Feature extraction (must match scripts/export_student_dataset.py)
# ---------------------------------------------------------------------------


def _compute_features(
    wav_path: Path,
    *,
    sample_rate: int = STARS_BILINGUAL_SAMPLE_RATE,
    hop_length: int = 128,
    n_mels: int = 80,
    n_fft: int = 512,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mel: (T, n_mels) log-mel, f0: (T,) Hz)`` on the student grid.

    F0 is extracted with RMVPE (same as the training corpus), guaranteeing
    identical feature distributions at train and inference time.  The old
    ``librosa.pyin`` path caused saturation because its voiced-frame density
    (~82%) was far higher than NanoPitch (~27%), pushing the technique head out
    of distribution.  RMVPE is the pitch estimator STARS itself uses internally,
    so it is both consistent and GPU-accelerated (~12 clips/s).
    """
    import librosa

    from vocal_coach.rmvpe_f0 import extract_f0_rmvpe

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
# Inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_student(
    *,
    metadata_path: Path,
    sample_id: str,
    item_name: Optional[str] = None,
    student_dir: Path = DEFAULT_STUDENT_DIR,
    device: str = "cuda",
    hop_seconds: float = STARS_BILINGUAL_HOP_SECONDS,
) -> StarsTrack:
    """Run the student on a single audio clip referenced by ``metadata_path``.

    Matches the contract of ``vocal_coach.stars_runner.run_stars`` so callers
    can swap between profiles by changing one function call.  F0 is always
    extracted with RMVPE so the inference feature is identical to the training
    feature (``scripts/regen_f0.py`` regenerated the corpus using the same
    ``vocal_coach.rmvpe_f0`` module).
    """
    raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"Empty stars metadata at {metadata_path}")
    if item_name is not None:
        entry = next((e for e in raw if e.get("item_name") == item_name), raw[0])
    else:
        entry = raw[0]
    entry = StarsMetadataEntry.model_validate(entry)
    wav_path = Path(entry.wav_fn)
    if not wav_path.is_file():
        raise FileNotFoundError(f"User wav not found: {wav_path}")

    model, phone_vocab, resolved_device = load_student(student_dir, device=device)
    phone_to_id = {p: i for i, p in enumerate(phone_vocab)}

    mel_np, f0_np = _compute_features(wav_path, device=resolved_device)
    mel = torch.from_numpy(mel_np).unsqueeze(0).to(resolved_device)
    f0 = torch.from_numpy(f0_np).unsqueeze(0).to(resolved_device)

    out = model(mel, f0, mask=None)
    phoneme_logits = out.phoneme_logits.squeeze(0).detach().cpu().numpy()
    # h stays on device so we can pool it per phoneme span for technique prediction.
    h = out.h  # (1, T, d_model)

    # Drop silence from the alignment target, like training, then re-insert
    # spans for the silence tokens by snapping to neighbour boundaries.
    target_phones = [p for p in entry.ph if p not in SILENCE_TOKENS]
    target_ids = [phone_to_id.get(p, phone_to_id.get("<UNK>", 0)) for p in target_phones]
    aligned_spans = viterbi_align_phones(
        phoneme_logits,
        target_phone_ids=target_ids,
        allow_blank=True,
        blank_index=BLANK_INDEX,
    )

    # Stitch back the original phone order with <SP>/<AP> tokens.
    spans_with_silence: list[tuple[str, tuple[int, int]]] = []
    nonsil_idx = 0
    prev_end = 0
    for ph in entry.ph:
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

    # Build phonemes.  Technique flags are predicted by pooling the Conformer
    # body output h over each phoneme span and running the technique head.
    # When the model was trained with phoneme_level_tech=True this matches
    # the training regime exactly.  For legacy checkpoints (phoneme_level_tech
    # =False) we fall back to pooling frame-level sigmoid probabilities.
    use_phoneme_level = getattr(model.config, "phoneme_level_tech", False)
    if not use_phoneme_level:
        # Legacy fallback: pool sigmoid probs from the frame-level head.
        _tech_probs_np = torch.sigmoid(out.technique_logits.squeeze(0)).detach().cpu().numpy()

    phonemes: list[StarsPhoneme] = []
    word_list = list(entry.word)
    ph2word = list(entry.ph2words)
    for i, (ph, (start_f, end_f)) in enumerate(spans_with_silence):
        start_s = float(start_f * hop_seconds)
        end_s = float(end_f * hop_seconds)
        techniques: dict[str, int] = {}
        if end_f > start_f:
            if use_phoneme_level:
                # Pool h, run technique head, threshold.
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

        w_idx = ph2word[i] if i < len(ph2word) else -1
        if 0 <= w_idx < len(word_list):
            word = word_list[w_idx]
        else:
            word = ph if ph in SILENCE_TOKENS else ""
        if ph in SILENCE_TOKENS:
            w_idx = -1
        phonemes.append(
            StarsPhoneme(
                index=i,
                phoneme=ph,
                word=word,
                word_index=w_idx,
                start_s=start_s,
                end_s=end_s,
                techniques=techniques,
            )
        )

    style = StarsStyle(
        language="unknown",
        gender="unknown",
        emotion="unknown",
        method="unknown",
        pace="unknown",
        range="unknown",
        technique_group="unknown",
    )
    return StarsTrack(
        sample_id=sample_id,
        sample_rate=STARS_BILINGUAL_SAMPLE_RATE,
        hop_seconds=hop_seconds,
        style=style,
        phonemes=phonemes,
        notes=[],  # student does not transcribe notes; pipeline uses UltraStar
    )


__all__ = [
    "DEFAULT_STUDENT_DIR",
    "load_student",
    "run_student",
]
