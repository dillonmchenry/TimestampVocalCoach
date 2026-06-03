"""Train the STARS student model from the teacher-exported corpus.

Inputs
------

    data/student_corpus/manifest.jsonl
        Produced by ``scripts/export_student_dataset.py``.
    configs/student.yaml
        Architecture + training knobs.

Outputs
-------

    stars_student/model_ckpt_steps_<N>.pt
    stars_student/best.pt
    stars_student/phone_vocab.json

The two-task objective is exactly what the plan specifies:

    L = phoneme_ctc + boundary_bce + technique_bce

There are no style classification losses (student has no style heads).

Usage::

    python scripts/train_student_stars.py
    python scripts/train_student_stars.py --config configs/student.yaml \\
        --corpus data/student_corpus --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vocal_coach.student.align import BLANK_INDEX  # noqa: E402
from vocal_coach.student.model import (  # noqa: E402
    STUDENT_TECH_NAMES,
    StudentConfig,
    StudentSTARS,
)


SILENCE_TOKENS = {"<SP>", "<AP>", "<UNK>"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class CorpusItem:
    clip_id: str
    feature_dir: Path
    num_frames: int
    split: str


def _load_manifest(manifest_path: Path) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        items.append(
            CorpusItem(
                clip_id=rec["clip_id"],
                feature_dir=Path(rec["feature_dir"]),
                num_frames=int(rec["num_frames"]),
                split=rec.get("split", "train"),
            )
        )
    return items


def _build_phone_vocab(corpus_root: Path, items: list[CorpusItem]) -> list[str]:
    """Walk every clip's labels.json and produce a stable vocab.

    Vocab order:
        index 0  : "<blank>"   (CTC blank)
        index 1  : "<SP>"
        index 2  : "<AP>"
        ...      : the rest in sorted order
    """
    seen: set[str] = set()
    for item in items:
        path = corpus_root / item.feature_dir / "labels.json"
        if not path.is_file():
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))
        for ph in labels.get("phones", []):
            seen.add(ph)
    seen.discard("<blank>")
    fixed_prefix = ["<blank>", "<SP>", "<AP>", "<UNK>"]
    rest = sorted(p for p in seen if p not in set(fixed_prefix))
    return fixed_prefix + rest


class StudentDataset(Dataset):
    """One item = one clip; we crop or pad to ``max_frames`` per __getitem__."""

    def __init__(
        self,
        corpus_root: Path,
        items: list[CorpusItem],
        phone_to_id: dict[str, int],
        n_techniques: int,
        max_frames: int,
        augment_mel_noise_std: float = 0.0,
    ):
        self.corpus_root = Path(corpus_root)
        self.items = items
        self.phone_to_id = phone_to_id
        self.n_techniques = n_techniques
        self.max_frames = int(max_frames)
        self.augment_mel_noise_std = float(augment_mel_noise_std)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        feat_dir = self.corpus_root / item.feature_dir
        mel = np.load(feat_dir / "mel.npy")  # (T, n_mels)
        f0 = np.load(feat_dir / "f0.npy")    # (T,)
        labels = json.loads((feat_dir / "labels.json").read_text(encoding="utf-8"))
        boundaries = np.asarray(labels["boundaries"], dtype=np.float32)  # (T,)

        T = min(mel.shape[0], f0.shape[0], boundaries.shape[0])
        mel = mel[:T]
        f0 = f0[:T]
        boundaries = boundaries[:T]

        ph_start_frames = list(labels.get("ph_start_frames", []))
        phones = list(labels.get("phones", []))
        techniques = labels.get("techniques", {})
        K = self.n_techniques

        # Per-phoneme technique labels: (L_ph, K).
        # This is the ground-truth label for each phoneme in the clip.
        n_phones = len(ph_start_frames)
        ph_tech = np.zeros((n_phones, K), dtype=np.float32)
        for p_idx in range(n_phones):
            for k, name in enumerate(STUDENT_TECH_NAMES[:K]):
                seq = techniques.get(name, [])
                if p_idx < len(seq) and seq[p_idx]:
                    ph_tech[p_idx, k] = 1.0

        # CTC target: drop silence so the alignment is on real phones only.
        target_phone_ids = [
            self.phone_to_id.get(p, self.phone_to_id.get("<UNK>", 0))
            for p in phones
            if p not in SILENCE_TOKENS
        ]
        if not target_phone_ids:
            target_phone_ids = [self.phone_to_id.get("<UNK>", 0)]

        # Random crop if longer than max_frames.
        crop_start = 0
        if T > self.max_frames:
            crop_start = random.randint(0, T - self.max_frames)
            mel = mel[crop_start : crop_start + self.max_frames]
            f0 = f0[crop_start : crop_start + self.max_frames]
            boundaries = boundaries[crop_start : crop_start + self.max_frames]
            T = self.max_frames
            # Keep only phonemes whose start frame falls within the crop window.
            crop_end_abs = crop_start + T
            keep = [
                (i, fs - crop_start)
                for i, fs in enumerate(ph_start_frames)
                if crop_start <= fs < crop_end_abs
            ]
            if keep:
                kept_idxs, adjusted_starts = zip(*keep)
                ph_start_frames = list(adjusted_starts)
                ph_tech = ph_tech[list(kept_idxs)]
            else:
                ph_start_frames = []
                ph_tech = np.zeros((0, K), dtype=np.float32)

        if self.augment_mel_noise_std > 0:
            mel = mel + np.random.randn(*mel.shape).astype(np.float32) * self.augment_mel_noise_std

        return {
            "mel": torch.from_numpy(mel.astype(np.float32)),
            "f0": torch.from_numpy(f0.astype(np.float32)),
            "boundaries": torch.from_numpy(boundaries.astype(np.float32)),
            "ph_start_frames": ph_start_frames,                              # list[int]
            "ph_tech_labels": torch.from_numpy(ph_tech.astype(np.float32)), # (L_ph, K)
            "target_phones": torch.tensor(target_phone_ids, dtype=torch.long),
            "n_frames": T,
            "clip_id": item.clip_id,
        }


def _collate(batch: list[dict]) -> dict:
    """Pad mel/f0/boundaries to the longest item; keep phoneme lists ragged."""
    T_max = max(b["n_frames"] for b in batch)
    n_mels = batch[0]["mel"].shape[-1]
    B = len(batch)

    mel = torch.zeros((B, T_max, n_mels), dtype=torch.float32)
    f0 = torch.zeros((B, T_max), dtype=torch.float32)
    boundaries = torch.zeros((B, T_max), dtype=torch.float32)
    mask = torch.ones((B, T_max), dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)

    target_phones: list[torch.Tensor] = []
    target_lengths: list[int] = []
    clip_ids: list[str] = []
    ph_start_frames: list[list[int]] = []    # ragged: one list[int] per clip
    ph_tech_labels: list[torch.Tensor] = []  # ragged: one (L_ph, K) per clip

    for i, b in enumerate(batch):
        T = b["n_frames"]
        mel[i, :T] = b["mel"]
        f0[i, :T] = b["f0"]
        boundaries[i, :T] = b["boundaries"]
        mask[i, :T] = False  # False = valid
        lengths[i] = T
        target_phones.append(b["target_phones"])
        target_lengths.append(int(b["target_phones"].shape[0]))
        clip_ids.append(b["clip_id"])
        ph_start_frames.append(b["ph_start_frames"])
        ph_tech_labels.append(b["ph_tech_labels"])

    target_phones_cat = torch.cat(target_phones, dim=0) if target_phones else torch.zeros(0, dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

    return {
        "mel": mel,
        "f0": f0,
        "boundaries": boundaries,
        "mask": mask,
        "lengths": lengths,
        "target_phones": target_phones_cat,
        "target_lengths": target_lengths_t,
        "clip_ids": clip_ids,
        "ph_start_frames": ph_start_frames,  # list[list[int]]
        "ph_tech_labels": ph_tech_labels,     # list[Tensor(L_ph, K)]
    }


# ---------------------------------------------------------------------------
# Loss + step
# ---------------------------------------------------------------------------


def _ctc_loss(
    phoneme_logits: torch.Tensor,
    target_phones: torch.Tensor,
    lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = BLANK_INDEX,
) -> torch.Tensor:
    # CTCLoss wants (T, B, C) log-probs.
    log_probs = F.log_softmax(phoneme_logits, dim=-1).transpose(0, 1)
    ctc = torch.nn.CTCLoss(blank=blank, zero_infinity=True)
    return ctc(log_probs, target_phones, lengths, target_lengths)


def _compute_tech_class_weights(
    corpus_root: Path,
    items: list[CorpusItem],
    n_techniques: int,
    cap: float = 20.0,
) -> torch.Tensor:
    """Compute per-technique positive class weights from the training corpus.

    Returns a (K,) tensor where weight[k] = min(cap, neg_k / pos_k).
    Techniques with no positive examples get weight=cap to still push the
    model toward them slightly.
    """
    pos = np.zeros(n_techniques, dtype=np.float64)
    total = np.zeros(n_techniques, dtype=np.float64)
    for item in items:
        path = corpus_root / item.feature_dir / "labels.json"
        if not path.is_file():
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))
        techniques = labels.get("techniques", {})
        for k, name in enumerate(STUDENT_TECH_NAMES[:n_techniques]):
            seq = techniques.get(name, [])
            pos[k] += sum(seq)
            total[k] += len(seq)
    neg = total - pos
    weights = np.where(pos > 0, neg / pos, cap)
    weights = np.clip(weights, 1.0, cap)
    return torch.tensor(weights, dtype=torch.float32)


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE with logits, ignoring positions where ``mask`` is True (PAD).

    ``pos_weight`` is a (K,) tensor of per-class positive weights as accepted
    by ``F.binary_cross_entropy_with_logits``; pass None for unweighted BCE.
    """
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    # Broadcast mask over the last dim if present.
    while loss.ndim > mask.ndim:
        mask = mask.unsqueeze(-1)
    loss = loss.masked_fill(mask, 0.0)
    denom = (~mask).sum().clamp_min(1).float()
    return loss.sum() / denom


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "student.yaml")
    p.add_argument("--corpus", type=Path, default=None, help="Override data.corpus_dir from the config.")
    p.add_argument("--output-dir", type=Path, default=None, help="Override checkpoint.out_dir.")
    p.add_argument("--device", default="cuda", help='"cuda", "cpu", or "cuda:N".')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint .pt.")
    return p.parse_args()


def _resolve_device(req: str) -> str:
    if req.lower() == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return req


def _data_checkpoint_or_exit(manifest_path: Path) -> None:
    if manifest_path.is_file() and manifest_path.stat().st_size > 0:
        return
    print(
        "[train] DATA CHECKPOINT — student corpus not found.\n\n"
        f"  expected: {manifest_path}\n\n"
        "Run `python scripts/export_student_dataset.py` first against at\n"
        "least one of:\n"
        "  - GTSinger English (HF dataset AaronZ345/GTSinger)\n"
        "  - NUS-48E (https://smsl.comp.nus.edu.sg/NUS48E/)\n"
        "  - data/songs/<song_id>/ bundles built via scripts/import_ultrastar.py\n",
        file=sys.stderr,
    )
    sys.exit(78)  # EX_CONFIG


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus_dir = Path(args.corpus or cfg["data"]["corpus_dir"]).resolve()
    output_dir = Path(args.output_dir or cfg["checkpoint"]["out_dir"]).resolve()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = corpus_dir / cfg["data"]["manifest"]
    _data_checkpoint_or_exit(manifest_path)

    items = _load_manifest(manifest_path)
    include = set(cfg["data"].get("include_sources") or [])
    if include:
        items = [it for it in items if (corpus_dir / it.feature_dir).parent.name in include]
    train_items = [it for it in items if it.split == "train"]
    held_items = [it for it in items if it.split == "held_out"]
    print(f"[train] corpus: train={len(train_items)} held_out={len(held_items)}")
    if not train_items:
        print("[train] no training items; aborting", file=sys.stderr)
        return 78

    # Compute per-technique positive class weights from training data to
    # counteract the severe imbalance in rare technique classes (vibrato,
    # bubble, breathe, etc. are positive in <5% of phonemes).
    pos_weight_cap = float(cfg["loss"].get("tech_pos_weight_cap", 20.0))
    tech_pos_weight = _compute_tech_class_weights(
        corpus_dir, train_items, len(STUDENT_TECH_NAMES), cap=pos_weight_cap
    )
    print(f"[train] technique pos_weight (cap={pos_weight_cap}):")
    for k, name in enumerate(STUDENT_TECH_NAMES):
        print(f"  {name:>12}: {tech_pos_weight[k]:.1f}x")

    # Build phone vocab from the full corpus (train+held).
    phone_vocab = _build_phone_vocab(corpus_dir, items)
    phone_to_id = {p: i for i, p in enumerate(phone_vocab)}
    print(f"[train] phone vocab size: {len(phone_vocab)}")

    # Build dataset + loader.
    ds = StudentDataset(
        corpus_root=corpus_dir,
        items=train_items,
        phone_to_id=phone_to_id,
        n_techniques=len(STUDENT_TECH_NAMES),
        max_frames=int(cfg["data"]["max_frames"]),
        augment_mel_noise_std=float(cfg["data"].get("augment_mel_noise_std", 0.0)),
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=False,
    )

    # Build model.
    model_cfg = StudentConfig(
        n_mels=int(cfg["model"]["n_mels"]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_blocks=int(cfg["model"]["n_blocks"]),
        conv_kernel=int(cfg["model"]["conv_kernel"]),
        ff_expansion=int(cfg["model"]["ff_expansion"]),
        dropout=float(cfg["model"]["dropout"]),
        phone_vocab_size=len(phone_vocab),
        n_techniques=len(STUDENT_TECH_NAMES),
        phoneme_level_tech=bool(cfg["model"].get("phoneme_level_tech", False)),
    )
    phoneme_level_tech = model_cfg.phoneme_level_tech
    print(f"[train] phoneme_level_tech: {phoneme_level_tech}")
    model = StudentSTARS(model_cfg)
    device = _resolve_device(args.device)
    model.to(device)
    print(f"[train] device: {device}")
    print(f"[train] params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer + scheduler.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    warmup_steps = int(cfg["train"]["warmup_steps"])
    max_steps = int(cfg["train"]["max_steps"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = bool(cfg["train"].get("amp", False)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(f"[train] resumed from {args.resume} at step {start_step}")

    # Persist phone vocab once so the runner can reload it standalone.
    (output_dir / "phone_vocab.json").write_text(
        json.dumps(phone_vocab, indent=2), encoding="utf-8"
    )

    step = start_step
    loss_log: dict[str, list[float]] = {"total": [], "ctc": [], "boundary": [], "tech": []}
    model.train()

    keep_last = int(cfg["checkpoint"].get("keep_last", 3))
    saved: list[Path] = []
    best_loss = float("inf")

    # Loop until max_steps.
    while step < max_steps:
        for batch in dl:
            step += 1
            mel = batch["mel"].to(device, non_blocking=True)
            f0 = batch["f0"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            boundaries = batch["boundaries"].to(device, non_blocking=True)
            target_phones = batch["target_phones"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)

            ph_start_frames_batch = batch["ph_start_frames"]   # list[list[int]]
            ph_tech_labels_batch  = batch["ph_tech_labels"]    # list[Tensor(L_ph,K)]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(mel, f0, mask=mask)
                ctc = _ctc_loss(
                    out.phoneme_logits,
                    target_phones=target_phones,
                    lengths=lengths,
                    target_lengths=target_lengths,
                )
                # Boundary head returns (B, T, 1); squeeze for BCE
                bnd = _masked_bce_with_logits(
                    out.boundary_logits.squeeze(-1),
                    boundaries,
                    mask,
                )

                if phoneme_level_tech:
                    # Pool Conformer h over GT phoneme spans, then predict.
                    # All pooled vectors are stacked into a flat (N_phones, d)
                    # tensor and passed through technique_head in one shot so
                    # gradients flow back through both the pooling and the head.
                    ph_h_list: list[torch.Tensor] = []
                    ph_lbl_list: list[torch.Tensor] = []
                    B_cur = mel.shape[0]
                    for b_i in range(B_cur):
                        starts = ph_start_frames_batch[b_i]
                        ph_lbl = ph_tech_labels_batch[b_i].to(device)  # (L_ph, K)
                        T_b = int(lengths[b_i])
                        n_ph = len(starts)
                        for p_i, sf in enumerate(starts):
                            ef = starts[p_i + 1] if p_i + 1 < n_ph else T_b
                            sf = int(max(0, min(T_b - 1, sf)))
                            ef = int(max(sf + 1, min(T_b, ef)))
                            ph_h_list.append(out.h[b_i, sf:ef].mean(dim=0))
                        if n_ph > 0:
                            ph_lbl_list.append(ph_lbl)
                    if ph_h_list:
                        ph_h_t = torch.stack(ph_h_list, dim=0)           # (N_ph, d)
                        ph_lbl_t = torch.cat(ph_lbl_list, dim=0)         # (N_ph, K)
                        ph_logits = model.technique_head(ph_h_t)         # (N_ph, K)
                        tch = F.binary_cross_entropy_with_logits(
                            ph_logits,
                            ph_lbl_t,
                            pos_weight=tech_pos_weight.to(device),
                        )
                    else:
                        tch = torch.tensor(0.0, device=device)
                else:
                    # Legacy: frame-level technique BCE (v3 and earlier).
                    techniques = batch["techniques"].to(device, non_blocking=True)
                    tch = _masked_bce_with_logits(
                        out.technique_logits,
                        techniques,
                        mask,
                        pos_weight=tech_pos_weight.to(device),
                    )

                loss = (
                    float(cfg["loss"]["phoneme_ctc_weight"]) * ctc
                    + float(cfg["loss"]["boundary_bce_weight"]) * bnd
                    + float(cfg["loss"]["technique_bce_weight"]) * tch
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_log["total"].append(float(loss.detach()))
            loss_log["ctc"].append(float(ctc.detach()))
            loss_log["boundary"].append(float(bnd.detach()))
            loss_log["tech"].append(float(tch.detach()))

            if step % int(cfg["train"]["log_every"]) == 0:
                avg_total = sum(loss_log["total"][-100:]) / max(1, len(loss_log["total"][-100:]))
                avg_ctc = sum(loss_log["ctc"][-100:]) / max(1, len(loss_log["ctc"][-100:]))
                avg_bnd = sum(loss_log["boundary"][-100:]) / max(1, len(loss_log["boundary"][-100:]))
                avg_tch = sum(loss_log["tech"][-100:]) / max(1, len(loss_log["tech"][-100:]))
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] step={step:6d} lr={lr:.2e} "
                    f"loss={avg_total:.4f} ctc={avg_ctc:.4f} bnd={avg_bnd:.4f} tch={avg_tch:.4f}"
                )

            if step % int(cfg["train"]["save_every"]) == 0:
                ckpt_path = output_dir / f"model_ckpt_steps_{step}.pt"
                model.save_checkpoint(ckpt_path, phone_vocab=phone_vocab, extra={"step": step})
                saved.append(ckpt_path)
                while len(saved) > keep_last:
                    old = saved.pop(0)
                    if old.is_file():
                        old.unlink()
                print(f"[train] checkpoint saved -> {ckpt_path}")
                if loss_log["total"]:
                    recent = sum(loss_log["total"][-100:]) / max(1, len(loss_log["total"][-100:]))
                    if recent < best_loss:
                        best_loss = recent
                        best_path = output_dir / "best.pt"
                        model.save_checkpoint(best_path, phone_vocab=phone_vocab, extra={"step": step})
                        print(f"[train]   new best -> {best_path} (loss={best_loss:.4f})")

            if step >= max_steps:
                break

    # Final checkpoint
    final_path = output_dir / "final.pt"
    model.save_checkpoint(final_path, phone_vocab=phone_vocab, extra={"step": step})
    print(f"[train] final checkpoint -> {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
