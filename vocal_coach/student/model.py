"""Compact 2-layer Conformer student that mimics STARS phoneme alignment + techniques.

Architecture (intentionally smaller and simpler than the teacher's
hierarchical CMU encoder + FreqMOE stack so it can run in-process on CPU
during interactive demos):

    Input:
        mel : (B, T, n_mels=80)         log-mel at 24 kHz / 128-hop
        f0  : (B, T)                    F0 in Hz (0 == unvoiced)

    Front-end:
        mel -> Linear(80, d) + LayerNorm
        f0  -> log1p(f0) -> Linear(1, d)
        h   = mel_proj + f0_proj + positional_embedding(T)

    Body:
        2x ConformerBlock(d, heads=4, conv_kernel=15)

    Heads (each is a small MLP on the body output):
        phoneme_logits     : (B, T, P)    P = phone vocab size (incl. blank)
        boundary_logits    : (B, T, 1)    BCE for "is this frame a phoneme start"
        technique_logits   : (B, T, 9)    frame-level BCE (legacy / fallback)

    When ``phoneme_level_tech=True`` (v4+), the technique head is NOT applied
    frame-by-frame.  Instead, the caller pools ``StudentOutputs.h`` over
    Viterbi-derived (inference) or GT (training) phoneme spans and passes the
    pooled (L_ph, d) tensor to ``model.technique_head`` directly.  This
    decouples technique detection from frame-level alignment noise.

The 9 technique names match STARS exactly so ``StarsTrack`` objects can be
built directly from this model's outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocal_coach.schemas import STARS_TECH_NAMES


STUDENT_TECH_NAMES: list[str] = list(STARS_TECH_NAMES)


# ---------------------------------------------------------------------------
# Conformer block (minimal, no convolutional subsampling needed)
# ---------------------------------------------------------------------------


class FeedForwardModule(nn.Module):
    """Half-step feed-forward block from the Conformer paper."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConvModule(nn.Module):
    """Depthwise-separable 1D conv module for the Conformer block."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        # Depthwise
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x).transpose(1, 2)  # (B, D, T)
        x = self.pointwise1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return residual + x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        # mask is (B, T) True at PAD positions; convert to key_padding_mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        return residual + self.dropout(attn_out)


class ConformerBlock(nn.Module):
    """Half-step FFN -> MHSA -> Conv -> Half-step FFN -> LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        conv_kernel: int = 15,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, mask=mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.ln(x)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def _sinusoidal_positional_encoding(length: int, d_model: int, device, dtype) -> torch.Tensor:
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------


@dataclass
class StudentConfig:
    """Hyperparameters for the student.

    The phoneme vocabulary is configurable; STARS's bilingual phone set has
    ~370 phones (including ``<SP>`` / ``<AP>`` / blank). The training script
    discovers the actual set from the labels and writes it into the
    checkpoint so the runner can rebuild the exact mapping at inference time.

    ``phoneme_level_tech=True`` (v4+): the technique head is trained on
    mean-pooled phoneme representations rather than individual frames.  At
    inference, call ``model.technique_head`` on spans pooled from ``h``
    instead of reading ``technique_logits`` from the forward output.
    """

    n_mels: int = 80
    d_model: int = 192
    n_heads: int = 4
    n_blocks: int = 2
    conv_kernel: int = 15
    ff_expansion: int = 4
    dropout: float = 0.1
    phone_vocab_size: int = 512  # placeholder; set from training data
    n_techniques: int = 9
    max_positional_length: int = 8192
    phoneme_level_tech: bool = False  # True for v4+ phoneme-pooled technique head


@dataclass
class StudentOutputs:
    """Forward-pass outputs.

    ``phoneme_logits`` and ``boundary_logits`` are always at the input frame
    rate.  ``technique_logits`` is frame-level (legacy; only meaningful when
    ``config.phoneme_level_tech=False``).  ``h`` is the Conformer body output
    and is used by the caller to compute phoneme-level technique predictions
    when ``config.phoneme_level_tech=True``.
    """

    phoneme_logits: torch.Tensor      # (B, T, P) — CTC-style logits
    boundary_logits: torch.Tensor     # (B, T, 1)
    technique_logits: torch.Tensor    # (B, T, K) — frame-level (legacy)
    h: torch.Tensor                   # (B, T, d_model) — Conformer body output


class StudentSTARS(nn.Module):
    """Two-head Conformer student.

    Inputs:
        mel : (B, T, n_mels)
        f0  : (B, T) Hz
        mask: optional (B, T) bool; True at PAD positions.
    """

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config

        self.mel_proj = nn.Sequential(
            nn.Linear(config.n_mels, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        # Single scalar (log1p Hz) projected into d_model.
        self.f0_proj = nn.Linear(1, config.d_model)

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    conv_kernel=config.conv_kernel,
                    ff_expansion=config.ff_expansion,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

        # Cached positional embeddings.
        self.register_buffer(
            "_positional_buf",
            _sinusoidal_positional_encoding(
                config.max_positional_length,
                config.d_model,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

        # Heads.
        self.phoneme_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.phone_vocab_size),
        )
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.technique_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.n_techniques),
        )

    def _positional(self, T: int, device, dtype) -> torch.Tensor:
        if T <= self._positional_buf.shape[0]:
            return self._positional_buf[:T].to(device=device, dtype=dtype)
        # Long input: regenerate on the fly.
        return _sinusoidal_positional_encoding(T, self.config.d_model, device, dtype)

    def forward(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> StudentOutputs:
        B, T, _ = mel.shape
        h_mel = self.mel_proj(mel)
        # log1p(f0) is a gentle scale; unvoiced (0) -> 0.
        f0_in = torch.log1p(f0.clamp_min(0.0)).unsqueeze(-1)
        h_f0 = self.f0_proj(f0_in)
        h = h_mel + h_f0 + self._positional(T, mel.device, mel.dtype).unsqueeze(0)

        for block in self.blocks:
            h = block(h, mask=mask)

        return StudentOutputs(
            phoneme_logits=self.phoneme_head(h),
            boundary_logits=self.boundary_head(h),
            technique_logits=self.technique_head(h),
            h=h,
        )

    # ------------------------------------------------------------------
    # Checkpoint serialization helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, *, phone_vocab: list[str], extra: Optional[dict] = None) -> None:
        """Write the model + config + phone vocab to ``path`` as a single .pt file."""
        payload = {
            "state_dict": self.state_dict(),
            "config": self.config.__dict__,
            "phone_vocab": list(phone_vocab),
            "tech_names": STUDENT_TECH_NAMES,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    @classmethod
    def load_checkpoint(cls, path, *, map_location="cpu") -> tuple["StudentSTARS", list[str]]:
        """Load a checkpoint written by ``save_checkpoint``."""
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        config = StudentConfig(**payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, list(payload["phone_vocab"])


__all__ = [
    "ConformerBlock",
    "STUDENT_TECH_NAMES",
    "StudentConfig",
    "StudentOutputs",
    "StudentSTARS",
]
