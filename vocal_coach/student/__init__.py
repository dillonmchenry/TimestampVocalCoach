"""Distilled STARS student model.

Two-head Conformer trained to mimic the STARS bilingual teacher's:

  1. per-frame phoneme logits + per-frame boundary probability (alignment), and
  2. per-phoneme 9-way technique multi-label classification.

Global style classification heads from the original STARS paper are explicitly
NOT modeled (see Sprint 3 plan: "Workstream A — STARS student model"); the
user-side analysis pipeline never consumes them.
"""

from vocal_coach.student.model import (
    STUDENT_TECH_NAMES,
    StudentConfig,
    StudentSTARS,
    StudentOutputs,
)
from vocal_coach.student.align import (
    viterbi_align_phones,
    spans_from_alignment,
)

__all__ = [
    "STUDENT_TECH_NAMES",
    "StudentConfig",
    "StudentOutputs",
    "StudentSTARS",
    "spans_from_alignment",
    "viterbi_align_phones",
]
