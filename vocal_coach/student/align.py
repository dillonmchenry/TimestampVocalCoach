"""Viterbi forced alignment for the student model.

At inference time we know:
  * the student's per-frame phoneme logits (B=1, T, P), and
  * the *target* phoneme sequence (from the song's reference annotation -
    same one we feed to the teacher today via `stars_metadata.json`).

We need to find the most likely frame-to-phoneme alignment under the
monotonic constraint that the phoneme sequence is preserved in order.

The DP is the standard "CTC-style" forced alignment with optional blank
insertion. Each phoneme in the target sequence becomes a state; we allow
a self-loop (stay on the same phoneme) and a forward transition (move to
the next phoneme). The blank token is permitted between any two phonemes
to model silence / sub-phoneme acoustic transitions.

Output: a list of ``(start_frame, end_frame_exclusive)`` per phoneme.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Sentinel index for the CTC blank. We assume the blank lives at index 0 of
# the phoneme vocabulary, matching the convention chosen in the training
# script (PHONE_VOCAB[0] == "<blank>").
BLANK_INDEX = 0


def _safe_log(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.log(np.maximum(probs, eps))


def viterbi_align_phones(
    phoneme_logits: np.ndarray,
    target_phone_ids: list[int],
    *,
    allow_blank: bool = True,
    blank_index: int = BLANK_INDEX,
) -> list[tuple[int, int]]:
    """Align frames to a target phoneme sequence using monotonic Viterbi DP.

    Parameters
    ----------
    phoneme_logits
        ``(T, P)`` float array of unnormalized per-frame phoneme logits.
    target_phone_ids
        Length-``L_p`` list of vocabulary indices for the phoneme sequence we
        want to align (the silence/<SP> tokens should be included if you
        want their spans returned).
    allow_blank
        When True, inserts an optional blank state between every pair of
        phonemes. This is the standard CTC-style alignment lattice.
    blank_index
        Vocabulary index for the blank token (defaults to ``BLANK_INDEX``).

    Returns
    -------
    spans
        List of ``(start_frame, end_frame_exclusive)`` tuples, one per entry
        of ``target_phone_ids``. Blank spans (when inserted) are NOT
        returned, only the requested phoneme spans.
    """
    T, P = phoneme_logits.shape
    L_p = len(target_phone_ids)
    if L_p == 0 or T == 0:
        return []

    # Convert logits to log-probs.
    logp = phoneme_logits - phoneme_logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))

    # Build the state sequence: interleave blanks if requested.
    if allow_blank:
        states: list[int] = [blank_index]
        for ph in target_phone_ids:
            states.append(int(ph))
            states.append(blank_index)
    else:
        states = [int(p) for p in target_phone_ids]
    S = len(states)

    NEG_INF = -1e30
    score = np.full((T, S), NEG_INF, dtype=np.float64)
    backptr = np.zeros((T, S), dtype=np.int32)  # 0 = stay, 1 = skip-from-prev,
    #                                              2 = skip-from-prev-prev (only valid for ctc with non-blank same neighbor)

    # Initialization: at t=0, we can be at state 0 (blank) or state 1 (first
    # real phone). For non-CTC mode (no blank), only state 0.
    score[0, 0] = logp[0, states[0]]
    if S > 1:
        score[0, 1] = logp[0, states[1]]

    for t in range(1, T):
        # Vectorize as 1-D operations over states.
        # Three predecessor options per state s:
        #   stay:        s     (self-loop)
        #   move:        s - 1
        #   skip-blank:  s - 2   (only when states[s-1] is blank AND states[s] != states[s-2])
        prev_self = score[t - 1, :]                                  # (S,)
        prev_left = np.concatenate(
            ([NEG_INF], score[t - 1, :-1])
        )                                                            # (S,) score[t-1, s-1]
        prev_skip = np.concatenate(
            ([NEG_INF, NEG_INF], score[t - 1, :-2])
        )                                                            # (S,) score[t-1, s-2]

        # Validity of the skip transition: only allowed when allow_blank
        # AND states[s-1] == blank AND states[s-2] != states[s].
        if allow_blank:
            skip_valid = np.zeros(S, dtype=bool)
            for s in range(2, S):
                if states[s - 1] == blank_index and states[s - 2] != states[s]:
                    skip_valid[s] = True
            prev_skip = np.where(skip_valid, prev_skip, NEG_INF)
        else:
            prev_skip = np.full(S, NEG_INF)

        candidates = np.stack([prev_self, prev_left, prev_skip], axis=0)  # (3, S)
        best = candidates.max(axis=0)
        which = candidates.argmax(axis=0).astype(np.int32)

        emit = np.array([logp[t, s_idx] for s_idx in states], dtype=np.float64)
        score[t] = best + emit
        backptr[t] = which

    # Backtrace from the best terminal state (last real phoneme, or last blank).
    end_candidates = [S - 1]
    if allow_blank and S >= 2:
        end_candidates.append(S - 2)
    best_end = max(end_candidates, key=lambda s: score[T - 1, s])

    path = [best_end] * T
    s = best_end
    for t in range(T - 1, 0, -1):
        move = backptr[t, s]
        if move == 1:
            s = s - 1
        elif move == 2:
            s = s - 2
        # move == 0 -> stay on same state
        path[t - 1] = s

    # Convert state-path -> spans for the original target phoneme list.
    spans: list[tuple[int, int]] = [(0, 0)] * L_p
    cur_phone_idx = -1
    for t, s in enumerate(path):
        if allow_blank:
            # State indices in the interleaved sequence: blank states at
            # even positions (0, 2, 4, ...), phone states at odd positions
            # (1, 3, 5, ...). Phone index in target_phone_ids = (s - 1) // 2
            # when s is odd.
            if s % 2 == 1:
                target_idx = (s - 1) // 2
            else:
                target_idx = -1
        else:
            target_idx = s
        if target_idx == -1:
            continue
        if target_idx != cur_phone_idx:
            # New phone span starts here.
            if cur_phone_idx >= 0:
                # Previous phone ends at t (exclusive).
                start, _ = spans[cur_phone_idx]
                spans[cur_phone_idx] = (start, t)
            spans[target_idx] = (t, t + 1)
            cur_phone_idx = target_idx
        else:
            start, _ = spans[target_idx]
            spans[target_idx] = (start, t + 1)

    # Phones that never matched any frame keep (0, 0); patch them to abut
    # their neighbours so callers don't see zero-width spans inside the
    # middle of the sequence.
    for i in range(L_p):
        if spans[i] == (0, 0) and i > 0:
            spans[i] = (spans[i - 1][1], spans[i - 1][1])

    return spans


def spans_from_alignment(
    spans: list[tuple[int, int]],
    *,
    hop_seconds: float,
) -> list[tuple[float, float]]:
    """Convert frame-index spans to seconds."""
    return [
        (start * hop_seconds, end * hop_seconds) for start, end in spans
    ]


__all__ = [
    "BLANK_INDEX",
    "spans_from_alignment",
    "viterbi_align_phones",
]
