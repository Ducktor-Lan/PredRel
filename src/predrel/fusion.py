"""Frozen Week 10 late-fusion control (pure NumPy, no training).

SupCon+student fusion (plan Section 9 SupCon+Readout, metric-only): per eval
query, rank the full support by student score and by SupCon cosine similarity
separately with average-tie ranks, average the two rank vectors elementwise,
and re-rank ascending. The fused score is the negative fused rank (higher is
better), so every downstream metric (top-K recall, spearman, NDCG, hit rate)
applies unchanged.

Ties: ranks are 1-based average ranks within each method. If the fused ranks
tie, top-K sets break ties by ascending support position, identical to the
tie-aware ordering used in metrics.topk_positions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class FusionError(ValueError):
    """Raised when the Week 10 fusion control cannot be scored safely."""


def _finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise FusionError(name + " must be a non-empty finite vector")
    return np.ascontiguousarray(vector)


def average_ranks_descending(scores: ArrayLike) -> NDArray[np.float64]:
    """1-based average ranks for descending scores (ties share the mean rank)."""

    vector = _finite_vector(scores, name="scores")
    order = np.argsort(-vector, kind="stable")
    ranks = np.empty(vector.size, dtype=np.float64)
    i = 0
    n = int(vector.size)
    while i < n:
        j = i + 1
        while j < n and vector[order[j]] == vector[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return np.ascontiguousarray(ranks)


def fuse_rank_average(student_scores: ArrayLike, supcon_scores: ArrayLike) -> NDArray[np.float64]:
    """Fuse two full-support score vectors into one fused score vector."""

    student = _finite_vector(student_scores, name="student_scores")
    supcon = _finite_vector(supcon_scores, name="supcon_scores")
    if student.shape != supcon.shape:
        raise FusionError("student/supcon score vectors do not align")
    fused_rank = (average_ranks_descending(student) + average_ranks_descending(supcon)) / 2.0
    return np.ascontiguousarray(-fused_rank)


__all__ = ["FusionError", "average_ranks_descending", "fuse_rank_average"]
