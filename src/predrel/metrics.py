"""Frozen evaluation metrics (pure NumPy).

For each frozen eval query the Teacher supplies its true-class beta block
and mean-raw true-class block (from one eval fit over the full support).
Each of the eight benchmark methods scores the same query against the full
support; only the true-class columns are compared unless noted:

- teacher_topk_recall@K: fraction of the teacher true-class top-K support
  positions recovered by the method true-class top-K (K = 1/5/10, capped by
  the block width).
- spearman_vs_beta: rank correlation of method true-class scores with the
  teacher beta block (Week 5's tie-aware formulation).
- spearman_vs_raw: rank correlation of method true-class scores with the
  teacher mean-raw true-class block (diagnostic).
- ndcg_at_10: NDCG with teacher beta as graded relevance over the method's
  full true-class ranking (K=10 capped by block width).
- teacher_top1_rank: 1-based rank of the teacher top-1 support row inside
  the method true-class ranking (ties share the average rank).
- label hit@1/hit@5: whether the query's true label appears among the top-1
  / top-5 scoring full-support columns (labels are metric-only).

Aggregation: mean over eval queries per seed, median over seeds per
dataset, mean over datasets globally (descriptive only).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import StudentError


METHOD_IDS = ("student", "supcon", "fusion", "raw", "pca", "mlp", "hidden", "readout_profile")


def _finite_block(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    block = np.asarray(values, dtype=np.float64)
    if block.ndim != 1 or block.size < 2:
        raise StudentError(name + " must be a vector with at least two entries")
    if not np.all(np.isfinite(block)):
        raise StudentError(name + " must be finite")
    return np.ascontiguousarray(block)


def _none_to_neginf(values: ArrayLike) -> NDArray[np.float64]:
    """Map None entries (readout-profile off-column) to -inf, keep floats."""

    raw = list(np.asarray(values).reshape(-1).tolist())
    return np.asarray([float(v) if v is not None else -np.inf for v in raw], dtype=np.float64)


def topk_positions(scores: ArrayLike, k: int, *, tie_tolerance: float = 1e-9) -> list[int]:
    """Top-K positions of one score vector (descending, tie-aware).

    Near-ties within ``tie_tolerance`` (absolute) are treated as tied and
    broken by ascending position index. Pure ``argsort(-v, stable)`` breaks
    exact ties by index but orders near-ties by float residue, which differs
    across processes at the ~1e-10 level on near-flat score blocks. The
    tolerance only affects which tied members enter the boundary window; all
    strictly separated ranks are unchanged.

    ``None`` entries (readout-profile off-column positions) map to -inf and
    can never enter the top-K unless every entry is -inf.
    """

    vector = _none_to_neginf(scores)
    if vector.size == 0 or not np.any(np.isfinite(vector)):
        raise StudentError("scores must be a non-empty finite vector")
    if not np.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0.0:
        raise StudentError("tie_tolerance must be a finite non-negative number")
    capped = int(max(1, min(int(k), vector.size)))
    if float(tie_tolerance) <= 0.0:
        return [int(v) for v in np.argsort(-vector, kind="stable")[:capped].tolist()]

    def _bucket(value: float) -> float:
        if math.isnan(value):
            raise StudentError("scores must be finite or None-mapped -inf")
        if math.isinf(value):
            return value  # -inf sorts last; +inf (should not occur) sorts first
        return float(-round(value / float(tie_tolerance)))

    order = sorted(range(vector.size), key=lambda i: (_bucket(float(vector[i])), i))
    return [int(v) for v in order[:capped]]


def topk_recall(teacher_scores: ArrayLike, student_scores: ArrayLike, k: int) -> float:
    """Fraction of the teacher top-K positions recovered by the method top-K."""

    teacher = _finite_block(teacher_scores, name="teacher_scores")
    student = _finite_block(student_scores, name="student_scores")
    if teacher.shape != student.shape:
        raise StudentError("teacher/method blocks do not align")
    capped = int(max(1, min(int(k), teacher.size)))
    teacher_top = set(topk_positions(teacher, capped))
    student_top = set(topk_positions(student, capped))
    return float(len(teacher_top & student_top) / float(capped))


def _average_ranks(values: NDArray[np.float64], *, tie_tolerance: float = 1e-9) -> NDArray[np.float64]:
    # Tie-aware ranking: values within tie_tolerance (absolute) share an
    # averaged rank. Without this, near-ties broken by float residue flip
    # ranks across processes at the ~1e-5 spearman level on near-flat blocks.
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = int(values.size)
    while i < n:
        j = i + 1
        while j < n and abs(float(sorted_vals[j]) - float(sorted_vals[i])) <= float(tie_tolerance):
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman_corr(left: ArrayLike, right: ArrayLike) -> float:
    """Spearman rank correlation for two equal-length vectors (no scipy)."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise StudentError("spearman needs two aligned vectors of length >= 2")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise StudentError("spearman inputs must be finite")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    cx = rx - rx.mean()
    cy = ry - ry.mean()
    denom = float(np.sqrt(float(cx @ cx) * float(cy @ cy)))
    if denom <= 0.0:
        return 0.0
    return float((cx @ cy) / denom)


def label_hit_at_k(full_scores: ArrayLike, support_labels: ArrayLike, query_label: int, k: int) -> int:
    """Whether the true label appears in the top-K full-support columns.

    ``None`` entries (readout-profile off-column positions) are treated as
    minus infinity: they can never enter the top-K. All other scores must
    be finite.
    """

    raw = list(np.asarray(full_scores).reshape(-1).tolist())
    scores = np.asarray([float(v) if v is not None else -np.inf for v in raw], dtype=np.float64)
    labels = np.asarray(support_labels).reshape(-1)
    if scores.shape != labels.shape or scores.size == 0:
        raise StudentError("scores and support labels do not align")
    if not np.all(np.isfinite(scores[np.isfinite(scores)])):
        raise StudentError("scores must be finite")
    if not np.any(np.isfinite(scores)):
        raise StudentError("scores have no finite entries")
    capped = int(max(1, min(int(k), scores.size)))
    # Same tie-aware ordering as topk_positions: near-ties at the K boundary
    # must resolve identically in runner and verifier processes.
    top = np.asarray(topk_positions(scores, capped), dtype=np.int64)
    return int(bool(np.any(labels[top] == int(query_label))))


def ndcg_at_k(method_scores: ArrayLike, teacher_beta: ArrayLike, k: int = 10) -> float:
    """NDCG of the method ranking with teacher beta as graded relevance."""

    method = _finite_block(method_scores, name="method_scores")
    relevance = _finite_block(teacher_beta, name="teacher_beta")
    if method.shape != relevance.shape:
        raise StudentError("method/teacher blocks do not align")
    if np.any(relevance < 0.0):
        raise StudentError("teacher beta relevance must be non-negative")
    capped = int(max(1, min(int(k), method.size)))
    order = topk_positions(method, method.size)
    gains = np.asarray([float(relevance[pos]) for pos in order], dtype=np.float64)
    discounts = np.log2(np.arange(gains.size, dtype=np.float64) + 2.0)
    dcg = float((gains / discounts)[:capped].sum())
    ideal = np.sort(relevance)[::-1]
    idcg = float(((ideal / discounts)[:capped]).sum())
    if idcg <= 0.0:
        return 0.0
    value = dcg / idcg
    if not math.isfinite(value):
        raise StudentError("NDCG is non-finite")
    return float(min(1.0, max(0.0, value)))


def teacher_top1_rank(method_scores: ArrayLike, teacher_beta: ArrayLike) -> float:
    """1-based average rank of the teacher top-1 row inside the method ranking."""

    method = _finite_block(method_scores, name="method_scores")
    relevance = _finite_block(teacher_beta, name="teacher_beta")
    if method.shape != relevance.shape:
        raise StudentError("method/teacher blocks do not align")
    teacher_top = int(topk_positions(relevance, 1)[0])
    ranks = _average_ranks(-method)
    return float(ranks[teacher_top] + 1.0)


def score_benchmark_query(
    *,
    method_true_scores: ArrayLike,
    teacher_beta_block: ArrayLike,
    teacher_raw_block: ArrayLike,
    full_method_scores: ArrayLike,
    support_labels: ArrayLike,
    query_label: int,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Score one frozen eval query for one benchmark method."""

    method_block = _finite_block(method_true_scores, name="method_true_scores")
    teacher_block = _finite_block(teacher_beta_block, name="teacher_beta_block")
    teacher_raw = _finite_block(teacher_raw_block, name="teacher_raw_block")
    if method_block.shape != teacher_block.shape or teacher_raw.shape != teacher_block.shape:
        raise StudentError("method/teacher true-class blocks do not align")
    record: dict[str, Any] = {
        "teacher_topk_recall_mean": {str(int(k)): topk_recall(teacher_block, method_block, int(k)) for k in ks},
        # Discrete content for exact verification: per-K top-K position sets
        # (block-relative indices). The verifier requires exact equality here;
        # float tolerance applies only to the continuous values above/below.
        "teacher_topk_positions": {str(int(k)): topk_positions(teacher_block, int(k)) for k in ks},
        "method_topk_positions": {str(int(k)): topk_positions(method_block, int(k)) for k in ks},
        "spearman_vs_beta": spearman_corr(method_block, teacher_block),
        "spearman_vs_raw": spearman_corr(method_block, teacher_raw),
        "ndcg_at_10": ndcg_at_k(method_block, teacher_block, 10),
        "teacher_top1_rank": teacher_top1_rank(method_block, teacher_block),
        "label_hit_at_1": label_hit_at_k(full_method_scores, support_labels, int(query_label), 1),
        "label_hit_at_5": label_hit_at_k(full_method_scores, support_labels, int(query_label), 5),
    }
    return record


__all__ = [
    "METHOD_IDS",
    "topk_positions",
    "topk_recall",
    "spearman_corr",
    "label_hit_at_k",
    "ndcg_at_k",
    "teacher_top1_rank",
    "score_benchmark_query",
    "score_eval_query",
]


def score_eval_query(
    *,
    student_true_scores,
    teacher_beta_block,
    full_student_scores,
    support_labels,
    query_label: int,
    ks=(1, 5, 10),
):
    """Legacy Stage 8/9 alias: score one query with the student-role naming.

    Numerics are identical to :func:`score_benchmark_query`; only the record
    keys keep the Stage 8/9 names (``teacher_topk_recall``,
    ``student_topk_positions``, ``spearman_student_vs_beta``) so old evidence
    stays readable. New code should call :func:`score_benchmark_query`.
    """

    record = score_benchmark_query(
        method_true_scores=student_true_scores,
        teacher_beta_block=teacher_beta_block,
        teacher_raw_block=teacher_beta_block,
        full_method_scores=full_student_scores,
        support_labels=support_labels,
        query_label=int(query_label),
        ks=ks,
    )
    return {
        "teacher_topk_recall": dict(record["teacher_topk_recall_mean"]),
        "teacher_topk_positions": dict(record["teacher_topk_positions"]),
        "student_topk_positions": dict(record["method_topk_positions"]),
        "spearman_student_vs_beta": float(record["spearman_vs_beta"]),
        "label_hit_at_1": int(record["label_hit_at_1"]),
        "label_hit_at_5": int(record["label_hit_at_5"]),
    }
