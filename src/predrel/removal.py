"""Same-class removal faithfulness machinery.

The Teacher prediction for a query is the alpha-weighted class vote, so a
full-support predictive distribution and every ablated distribution are plain
probability vectors over classes.  This module builds matched-cardinality
removal sets (top-beta, random-same-class, bottom-beta, SupCon-guided) and
scores their behavioral effect without ever touching a query label inside a
support context: labels are released only after the full-support extraction,
exactly as in Week 3, and every ablated re-run receives support features,
support labels, and query features alone.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .label_anatomy import AnatomyError, decompose_alpha


class RemovalError(ValueError):
    """Raised when a removal plan cannot be built or scored safely."""


CONDITIONS = ("top_beta", "random_same_class", "bottom_beta", "supcon_top")


def select_removal_queries(query_labels: ArrayLike, *, max_queries: int) -> list[int]:
    """Deterministically subsample query positions with label round-robin.

    Groups query positions by label in ascending label order, then takes
    rows round-robin so small budgets stay class-balanced.  The output is
    sorted back into split order for stable evidence.
    """

    labels = np.asarray(query_labels).reshape(-1)
    if labels.size == 0:
        raise RemovalError("cannot select removal queries from an empty split")
    if max_queries < 1:
        raise RemovalError("max_queries must be positive")
    by_label: dict[int, list[int]] = {}
    for position, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), []).append(int(position))
    ordered_labels = sorted(by_label)
    selected: list[int] = []
    cursor = 0
    while len(selected) < min(max_queries, labels.size):
        progressed = False
        for label in ordered_labels:
            bucket = by_label[label]
            if cursor < len(bucket) and len(selected) < min(max_queries, labels.size):
                selected.append(bucket[cursor])
                progressed = True
        cursor += 1
        if not progressed:
            break
    return sorted(selected)


def _descending(beta: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.int64]:
    order = np.argsort(-np.asarray(beta[indices], dtype=np.float64), kind="stable")
    return np.asarray(indices[order], dtype=np.int64)


def _ascending(beta: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.int64]:
    order = np.argsort(np.asarray(beta[indices], dtype=np.float64), kind="stable")
    return np.asarray(indices[order], dtype=np.int64)


def build_removal_sets(
    beta_row: ArrayLike,
    support_labels: ArrayLike,
    query_label: int,
    *,
    k: int,
    rng: np.random.Generator,
    random_repeats: int,
    supcon_scores: ArrayLike | None = None,
) -> dict[str, Any]:
    """Build matched-cardinality same-class removal sets for one query.

    Returns support positions (not IDs) so the caller keeps ID tracking in
    one place.  At most ``count - 1`` rows of a class are ever removed, so an
    ablated support never loses a class.
    """

    if k < 1:
        raise RemovalError("k must be positive")
    if random_repeats < 1:
        raise RemovalError("random_repeats must be positive")
    beta = np.asarray(beta_row, dtype=np.float64).reshape(-1)
    supports = np.asarray(support_labels).reshape(-1)
    if beta.shape != supports.shape:
        raise RemovalError("beta row does not match support labels")
    if not np.all(np.isfinite(beta)):
        raise RemovalError("beta row is not finite")
    same = np.flatnonzero(supports == int(query_label))
    count = int(same.size)
    if count < 2:
        raise RemovalError("same-class support is too small for a removal experiment")
    actual_k = int(min(k, count - 1))
    top = _descending(beta, same)[:actual_k].tolist()
    bottom = _ascending(beta, same)[:actual_k].tolist()
    repeats: list[list[int]] = []
    for _ in range(random_repeats):
        draw = rng.choice(same, size=actual_k, replace=False)
        repeats.append([int(value) for value in np.sort(draw).tolist()])
    supcon_top: list[int] | None = None
    if supcon_scores is not None:
        scores = np.asarray(supcon_scores, dtype=np.float64).reshape(-1)
        if scores.shape != supports.shape:
            raise RemovalError("supcon scores do not match support labels")
        if not np.all(np.isfinite(scores)):
            raise RemovalError("supcon scores are not finite")
        supcon_top = _descending(scores, same)[:actual_k].tolist()
    return {
        "top_beta": top,
        "bottom_beta": bottom,
        "random_same_class": repeats,
        "supcon_top": supcon_top,
        "actual_k": actual_k,
        "same_class_count": count,
    }


def class_distribution(alpha: ArrayLike, support_labels: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return (classes, class_mass) predictive distributions for queries."""

    try:
        decomposition = decompose_alpha(alpha, support_labels)
    except AnatomyError as error:
        raise RemovalError(str(error)) from error
    return np.asarray(decomposition.classes), np.asarray(decomposition.class_mass, dtype=np.float64)


def removal_effect(
    full_distribution: ArrayLike,
    full_classes: ArrayLike,
    ablated_distribution: ArrayLike,
    ablated_classes: ArrayLike,
    *,
    removed_alpha_mass: float,
    actual_k: int,
    epsilon: float = 1.0e-12,
) -> dict[str, float | int]:
    """Score one ablated prediction against its full-support reference."""

    full = np.asarray(full_distribution, dtype=np.float64).reshape(-1)
    ablated = np.asarray(ablated_distribution, dtype=np.float64).reshape(-1)
    full_labels = np.asarray(full_classes).reshape(-1)
    ablated_labels = np.asarray(ablated_classes).reshape(-1)
    if full.shape != full_labels.shape or ablated.shape != ablated_labels.shape:
        raise RemovalError("distributions do not match their class labels")
    union = sorted({int(value) for value in full_labels.tolist()} | {int(value) for value in ablated_labels.tolist()})
    if not union:
        raise RemovalError("empty class union")
    full_aligned = np.zeros(len(union), dtype=np.float64)
    ablated_aligned = np.zeros(len(union), dtype=np.float64)
    for position, label in enumerate(union):
        if label in set(full_labels.tolist()):
            full_aligned[position] = float(full[np.flatnonzero(full_labels == label)[0]])
        if label in set(ablated_labels.tolist()):
            ablated_aligned[position] = float(ablated[np.flatnonzero(ablated_labels == label)[0]])
    if not np.all(np.isfinite(full_aligned)) or not np.all(np.isfinite(ablated_aligned)) or np.any(full_aligned < 0.0) or np.any(ablated_aligned < 0.0):
        raise RemovalError("predictive distributions are not finite")
    if full_aligned.sum() <= 0.0 or ablated_aligned.sum() <= 0.0:
        raise RemovalError("predictive distributions are degenerate")
    full_aligned = full_aligned / full_aligned.sum()
    ablated_aligned = ablated_aligned / ablated_aligned.sum()
    l1 = float(np.abs(full_aligned - ablated_aligned).sum())
    predicted = int(np.argmax(full_aligned))
    flip = int(np.argmax(ablated_aligned) != predicted)
    delta_logprob = float(
        np.log(max(full_aligned[predicted], epsilon)) - np.log(max(ablated_aligned[predicted], epsilon))
    )
    return {
        "tv": 0.5 * l1,
        "l1": l1,
        "flip": flip,
        "delta_logprob": delta_logprob,
        "removed_alpha_mass": float(removed_alpha_mass),
        "actual_k": int(actual_k),
    }


def summarize_effects(effects: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    """Aggregate per-query effects of one condition into a frozen summary."""

    rows = list(effects)
    if not rows:
        raise RemovalError("cannot summarize an empty effect set")
    tv = np.asarray([float(row["tv"]) for row in rows], dtype=np.float64)
    flips = np.asarray([int(row["flip"]) for row in rows], dtype=np.float64)
    dropped = np.asarray([float(row["delta_logprob"]) for row in rows], dtype=np.float64)
    mass = np.asarray([float(row["removed_alpha_mass"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(tv)) or not np.all(np.isfinite(dropped)) or not np.all(np.isfinite(mass)):
        raise RemovalError("effect set contains non-finite values")
    return {
        "count": int(len(rows)),
        "mean_tv": float(tv.mean()),
        "median_tv": float(np.median(tv)),
        "flip_rate": float(flips.mean()),
        "mean_delta_logprob": float(dropped.mean()),
        "mean_removed_alpha_mass": float(mass.mean()),
    }


def paired_comparison(
    primary_tv_by_query: Mapping[int, float],
    reference_tv_by_query: Mapping[int, float],
) -> dict[str, float | int]:
    """Compare two matched-cardinality conditions query by query."""

    shared = sorted(set(primary_tv_by_query) & set(reference_tv_by_query))
    if not shared:
        raise RemovalError("paired comparison needs shared queries")
    diffs = np.asarray(
        [float(primary_tv_by_query[q]) - float(reference_tv_by_query[q]) for q in shared],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(diffs)):
        raise RemovalError("paired comparison contains non-finite TV values")
    return {
        "count": int(len(shared)),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "win_rate": float((diffs > 0.0).mean()),
    }


def teacher_topk_recall(
    teacher_top_positions: Sequence[int],
    ranked_positions: Sequence[int],
    *,
    at: int,
) -> float:
    """Recall of Teacher top-K support rows inside the first ``at`` ranks."""

    teacher = list(teacher_top_positions)
    ranked = list(ranked_positions)
    if not teacher:
        raise RemovalError("Teacher top-K set is empty")
    if at < 1:
        raise RemovalError("recall rank must be positive")
    retrieved = set(ranked[:at])
    hits = sum(1 for position in teacher if position in retrieved)
    return float(hits) / float(len(teacher))


__all__ = [
    "CONDITIONS",
    "RemovalError",
    "build_removal_sets",
    "class_distribution",
    "paired_comparison",
    "removal_effect",
    "select_removal_queries",
    "summarize_effects",
    "teacher_topk_recall",
]