"""Teacher-target decomposition (beta) and beyond-label/beyond-SupCon analysis.

The module treats alpha as a [query, support] probability matrix. Query
labels are used only by the analyze_* functions after Teacher extraction;
the Teacher bridge itself has no query-label parameter.

``analyze_beyond_labels`` (Stage 3 name) and ``analyze_beyond_supcon``
(Stage 4+ name) share one frozen implementation; only the entry-point name
changed between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class AnatomyError(ValueError):
    """Raised when a predictive-relation array violates its declared layout."""


def _probability_matrix(values: ArrayLike, *, name: str, epsilon: float) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise AnatomyError(f"{name} must be a non-empty [query, support] matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < -epsilon):
        raise AnatomyError(f"{name} must contain finite non-negative values")
    totals = matrix.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=max(epsilon * 100.0, 1.0e-6)):
        raise AnatomyError(f"{name} rows must sum to one; observed range {totals.min()}..{totals.max()}")
    return np.clip(matrix, 0.0, None) / totals[:, np.newaxis]


@dataclass(frozen=True, slots=True)
class AlphaDecomposition:
    """Explicit alpha = M * beta decomposition aligned to support columns."""

    classes: NDArray[np.int64]
    class_mass: NDArray[np.float64]
    beta: NDArray[np.float64]
    class_only_null: NDArray[np.float64]


def decompose_alpha(alpha: ArrayLike, support_labels: ArrayLike, *, epsilon: float = 1.0e-12) -> AlphaDecomposition:
    """Compute class mass, beta, and the class-only null model.

    For each support item i, the null is M[q, y_i] / N[y_i]. The method does
    not use a query label.
    """

    if epsilon <= 0.0:
        raise AnatomyError("epsilon must be positive")
    matrix = _probability_matrix(alpha, name="alpha", epsilon=epsilon)
    labels = np.asarray(support_labels)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[1]:
        raise AnatomyError("support_labels must have one value per support column")
    if not np.issubdtype(labels.dtype, np.integer):
        raise AnatomyError("support_labels must be integral")
    classes = np.unique(labels.astype(np.int64, copy=False))
    class_mass = np.empty((matrix.shape[0], classes.shape[0]), dtype=np.float64)
    beta = np.zeros_like(matrix)
    null = np.zeros_like(matrix)
    for class_position, class_label in enumerate(classes):
        mask = labels == class_label
        count = int(mask.sum())
        if count <= 0:
            raise AnatomyError("support class is unexpectedly empty")
        mass = matrix[:, mask].sum(axis=1)
        class_mass[:, class_position] = mass
        beta[:, mask] = matrix[:, mask] / np.maximum(mass, epsilon)[:, np.newaxis]
        null[:, mask] = mass[:, np.newaxis] / float(count)
    return AlphaDecomposition(classes=classes, class_mass=class_mass, beta=beta, class_only_null=null)


def js_divergence(left: ArrayLike, right: ArrayLike, *, epsilon: float = 1.0e-12) -> NDArray[np.float64]:
    """Return rowwise Jensen-Shannon divergence in nats."""

    p = _probability_matrix(left, name="left", epsilon=epsilon)
    q = _probability_matrix(right, name="right", epsilon=epsilon)
    if p.shape != q.shape:
        raise AnatomyError("left and right have different shapes")
    midpoint = 0.5 * (p + q)

    def _kl(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        output = np.zeros_like(a)
        positive = a > epsilon
        output[positive] = a[positive] * (
            np.log(a[positive]) - np.log(np.maximum(b[positive], epsilon))
        )
        return output.sum(axis=1)

    return 0.5 * (_kl(p, midpoint) + _kl(q, midpoint))


def _class_profile(
    beta: NDArray[np.float64],
    support_labels: NDArray[np.int64],
    *,
    query_index: int,
    class_label: int,
    top_k: int,
    epsilon: float,
) -> dict[str, float | int]:
    values = beta[query_index, support_labels == class_label]
    count = int(values.shape[0])
    if count <= 0:
        raise AnatomyError("attempted to profile an absent support class")
    normalized = values / np.maximum(values.sum(), epsilon)
    uniform_value = 1.0 / float(count)
    entropy = 1.0 if count == 1 else float(
        -np.sum(np.where(normalized > epsilon, normalized * np.log(np.maximum(normalized, epsilon)), 0.0))
        / np.log(float(count))
    )
    effective_k = min(top_k, count)
    top_concentration = float(np.sort(normalized)[-effective_k:].sum())
    uniform_top_concentration = float(effective_k) / float(count)
    residual = normalized - uniform_value
    return {
        "support_class": int(class_label),
        "support_class_count": count,
        "normalized_beta_entropy": entropy,
        "top_beta_concentration": top_concentration,
        "uniform_top_beta_concentration": uniform_top_concentration,
        "top_beta_excess_over_uniform": top_concentration - uniform_top_concentration,
        "beta_l1_residual_from_uniform": float(np.abs(residual).sum()),
        "beta_max_residual_from_uniform": float(np.abs(residual).max()),
    }


def _finite_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise AnatomyError("cannot summarize an empty or non-finite metric")
    return {"mean": float(array.mean()), "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())}


def analyze_beyond_supcon(
    alpha: ArrayLike,
    *,
    support_labels: ArrayLike,
    query_labels: ArrayLike,
    support_ids: tuple[str, ...],
    query_ids: tuple[str, ...],
    top_k: int,
    epsilon: float = 1.0e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute all Week 4 metrics after an alpha extraction.

    Profiles include every query/support-class combination. A true-query-class
    view is summarized separately because it is the primary Week 4 gate.
    """

    if top_k < 1:
        raise AnatomyError("top_k must be positive")
    matrix = _probability_matrix(alpha, name="alpha", epsilon=epsilon)
    supports = np.asarray(support_labels, dtype=np.int64)
    queries = np.asarray(query_labels, dtype=np.int64)
    if supports.shape != (matrix.shape[1],) or queries.shape != (matrix.shape[0],):
        raise AnatomyError("label vectors do not match alpha")
    if len(support_ids) != matrix.shape[1] or len(query_ids) != matrix.shape[0]:
        raise AnatomyError("sample IDs do not match alpha")
    if len(set(support_ids)) != len(support_ids) or len(set(query_ids)) != len(query_ids):
        raise AnatomyError("sample IDs must be unique within a split")
    if set(support_ids).intersection(query_ids):
        raise AnatomyError("support and query IDs overlap")

    decomposition = decompose_alpha(matrix, supports, epsilon=epsilon)
    divergence = js_divergence(matrix, decomposition.class_only_null, epsilon=epsilon)
    class_position = {int(label): index for index, label in enumerate(decomposition.classes.tolist())}
    profiles: list[dict[str, Any]] = []
    true_profiles: list[dict[str, Any]] = []
    all_entropy: list[float] = []
    for query_index, query_id in enumerate(query_ids):
        true_label = int(queries[query_index])
        if true_label not in class_position:
            raise AnatomyError(f"query label {true_label} has no support rows")
        for class_label in decomposition.classes.tolist():
            profile = _class_profile(
                decomposition.beta,
                supports,
                query_index=query_index,
                class_label=int(class_label),
                top_k=top_k,
                epsilon=epsilon,
            )
            profile.update(
                {
                    "query_id": query_id,
                    "query_index": query_index,
                    "query_true_label": true_label,
                    "is_true_query_class": bool(int(class_label) == true_label),
                    "class_mass": float(decomposition.class_mass[query_index, class_position[int(class_label)]]),
                    "js_alpha_vs_class_only_null": float(divergence[query_index]),
                }
            )
            profiles.append(profile)
            all_entropy.append(float(profile["normalized_beta_entropy"]))
            if profile["is_true_query_class"]:
                true_profiles.append(profile)

    if len(true_profiles) != matrix.shape[0]:
        raise AnatomyError("every query must yield exactly one true-class profile")
    true_entropy = [float(profile["normalized_beta_entropy"]) for profile in true_profiles]
    true_excess = [float(profile["top_beta_excess_over_uniform"]) for profile in true_profiles]
    true_top = [float(profile["top_beta_concentration"]) for profile in true_profiles]
    true_l1 = [float(profile["beta_l1_residual_from_uniform"]) for profile in true_profiles]
    summary: dict[str, Any] = {
        "alpha_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "alpha_row_sum_min": float(matrix.sum(axis=1).min()),
        "alpha_row_sum_max": float(matrix.sum(axis=1).max()),
        "support_class_labels": decomposition.classes.tolist(),
        "js_alpha_vs_class_only_null": _finite_summary(divergence.tolist()),
        "normalized_within_class_beta_entropy_all_classes": _finite_summary(all_entropy),
        "true_class_normalized_beta_entropy": _finite_summary(true_entropy),
        "true_class_top_beta_concentration": _finite_summary(true_top),
        "true_class_top_beta_excess_over_uniform": _finite_summary(true_excess),
        "true_class_beta_l1_residual_from_uniform": _finite_summary(true_l1),
        "class_residual_profile_count": len(profiles),
        "true_class_profile_count": len(true_profiles),
    }
    return summary, profiles



def analyze_beyond_labels(
    alpha,
    *,
    support_labels,
    query_labels,
    support_ids: tuple[str, ...],
    query_ids: tuple[str, ...],
    top_k: int,
    epsilon: float = 1.0e-12,
):
    """Stage 3 alias of :func:`analyze_beyond_supcon` (identical numerics)."""

    return analyze_beyond_supcon(
        alpha,
        support_labels=support_labels,
        query_labels=query_labels,
        support_ids=support_ids,
        query_ids=query_ids,
        top_k=top_k,
        epsilon=epsilon,
    )
