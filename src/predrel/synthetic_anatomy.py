"""Synthetic-task anatomy metrics (Stage 2; validates methods before real-data claims)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class AnatomyError(ValueError):
    """Raised when an analysis target does not satisfy its documented layout."""


def _as_probability_matrix(values: ArrayLike, *, name: str, epsilon: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise AnatomyError(f"{name} must be a non-empty [query, support] matrix")
    if not np.isfinite(array).all() or np.any(array < -epsilon):
        raise AnatomyError(f"{name} must be finite and non-negative")
    row_sums = array.sum(axis=1)
    # Week 1 validates raw-score parity at 1e-6.  TabPFN's float32 decoder
    # aggregation can differ from exact one by roughly 1e-7, so reject only a
    # materially invalid probability row and normalize harmless roundoff.
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=max(epsilon * 100.0, 1.0e-6)):
        raise AnatomyError(f"{name} rows must sum to one; observed range {row_sums.min()}..{row_sums.max()}")
    return np.clip(array, 0.0, None) / row_sums[:, np.newaxis]


@dataclass(frozen=True, slots=True)
class AlphaDecomposition:
    """`alpha = class_mass * beta` with explicit support-column alignment."""

    classes: NDArray[np.int64]
    class_mass: NDArray[np.float64]
    beta: NDArray[np.float64]
    class_only_null: NDArray[np.float64]


def decompose_alpha(alpha: ArrayLike, support_labels: ArrayLike, *, epsilon: float = 1.0e-12) -> AlphaDecomposition:
    """Compute class mass, within-class beta, and the class-only null model."""

    if epsilon <= 0:
        raise AnatomyError("epsilon must be positive")
    matrix = _as_probability_matrix(alpha, name="alpha", epsilon=epsilon)
    labels = np.asarray(support_labels)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[1]:
        raise AnatomyError("support_labels must have one value per alpha support column")
    if not np.issubdtype(labels.dtype, np.integer):
        raise AnatomyError("support_labels must be integral")
    classes = np.unique(labels.astype(np.int64, copy=False))
    class_mass = np.empty((matrix.shape[0], classes.shape[0]), dtype=np.float64)
    beta = np.zeros_like(matrix)
    null = np.zeros_like(matrix)
    for class_index, label in enumerate(classes):
        mask = labels == label
        count = int(mask.sum())
        mass = matrix[:, mask].sum(axis=1)
        class_mass[:, class_index] = mass
        denominator = np.maximum(mass, epsilon)
        beta[:, mask] = matrix[:, mask] / denominator[:, np.newaxis]
        null[:, mask] = mass[:, np.newaxis] / count
    return AlphaDecomposition(classes=classes, class_mass=class_mass, beta=beta, class_only_null=null)


def class_only_null(alpha: ArrayLike, support_labels: ArrayLike, *, epsilon: float = 1.0e-12) -> NDArray[np.float64]:
    """Return `M[q, y_i] / N_{y_i}` for every support column."""

    return decompose_alpha(alpha, support_labels, epsilon=epsilon).class_only_null


def js_divergence(left: ArrayLike, right: ArrayLike, *, epsilon: float = 1.0e-12) -> NDArray[np.float64]:
    """Compute rowwise Jensen-Shannon divergence in nats for two distributions."""

    p = _as_probability_matrix(left, name="left", epsilon=epsilon)
    q = _as_probability_matrix(right, name="right", epsilon=epsilon)
    if p.shape != q.shape:
        raise AnatomyError("left and right must have equal shape")
    midpoint = 0.5 * (p + q)

    def kl(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        terms = np.zeros_like(a)
        positive = a > epsilon
        terms[positive] = a[positive] * (np.log(a[positive]) - np.log(np.maximum(b[positive], epsilon)))
        return terms.sum(axis=1)

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


def normalized_within_class_entropy(
    beta: ArrayLike, support_labels: ArrayLike, *, epsilon: float = 1.0e-12
) -> dict[int, NDArray[np.float64]]:
    """Return entropy/log(class size) of each beta class distribution per query."""

    matrix = np.asarray(beta, dtype=np.float64)
    labels = np.asarray(support_labels)
    if matrix.ndim != 2 or labels.ndim != 1 or matrix.shape[1] != labels.shape[0]:
        raise AnatomyError("beta and support_labels have incompatible shapes")
    output: dict[int, NDArray[np.float64]] = {}
    for label in np.unique(labels.astype(np.int64, copy=False)):
        values = matrix[:, labels == label]
        count = values.shape[1]
        if count == 1:
            output[int(label)] = np.ones(matrix.shape[0], dtype=np.float64)
            continue
        normalised = values / np.maximum(values.sum(axis=1, keepdims=True), epsilon)
        entropy = -(np.where(normalised > epsilon, normalised * np.log(np.maximum(normalised, epsilon)), 0.0)).sum(axis=1)
        output[int(label)] = entropy / np.log(float(count))
    return output


def _rank_descending(values: ArrayLike) -> NDArray[np.float64]:
    """Average ranks with rank 1 assigned to the largest value."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise AnatomyError("ranking expects a 1-D vector")
    order = np.argsort(-array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < array.shape[0]:
        end = start + 1
        while end < array.shape[0] and np.isclose(array[order[end]], array[order[start]], rtol=0.0, atol=1.0e-12):
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman_correlation(left: ArrayLike, right: ArrayLike) -> float:
    """Spearman rho without a scipy dependency; returns NaN for constant ranks."""

    x = _rank_descending(left)
    y = _rank_descending(right)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def kendall_tau(left: ArrayLike, right: ArrayLike, *, epsilon: float = 1.0e-12) -> float:
    """Kendall tau-a over non-tied pairs, sufficient for fixed-anchor diagnostics."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise AnatomyError("Kendall tau expects equal 1-D vectors")
    concordant = 0
    discordant = 0
    for first, second in combinations(range(x.shape[0]), 2):
        left_delta = x[first] - x[second]
        right_delta = y[first] - y[second]
        if abs(left_delta) <= epsilon or abs(right_delta) <= epsilon:
            continue
        if np.sign(left_delta) == np.sign(right_delta):
            concordant += 1
        else:
            discordant += 1
    denominator = concordant + discordant
    return float("nan") if denominator == 0 else (concordant - discordant) / denominator


def top_k_jaccard(left: ArrayLike, right: ArrayLike, *, k: int) -> float:
    """Return Jaccard overlap between the top-k indices of two score vectors."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or not 1 <= k <= x.shape[0]:
        raise AnatomyError("top_k_jaccard expects equal vectors and a valid k")
    x_top = set(np.argsort(-x, kind="mergesort")[:k].tolist())
    y_top = set(np.argsort(-y, kind="mergesort")[:k].tolist())
    return len(x_top.intersection(y_top)) / len(x_top.union(y_top))


def preference_flip_rate(reference: ArrayLike, candidate: ArrayLike, *, epsilon: float = 1.0e-12) -> float:
    """Fraction of non-tied anchor pairs whose ordering reverses."""

    baseline = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(candidate, dtype=np.float64)
    if baseline.ndim != 1 or observed.ndim != 1 or baseline.shape != observed.shape:
        raise AnatomyError("preference_flip_rate expects equal 1-D vectors")
    comparable = 0
    flips = 0
    for first, second in combinations(range(baseline.shape[0]), 2):
        left_delta = baseline[first] - baseline[second]
        right_delta = observed[first] - observed[second]
        if abs(left_delta) <= epsilon or abs(right_delta) <= epsilon:
            continue
        comparable += 1
        flips += int(np.sign(left_delta) != np.sign(right_delta))
    return float("nan") if comparable == 0 else flips / comparable


def context_stability(
    alpha_by_context: ArrayLike,
    raw_by_context: ArrayLike,
    *,
    top_k: int,
) -> dict[str, float]:
    """Separate alpha variability from raw-score stability for fixed anchors.

    Each input has layout ``[contexts, anchors]``.  ``alpha_by_context`` should
    be normalized among the anchors so changes in total anchor mass do not mask
    rank changes caused by context competition.
    """

    alpha = np.asarray(alpha_by_context, dtype=np.float64)
    raw = np.asarray(raw_by_context, dtype=np.float64)
    if alpha.ndim != 2 or raw.ndim != 2 or alpha.shape != raw.shape or alpha.shape[0] < 2:
        raise AnatomyError("context arrays must be matching [contexts, anchors] matrices with >=2 contexts")
    if not np.isfinite(alpha).all() or not np.isfinite(raw).all() or np.any(alpha < 0.0):
        raise AnatomyError("context arrays must be finite and alpha non-negative")
    alpha_sums = alpha.sum(axis=1)
    if np.any(alpha_sums <= 0.0):
        raise AnatomyError("each context must retain positive total anchor alpha")
    alpha = alpha / alpha_sums[:, np.newaxis]
    if not 1 <= top_k <= alpha.shape[1]:
        raise AnatomyError("context top_k is invalid")
    pair_indices = list(combinations(range(alpha.shape[0]), 2))
    alpha_jaccard = [top_k_jaccard(alpha[first], alpha[second], k=top_k) for first, second in pair_indices]
    raw_jaccard = [top_k_jaccard(raw[first], raw[second], k=top_k) for first, second in pair_indices]
    alpha_kendall = [kendall_tau(alpha[first], alpha[second]) for first, second in pair_indices]
    raw_kendall = [kendall_tau(raw[first], raw[second]) for first, second in pair_indices]
    alpha_spearman = [spearman_correlation(alpha[first], alpha[second]) for first, second in pair_indices]
    raw_spearman = [spearman_correlation(raw[first], raw[second]) for first, second in pair_indices]
    return {
        "contexts": float(alpha.shape[0]),
        "anchors": float(alpha.shape[1]),
        "mean_anchor_alpha_std": float(np.std(alpha, axis=0).mean()),
        "mean_anchor_raw_score_std": float(np.std(raw, axis=0).mean()),
        "mean_pair_alpha_top_k_jaccard": float(np.mean(alpha_jaccard)),
        "mean_pair_raw_top_k_jaccard": float(np.mean(raw_jaccard)),
        "mean_pair_alpha_kendall_tau": float(np.nanmean(alpha_kendall)),
        "mean_pair_raw_kendall_tau": float(np.nanmean(raw_kendall)),
        "mean_pair_alpha_spearman": float(np.nanmean(alpha_spearman)),
        "mean_pair_raw_spearman": float(np.nanmean(raw_spearman)),
        "mean_alpha_preference_flip_rate_vs_first": float(
            np.nanmean([preference_flip_rate(alpha[0], alpha[index]) for index in range(1, alpha.shape[0])])
        ),
        "mean_raw_preference_flip_rate_vs_first": float(
            np.nanmean([preference_flip_rate(raw[0], raw[index]) for index in range(1, raw.shape[0])])
        ),
    }


def fixture_metrics(
    alpha: ArrayLike,
    *,
    support_labels: ArrayLike,
    query_labels: ArrayLike,
    support_groups: Mapping[str, ArrayLike],
    query_groups: Mapping[str, ArrayLike],
    top_k: int,
    epsilon: float = 1.0e-12,
) -> dict[str, object]:
    """Compute general and fixture-specific metrics from one aligned alpha matrix."""

    values = _as_probability_matrix(alpha, name="alpha", epsilon=epsilon)
    labels = np.asarray(support_labels, dtype=np.int64)
    held_out = np.asarray(query_labels, dtype=np.int64)
    if labels.shape != (values.shape[1],) or held_out.shape != (values.shape[0],):
        raise AnatomyError("label vectors are incompatible with alpha")
    decomposition = decompose_alpha(values, labels, epsilon=epsilon)
    null_js = js_divergence(values, decomposition.class_only_null, epsilon=epsilon)
    entropy = normalized_within_class_entropy(decomposition.beta, labels, epsilon=epsilon)
    report: dict[str, object] = {
        "alpha_shape": [int(values.shape[0]), int(values.shape[1])],
        "alpha_row_sum_min": float(values.sum(axis=1).min()),
        "alpha_row_sum_max": float(values.sum(axis=1).max()),
        "class_labels": decomposition.classes.tolist(),
        "mean_js_alpha_vs_class_only_null": float(null_js.mean()),
        "median_js_alpha_vs_class_only_null": float(np.median(null_js)),
        "per_query_js_alpha_vs_class_only_null": null_js.tolist(),
        "mean_normalized_within_class_entropy": {
            str(label): float(values_for_label.mean()) for label, values_for_label in entropy.items()
        },
    }
    if "prototype" in support_groups and "prototype" in query_groups:
        support_prototype = np.asarray(support_groups["prototype"], dtype=np.int64)
        query_prototype = np.asarray(query_groups["prototype"], dtype=np.int64)
        matching: list[float] = []
        nonmatching: list[float] = []
        for query_index, (label, prototype) in enumerate(zip(held_out, query_prototype, strict=True)):
            class_mask = labels == label
            matching_mask = class_mask & (support_prototype == prototype)
            other_mask = class_mask & ~matching_mask
            class_mass = float(values[query_index, class_mask].sum())
            matching.append(float(values[query_index, matching_mask].sum() / max(class_mass, epsilon)))
            nonmatching.append(float(values[query_index, other_mask].sum() / max(class_mass, epsilon)))
        report["prototype"] = {
            "mean_matching_prototype_beta_mass": float(np.mean(matching)),
            "mean_nonmatching_same_label_beta_mass": float(np.mean(nonmatching)),
            "mean_matching_minus_nonmatching": float(np.mean(np.asarray(matching) - np.asarray(nonmatching))),
        }
    if "boundary_distance" in support_groups:
        support_distance = np.asarray(support_groups["boundary_distance"], dtype=np.float64)
        if support_distance.shape != (values.shape[1],):
            raise AnatomyError("boundary_distance support metadata has incorrect length")
        weighted_distances = values @ support_distance
        correlations = [spearman_correlation(values[index], -support_distance) for index in range(values.shape[0])]
        report["boundary"] = {
            "unweighted_support_distance_mean": float(support_distance.mean()),
            "mean_alpha_weighted_support_distance": float(weighted_distances.mean()),
            "per_query_alpha_weighted_support_distance": weighted_distances.tolist(),
            "mean_spearman_alpha_vs_negative_support_distance": float(np.nanmean(correlations)),
            "per_query_spearman_alpha_vs_negative_support_distance": correlations,
        }
    # Top-beta concentration is evaluated inside each held-out query's true class.
    concentration: list[float] = []
    for query_index, label in enumerate(held_out):
        mask = labels == label
        local = decomposition.beta[query_index, mask]
        local_k = min(top_k, local.shape[0])
        concentration.append(float(np.sort(local)[-local_k:].sum()))
    report["mean_true_class_top_beta_concentration"] = float(np.mean(concentration))
    report["per_query_true_class_top_beta_concentration"] = concentration
    return report
