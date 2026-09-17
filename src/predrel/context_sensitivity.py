"""Leakage-safe context-sensitivity protocol.

The only objects compared across contexts are a fixed held-out query and a
fixed universe of support anchors.  Non-anchor support rows are deliberately
changed.  This distinction matters: a Top-K comparison over whole contexts
would mostly compare different background rows, not a changing relation.

The module preserves the Teacher's exact raw-score surfaces ``[estimator,
head, query, support]``.  Raw rank metrics are computed per estimator/head
and then macro-aggregated, rather than taking a softmax or a mean score first.
The average raw-score diagnostic is also retained for readability.  Alpha is
reported separately and normalized only *within fixed anchors* for its own
diagnostic; alpha movement alone cannot establish a true context shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ContextSensitivityError(ValueError):
    """Raised when a fixed-anchor context comparison is ill-defined."""


@dataclass(frozen=True, slots=True)
class ContextDefinition:
    """One frozen support context represented by original support positions."""

    context_index: int
    support_positions: tuple[int, ...]
    anchor_positions: tuple[int, ...]
    background_positions: tuple[int, ...]
    background_class_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class AnchorObservation:
    """Aligned alpha/raw information for fixed anchors in one Teacher fit."""

    anchor_indices: tuple[int, ...]
    anchor_alpha: NDArray[np.float64]
    anchor_alpha_normalized: NDArray[np.float64]
    anchor_alpha_total: float
    anchor_raw_scores: NDArray[np.float64]
    anchor_raw_conditional: NDArray[np.float64]
    anchor_mean_raw_scores: NDArray[np.float64]
    pair_indices: tuple[tuple[int, int], ...]
    pairwise_raw_score_differences: NDArray[np.float64]
    pairwise_mean_raw_score_differences: NDArray[np.float64]
    raw_score_ranking: tuple[str, ...]
    anchor_raw_score_ranking: tuple[str, ...]


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable non-secret uint32 seed without Python hash randomization."""

    if not isinstance(base_seed, int) or base_seed < 0:
        raise ContextSensitivityError("base_seed must be a non-negative integer")
    material = "\0".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], byteorder="big") % (2**32)


def _as_labels(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise ContextSensitivityError(name + " must be a non-empty one-dimensional vector")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ContextSensitivityError(name + " must contain integral labels")
    return labels.astype(np.int64, copy=False)


def select_query_positions(query_count: int, *, available: int, seed: int) -> tuple[int, ...]:
    """Select fixed queries without inspecting query labels."""

    if not isinstance(query_count, int) or query_count < 1 or query_count > available:
        raise ContextSensitivityError("query_count must be between one and the available query count")
    rng = np.random.default_rng(derive_seed(seed, "query-selection", available, query_count))
    return tuple(int(value) for value in sorted(rng.choice(available, size=query_count, replace=False).tolist()))


def select_anchor_positions(support_labels: ArrayLike, *, anchor_count: int) -> tuple[int, ...]:
    """Pick fixed support anchors by deterministic label round-robin.

    Only support labels are used here.  They are legal Teacher-context inputs;
    held-out query labels do not participate in any sampling or fit.
    """

    labels = _as_labels(support_labels, name="support_labels")
    if not isinstance(anchor_count, int) or anchor_count < 3 or anchor_count >= labels.size:
        raise ContextSensitivityError("anchor_count must be at least three and leave a non-anchor background")
    buckets: dict[int, list[int]] = {}
    for position, label in enumerate(labels.tolist()):
        buckets.setdefault(int(label), []).append(position)
    chosen: list[int] = []
    cursor = 0
    ordered_labels = sorted(buckets)
    while len(chosen) < anchor_count:
        progressed = False
        for label in ordered_labels:
            bucket = buckets[label]
            if cursor < len(bucket) and len(chosen) < anchor_count:
                chosen.append(bucket[cursor])
                progressed = True
        cursor += 1
        if not progressed:  # defensive; anchor_count is bounded by support length
            raise ContextSensitivityError("unable to select the requested anchor count")
    return tuple(sorted(int(value) for value in chosen))


def _proportional_quotas(labels: NDArray[np.int64], *, total: int) -> dict[int, int]:
    """Allocate a fixed background count across support classes proportionally."""

    if total < 0 or total > labels.size:
        raise ContextSensitivityError("background total is outside its available pool")
    classes, counts = np.unique(labels, return_counts=True)
    if total == 0:
        return {int(label): 0 for label in classes.tolist()}
    raw = counts.astype(np.float64) * (float(total) / float(labels.size))
    allocated = np.floor(raw).astype(np.int64)
    remainder = int(total - int(allocated.sum()))
    # Largest remainders; break ties by numerical class label for determinism.
    order = sorted(
        range(len(classes)),
        key=lambda index: (-(raw[index] - allocated[index]), int(classes[index])),
    )
    for index in order:
        if remainder <= 0:
            break
        if allocated[index] < counts[index]:
            allocated[index] += 1
            remainder -= 1
    if remainder != 0 or np.any(allocated > counts):
        raise ContextSensitivityError("unable to construct a capacity-safe proportional background quota")
    return {int(label): int(count) for label, count in zip(classes.tolist(), allocated.tolist(), strict=True)}


def sample_contexts(
    support_labels: ArrayLike,
    *,
    anchor_positions: Sequence[int],
    context_size: int,
    context_count: int,
    seed: int,
) -> tuple[ContextDefinition, ...]:
    """Create distinct contexts with invariant anchors and class quotas.

    Every context has the same size and non-anchor class counts.  Thus a
    changed context is a composition change, not a class-count change.  Rows
    are returned in original support order to remove support-order noise.
    """

    labels = _as_labels(support_labels, name="support_labels")
    anchors = tuple(sorted(int(value) for value in anchor_positions))
    if not anchors or len(set(anchors)) != len(anchors):
        raise ContextSensitivityError("anchor_positions must be unique and non-empty")
    if any(value < 0 or value >= labels.size for value in anchors):
        raise ContextSensitivityError("anchor_positions are outside the support split")
    if not isinstance(context_size, int) or not isinstance(context_count, int):
        raise ContextSensitivityError("context_size and context_count must be integers")
    if context_count < 2:
        raise ContextSensitivityError("at least two contexts are needed for a sensitivity analysis")
    if context_size <= len(anchors) or context_size >= labels.size:
        raise ContextSensitivityError("context_size must retain anchors and omit at least one background row")
    non_anchor = np.asarray([i for i in range(labels.size) if i not in set(anchors)], dtype=np.int64)
    background_size = context_size - len(anchors)
    if background_size > non_anchor.size:
        raise ContextSensitivityError("requested context cannot be filled after holding anchors fixed")
    quotas = _proportional_quotas(labels[non_anchor], total=background_size)
    positions_by_label = {
        int(label): non_anchor[np.flatnonzero(labels[non_anchor] == label)]
        for label in np.unique(labels[non_anchor]).tolist()
    }
    rng = np.random.default_rng(derive_seed(seed, "background-contexts", labels.size, context_size, context_count))
    seen: set[tuple[int, ...]] = set()
    contexts: list[ContextDefinition] = []
    for context_index in range(context_count):
        for _attempt in range(256):
            background: list[int] = []
            for label in sorted(positions_by_label):
                count = quotas.get(label, 0)
                if count:
                    pool = positions_by_label[label]
                    background.extend(int(value) for value in rng.choice(pool, size=count, replace=False).tolist())
            background_tuple = tuple(sorted(background))
            support_tuple = tuple(sorted((*anchors, *background_tuple)))
            if support_tuple not in seen:
                seen.add(support_tuple)
                contexts.append(
                    ContextDefinition(
                        context_index=context_index,
                        support_positions=support_tuple,
                        anchor_positions=anchors,
                        background_positions=background_tuple,
                        background_class_counts={str(label): count for label, count in sorted(quotas.items())},
                    )
                )
                break
        else:
            raise ContextSensitivityError("could not sample enough distinct contexts from the frozen support split")
    return tuple(contexts)


def _rank_descending(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average descending ranks, with rank one assigned to the largest score."""

    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and np.isclose(values[order[end]], values[order[start]], rtol=0.0, atol=1e-12):
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman_correlation(left: ArrayLike, right: ArrayLike) -> float:
    """Rank correlation without SciPy; NaN means no nonconstant rank signal."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContextSensitivityError("spearman inputs must be aligned finite vectors with length at least three")
    rx = _rank_descending(x)
    ry = _rank_descending(y)
    if np.std(rx) <= 0.0 or np.std(ry) <= 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau(left: ArrayLike, right: ArrayLike, *, epsilon: float = 1.0e-12) -> float:
    """Kendall tau-a over non-tied fixed-anchor pairs."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContextSensitivityError("kendall inputs must be aligned finite vectors with length at least three")
    concordant = 0
    discordant = 0
    for first, second in combinations(range(x.size), 2):
        dx = x[first] - x[second]
        dy = y[first] - y[second]
        if abs(dx) <= epsilon or abs(dy) <= epsilon:
            continue
        if np.sign(dx) == np.sign(dy):
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant
    return float("nan") if total == 0 else float(concordant - discordant) / float(total)


def top_k_jaccard(left: ArrayLike, right: ArrayLike, *, k: int) -> float:
    """Jaccard agreement of two fixed-anchor top-k sets."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or not 1 <= int(k) <= x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContextSensitivityError("top_k_jaccard needs aligned finite scores and a valid k")
    a = set(np.argsort(-x, kind="stable")[: int(k)].tolist())
    b = set(np.argsort(-y, kind="stable")[: int(k)].tolist())
    return float(len(a & b)) / float(len(a | b))


def preference_flip_rate(reference: ArrayLike, candidate: ArrayLike, *, epsilon: float = 1.0e-12) -> float:
    """Fraction of comparable anchor pairs whose preference reverses."""

    base = np.asarray(reference, dtype=np.float64).reshape(-1)
    observed = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if base.shape != observed.shape or base.size < 3 or not np.isfinite(base).all() or not np.isfinite(observed).all():
        raise ContextSensitivityError("preference flip inputs must be aligned finite vectors with length at least three")
    comparable = 0
    flips = 0
    for first, second in combinations(range(base.size), 2):
        before = base[first] - base[second]
        after = observed[first] - observed[second]
        if abs(before) <= epsilon or abs(after) <= epsilon:
            continue
        comparable += 1
        flips += int(np.sign(before) != np.sign(after))
    return float("nan") if comparable == 0 else float(flips) / float(comparable)


def pairwise_differences(values: ArrayLike) -> tuple[tuple[tuple[int, int], ...], NDArray[np.float64]]:
    """Return every unordered anchor score difference along the final axis."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim < 1 or scores.shape[-1] < 3 or not np.isfinite(scores).all():
        raise ContextSensitivityError("pairwise differences require finite scores with at least three anchors")
    pairs = tuple((first, second) for first, second in combinations(range(scores.shape[-1]), 2))
    differences = np.stack([scores[..., first] - scores[..., second] for first, second in pairs], axis=-1)
    return pairs, np.asarray(differences, dtype=np.float64)


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float("nan") if finite.size == 0 else float(finite.mean())


def _nanmedian(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float("nan") if finite.size == 0 else float(np.median(finite))


def rank_stability(scores_by_context: ArrayLike, *, top_k: int) -> dict[str, float]:
    """Compute all cross-context diagnostics on one invariant anchor universe."""

    scores = np.asarray(scores_by_context, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 3 or not np.isfinite(scores).all():
        raise ContextSensitivityError("scores must be a finite [contexts, anchors] array with contexts>=2, anchors>=3")
    if not 1 <= int(top_k) <= scores.shape[1]:
        raise ContextSensitivityError("top_k is invalid for the fixed anchor universe")
    comparisons = list(combinations(range(scores.shape[0]), 2))
    _, differences = pairwise_differences(scores)
    jaccard = [top_k_jaccard(scores[first], scores[second], k=int(top_k)) for first, second in comparisons]
    kendall = [kendall_tau(scores[first], scores[second]) for first, second in comparisons]
    spearman = [spearman_correlation(scores[first], scores[second]) for first, second in comparisons]
    flips = [preference_flip_rate(scores[0], scores[index]) for index in range(1, scores.shape[0])]
    return {
        "contexts": float(scores.shape[0]),
        "anchors": float(scores.shape[1]),
        "mean_anchor_score_std": float(np.std(scores, axis=0).mean()),
        "mean_pairwise_score_difference_std": float(np.std(differences, axis=0).mean()),
        "mean_pair_top_k_jaccard": _nanmean(jaccard),
        "mean_pair_kendall_tau": _nanmean(kendall),
        "mean_pair_spearman": _nanmean(spearman),
        "mean_preference_flip_rate_vs_first": _nanmean(flips),
    }


def _rename_metrics(metrics: Mapping[str, float], *, prefix: str) -> dict[str, float]:
    mapping = {
        "mean_anchor_score_std": "mean_anchor_" + prefix + "_std",
        "mean_pairwise_score_difference_std": "mean_pairwise_" + prefix + "_difference_std",
        "mean_pair_top_k_jaccard": "mean_pair_" + prefix + "_top_k_jaccard",
        "mean_pair_kendall_tau": "mean_pair_" + prefix + "_kendall_tau",
        "mean_pair_spearman": "mean_pair_" + prefix + "_spearman",
        "mean_preference_flip_rate_vs_first": "mean_" + prefix + "_preference_flip_rate_vs_first",
    }
    return {new: float(metrics[old]) for old, new in mapping.items()}


def headwise_raw_score_stability(raw_scores_by_context: ArrayLike, *, top_k: int) -> dict[str, Any]:
    """Score stability per exact Teacher surface, plus macro summaries."""

    raw = np.asarray(raw_scores_by_context, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[0] < 2 or raw.shape[-1] < 3 or not np.isfinite(raw).all():
        raise ContextSensitivityError("raw scores must have layout [contexts, estimators, heads, anchors]")
    surfaces: list[dict[str, Any]] = []
    fields = (
        "mean_anchor_score_std",
        "mean_pairwise_score_difference_std",
        "mean_pair_top_k_jaccard",
        "mean_pair_kendall_tau",
        "mean_pair_spearman",
        "mean_preference_flip_rate_vs_first",
    )
    values_by_field: dict[str, list[float]] = {field: [] for field in fields}
    for estimator in range(raw.shape[1]):
        for head in range(raw.shape[2]):
            metrics = rank_stability(raw[:, estimator, head, :], top_k=top_k)
            surfaces.append({"estimator_index": estimator, "head_index": head, **metrics})
            for field in fields:
                values_by_field[field].append(float(metrics[field]))
    macro_mean = {field: _nanmean(values) for field, values in values_by_field.items()}
    macro_median = {field: _nanmedian(values) for field, values in values_by_field.items()}
    return {
        "surface_count": int(raw.shape[1] * raw.shape[2]),
        "macro_mean": macro_mean,
        "macro_median": macro_median,
        "surfaces": surfaces,
    }


def observe_anchors(
    *,
    alpha: ArrayLike,
    raw_scores: ArrayLike,
    support_ids: Sequence[str],
    anchor_ids: Sequence[str],
) -> AnchorObservation:
    """Align one Teacher extraction to its fixed anchor IDs and preserve axes."""

    ids = tuple(str(value) for value in support_ids)
    anchors = tuple(str(value) for value in anchor_ids)
    if len(ids) != len(set(ids)) or len(anchors) < 3 or len(anchors) != len(set(anchors)):
        raise ContextSensitivityError("support and anchor IDs must be unique, with at least three anchors")
    missing = [value for value in anchors if value not in ids]
    if missing:
        raise ContextSensitivityError("fixed anchors are missing from a context: " + ",".join(missing))
    alpha_array = np.asarray(alpha, dtype=np.float64)
    raw_array = np.asarray(raw_scores, dtype=np.float64)
    if alpha_array.shape != (1, len(ids)):
        raise ContextSensitivityError("alpha must have layout [1, support] aligned to support_ids")
    if raw_array.ndim != 4 or raw_array.shape[2:] != (1, len(ids)):
        raise ContextSensitivityError("raw_scores must have layout [estimator, head, 1, support]")
    if not np.isfinite(alpha_array).all() or not np.isfinite(raw_array).all() or np.any(alpha_array < 0.0):
        raise ContextSensitivityError("Teacher alpha/raw scores must be finite, with non-negative alpha")
    alpha_sum = float(alpha_array.sum())
    if not np.isclose(alpha_sum, 1.0, rtol=0.0, atol=1e-6):
        raise ContextSensitivityError("Teacher alpha does not form a probability simplex")
    indices = tuple(ids.index(anchor) for anchor in anchors)
    anchor_alpha = np.take(alpha_array[0], indices, axis=0).astype(np.float64, copy=False)
    anchor_total = float(anchor_alpha.sum())
    if anchor_total <= 0.0:
        raise ContextSensitivityError("Teacher assigned no alpha mass to the fixed anchors")
    anchor_raw = np.take(raw_array[:, :, 0, :], indices, axis=2).astype(np.float64, copy=False)
    shifted = anchor_raw - anchor_raw.max(axis=2, keepdims=True)
    conditional = np.exp(shifted)
    conditional = conditional / conditional.sum(axis=2, keepdims=True)
    mean_raw = anchor_raw.mean(axis=(0, 1))
    pairs, pairwise_raw = pairwise_differences(anchor_raw)
    _, pairwise_mean = pairwise_differences(mean_raw)
    full_mean = raw_array[:, :, 0, :].mean(axis=(0, 1))
    ranked = tuple(ids[index] for index in np.argsort(-full_mean, kind="stable").tolist())
    anchor_ranked = tuple(anchors[index] for index in np.argsort(-mean_raw, kind="stable").tolist())
    return AnchorObservation(
        anchor_indices=indices,
        anchor_alpha=np.ascontiguousarray(anchor_alpha),
        anchor_alpha_normalized=np.ascontiguousarray(anchor_alpha / anchor_total),
        anchor_alpha_total=anchor_total,
        anchor_raw_scores=np.ascontiguousarray(anchor_raw),
        anchor_raw_conditional=np.ascontiguousarray(conditional),
        anchor_mean_raw_scores=np.ascontiguousarray(mean_raw),
        pair_indices=pairs,
        pairwise_raw_score_differences=np.ascontiguousarray(pairwise_raw),
        pairwise_mean_raw_score_differences=np.ascontiguousarray(pairwise_mean),
        raw_score_ranking=ranked,
        anchor_raw_score_ranking=anchor_ranked,
    )


def summarize_contexts(
    anchor_alpha_by_context: ArrayLike,
    anchor_raw_by_context: ArrayLike,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Separate alpha competition from raw-score context sensitivity."""

    alpha = np.asarray(anchor_alpha_by_context, dtype=np.float64)
    raw = np.asarray(anchor_raw_by_context, dtype=np.float64)
    if alpha.ndim != 2 or raw.ndim != 4 or alpha.shape[0] != raw.shape[0] or alpha.shape[1] != raw.shape[-1]:
        raise ContextSensitivityError("alpha/raw context arrays must align on [contexts, anchors]")
    if not np.isfinite(alpha).all() or np.any(alpha < 0.0):
        raise ContextSensitivityError("anchor alpha must be finite and non-negative")
    alpha_mass = alpha.sum(axis=1)
    if np.any(alpha_mass <= 0.0):
        raise ContextSensitivityError("every context must preserve positive alpha mass on fixed anchors")
    alpha_normalized = alpha / alpha_mass[:, np.newaxis]
    alpha_metrics = _rename_metrics(rank_stability(alpha_normalized, top_k=top_k), prefix="alpha")
    mean_raw = raw.mean(axis=(1, 2))
    mean_raw_metrics = _rename_metrics(rank_stability(mean_raw, top_k=top_k), prefix="mean_raw_score")
    headwise = headwise_raw_score_stability(raw, top_k=top_k)
    return {
        "contexts": int(alpha.shape[0]),
        "anchors": int(alpha.shape[1]),
        "anchor_alpha_total_mean": float(alpha_mass.mean()),
        "anchor_alpha_total_std": float(alpha_mass.std()),
        "alpha_stability": alpha_metrics,
        "mean_raw_score_stability": mean_raw_metrics,
        "headwise_raw_score_stability": headwise,
    }


def classify_context_regime(summary: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen raw-score-only Week 6 regime rule to one query summary."""

    try:
        primary = summary["headwise_raw_score_stability"]["macro_mean"]
        weak = gate["weak"]
        strong = gate["strong"]
        kendall = float(primary["mean_pair_kendall_tau"])
        spearman = float(primary["mean_pair_spearman"])
        jaccard = float(primary["mean_pair_top_k_jaccard"])
        flips = float(primary["mean_preference_flip_rate_vs_first"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContextSensitivityError("context summary or frozen gate is incomplete") from error
    values = (kendall, spearman, jaccard, flips)
    finite = all(np.isfinite(value) for value in values)
    weak_conditions = {
        "raw_kendall": finite and kendall >= float(weak["min_raw_kendall_tau"]),
        "raw_spearman": finite and spearman >= float(weak["min_raw_spearman"]),
        "raw_top_k_jaccard": finite and jaccard >= float(weak["min_raw_top_k_jaccard"]),
        "raw_preference_flip": finite and flips <= float(weak["max_raw_preference_flip_rate"]),
    }
    strong_conditions = {
        "raw_kendall": finite and kendall <= float(strong["max_raw_kendall_tau"]),
        "raw_spearman": finite and spearman <= float(strong["max_raw_spearman"]),
        "raw_top_k_jaccard": finite and jaccard <= float(strong["max_raw_top_k_jaccard"]),
        "raw_preference_flip": finite and flips >= float(strong["min_raw_preference_flip_rate"]),
    }
    is_weak = bool(all(weak_conditions.values()))
    strong_trigger_count = int(sum(strong_conditions.values()))
    is_strong = bool(strong_trigger_count >= int(strong["minimum_trigger_count"]))
    if is_weak and is_strong:
        raise ContextSensitivityError("frozen weak and strong context rules overlap")
    regime = "weak" if is_weak else ("strong" if is_strong else "moderate")
    return {
        "regime": regime,
        "weak": is_weak,
        "strong": is_strong,
        "raw_metric_finite": finite,
        "primary_raw_metrics": {
            "mean_pair_kendall_tau": kendall,
            "mean_pair_spearman": spearman,
            "mean_pair_top_k_jaccard": jaccard,
            "mean_preference_flip_rate_vs_first": flips,
        },
        "weak_conditions": weak_conditions,
        "strong_conditions": strong_conditions,
        "strong_trigger_count": strong_trigger_count,
    }


__all__ = [
    "AnchorObservation",
    "ContextDefinition",
    "ContextSensitivityError",
    "classify_context_regime",
    "derive_seed",
    "headwise_raw_score_stability",
    "kendall_tau",
    "observe_anchors",
    "pairwise_differences",
    "preference_flip_rate",
    "rank_stability",
    "sample_contexts",
    "select_anchor_positions",
    "select_query_positions",
    "spearman_correlation",
    "summarize_contexts",
    "top_k_jaccard",
]
