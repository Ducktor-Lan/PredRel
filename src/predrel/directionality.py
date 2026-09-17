"""Directed-relation machinery (directed matrix R vs symmetric S).

Week 5 asks whether the TabPFN decoder readout is fundamentally directed,
i.e. whether the query-to-support relation R(i,j) differs systematically
from R(j,i).  A static cosine geometry is symmetric by construction, so
strong directed structure would be evidence a symmetric student cannot
reproduce the Teacher behavior.

Protocol (frozen in the dataset manifest, leakage-safe):
  - From each frozen support split, take a deterministic stratified probe
    pool P of at most M rows (M=48 full, M=16 smoke).
  - For each ordered pair (i,j) with i != j, hold out pool row i as a
    single query and keep P \\ {i} as support (j stays in support).
    One frozen Teacher fit per held-out row yields row i of R.
  - Query labels are never supplied to the Teacher.  Labels are released
    only after all R rows are extracted, for stratified analysis.
  - The matched SupCon baseline (trained on support rows only, same as
    Week 4) gives a symmetric cosine matrix S on the same pool P.  The
    non-removal probe compares R against S (Top-K overlap, Spearman).

This module is pure NumPy so the local test suite runs without torch or
TabPFN.  The server runner imports the same functions for evidence.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DirectionalityError(ValueError):
    """Raised when a directed-relation computation is unsafe."""


def select_probe_pool(
    support_labels: ArrayLike, *, max_pool: int, seed: int
) -> list[int]:
    """Deterministically subsample probe positions with label round-robin.

    Groups support positions by label in ascending label order, then takes
    rows round-robin so small pools stay class-balanced.  Within each class
    bucket the original support order is kept, and a seeded shuffle of the
    bucket order is NOT applied: the output is fully determined by
    (support_labels, max_pool).  The seed is recorded by the caller for
    provenance but does not change the selection, which keeps smoke and
    full pools nested (the first min(len) entries of the full pool equal
    the smoke pool when max_pool grows).
    """
    labels = np.asarray(support_labels).reshape(-1)
    if labels.size == 0:
        raise DirectionalityError("cannot select a probe pool from an empty split")
    if int(max_pool) < 3:
        raise DirectionalityError("max_pool must be at least 3 for a directed analysis")
    if int(seed) < 0:
        raise DirectionalityError("seed must be non-negative")
    by_label: dict[int, list[int]] = {}
    for position, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), []).append(int(position))
    ordered_labels = sorted(by_label)
    selected: list[int] = []
    cursor = 0
    budget = int(min(int(max_pool), int(labels.size)))
    while len(selected) < budget:
        progressed = False
        for label in ordered_labels:
            bucket = by_label[label]
            if cursor < len(bucket) and len(selected) < budget:
                selected.append(bucket[cursor])
                progressed = True
        cursor += 1
        if not progressed:
            break
    return sorted(selected)


def _check_square(matrix: ArrayLike, *, name: str = "R") -> NDArray[np.float64]:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 3:
        raise DirectionalityError(name + " must be a square matrix with n >= 3")
    if not np.all(np.isfinite(mat)):
        raise DirectionalityError(name + " contains non-finite values")
    if np.any(mat < 0.0):
        raise DirectionalityError(name + " contains negative entries")
    return mat


def check_row_stochastic(
    matrix: ArrayLike, *, atol: float = 1e-6, name: str = "R"
) -> dict[str, float]:
    """Verify each directed row sums to one (alpha simplex per held-out query)."""
    mat = _check_square(matrix, name=name)
    row_sums = mat.sum(axis=1)
    deviation = np.abs(row_sums - 1.0)
    return {
        "max_abs_deviation": float(deviation.max()),
        "mean_abs_deviation": float(deviation.mean()),
        "within_tolerance": bool(np.all(deviation <= float(atol))),
    }


def asymmetry_metrics(matrix: ArrayLike, *, epsilon: float = 1e-12) -> dict[str, float]:
    """Core directed-vs-symmetric summary for one relation matrix."""
    mat = _check_square(matrix)
    n = int(mat.shape[0])
    fro = float(np.linalg.norm(mat, ord="fro"))
    diff = mat - mat.T
    asym_fro = float(np.linalg.norm(diff, ord="fro"))
    a_f = float(asym_fro / (fro + float(epsilon)))
    abs_diff = np.abs(diff)
    mask = ~np.eye(n, dtype=bool)
    off = abs_diff[mask]
    scale = np.maximum(np.maximum(mat[mask], mat.T[mask]), float(epsilon))
    rel = off / scale
    return {
        "n": float(n),
        "fro_norm": fro,
        "asym_fro": asym_fro,
        "A_F": float(min(max(a_f, 0.0), 2.0)),
        "mean_abs_asym": float(off.mean()),
        "max_abs_asym": float(off.max()),
        "median_abs_asym": float(np.median(off)),
        "mean_relative_asym": float(rel.mean()),
        "median_relative_asym": float(np.median(rel)),
        "pairwise_asym_rate_tol_1e9": float((off > 1e-9).mean()),
        "pairwise_asym_rate_tol_1e6": float((off > 1e-6).mean()),
    }


def topk_out(matrix: ArrayLike, k: int) -> list[list[int]]:
    """Top-K outgoing neighbours per row (excluding the diagonal)."""
    mat = _check_square(matrix)
    n = int(mat.shape[0])
    kk = int(max(1, min(int(k), n - 1)))
    out: list[list[int]] = []
    for i in range(n):
        row = mat[i].copy()
        row[i] = -np.inf
        order = np.argsort(-row, kind="stable")[:kk]
        out.append([int(v) for v in order.tolist()])
    return out


def topk_in(matrix: ArrayLike, k: int) -> list[list[int]]:
    """Top-K incoming neighbours per column (excluding the diagonal)."""
    mat = _check_square(matrix)
    return topk_out(mat.T, k)


def reciprocity_at_k(matrix: ArrayLike, k: int) -> dict[str, float]:
    """Overlap between outgoing and incoming Top-K sets per node."""
    mat = _check_square(matrix)
    n = int(mat.shape[0])
    kk = int(max(1, min(int(k), n - 1)))
    out_sets = [set(v) for v in topk_out(mat, kk)]
    in_sets = [set(v) for v in topk_in(mat, kk)]
    scores = np.asarray(
        [len(o & i) / float(kk) for o, i in zip(out_sets, in_sets)], dtype=np.float64
    )
    return {
        "k": float(kk),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "min": float(scores.min()),
        "fraction_zero_overlap": float((scores <= 0.0).mean()),
        "fraction_full_overlap": float((scores >= 1.0 - 1e-12).mean()),
    }


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = int(values.size)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_corr(a: ArrayLike, b: ArrayLike) -> float:
    """Spearman rank correlation for two equal-length vectors (no scipy)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3:
        raise DirectionalityError("spearman needs two aligned vectors of length >= 3")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise DirectionalityError("spearman inputs must be finite")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    cx = rx - rx.mean()
    cy = ry - ry.mean()
    denom = float(np.sqrt(float(cx @ cx) * float(cy @ cy)))
    if denom <= 0.0:
        return 0.0
    return float((cx @ cy) / denom)


def kendall_tau(a: ArrayLike, b: ArrayLike) -> float:
    """Kendall tau-b for two equal-length vectors (pure NumPy, O(n^2))."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3:
        raise DirectionalityError("kendall needs two aligned vectors of length >= 3")
    n = int(x.size)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                ties_x += 1
                continue
            if dy == 0.0:
                ties_y += 1
                continue
            if dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denom = float(np.sqrt(float(concordant + discordant + ties_x) * float(concordant + discordant + ties_y)))
    if denom <= 0.0:
        return 0.0
    return float((concordant - discordant) / denom)


def rowwise_rank_agreement(matrix: ArrayLike) -> dict[str, float]:
    """For each node, compare outgoing ranking vs incoming ranking."""
    mat = _check_square(matrix)
    n = int(mat.shape[0])
    spears: list[float] = []
    taus: list[float] = []
    reversals: list[float] = []
    for i in range(n):
        out_row = np.delete(mat[i], i)
        in_col = np.delete(mat[:, i], i)
        spears.append(spearman_corr(out_row, in_col))
        taus.append(kendall_tau(out_row, in_col))
        order_out = np.argsort(-out_row, kind="stable")
        order_in = np.argsort(-in_col, kind="stable")
        pos_in = np.empty_like(order_in)
        pos_in[order_in] = np.arange(order_in.size)
        discord = 0
        total = 0
        m = int(order_out.size)
        for a in range(m):
            for b in range(a + 1, m):
                total += 1
                if pos_in[order_out[a]] > pos_in[order_out[b]]:
                    discord += 1
        reversals.append(float(discord) / float(total) if total else 0.0)
    spears_arr = np.asarray(spears, dtype=np.float64)
    taus_arr = np.asarray(taus, dtype=np.float64)
    rev_arr = np.asarray(reversals, dtype=np.float64)
    return {
        "mean_spearman_out_vs_in": float(spears_arr.mean()),
        "median_spearman_out_vs_in": float(np.median(spears_arr)),
        "mean_kendall_out_vs_in": float(taus_arr.mean()),
        "median_kendall_out_vs_in": float(np.median(taus_arr)),
        "mean_rank_reversal_rate": float(rev_arr.mean()),
        "median_rank_reversal_rate": float(np.median(rev_arr)),
    }


def global_transpose_agreement(matrix: ArrayLike) -> dict[str, float]:
    """Flattened off-diagonal agreement between R and R-transpose."""
    mat = _check_square(matrix)
    n = int(mat.shape[0])
    mask = ~np.eye(n, dtype=bool)
    flat = mat[mask].reshape(-1)
    flat_t = mat.T[mask].reshape(-1)
    return {
        "spearman_R_vs_RT": spearman_corr(flat, flat_t),
        "kendall_R_vs_RT": kendall_tau(flat, flat_t),
    }


def heatmap_stats(matrix: ArrayLike) -> dict[str, Any]:
    """Compact R - R^T distribution summary for the report (no figure files)."""
    mat = _check_square(matrix)
    diff = mat - mat.T
    n = int(mat.shape[0])
    mask = ~np.eye(n, dtype=bool)
    vals = diff[mask].reshape(-1)
    return {
        "diff_mean": float(vals.mean()),
        "diff_std": float(vals.std()),
        "diff_min": float(vals.min()),
        "diff_max": float(vals.max()),
        "diff_quantiles": {
            "p01": float(np.quantile(vals, 0.01)),
            "p05": float(np.quantile(vals, 0.05)),
            "p25": float(np.quantile(vals, 0.25)),
            "p50": float(np.quantile(vals, 0.50)),
            "p75": float(np.quantile(vals, 0.75)),
            "p95": float(np.quantile(vals, 0.95)),
            "p99": float(np.quantile(vals, 0.99)),
        },
        "row_mean_out": [float(v) for v in mat.mean(axis=1).tolist()],
        "col_mean_in": [float(v) for v in mat.mean(axis=0).tolist()],
    }


def retrieval_overlap(
    teacher_matrix: ArrayLike, baseline_matrix: ArrayLike, ks: Sequence[int]
) -> dict[str, Any]:
    """Non-removal probe: Top-K retrieval agreement between R and S.

    For each query row, compare the Teacher Top-K outgoing set against the
    symmetric-baseline Top-K set (Jaccard + recall both directions) and the
    global flattened Spearman.  A symmetric baseline that fully captures the
    Teacher should score near 1; systematic gaps are the Week 5
    beyond-symmetric signal.
    """
    r_mat = _check_square(teacher_matrix, name="R")
    s_mat = _check_square(baseline_matrix, name="S")
    if r_mat.shape != s_mat.shape:
        raise DirectionalityError("R and S must share the same probe pool shape")
    n = int(r_mat.shape[0])
    out: dict[str, Any] = {"n": n, "per_k": {}}
    flat_r = r_mat[~np.eye(n, dtype=bool)].reshape(-1)
    flat_s = s_mat[~np.eye(n, dtype=bool)].reshape(-1)
    out["spearman_R_vs_S"] = spearman_corr(flat_r, flat_s)
    per_k: dict[str, Any] = {}
    for k in [int(v) for v in ks]:
        kk = int(max(1, min(k, n - 1)))
        r_top = [set(v) for v in topk_out(r_mat, kk)]
        s_top = [set(v) for v in topk_out(s_mat, kk)]
        jacc: list[float] = []
        rec_r_in_s: list[float] = []
        rec_s_in_r: list[float] = []
        for a, b in zip(r_top, s_top):
            union = len(a | b)
            jacc.append(float(len(a & b) / union) if union else 1.0)
            rec_r_in_s.append(float(len(a & b) / len(a)) if a else 1.0)
            rec_s_in_r.append(float(len(a & b) / len(b)) if b else 1.0)
        jacc_arr = np.asarray(jacc, dtype=np.float64)
        per_k[str(kk)] = {
            "k": kk,
            "mean_jaccard": float(jacc_arr.mean()),
            "median_jaccard": float(np.median(jacc_arr)),
            "mean_recall_R_in_S": float(np.mean(rec_r_in_s)),
            "mean_recall_S_in_R": float(np.mean(rec_s_in_r)),
        }
    out["per_k"] = per_k
    return out


def summarize_directed_matrix(
    matrix: ArrayLike, *, ks: Sequence[int] = (1, 5, 10)
) -> dict[str, Any]:
    """One-call frozen summary for a directed Teacher matrix R."""
    mat = _check_square(matrix)
    summary: dict[str, Any] = {"stochastic": check_row_stochastic(mat)}
    summary["asymmetry"] = asymmetry_metrics(mat)
    summary["reciprocity"] = {str(k): reciprocity_at_k(mat, int(k)) for k in ks}
    summary["rank_agreement"] = rowwise_rank_agreement(mat)
    summary["transpose_agreement"] = global_transpose_agreement(mat)
    summary["heatmap"] = heatmap_stats(mat)
    return summary


def summarize_baseline_matrix(
    matrix: ArrayLike, *, ks: Sequence[int] = (1, 5, 10)
) -> dict[str, Any]:
    """Same summary for the symmetric SupCon matrix S (reference scale)."""
    return summarize_directed_matrix(matrix, ks=ks)


__all__ = [
    "DirectionalityError",
    "select_probe_pool",
    "check_row_stochastic",
    "asymmetry_metrics",
    "topk_out",
    "topk_in",
    "reciprocity_at_k",
    "spearman_corr",
    "kendall_tau",
    "rowwise_rank_agreement",
    "global_transpose_agreement",
    "heatmap_stats",
    "retrieval_overlap",
    "summarize_directed_matrix",
    "summarize_baseline_matrix",
]
