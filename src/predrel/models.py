"""Route B static-directed student (pure NumPy, no torch import).

Architecture (plan Section 8, Week 7 hand-off):
  features -> standardized MLP (hidden 128, ReLU) -> h (128)
  q = W_Q h (64), k = W_K h (64)
  s_ij = q_i . k_j / sqrt(64)

The optional simple context adapter uses the mean-pool additive form:
  c = mean of support h rows; q'_i = q_i + A_q c; k'_j = k_j + A_k c.
DeepSets is intentionally NOT implemented in v1.

Standardization uses the pseudo-support statistics only (leakage boundary);
pseudo-queries, full support, and holdout queries all reuse them.

Training (frozen for v1): full-batch Adam on CPU, beta-KL first with a
pairwise logistic ranking term on the teacher mean-raw true-class block,
weight 0.5. The teacher target for training selects the train pseudo-query's
true-class block only; holdout queries never enter training.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class StudentError(ValueError):
    """Raised when the Route B student cannot be built, trained, or scored safely."""


HIDDEN_DIM = 128
H_DIM = 128
QK_DIM = 64
EPSILON = 1e-12


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable non-secret uint32 seed without Python hash randomization."""

    if not isinstance(base_seed, int) or base_seed < 0:
        raise StudentError("base_seed must be a non-negative integer")
    material = "\0".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], byteorder="big") % (2**32)


def _finite_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise StudentError(name + " must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise StudentError(name + " must be finite")
    return np.ascontiguousarray(matrix)


def _finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise StudentError(name + " must be a non-empty finite vector")
    return np.ascontiguousarray(vector)


def _finite_labels(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise StudentError(name + " must be a non-empty one-dimensional vector")
    if not np.issubdtype(labels.dtype, np.integer):
        raise StudentError(name + " must contain integral labels")
    return np.asarray(labels, dtype=np.int64)


def select_pseudo_queries(
    support_count: int,
    *,
    query_fraction: float,
    max_queries: int,
    min_queries: int,
    seed: int,
) -> tuple[int, ...]:
    """Select pseudo-query positions without inspecting any labels."""

    if (
        not isinstance(support_count, int)
        or support_count < 4
        or not 0.0 < float(query_fraction) < 1.0
        or int(max_queries) < 1
        or int(min_queries) < 1
        or int(min_queries) > int(max_queries)
    ):
        raise StudentError("pseudo-query selection arguments are invalid")
    budget = int(min(int(max_queries), max(int(min_queries), int(round(support_count * float(query_fraction))))))
    budget = int(min(budget, support_count - 1))
    if budget < int(min_queries):
        raise StudentError("support split is too small for the frozen pseudo-query minimum")
    rng = np.random.default_rng(derive_seed(int(seed), "pseudo-queries", support_count, budget))
    chosen = rng.choice(support_count, size=budget, replace=False)
    return tuple(int(value) for value in sorted(chosen.tolist()))


@dataclass(frozen=True, slots=True)
class StudentParams:
    """Frozen NumPy parameters for one Route B student."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    hidden_dims: tuple[int, ...]
    weights: tuple[NDArray[np.float64], ...]
    biases: tuple[NDArray[np.float64], ...]
    w_q: NDArray[np.float64]
    w_k: NDArray[np.float64]
    adapter: bool
    a_q: NDArray[np.float64] | None
    a_k: NDArray[np.float64] | None
    seed: int
    epochs: int


def init_student_params(
    feature_dim: int,
    *,
    hidden_dims: Sequence[int] = (HIDDEN_DIM,),
    qk_dim: int = QK_DIM,
    adapter: bool = False,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    """He-initialize the MLP backbone plus Q/K heads (and adapter when requested)."""

    dims = [int(feature_dim), *[int(value) for value in hidden_dims], H_DIM]
    if len(dims) < 2 or any(value < 1 for value in dims) or int(qk_dim) < 1:
        raise StudentError("student dimensions are invalid")
    if int(seed) < 0 or int(epochs) < 1:
        raise StudentError("student seed/epochs are invalid")
    rng = np.random.default_rng(int(seed))
    weights: list[NDArray[np.float64]] = []
    biases: list[NDArray[np.float64]] = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:], strict=True):
        weights.append(np.ascontiguousarray(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))))
        biases.append(np.zeros(fan_out, dtype=np.float64))
    w_q = np.ascontiguousarray(rng.normal(0.0, np.sqrt(2.0 / H_DIM), size=(H_DIM, int(qk_dim))))
    w_k = np.ascontiguousarray(rng.normal(0.0, np.sqrt(2.0 / H_DIM), size=(H_DIM, int(qk_dim))))
    params: dict[str, Any] = {
        "weights": weights,
        "biases": biases,
        "w_q": w_q,
        "w_k": w_k,
        "adapter": bool(adapter),
        "a_q": None,
        "a_k": None,
        "layer_dims": dims,
        "qk_dim": int(qk_dim),
        "seed": int(seed),
        "epochs": int(epochs),
    }
    if adapter:
        scale = np.sqrt(2.0 / H_DIM)
        params["a_q"] = np.ascontiguousarray(rng.normal(0.0, scale, size=(H_DIM, int(qk_dim))))
        params["a_k"] = np.ascontiguousarray(rng.normal(0.0, scale, size=(H_DIM, int(qk_dim))))
    return params


def _backbone_h(params: Mapping[str, Any], standardized: NDArray[np.float64]) -> NDArray[np.float64]:
    weights = params["weights"]
    biases = params["biases"]
    if len(weights) != len(biases) or not weights:
        raise StudentError("student MLP parameters are malformed")
    hidden = np.asarray(standardized, dtype=np.float64)
    for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        hidden = hidden @ np.asarray(weight, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
        if index < len(weights) - 1:
            hidden = np.maximum(hidden, 0.0)
    if hidden.shape[1] != H_DIM:
        raise StudentError("student backbone must emit h with width 128")
    return np.ascontiguousarray(hidden)


def _heads(
    params: Mapping[str, Any], hidden: NDArray[np.float64], *, support_hidden: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    queries = np.asarray(hidden @ np.asarray(params["w_q"], dtype=np.float64))
    keys = np.asarray(hidden @ np.asarray(params["w_k"], dtype=np.float64))
    if bool(params.get("adapter", False)):
        if support_hidden is None or params.get("a_q") is None or params.get("a_k") is None:
            raise StudentError("adapter context is missing")
        context = np.asarray(support_hidden, dtype=np.float64).mean(axis=0)
        queries = queries + context @ np.asarray(params["a_q"], dtype=np.float64)
        keys = keys + context @ np.asarray(params["a_k"], dtype=np.float64)
    return np.ascontiguousarray(queries), np.ascontiguousarray(keys)


def directed_scores_numpy(params: Mapping[str, Any], features: ArrayLike) -> NDArray[np.float64]:
    """Score every row against every row with the frozen static student."""

    matrix = _finite_matrix(features, name="features")
    mean = np.asarray(params["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(params["std"], dtype=np.float64).reshape(-1)
    if matrix.shape[1] != mean.shape[0]:
        raise StudentError("feature width does not match student statistics")
    standardized = (matrix - mean) / np.where(std > 0.0, std, 1.0)
    hidden = _backbone_h(params, np.ascontiguousarray(standardized))
    queries, keys = _heads(params, hidden, support_hidden=hidden)
    scale = float(np.sqrt(float(params.get("qk_dim", QK_DIM))))
    return np.ascontiguousarray(queries @ keys.T / scale)


def directed_scores_with_support_context(
    params: Mapping[str, Any], query_features: ArrayLike, support_features: ArrayLike
) -> NDArray[np.float64]:
    """Score queries against supports; the adapter context comes from the support set."""

    queries_in = _finite_matrix(query_features, name="query_features")
    supports_in = _finite_matrix(support_features, name="support_features")
    if queries_in.shape[1] != supports_in.shape[1]:
        raise StudentError("query/support feature widths do not match")
    mean = np.asarray(params["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(params["std"], dtype=np.float64).reshape(-1)
    if queries_in.shape[1] != mean.shape[0]:
        raise StudentError("feature width does not match student statistics")
    query_std = (queries_in - mean) / np.where(std > 0.0, std, 1.0)
    support_std = (supports_in - mean) / np.where(std > 0.0, std, 1.0)
    query_hidden = _backbone_h(params, np.ascontiguousarray(query_std))
    support_hidden = _backbone_h(params, np.ascontiguousarray(support_std))
    query_q, _ = _heads(params, query_hidden, support_hidden=support_hidden)
    _, support_k = _heads(params, support_hidden, support_hidden=support_hidden)
    scale = float(np.sqrt(float(params.get("qk_dim", QK_DIM))))
    return np.ascontiguousarray(query_q @ support_k.T / scale)


def student_params_to_record(params: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize frozen parameters to JSON-safe lists for evidence."""

    return {
        "mean": np.asarray(params["mean"], dtype=np.float64).tolist(),
        "std": np.asarray(params["std"], dtype=np.float64).tolist(),
        "layer_dims": [int(value) for value in params["layer_dims"]],
        "qk_dim": int(params["qk_dim"]),
        "weights": [np.asarray(w, dtype=np.float64).tolist() for w in params["weights"]],
        "biases": [np.asarray(b, dtype=np.float64).tolist() for b in params["biases"]],
        "w_q": np.asarray(params["w_q"], dtype=np.float64).tolist(),
        "w_k": np.asarray(params["w_k"], dtype=np.float64).tolist(),
        "adapter": bool(params.get("adapter", False)),
        "a_q": None if params.get("a_q") is None else np.asarray(params["a_q"], dtype=np.float64).tolist(),
        "a_k": None if params.get("a_k") is None else np.asarray(params["a_k"], dtype=np.float64).tolist(),
        "seed": int(params["seed"]),
        "epochs": int(params["epochs"]),
    }


def student_params_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Restore frozen parameters from an evidence record."""

    try:
        params: dict[str, Any] = {
            "mean": np.asarray(record["mean"], dtype=np.float64),
            "std": np.asarray(record["std"], dtype=np.float64),
            "layer_dims": [int(value) for value in record["layer_dims"]],
            "qk_dim": int(record["qk_dim"]),
            "weights": [np.asarray(w, dtype=np.float64) for w in record["weights"]],
            "biases": [np.asarray(b, dtype=np.float64) for b in record["biases"]],
            "w_q": np.asarray(record["w_q"], dtype=np.float64),
            "w_k": np.asarray(record["w_k"], dtype=np.float64),
            "adapter": bool(record["adapter"]),
            "a_q": None if record.get("a_q") is None else np.asarray(record["a_q"], dtype=np.float64),
            "a_k": None if record.get("a_k") is None else np.asarray(record["a_k"], dtype=np.float64),
            "seed": int(record["seed"]),
            "epochs": int(record["epochs"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise StudentError("student parameter record is malformed") from error
    return params


__all__ = [
    "StudentError",
    "HIDDEN_DIM",
    "H_DIM",
    "QK_DIM",
    "EPSILON",
    "derive_seed",
    "select_pseudo_queries",
    "StudentParams",
    "init_student_params",
    "directed_scores_numpy",
    "directed_scores_with_support_context",
    "student_params_to_record",
    "student_params_from_record",
]
