"""Frozen Week 10 non-Teacher baselines (pure NumPy except the MLP trainer).

All baselines train/fit on pseudo-support rows ONLY (train split). Queries
(pseudo or holdout) reuse frozen pseudo-support statistics. Holdout query
labels are never consumed here; they are released only for metric
computation in runner.py.

Methods (manifest methods.classical):
  raw: score = negative Euclidean distance on standardized features.
  pca: PCA(n_components=min(32, n_features, n_support-1)) fit on
       pseudo-support only; score = negative Euclidean distance in PCA space.
  mlp: supervised MLP classifier (hidden 128, ReLU) trained full-batch Adam
       on pseudo-support only, 50 epochs (5 smoke), lr 0.001, seed 0, CPU;
       score = negative Euclidean distance between penultimate-layer
       embeddings (h, pre-logits) of query and support rows.

Standardization mirrors models.py: mean/variance from the scored support
set (pseudo-support at train time, full support at eval time); queries
reuse those frozen statistics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaselineError(ValueError):
    """Raised when a Week 10 classical baseline cannot be built or scored safely."""


MLP_HIDDEN = 128
MLP_SEED = 0
MLP_LR = 0.001


def _finite_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise BaselineError(name + " must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise BaselineError(name + " must be finite")
    return np.ascontiguousarray(matrix)


def _finite_labels(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise BaselineError(name + " must be a non-empty one-dimensional vector")
    if not np.issubdtype(labels.dtype, np.integer):
        raise BaselineError(name + " must contain integral labels")
    return np.asarray(labels, dtype=np.int64)


def standardize_like_student(
    support: NDArray[np.float64], queries: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Standardize with support statistics only; queries reuse them."""

    mean = support.mean(axis=0)
    std = support.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return (
        np.ascontiguousarray((support - mean) / std),
        np.ascontiguousarray((queries - mean) / std),
        np.ascontiguousarray(mean),
        np.ascontiguousarray(std),
    )


def negative_euclidean_scores(
    query_vectors: ArrayLike, support_vectors: ArrayLike
) -> NDArray[np.float64]:
    """Score queries against supports by negative Euclidean distance."""

    queries = _finite_matrix(query_vectors, name="query_vectors")
    supports = _finite_matrix(support_vectors, name="support_vectors")
    if queries.shape[1] != supports.shape[1]:
        raise BaselineError("query/support vector widths do not match")
    diff = queries[:, np.newaxis, :] - supports[np.newaxis, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0))
    return np.ascontiguousarray(-dist)


def raw_scores(support_features: ArrayLike, query_features: ArrayLike) -> NDArray[np.float64]:
    """Raw baseline: negative Euclidean distance on standardized features."""

    support = _finite_matrix(support_features, name="support_features")
    queries = _finite_matrix(query_features, name="query_features")
    if support.shape[1] != queries.shape[1]:
        raise BaselineError("support/query feature widths do not match")
    support_std, queries_std, _, _ = standardize_like_student(support, queries)
    return negative_euclidean_scores(queries_std, support_std)


def fit_pca(
    support_standardized: ArrayLike, *, n_components: int = 32
) -> dict[str, Any]:
    """Fit a frozen PCA projector on standardized pseudo-support rows (NumPy SVD)."""

    matrix = _finite_matrix(support_standardized, name="support_standardized")
    n_support, n_features = matrix.shape
    rank = int(min(int(n_components), n_features, n_support - 1))
    if rank < 1:
        raise BaselineError("PCA needs at least one component")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.ascontiguousarray(vt[:rank].T)
    return {"components": components, "mean": np.ascontiguousarray(mean), "n_components": rank}


def apply_pca(projector: Mapping[str, Any], vectors: ArrayLike) -> NDArray[np.float64]:
    """Project standardized rows with a frozen PCA projector."""

    matrix = _finite_matrix(vectors, name="vectors")
    components = np.asarray(projector["components"], dtype=np.float64)
    mean = np.asarray(projector["mean"], dtype=np.float64).reshape(-1)
    if matrix.shape[1] != mean.shape[0] or components.shape[0] != mean.shape[0]:
        raise BaselineError("PCA projector does not match vector width")
    return np.ascontiguousarray((matrix - mean) @ components)


def pca_scores(
    support_features: ArrayLike, query_features: ArrayLike, *, n_components: int = 32
) -> NDArray[np.float64]:
    """PCA baseline: fit on pseudo-support only, negative distance in PCA space."""

    support = _finite_matrix(support_features, name="support_features")
    queries = _finite_matrix(query_features, name="query_features")
    if support.shape[1] != queries.shape[1]:
        raise BaselineError("support/query feature widths do not match")
    support_std, queries_std, _, _ = standardize_like_student(support, queries)
    projector = fit_pca(support_std, n_components=int(n_components))
    return negative_euclidean_scores(
        apply_pca(projector, queries_std), apply_pca(projector, support_std)
    )


def train_mlp_classifier(
    support_features: ArrayLike,
    support_labels: ArrayLike,
    *,
    hidden_dim: int = MLP_HIDDEN,
    epochs: int = 50,
    learning_rate: float = MLP_LR,
    seed: int = MLP_SEED,
) -> dict[str, Any]:
    """Train the frozen supervised MLP classifier on pseudo-support only; needs torch."""

    try:
        import torch
    except ImportError as error:
        raise BaselineError("torch is required to train the Week 10 MLP baseline") from error
    support = _finite_matrix(support_features, name="support_features")
    labels = _finite_labels(support_labels, name="support_labels")
    if support.shape[0] != labels.shape[0]:
        raise BaselineError("support features and labels do not align")
    classes = np.unique(labels)
    if classes.size < 2:
        raise BaselineError("MLP training needs at least two classes")
    remap = {int(v): i for i, v in enumerate(classes.tolist())}
    targets = np.asarray([remap[int(v)] for v in labels.tolist()], dtype=np.int64)
    if int(epochs) < 1 or float(learning_rate) <= 0.0 or int(hidden_dim) < 1:
        raise BaselineError("MLP hyperparameters are invalid")
    import os as _os
    import random as _random

    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    _random.seed(int(seed))
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    mean = support.mean(axis=0)
    std = support.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    support_std = (support - mean) / std
    n_features = support.shape[1]
    n_classes = int(classes.size)
    backbone = torch.nn.Sequential(
        torch.nn.Linear(n_features, int(hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
    ).double()
    head = torch.nn.Linear(int(hidden_dim), n_classes).double()
    for module in list(backbone) + [head]:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            torch.nn.init.zeros_(module.bias)
    backbone.train()
    head.train()
    optimizer = torch.optim.Adam([*backbone.parameters(), *head.parameters()], lr=float(learning_rate))
    inputs = torch.from_numpy(np.ascontiguousarray(support_std, dtype=np.float64)).double()
    target_t = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.int64))
    history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        hidden = backbone(inputs)
        logits = head(hidden)
        loss = torch.nn.functional.cross_entropy(logits, target_t)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    params: dict[str, Any] = {
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "classes": [int(v) for v in classes.tolist()],
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "loss_history": [float(v) for v in history],
        "weights": [],
        "biases": [],
    }
    with torch.no_grad():
        for module in list(backbone) + [head]:
            if isinstance(module, torch.nn.Linear):
                params["weights"].append(np.asarray(module.weight.detach().cpu().T, dtype=np.float64))
                params["biases"].append(np.asarray(module.bias.detach().cpu(), dtype=np.float64))
    if len(params["weights"]) != 3 or len(params["biases"]) != 3:
        raise BaselineError("MLP parameter capture is incomplete")
    return params


def mlp_penultimate_embeddings(params: Mapping[str, Any], features: ArrayLike) -> NDArray[np.float64]:
    """Embed rows with the frozen MLP penultimate layer (NumPy, no torch)."""

    matrix = _finite_matrix(features, name="features")
    mean = np.asarray(params["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(params["std"], dtype=np.float64).reshape(-1)
    if matrix.shape[1] != mean.shape[0]:
        raise BaselineError("feature width does not match MLP statistics")
    weights = params["weights"]
    biases = params["biases"]
    if len(weights) != 3 or len(biases) != 3:
        raise BaselineError("MLP parameters are malformed")
    hidden = (matrix - mean) / np.where(std > 0.0, std, 1.0)
    # Layers 0-1 form the penultimate embedding; layer 2 is the logit head.
    for index in (0, 1):
        hidden = hidden @ np.asarray(weights[index], dtype=np.float64) + np.asarray(
            biases[index], dtype=np.float64
        )
        hidden = np.maximum(hidden, 0.0)
    return np.ascontiguousarray(hidden)


def mlp_scores(
    params: Mapping[str, Any], support_features: ArrayLike, query_features: ArrayLike
) -> NDArray[np.float64]:
    """MLP baseline: negative distance between penultimate embeddings."""

    support = _finite_matrix(support_features, name="support_features")
    queries = _finite_matrix(query_features, name="query_features")
    if support.shape[1] != queries.shape[1]:
        raise BaselineError("support/query feature widths do not match")
    return negative_euclidean_scores(
        mlp_penultimate_embeddings(params, queries), mlp_penultimate_embeddings(params, support)
    )


__all__ = [
    "BaselineError",
    "MLP_HIDDEN",
    "MLP_SEED",
    "MLP_LR",
    "standardize_like_student",
    "negative_euclidean_scores",
    "raw_scores",
    "fit_pca",
    "apply_pca",
    "pca_scores",
    "train_mlp_classifier",
    "mlp_penultimate_embeddings",
    "mlp_scores",
]
