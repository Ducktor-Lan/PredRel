"""Matched SupCon baseline encoder (matched symmetric baseline).

A small MLP trained with supervised contrastive loss on support rows only.
Query rows never participate in training or standardization statistics.  The
torch training path runs on the server; every numerical helper used by the
local test suite is pure NumPy, and inference from frozen parameters is also
pure NumPy so evidence can be rechecked without torch.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SupconError(ValueError):
    """Raised when the SupCon baseline cannot be trained or applied safely."""


def standardize_with_support_stats(
    support_features: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Standardize support rows; queries must reuse the returned statistics."""

    matrix = np.asarray(support_features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise SupconError("support features must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise SupconError("support features are not finite")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return (matrix - mean) / std, mean, std


def apply_support_stats(
    features: ArrayLike, mean: ArrayLike, std: ArrayLike
) -> NDArray[np.float64]:
    """Standardize new rows with frozen support statistics."""

    matrix = np.asarray(features, dtype=np.float64)
    mean_vector = np.asarray(mean, dtype=np.float64).reshape(-1)
    std_vector = np.asarray(std, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != mean_vector.shape[0]:
        raise SupconError("feature width does not match support statistics")
    if not np.all(np.isfinite(matrix)):
        raise SupconError("features are not finite")
    return (matrix - mean_vector) / np.where(std_vector > 0.0, std_vector, 1.0)


def init_params_numpy(layer_dims: Sequence[int], rng: np.random.Generator) -> dict[str, Any]:
    """He-initialize MLP parameters as plain NumPy arrays."""

    dims = [int(value) for value in layer_dims]
    if len(dims) < 2 or any(value < 1 for value in dims):
        raise SupconError("layer dimensions are invalid")
    weights: list[NDArray[np.float64]] = []
    biases: list[NDArray[np.float64]] = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:], strict=True):
        weights.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)))
        biases.append(np.zeros(fan_out, dtype=np.float64))
    return {"weights": weights, "biases": biases}


def mlp_forward_numpy(params: Mapping[str, Any], features: ArrayLike) -> NDArray[np.float64]:
    """MLP forward pass with ReLU hidden layers and an L2-normalized head."""

    matrix = np.asarray(features, dtype=np.float64)
    weights = params["weights"]
    biases = params["biases"]
    if len(weights) != len(biases) or not weights:
        raise SupconError("MLP parameters are malformed")
    hidden = matrix
    for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        hidden = hidden @ np.asarray(weight, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
        if index < len(weights) - 1:
            hidden = np.maximum(hidden, 0.0)
    norms = np.linalg.norm(hidden, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return np.ascontiguousarray(hidden / norms)


def supcon_loss_numpy(
    embeddings: ArrayLike, labels: ArrayLike, *, temperature: float
) -> float:
    """Supervised contrastive loss over L2-normalized embeddings (Khosla et al.).

    Anchors without a same-class partner in the batch are excluded, and the
    loss raises if no anchor has a positive.
    """

    if temperature <= 0.0:
        raise SupconError("temperature must be positive")
    vectors = np.asarray(embeddings, dtype=np.float64)
    targets = np.asarray(labels).reshape(-1)
    if vectors.ndim != 2 or vectors.shape[0] != targets.shape[0] or vectors.shape[0] < 2:
        raise SupconError("embeddings and labels do not form a valid batch")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms > 0.0, norms, 1.0)
    logits = (unit @ unit.T) / float(temperature)
    np.fill_diagonal(logits, -np.inf)
    log_denom = np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1))
    losses: list[float] = []
    for anchor in range(vectors.shape[0]):
        positives = np.flatnonzero((targets == targets[anchor]) & (np.arange(vectors.shape[0]) != anchor))
        if positives.size == 0:
            continue
        log_probs = logits[anchor, positives] - logits[anchor].max() - log_denom[anchor]
        losses.append(float(-log_probs.mean()))
    if not losses:
        raise SupconError("batch has no anchor with a same-class positive")
    return float(np.mean(losses))


def cosine_similarities(
    query_embeddings: ArrayLike, support_embeddings: ArrayLike
) -> NDArray[np.float64]:
    """Cosine similarity of L2-normalized query rows against support rows."""

    queries = np.asarray(query_embeddings, dtype=np.float64)
    supports = np.asarray(support_embeddings, dtype=np.float64)
    if queries.ndim != 2 or supports.ndim != 2 or queries.shape[1] != supports.shape[1]:
        raise SupconError("embedding widths do not match")
    return np.ascontiguousarray(queries @ supports.T)


def rank_supports_by_similarity(similarities: ArrayLike) -> list[list[int]]:
    """Descending support rank per query with stable tie-breaking."""

    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2:
        raise SupconError("similarities must be a 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise SupconError("similarities are not finite")
    return [list(np.argsort(-row, kind="stable").tolist()) for row in matrix]


def deterministic_batches(sample_count: int, batch_size: int, seed: int, epoch: int) -> list[NDArray[np.int64]]:
    """Deterministic shuffled batch index sets for one epoch."""

    if sample_count < 1 or batch_size < 1:
        raise SupconError("batch arguments are invalid")
    rng = np.random.default_rng([int(seed), int(epoch)])
    permutation = rng.permutation(sample_count)
    size = int(min(batch_size, sample_count))
    return [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(permutation, max(1, sample_count // size))]


def train_supcon_encoder(
    support_features: ArrayLike,
    support_labels: ArrayLike,
    *,
    hidden_dims: Sequence[int],
    embedding_dim: int,
    temperature: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Train the frozen SupCon MLP on support rows only; requires torch."""

    try:
        import torch
    except ImportError as error:
        raise SupconError("torch is required to train the SupCon baseline") from error
    matrix = np.asarray(support_features, dtype=np.float64)
    labels = np.asarray(support_labels).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != labels.shape[0]:
        raise SupconError("support features and labels do not align")
    if epochs < 1 or learning_rate <= 0.0 or temperature <= 0.0 or embedding_dim < 1:
        raise SupconError("SupCon hyperparameters are invalid")
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        raise SupconError("SupCon training needs at least two classes with two rows each")
    import os as _os
    import random as _random

    # NOTE (Week 8 v3 lesson): torch.use_deterministic_algorithms(True) is
    # deliberately NOT enabled. The Week 1 Teacher (TabPFN preprocessing SVD
    # on CUDA) has non-deterministic CuBLAS kernels without which it refuses
    # to run; the Teacher is an upstream frozen dependency we must not break.
    # Student determinism rests on: fixed seeds, single-threaded CPU training
    # (torch CPU kernels used here are deterministic), and the verifier's
    # 1e-6 cross-process recomputation tolerance for float residue.
    _os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    _random.seed(int(seed))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    standardized, mean, std = standardize_with_support_stats(matrix)
    dims = [matrix.shape[1], *[int(value) for value in hidden_dims], int(embedding_dim)]
    layers: list[Any] = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:], strict=True):
        linear = torch.nn.Linear(fan_in, fan_out)
        torch.nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        torch.nn.init.zeros_(linear.bias)
        layers.append(linear)
        if fan_out != dims[-1]:
            layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers).double()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    inputs = torch.from_numpy(np.ascontiguousarray(standardized, dtype=np.float64)).double()
    targets = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    count = matrix.shape[0]
    for epoch in range(int(epochs)):
        for batch in deterministic_batches(count, int(batch_size), int(seed), epoch):
            index = torch.from_numpy(np.ascontiguousarray(batch))
            batch_inputs = inputs[index]
            batch_targets = targets[index]
            vectors = torch.nn.functional.normalize(model(batch_inputs), p=2, dim=1)
            logits = (vectors @ vectors.T) / float(temperature)
            logits.fill_diagonal_(-float("inf"))
            log_denom = torch.logsumexp(logits, dim=1)
            anchor_losses: list[Any] = []
            for anchor in range(batch_inputs.shape[0]):
                positives = ((batch_targets == batch_targets[anchor]) & (torch.arange(batch_inputs.shape[0]) != anchor)).nonzero(as_tuple=True)[0]
                if positives.numel() == 0:
                    continue
                log_probs = logits[anchor, positives] - log_denom[anchor]
                anchor_losses.append(-log_probs.mean())
            if not anchor_losses:
                continue
            optimizer.zero_grad()
            torch.stack(anchor_losses).mean().backward()
            optimizer.step()
    params: dict[str, Any] = {"weights": [], "biases": []}
    with torch.no_grad():
        for module in model:
            if isinstance(module, torch.nn.Linear):
                params["weights"].append(np.asarray(module.weight.detach().cpu().T, dtype=np.float64))
                params["biases"].append(np.asarray(module.bias.detach().cpu(), dtype=np.float64))
    return {
        "weights": params["weights"],
        "biases": params["biases"],
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "layer_dims": dims,
        "temperature": float(temperature),
        "epochs": int(epochs),
        "seed": int(seed),
    }


def embed_with_params(params: Mapping[str, Any], features: ArrayLike) -> NDArray[np.float64]:
    """Embed rows with frozen parameters using support statistics."""

    return mlp_forward_numpy(params, apply_support_stats(features, params["mean"], params["std"]))


__all__ = [
    "SupconError",
    "apply_support_stats",
    "cosine_similarities",
    "deterministic_batches",
    "embed_with_params",
    "init_params_numpy",
    "mlp_forward_numpy",
    "rank_supports_by_similarity",
    "standardize_with_support_stats",
    "supcon_loss_numpy",
    "train_supcon_encoder",
]