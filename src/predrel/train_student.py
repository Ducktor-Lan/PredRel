"""Torch training loop for one ablation cell (server only).

The loop mirrors ``models.py``/``losses.py`` exactly:
  standardized MLP (ReLU) -> h -> Q/K heads -> s = Q K^T / sqrt(qk_dim),
  optional mean-pool additive adapter, per-cell target/loss from the frozen
  grid (``provenance/dataset_manifest.json``). Full-batch Adam on CPU.

Score scale note: the score is ``q_i . k_j / sqrt(qk_dim)`` with NO q/k
normalization, identical to Week 8. Raw-MSE cells therefore train student
scores directly against the teacher native pre-softmax scale (~10-18);
no teacher-side rescaling is applied (frozen Week 10 decision).

The module imports torch lazily so the package stays importable without it.
NumPy reference parity is covered by tests (forward scores and loss values).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .losses import EPSILON
from .models import H_DIM, StudentError, init_student_params


def _as_float_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise StudentError(name + " must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise StudentError(name + " must be finite")
    return np.ascontiguousarray(matrix)


def _finite_block_row(values: object, *, name: str) -> NDArray[np.float64]:
    row = np.asarray(values, dtype=np.float64).reshape(-1)
    if row.size < 2 or not np.all(np.isfinite(row)):
        raise StudentError(name + " rows must be finite vectors with at least two entries")
    return np.ascontiguousarray(row)


def train_ablation_cell(
    *,
    support_features: ArrayLike,
    pseudo_query_features: ArrayLike,
    teacher_alpha_rows: ArrayLike | None = None,
    teacher_beta_blocks: ArrayLike | None = None,
    teacher_raw_blocks: ArrayLike | None = None,
    true_positions: list[list[int]] | tuple[tuple[int, ...], ...],
    target: str,
    loss: str,
    hidden_dims: list[int] | tuple[int, ...] = (128,),
    qk_dim: int = 64,
    adapter: bool = False,
    learning_rate: float = 0.001,
    epochs: int = 50,
    lambda_ranking: float = 0.5,
    seed: int = 0,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Train one Route B student cell on pseudo-queries; requires torch."""

    try:
        import torch
    except ImportError as error:
        raise StudentError("torch is required to train the Week 10 ablation cell") from error
    if target not in ("alpha", "beta", "raw", "centered"):
        raise StudentError("target must be one of alpha/beta/raw/centered")
    if loss not in ("KL", "rank", "KL+rank", "MSE", "MSE+rank"):
        raise StudentError("loss must be one of KL/rank/KL+rank/MSE/MSE+rank")
    if target == "alpha" and (loss != "KL" or adapter):
        raise StudentError("alpha target requires the KL loss on the static variant")
    if target in ("raw", "centered") and loss not in ("MSE", "MSE+rank"):
        raise StudentError(target + " target requires the MSE loss family")
    if target == "beta" and loss not in ("KL", "rank", "KL+rank"):
        raise StudentError("beta target requires the KL/ranking loss family")
    support = _as_float_matrix(support_features, name="support_features")
    queries = _as_float_matrix(pseudo_query_features, name="pseudo_query_features")
    blocks = [tuple(int(v) for v in row) for row in true_positions]
    if queries.shape[0] != len(blocks):
        raise StudentError("pseudo-query rows do not align across inputs")
    beta_rows = (
        [_finite_block_row(row, name="teacher_beta_blocks") for row in teacher_beta_blocks]
        if teacher_beta_blocks is not None
        else None
    )
    raw_rows = (
        [_finite_block_row(row, name="teacher_raw_blocks") for row in teacher_raw_blocks]
        if teacher_raw_blocks is not None
        else None
    )
    alpha_rows = (
        [_finite_block_row(row, name="teacher_alpha_rows") for row in teacher_alpha_rows]
        if teacher_alpha_rows is not None
        else None
    )
    n_queries = len(blocks)
    if target == "alpha":
        if alpha_rows is None or len(alpha_rows) != n_queries:
            raise StudentError("alpha target needs one full teacher row per pseudo-query")
        for row in alpha_rows:
            if row.shape[0] != support.shape[0]:
                raise StudentError("teacher alpha rows must cover the full pseudo-support")
    else:
        if beta_rows is not None and len(beta_rows) != n_queries:
            raise StudentError("pseudo-query rows do not align across inputs")
        if raw_rows is not None and len(raw_rows) != n_queries:
            raise StudentError("pseudo-query rows do not align across inputs")
        if loss in ("KL", "KL+rank") and target == "beta" and beta_rows is None:
            raise StudentError("beta-KL needs teacher beta blocks")
        if loss in ("MSE", "MSE+rank", "rank", "KL+rank") and raw_rows is None:
            raise StudentError("ranking/MSE needs teacher raw blocks")
    if any(len(row) < 2 for row in blocks):
        raise StudentError("every train pseudo-query needs at least two true-class supports")
    for qi, row in enumerate(blocks):
        if any(v < 0 or v >= support.shape[0] for v in row) or len(set(row)) != len(row):
            raise StudentError("true-class support positions are invalid")
        if beta_rows is not None and len(row) != beta_rows[qi].shape[0]:
            raise StudentError("true-class block width does not match teacher beta blocks")
        if raw_rows is not None and len(row) != raw_rows[qi].shape[0]:
            raise StudentError("true-class block width does not match teacher raw blocks")
    if support.shape[1] != queries.shape[1]:
        raise StudentError("support/query feature widths do not match")
    if int(epochs) < 1 or float(learning_rate) <= 0.0 or int(qk_dim) < 1:
        raise StudentError("student hyperparameters are invalid")
    if float(epsilon) <= 0.0 or float(lambda_ranking) < 0.0:
        raise StudentError("loss hyperparameters are invalid")

    import os as _os
    import random as _random

    # NOTE (Week 8 v3 lesson, carried into Week 10): torch.use_deterministic_algorithms(True)
    # is deliberately NOT enabled. The Week 1 Teacher (TabPFN preprocessing SVD
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

    reference = init_student_params(
        support.shape[1], hidden_dims=list(hidden_dims), qk_dim=int(qk_dim),
        adapter=bool(adapter), seed=int(seed), epochs=int(epochs),
    )
    mean = support.mean(axis=0)
    std = support.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    support_std = (support - mean) / std
    queries_std = (queries - mean) / std

    dims: list[int] = [int(v) for v in reference["layer_dims"]]
    layers: list[Any] = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:], strict=True):
        linear = torch.nn.Linear(fan_in, fan_out)
        torch.nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        torch.nn.init.zeros_(linear.bias)
        layers.append(linear)
        if fan_out != dims[-1]:
            layers.append(torch.nn.ReLU())
    backbone = torch.nn.Sequential(*layers).double()
    w_q = torch.nn.Linear(H_DIM, int(qk_dim), bias=False).double()
    w_k = torch.nn.Linear(H_DIM, int(qk_dim), bias=False).double()
    torch.nn.init.kaiming_normal_(w_q.weight, nonlinearity="linear")
    torch.nn.init.kaiming_normal_(w_k.weight, nonlinearity="linear")
    a_q = torch.nn.Linear(H_DIM, int(qk_dim), bias=False).double() if adapter else None
    a_k = torch.nn.Linear(H_DIM, int(qk_dim), bias=False).double() if adapter else None
    if a_q is not None and a_k is not None:
        torch.nn.init.kaiming_normal_(a_q.weight, nonlinearity="linear")
        torch.nn.init.kaiming_normal_(a_k.weight, nonlinearity="linear")

    backbone.train()
    w_q.train()
    w_k.train()
    parameters: list[Any] = [*list(backbone.parameters()), *list(w_q.parameters()), *list(w_k.parameters())]
    if a_q is not None and a_k is not None:
        a_q.train()
        a_k.train()
        parameters.extend([*list(a_q.parameters()), *list(a_k.parameters())])
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))

    support_t = torch.from_numpy(np.ascontiguousarray(support_std, dtype=np.float64)).double()
    queries_t = torch.from_numpy(np.ascontiguousarray(queries_std, dtype=np.float64)).double()
    beta_rows_t = (
        [torch.from_numpy(np.ascontiguousarray(row, dtype=np.float64)).double() for row in beta_rows]
        if beta_rows is not None
        else None
    )
    raw_rows_t = (
        [torch.from_numpy(np.ascontiguousarray(row, dtype=np.float64)).double() for row in raw_rows]
        if raw_rows is not None
        else None
    )
    alpha_rows_t = (
        [torch.from_numpy(np.ascontiguousarray(row, dtype=np.float64)).double() for row in alpha_rows]
        if alpha_rows is not None
        else None
    )
    index_rows = [torch.as_tensor(np.asarray(row, dtype=np.int64)) for row in blocks]
    support_size = int(support.shape[0])
    # Ragged ranking/MSE tensors: pad per-query blocks to Kmax once; a boolean
    # mask keeps only real (non-padded) comparable pairs. Mathematically
    # identical to the per-pair loop (same pairs, same mean), but fully
    # vectorized: ~0.1 s/epoch instead of hours at full scale.
    k_max = int(max(len(row) for row in blocks))
    padded_index = torch.zeros((n_queries, k_max), dtype=torch.int64)
    padded_beta = torch.zeros((n_queries, k_max), dtype=torch.float64)
    padded_raw = torch.zeros((n_queries, k_max), dtype=torch.float64)
    block_valid = torch.zeros((n_queries, k_max), dtype=torch.bool)
    for qi, row in enumerate(blocks):
        columns = index_rows[qi]
        width = int(columns.numel())
        padded_index[qi, :width] = columns
        if beta_rows_t is not None:
            padded_beta[qi, :width] = beta_rows_t[qi]
        if raw_rows_t is not None:
            padded_raw[qi, :width] = raw_rows_t[qi]
        block_valid[qi, :width] = True
    padded_alpha = torch.zeros((n_queries, support_size), dtype=torch.float64)
    if alpha_rows_t is not None:
        for qi in range(n_queries):
            padded_alpha[qi, :] = alpha_rows_t[qi]
    scale = float(np.sqrt(float(qk_dim)))
    eps = float(epsilon)
    history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        support_h = backbone(support_t)
        queries_h = backbone(queries_t)
        context = support_h.mean(dim=0, keepdim=True)
        if a_q is not None and a_k is not None:
            query_q = w_q(queries_h) + a_q(context).expand(queries_h.shape[0], -1)
            support_k = w_k(support_h) + a_k(context).expand(support_h.shape[0], -1)
        else:
            query_q = w_q(queries_h)
            support_k = w_k(support_h)
        scores = (query_q @ support_k.T) / scale
        main = torch.zeros((), dtype=torch.float64)
        if loss in ("KL", "KL+rank"):
            if target == "alpha":
                assert alpha_rows_t is not None
                kl_terms: list[Any] = []
                for qi in range(n_queries):
                    log_prob = torch.nn.functional.log_softmax(scores[qi, :].unsqueeze(0), dim=1)
                    kl_terms.append(
                        torch.nn.functional.kl_div(
                            log_prob, padded_alpha[qi, :].clamp_min(eps).unsqueeze(0), reduction="batchmean"
                        )
                    )
                main = torch.stack(kl_terms).mean()
            else:
                # Beta-KL against the frozen true-class support columns only. Each
                # train pseudo-query has its own ragged true-class column map, so
                # per-query KL terms are averaged (batchmean over pseudo-queries).
                assert beta_rows_t is not None
                kl_terms = []
                student_padded = scores[
                    torch.arange(n_queries).unsqueeze(1).expand(n_queries, k_max), padded_index
                ]
                for qi in range(n_queries):
                    width = int(block_valid[qi].sum())
                    student_row = student_padded[qi, :width]
                    log_prob = torch.nn.functional.log_softmax(student_row.unsqueeze(0), dim=1)
                    kl_terms.append(
                        torch.nn.functional.kl_div(
                            log_prob, padded_beta[qi, :width].clamp_min(eps).unsqueeze(0), reduction="batchmean"
                        )
                    )
                main = torch.stack(kl_terms).mean()
        elif loss in ("MSE", "MSE+rank"):
            assert raw_rows_t is not None
            student_padded = scores[
                torch.arange(n_queries).unsqueeze(1).expand(n_queries, k_max), padded_index
            ]
            if target == "raw":
                diff = (student_padded - padded_raw) * block_valid.to(torch.float64)
            else:
                student_mean = (student_padded * block_valid.to(torch.float64)).sum(dim=1, keepdim=True) / block_valid.sum(dim=1, keepdim=True).to(torch.float64).unsqueeze(1)
                teacher_mean = (padded_raw * block_valid.to(torch.float64)).sum(dim=1, keepdim=True) / block_valid.sum(dim=1, keepdim=True).to(torch.float64).unsqueeze(1)
                diff = ((student_padded - student_mean) - (padded_raw - teacher_mean)) * block_valid.to(torch.float64)
            denom = float(block_valid.sum().to(torch.float64))
            if denom <= 0.0:
                raise StudentError("MSE has no valid teacher entries")
            main = (diff * diff).sum() / denom
        rank = torch.zeros((), dtype=torch.float64)
        if loss in ("rank", "KL+rank", "MSE+rank"):
            assert raw_rows_t is not None
            # Vectorized pairwise logistic ranking on the teacher mean-raw
            # blocks: all unordered pairs at once, masked to real comparable
            # pairs, mean over the same pair set as the reference loop.
            student_padded = scores[
                torch.arange(n_queries).unsqueeze(1).expand(n_queries, k_max), padded_index
            ]
            teacher_diff = padded_raw.unsqueeze(2) - padded_raw.unsqueeze(1)
            student_diff = student_padded.unsqueeze(2) - student_padded.unsqueeze(1)
            pair_valid = block_valid.unsqueeze(2) & block_valid.unsqueeze(1)
            rank_mask = ((teacher_diff.abs() > eps) & pair_valid).triu(diagonal=1)
            if not bool(rank_mask.any()):
                raise StudentError("ranking loss has no comparable teacher pair (all ties)")
            rank_terms = torch.nn.functional.softplus(-torch.sign(teacher_diff) * student_diff)
            rank = rank_terms[rank_mask].mean()
        if loss == "rank":
            total = rank
        elif loss in ("KL+rank", "MSE+rank"):
            total = main + float(lambda_ranking) * rank
        else:
            total = main
        total.backward()
        optimizer.step()
        history.append(float(total.detach().cpu()))

    params: dict[str, Any] = {
        "weights": [],
        "biases": [],
        "w_q": np.asarray(w_q.weight.detach().cpu().T, dtype=np.float64),
        "w_k": np.asarray(w_k.weight.detach().cpu().T, dtype=np.float64),
        "adapter": bool(adapter),
        "a_q": None,
        "a_k": None,
        "layer_dims": dims,
        "qk_dim": int(qk_dim),
        "seed": int(seed),
        "epochs": int(epochs),
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "true_positions": [tuple(int(v) for v in row) for row in blocks],
        "target": str(target),
        "loss": str(loss),
        "lambda_ranking": float(lambda_ranking),
    }
    with torch.no_grad():
        for module in backbone:
            if isinstance(module, torch.nn.Linear):
                params["weights"].append(np.asarray(module.weight.detach().cpu().T, dtype=np.float64))
                params["biases"].append(np.asarray(module.bias.detach().cpu(), dtype=np.float64))
    if a_q is not None and a_k is not None:
        with torch.no_grad():
            params["a_q"] = np.asarray(a_q.weight.detach().cpu().T, dtype=np.float64)
            params["a_k"] = np.asarray(a_k.weight.detach().cpu().T, dtype=np.float64)
    params["loss_history"] = [float(v) for v in history]
    return params


def cell_spec(cell_id: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Look up one frozen grid cell by its cell_id."""

    for cell in manifest["student"]["grid"]["cells"]:
        if str(cell.get("cell_id")) == str(cell_id):
            return dict(cell)
    raise StudentError("unknown ablation cell: " + str(cell_id))


def winner_cell_spec() -> dict[str, Any]:
    """Return the frozen Week 10 student cell (Week 9 winner beta-rank static qk64)."""

    return {"cell_id": "beta-rank_qk64_static", "target": "beta", "loss": "rank", "qk_dim": 64, "variant": "static"}


__all__ = ["train_ablation_cell", "train_route_b_student", "cell_spec", "winner_cell_spec"]


def train_route_b_student(
    *,
    support_features,
    pseudo_query_features,
    teacher_beta_blocks,
    teacher_raw_blocks,
    true_positions,
    hidden_dims=(128,),
    qk_dim: int = 64,
    adapter: bool = False,
    learning_rate: float = 0.001,
    epochs: int = 50,
    lambda_ranking: float = 0.5,
    seed: int = 0,
    epsilon: float = EPSILON,
):
    """Legacy Stage 8 alias: beta-KL + ranking static/adapter training.

    Delegates to :func:`train_ablation_cell` with the frozen Stage 8 cell
    (``target="beta"``, ``loss="KL+rank"``); numerics are identical.
    """

    return train_ablation_cell(
        support_features=support_features,
        pseudo_query_features=pseudo_query_features,
        teacher_beta_blocks=teacher_beta_blocks,
        teacher_raw_blocks=teacher_raw_blocks,
        true_positions=true_positions,
        target="beta",
        loss="KL+rank",
        hidden_dims=list(hidden_dims),
        qk_dim=int(qk_dim),
        adapter=bool(adapter),
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        lambda_ranking=float(lambda_ranking),
        seed=int(seed),
        epsilon=float(epsilon),
    )
