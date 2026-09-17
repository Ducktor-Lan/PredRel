"""Run the frozen formal benchmark (20 datasets x 8 methods).

Protocol per (dataset, seed), fixed by provenance/dataset_manifest.json:
  1. Build the frozen stratified split (same contract as Weeks 3-6/8/9).
  2. Hold out every query row. From the support, pick label-blind
     pseudo-queries (20 percent, 8..64, seed 20260914); the rest is the
     pseudo-support.
  3. One frozen TRAIN Teacher fit over the pseudo-split yields the student
     training blocks (beta/raw) AND the readout-profile support profiles.
     One frozen EVAL Teacher fit over the FULL support with all holdout
     queries yields per-query teacher beta/raw blocks, hidden embeddings,
     and query readout profiles.
  4. Train the single frozen student cell (beta-rank static qk64, the Week 9
     global winner), the matched SupCon baseline, and the supervised MLP
     baseline on the train split only. Fit PCA on the pseudo-support only.
     Raw needs no training. Hidden/readout-profile/fusion are Teacher-side
     only (no training).
  5. Freeze everything. Score all eight methods on every holdout query
     (labels metric-only) with the frozen metric set.

Smoke mode executes a reduced contract: the frozen smoke dataset/seed/caps,
all eight methods at smoke epochs, and no scientific gate.

Leakage boundary: query labels (train or holdout) never enter any support
context or any Teacher fit. Train pseudo-query labels select the
true-class block only; classical/MLP/PCA/SupCon/student train on
pseudo-support rows only; hidden/readout-profile/fusion consume Teacher
outputs only; holdout query labels are released only for metric computation.

This module orchestrates NumPy/Torch training plus Teacher fits; the
evidence writer persists method metrics, per-method-query metric rows (no
raw feature matrices, no teacher blocks, no parameter tensors -- only
recomputable fingerprints), provenance, and the frozen method set so the
verifier can re-derive and re-score every number from the manifest plus
the Teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..baselines import (
    BaselineError,
    mlp_penultimate_embeddings,
    mlp_scores,
    negative_euclidean_scores,
    pca_scores,
    raw_scores,
    train_mlp_classifier,
)
from ..data import (
    DatasetError,
    RealDataset,
    load_manifest,
    make_stratified_split,
    materialize_manifest,
    verify_dataset_lock,
)
from ..fusion import FusionError, fuse_rank_average
from ..label_anatomy import AnatomyError, decompose_alpha
from ..losses import EPSILON
from ..metrics import METHOD_IDS, score_benchmark_query
from ..models import StudentError, select_pseudo_queries, student_params_to_record
from ..supcon import SupconError, cosine_similarities, embed_with_params, train_supcon_encoder
from ..teacher_bridge import WEEK01_SOURCE_SHA256, TeacherBridgeError, extract_readout
from ..train_student import cell_spec, train_ablation_cell, winner_cell_spec


class RunnerError(RuntimeError):
    """Raised when the benchmark cannot make technically valid evidence."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """All fresh serializable evidence produced by one successful execution."""

    evidence: dict[str, Any]
    eval_rows: list[dict[str, Any]]
    dataset_provenance: dict[str, Any]
    dataset_lock: dict[str, Any] | None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


BENCHMARK_MANIFEST_ID = "predrel-benchmark-v1"


def _load_benchmark_provenance() -> dict[str, Any]:
    """Read the unified benchmark provenance (route + frozen winner cell)."""

    path = _project_root() / "provenance" / "provenance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError("benchmark provenance record is unreadable: " + str(error)) from error
    if not isinstance(payload, dict):
        raise RunnerError("benchmark provenance record must be a JSON object")
    if payload.get("recommended_route") != "Route B: Static Directed with simple context adapter first":
        raise RunnerError("benchmark provenance does not authorize the Route B benchmark")
    if str(payload.get("benchmark_winner_cell", "")) != "beta-rank_qk64_static":
        raise RunnerError("benchmark provenance lacks the frozen winner cell")
    if str(payload.get("benchmark_manifest_id", "")) != BENCHMARK_MANIFEST_ID:
        raise RunnerError("benchmark provenance lacks the frozen manifest id")
    return payload


_load_week10_dependency = _load_benchmark_provenance


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _require_finite(value: object, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RunnerError(name + " is not numeric") from error
    if not math.isfinite(numeric):
        raise RunnerError(name + " is non-finite")
    return numeric


def _fingerprint_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse one trained model to a recomputable fingerprint (no tensors)."""

    def _arr(name: str) -> np.ndarray:
        value = np.asarray(params[name], dtype=np.float64)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise RunnerError("method fingerprint needs finite " + name)
        return value

    arrays: dict[str, np.ndarray] = {
        "mean": _arr("mean"),
        "std": _arr("std"),
    }
    # Student-style params carry Q/K heads + MLP backbone.
    if "w_q" in params:
        arrays["w_q"] = _arr("w_q")
        arrays["w_k"] = _arr("w_k")
        for i, w in enumerate(params["weights"]):
            arrays["weights_%d" % i] = np.asarray(w, dtype=np.float64)
        for i, b in enumerate(params["biases"]):
            arrays["biases_%d" % i] = np.asarray(b, dtype=np.float64)
        if params.get("a_q") is not None:
            arrays["a_q"] = _arr("a_q")
        if params.get("a_k") is not None:
            arrays["a_k"] = _arr("a_k")
    elif "weights" in params:
        # SupCon/MLP-style params carry a plain MLP stack.
        for i, w in enumerate(params["weights"]):
            arrays["weights_%d" % i] = np.asarray(w, dtype=np.float64)
        for i, b in enumerate(params["biases"]):
            arrays["biases_%d" % i] = np.asarray(b, dtype=np.float64)
    digest = sha256()
    for key in sorted(arrays):
        block = np.ascontiguousarray(arrays[key], dtype=np.float64)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(list(block.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(block.tobytes(order="C"))
        digest.update(b"\n")
    history = [float(v) for v in params.get("loss_history", [])]
    if not history or not all(math.isfinite(v) for v in history):
        raise RunnerError("method fingerprint needs a finite loss history")
    record: dict[str, Any] = {
        "sha256": digest.hexdigest(),
        "seed": int(params.get("seed")),
        "epochs": int(params.get("epochs")),
        "loss_first": float(history[0]),
        "loss_last": float(history[-1]),
        "loss_min": float(min(history)),
        "loss_history_len": len(history),
    }
    for key in ("target", "loss", "qk_dim", "adapter", "layer_dims", "lambda_ranking",
                "hidden_dim", "learning_rate", "classes", "temperature"):
        if key in params:
            value = params[key]
            record[key] = [int(v) for v in value] if key in ("layer_dims", "classes") else (
                bool(value) if key == "adapter" else (float(value) if key in ("lambda_ranking", "learning_rate", "temperature") else (int(value) if key in ("qk_dim", "hidden_dim", "seed", "epochs") else str(value))))
    if "true_positions" in params:
        record["true_positions"] = [[int(v) for v in row] for row in params.get("true_positions", [])]
    return record


def _split_config(manifest: Mapping[str, Any], *, mode: str) -> tuple[list[int], dict[str, Any]]:
    split = manifest["split"]
    if mode == "full":
        return list(split["seeds"]), {
            "max_support_size": int(split["max_support_size"]),
            "max_query_size": int(split["max_query_size"]),
            "train_fraction": float(split["train_fraction"]),
            "min_support_per_class": int(split["min_support_per_class"]),
            "min_query_per_class": int(split["min_query_per_class"]),
        }
    smoke = manifest["smoke"]
    return [int(smoke["seed"])], {
        "max_support_size": int(smoke["max_support_size"]),
        "max_query_size": int(smoke["max_query_size"]),
        "train_fraction": float(split["train_fraction"]),
        "min_support_per_class": int(split["min_support_per_class"]),
        "min_query_per_class": int(split["min_query_per_class"]),
    }


def _method_config(manifest: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    student = manifest["methods"]["student"]
    supcon = manifest["methods"]["supcon"]
    smoke_epochs = int(manifest["smoke"].get("smoke_epochs", 5))
    return {
        "hidden_dims": [int(v) for v in student["hidden_dims"]],
        "learning_rate": float(student["learning_rate"]),
        "epochs": int(student["epochs"] if mode == "full" else smoke_epochs),
        "lambda_ranking": float(manifest["methods"]["student"].get("lambda_ranking", 0.5)),
        "seed": int(student["seed"]),
        "pseudo_query_fraction": float(student["pseudo_query_fraction"]),
        "max_pseudo_queries": int(student["max_pseudo_queries"]),
        "min_pseudo_queries": int(student["min_pseudo_queries"]),
        "pseudo_query_seed": int(student["pseudo_query_seed"]),
        "supcon_hidden_dims": [int(v) for v in supcon["hidden_dims"]],
        "supcon_embedding_dim": int(supcon["embedding_dim"]),
        "supcon_temperature": float(supcon["temperature"]),
        "supcon_learning_rate": float(supcon["learning_rate"]),
        "supcon_epochs": int(supcon["epochs"] if mode == "full" else smoke_epochs),
        "supcon_batch_size": int(supcon["batch_size"]),
        "supcon_seed": int(supcon["seed"]),
        "mlp_hidden_dim": 128,
        "mlp_learning_rate": 0.001,
        "mlp_epochs": int(50 if mode == "full" else smoke_epochs),
        "mlp_seed": 0,
        "pca_components": 32,
    }


def _train_blocks(
    *,
    alpha: np.ndarray,
    raw_scores: np.ndarray,
    support_labels: np.ndarray,
    pseudo_query_labels: np.ndarray,
    true_positions: list[list[int]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Extract per-pseudo-query teacher alpha rows, beta blocks, mean-raw blocks."""

    for positions in true_positions:
        if len(positions) < 2 or len(set(int(v) for v in positions)) != len(positions):
            raise RunnerError("train true-class blocks need at least two distinct supports")
        if any(int(v) < 0 or int(v) >= len(support_labels) for v in positions):
            raise RunnerError("train true-class positions are outside the pseudo-support")
    try:
        decomposition = decompose_alpha(alpha, support_labels, epsilon=EPSILON)
    except AnatomyError as error:
        raise RunnerError("train decomposition failed: " + str(error)) from error
    mean_raw = np.asarray(raw_scores, dtype=np.float64).mean(axis=(0, 1))
    if mean_raw.shape != decomposition.beta.shape:
        raise RunnerError("train raw scores do not align with alpha")
    alpha_rows: list[np.ndarray] = []
    beta_blocks: list[np.ndarray] = []
    raw_blocks: list[np.ndarray] = []
    for qi, positions in enumerate(true_positions):
        alpha_rows.append(np.ascontiguousarray(np.asarray(alpha[qi], dtype=np.float64)))
        beta_blocks.append(np.ascontiguousarray(np.asarray(decomposition.beta[qi][positions], dtype=np.float64)))
        raw_blocks.append(np.ascontiguousarray(np.asarray(mean_raw[qi][positions], dtype=np.float64)))
    if len(beta_blocks) != int(np.asarray(pseudo_query_labels).shape[0]):
        raise RunnerError("train blocks do not align with pseudo-queries")
    return alpha_rows, beta_blocks, raw_blocks


def _cosine_rows(queries: np.ndarray, supports: np.ndarray) -> np.ndarray:
    queries = np.asarray(queries, dtype=np.float64)
    supports = np.asarray(supports, dtype=np.float64)
    if queries.ndim != 2 or supports.ndim != 2 or queries.shape[1] != supports.shape[1]:
        raise RunnerError("hidden embedding widths do not match")
    qn = np.linalg.norm(queries, axis=1, keepdims=True)
    sn = np.linalg.norm(supports, axis=1, keepdims=True)
    qn = np.where(qn > 0.0, qn, 1.0)
    sn = np.where(sn > 0.0, sn, 1.0)
    sims = (queries / qn) @ (supports / sn).T
    if not np.all(np.isfinite(sims)):
        raise RunnerError("hidden cosine similarities are non-finite")
    return np.ascontiguousarray(sims)


def _readout_profile_scores(
    *,
    train_alpha: np.ndarray,
    eval_alpha: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between eval query profiles and support profiles.

    Support profiles are TRAIN-fit alpha rows of each pseudo-support row held
    out as a query, over the pseudo-support columns. Query profiles are
    EVAL-fit alpha rows restricted to the same pseudo-support columns. Both
    profile sets therefore share one column space; cosine similarity gives
    the readout-profile score matrix [n_eval_queries, n_pseudo_support_full?].

    NOTE: this helper scores queries against the TRAIN pseudo-support only
    when called with matching column maps; the caller is responsible for
    aligning columns. Here both alpha matrices must already share columns.
    """

    train = np.asarray(train_alpha, dtype=np.float64)
    eval_q = np.asarray(eval_alpha, dtype=np.float64)
    if train.ndim != 2 or eval_q.ndim != 2 or train.shape[1] != eval_q.shape[1]:
        raise RunnerError("readout-profile column spaces do not match")
    if train.shape[0] == 0 or eval_q.shape[0] == 0:
        raise RunnerError("readout-profile needs non-empty profile sets")
    return _cosine_rows(eval_q, train)


def _run_seed(
    *,
    dataset: RealDataset,
    seed: int,
    split_args: Mapping[str, Any],
    method_cfg: Mapping[str, Any],
    manifest: Mapping[str, Any],
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from ..models import directed_scores_with_support_context, student_params_from_record

    split = make_stratified_split(dataset, seed=int(seed), **dict(split_args))
    support_count = int(split.support_features.shape[0])
    pseudo_positions = select_pseudo_queries(
        support_count,
        query_fraction=float(method_cfg["pseudo_query_fraction"]),
        max_queries=int(method_cfg["max_pseudo_queries"]),
        min_queries=int(method_cfg["min_pseudo_queries"]),
        seed=int(derive_seed_for_split(method_cfg, dataset.dataset_id, seed)),
    )
    pseudo_mask = np.zeros(support_count, dtype=bool)
    pseudo_mask[np.asarray(pseudo_positions, dtype=np.int64)] = True
    keep = np.flatnonzero(~pseudo_mask)
    pseudo_support_ids = tuple(split.support_ids[i] for i in keep.tolist())
    pseudo_query_ids = tuple(split.support_ids[i] for i in pseudo_positions)
    train_extraction = extract_readout(
        support_features=np.asarray(split.support_features)[keep],
        support_labels=np.asarray(split.support_labels)[keep],
        support_ids=pseudo_support_ids,
        query_features=np.asarray(split.support_features)[list(pseudo_positions)],
        query_ids=pseudo_query_ids,
        seed=seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    if train_extraction.support_ids != pseudo_support_ids or train_extraction.query_ids != pseudo_query_ids:
        raise RunnerError("train Teacher output IDs are not aligned to the pseudo-split")
    if set(pseudo_support_ids) & set(pseudo_query_ids):
        raise RunnerError("pseudo-support and pseudo-queries overlap")
    if set(pseudo_support_ids) | set(pseudo_query_ids) != set(split.support_ids):
        raise RunnerError("pseudo-split does not partition the frozen support")
    pseudo_labels = np.asarray(split.support_labels)[list(pseudo_positions)]
    pseudo_support_labels = np.asarray(split.support_labels)[keep]
    true_positions = _check_true_positions(
        len(keep),
        len(pseudo_positions),
        [
            [pos for pos in range(len(keep)) if int(pseudo_support_labels[pos]) == int(pseudo_labels[qi])]
            for qi in range(len(pseudo_positions))
        ],
    )
    for qi in range(len(pseudo_positions)):
        if len(true_positions[qi]) < 2:
            raise RunnerError("a train pseudo-query has fewer than two true-class supports")
    _, beta_blocks, raw_blocks = _train_blocks(
        alpha=np.asarray(train_extraction.alpha, dtype=np.float64),
        raw_scores=np.asarray(train_extraction.raw_scores, dtype=np.float64),
        support_labels=pseudo_support_labels,
        pseudo_query_labels=pseudo_labels,
        true_positions=true_positions,
    )
    student_spec = manifest["methods"]["student"]
    frozen_winner = winner_cell_spec()
    if (
        str(student_spec["target"]) != frozen_winner["target"]
        or str(student_spec["loss"]) != frozen_winner["loss"]
        or int(student_spec["qk_dim"]) != int(frozen_winner["qk_dim"])
        or str(student_spec["variant"]) != frozen_winner["variant"]
    ):
        raise RunnerError("manifest student cell differs from the frozen Week 9 winner beta-rank_qk64_static")
    student_params = train_ablation_cell(
        support_features=np.asarray(split.support_features)[keep],
        pseudo_query_features=np.asarray(split.support_features)[list(pseudo_positions)],
        teacher_beta_blocks=None,
        teacher_raw_blocks=raw_blocks,
        true_positions=true_positions,
        target=str(student_spec["target"]),
        loss=str(student_spec["loss"]),
        hidden_dims=[int(v) for v in method_cfg["hidden_dims"]],
        qk_dim=int(student_spec["qk_dim"]),
        adapter=False,
        learning_rate=float(method_cfg["learning_rate"]),
        epochs=int(method_cfg["epochs"]),
        lambda_ranking=float(method_cfg["lambda_ranking"]),
        seed=int(method_cfg["seed"]),
    )
    if [list(map(int, row)) for row in student_params["true_positions"]] != [list(map(int, row)) for row in true_positions]:
        raise RunnerError("trainer did not echo the frozen true-class column maps")
    eval_student = student_params_from_record(student_params_to_record(student_params))
    try:
        supcon_params = train_supcon_encoder(
            np.asarray(split.support_features)[keep],
            pseudo_support_labels,
            hidden_dims=[int(v) for v in method_cfg["supcon_hidden_dims"]],
            embedding_dim=int(method_cfg["supcon_embedding_dim"]),
            temperature=float(method_cfg["supcon_temperature"]),
            learning_rate=float(method_cfg["supcon_learning_rate"]),
            epochs=int(method_cfg["supcon_epochs"]),
            batch_size=int(method_cfg["supcon_batch_size"]),
            seed=int(method_cfg["supcon_seed"]),
        )
    except SupconError as error:
        raise RunnerError("SupCon baseline failed: " + str(error)) from error
    try:
        mlp_params = train_mlp_classifier(
            np.asarray(split.support_features)[keep],
            pseudo_support_labels,
            hidden_dim=int(method_cfg["mlp_hidden_dim"]),
            epochs=int(method_cfg["mlp_epochs"]),
            learning_rate=float(method_cfg["mlp_learning_rate"]),
            seed=int(method_cfg["mlp_seed"]),
        )
    except BaselineError as error:
        raise RunnerError("MLP baseline failed: " + str(error)) from error
    try:
        pseudo_support_std = (
            np.asarray(split.support_features)[keep]
            - np.asarray(split.support_features)[keep].mean(axis=0)
        ) / np.where(
            np.asarray(split.support_features)[keep].std(axis=0) > 0.0,
            np.asarray(split.support_features)[keep].std(axis=0),
            1.0,
        )
        from ..baselines import fit_pca as _fit_pca

        pca_projector = _fit_pca(pseudo_support_std, n_components=int(method_cfg["pca_components"]))
    except BaselineError as error:
        raise RunnerError("PCA baseline failed: " + str(error)) from error

    # ---- frozen eval fit over the FULL support with all holdout queries ----
    eval_extraction = extract_readout(
        support_features=split.support_features,
        support_labels=split.support_labels,
        support_ids=split.support_ids,
        query_features=split.query_features,
        query_ids=split.query_ids,
        seed=seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    if eval_extraction.support_ids != split.support_ids or eval_extraction.query_ids != split.query_ids:
        raise RunnerError("eval Teacher output IDs are not aligned to the frozen split")
    try:
        eval_decomposition = decompose_alpha(
            np.asarray(eval_extraction.alpha, dtype=np.float64), np.asarray(split.support_labels), epsilon=EPSILON
        )
    except AnatomyError as error:
        raise RunnerError("eval decomposition failed: " + str(error)) from error
    eval_mean_raw = np.asarray(eval_extraction.raw_scores, dtype=np.float64).mean(axis=(0, 1))
    if eval_extraction.hidden_support_aggregated is None or eval_extraction.hidden_query_aggregated is None:
        raise RunnerError("eval Teacher hidden embeddings are unavailable")
    hidden_support = np.asarray(eval_extraction.hidden_support_aggregated, dtype=np.float64)
    hidden_queries = np.asarray(eval_extraction.hidden_query_aggregated, dtype=np.float64)
    if hidden_support.shape[0] != len(split.support_ids) or hidden_queries.shape[0] != len(split.query_ids):
        raise RunnerError("eval Teacher hidden embeddings do not align to the frozen split")
    hidden_scores = _cosine_rows(hidden_queries, hidden_support)
    # Readout-profile column contract: support profiles come from the TRAIN
    # fit (pseudo-support rows as queries over pseudo-support columns);
    # query profiles come from the EVAL fit restricted to the pseudo-support
    # columns (positions ``keep`` in full-support order).
    train_alpha = np.asarray(train_extraction.alpha, dtype=np.float64)
    eval_alpha = np.asarray(eval_extraction.alpha, dtype=np.float64)
    if train_alpha.shape != (len(pseudo_query_ids), len(keep)):
        raise RunnerError("train Teacher alpha shape differs from the pseudo-split")
    # Support profiles: for each pseudo-support row, its TRAIN-fit alpha row
    # when held out is unavailable (it was support, not query, in that fit).
    # Frozen fallback: profile of pseudo-support row j = TRAIN-fit alpha row
    # of the pseudo-query most similar in label? NO -- that would use labels
    # as similarity. Instead use the EVAL-fit alpha submatrix restricted to
    # pseudo-support columns for BOTH sides: support profile j = eval alpha
    # row of... eval queries are holdout rows, not supports. Correct frozen
    # construction: support profiles from a dedicated SUPPORT-profile Teacher
    # fit is one extra fit per seed; to keep Teacher fits at two per seed,
    # support profiles reuse the TRAIN fit transposed view is invalid.
    #
    # Frozen resolution (manifest teacher_derived.readout_profile): support
    # profiles ARE the train-fit alpha rows of pseudo-queries is wrong for
    # supports. So the benchmark performs exactly one extra frozen SUPPORT-profile
    # fit: every pseudo-support row as a single query against the
    # pseudo-support-minus-self? That changes columns per row.
    #
    # Simplest column-consistent construction with exactly one extra fit:
    # fit Teacher on (support=pseudo-support, queries=pseudo-support) --
    # self-attention profiles where query j may attend to itself. Both
    # support and eval-query profiles then share the pseudo-support column
    # space only if eval queries are ALSO profiled against pseudo-support
    # columns. That needs a fourth fit (queries=holdout, support=pseudo-
    # support). Total four frozen Teacher fits per seed; all label-free.
    profile_support_extraction = extract_readout(
        support_features=np.asarray(split.support_features)[keep],
        support_labels=pseudo_support_labels,
        support_ids=pseudo_support_ids,
        query_features=np.asarray(split.support_features)[keep],
        query_ids=tuple(s + ":profile" for s in pseudo_support_ids),
        seed=seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    profile_query_extraction = extract_readout(
        support_features=np.asarray(split.support_features)[keep],
        support_labels=pseudo_support_labels,
        support_ids=pseudo_support_ids,
        query_features=split.query_features,
        query_ids=tuple(s + ":profile" for s in split.query_ids),
        seed=seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    support_profiles = np.asarray(profile_support_extraction.alpha, dtype=np.float64)
    query_profiles = np.asarray(profile_query_extraction.alpha, dtype=np.float64)
    if support_profiles.shape != (len(keep), len(keep)):
        raise RunnerError("support readout profiles have an unexpected shape")
    if query_profiles.shape != (len(split.query_ids), len(keep)):
        raise RunnerError("query readout profiles have an unexpected shape")
    profile_scores_full: np.ndarray | None = None
    try:
        profile_scores_pseudo = _readout_profile_scores(
            train_alpha=support_profiles, eval_alpha=query_profiles
        )
        # Map pseudo-support columns back to full-support columns for the
        # unified scorer: non-pseudo columns receive -inf (never top-K).
        profile_scores_full = np.full(
            (len(split.query_ids), len(split.support_ids)), -np.inf, dtype=np.float64
        )
        profile_scores_full[:, np.asarray(keep, dtype=np.int64)] = profile_scores_pseudo
    except RunnerError as error:
        raise RunnerError("readout-profile scoring failed: " + str(error)) from error
    assert profile_scores_full is not None

    support_labels = np.asarray(split.support_labels)
    supcon_support_emb = embed_with_params(supcon_params, split.support_features)
    supcon_query_emb = embed_with_params(supcon_params, split.query_features)
    supcon_scores = np.asarray(cosine_similarities(supcon_query_emb, supcon_support_emb), dtype=np.float64)
    eval_ks = [int(v) for v in manifest["evaluation"]["ks"]]
    rows: list[dict[str, Any]] = []
    per_method_queries: dict[str, list[dict[str, Any]]] = {m: [] for m in METHOD_IDS}
    for qi in range(len(split.query_ids)):
        query_label = int(split.query_labels[qi])
        true_cols = [pos for pos in range(len(split.support_ids)) if int(support_labels[pos]) == query_label]
        if len(true_cols) < 2:
            raise RunnerError("an eval query has fewer than two true-class supports")
        teacher_beta = np.asarray(eval_decomposition.beta[qi][true_cols], dtype=np.float64)
        teacher_raw = np.asarray(eval_mean_raw[qi][true_cols], dtype=np.float64)
        full_support = np.asarray(split.support_features)
        query_row = np.asarray(split.query_features[qi]).reshape(1, -1)
        student_full = directed_scores_with_support_context(eval_student, query_row, full_support)[0]
        supcon_full = np.asarray(supcon_scores[qi], dtype=np.float64)
        try:
            fused_full = fuse_rank_average(student_full, supcon_full)
        except FusionError as error:
            raise RunnerError("fusion scoring failed: " + str(error)) from error
        try:
            raw_full = np.asarray(raw_scores(full_support, query_row)[0], dtype=np.float64)
            pca_full = np.asarray(
                pca_scores(full_support, query_row, n_components=int(method_cfg["pca_components"]))[0],
                dtype=np.float64,
            )
            mlp_full = np.asarray(mlp_scores(mlp_params, full_support, query_row)[0], dtype=np.float64)
        except BaselineError as error:
            raise RunnerError("classical baseline scoring failed: " + str(error)) from error
        hidden_full = np.asarray(hidden_scores[qi], dtype=np.float64)
        profile_full = np.asarray(profile_scores_full[qi], dtype=np.float64)
        if not np.all(np.isfinite(profile_full[np.asarray(keep, dtype=np.int64)])):
            raise RunnerError("readout-profile scores are non-finite on pseudo columns")
        method_full_scores: dict[str, np.ndarray] = {
            "student": np.asarray(student_full, dtype=np.float64),
            "supcon": supcon_full,
            "fusion": np.asarray(fused_full, dtype=np.float64),
            "raw": raw_full,
            "pca": pca_full,
            "mlp": mlp_full,
            "hidden": hidden_full,
            "readout_profile": profile_full,
        }
        row_record: dict[str, Any] = {
            "dataset_id": dataset.dataset_id,
            "seed": int(seed),
            "split_id": split.split_id,
            "query_id": split.query_ids[qi],
            "query_true_label": query_label,
            "support_ids": list(split.support_ids),
            "support_labels": [int(v) for v in support_labels.tolist()],
            "true_support_positions": [int(v) for v in true_cols],
            "teacher_beta_block": [float(v) for v in teacher_beta.tolist()],
            "teacher_mean_raw_block": [float(v) for v in np.asarray(teacher_raw).tolist()],
            "methods": {},
        }
        for method_id in METHOD_IDS:
            full_scores = method_full_scores[method_id]
            if method_id == "readout_profile":
                # Non-pseudo columns are -inf by construction (the profile
                # column space is the pseudo-support). Score readout-profile
                # ONLY on the finite pseudo-support restriction of both the
                # method block and the teacher blocks; the restriction is
                # recorded so the verifier re-derives it exactly.
                finite_positions = [pos for pos in true_cols if math.isfinite(float(full_scores[pos]))]
                if len(finite_positions) < 2:
                    raise RunnerError("readout-profile has fewer than two finite true-class supports")
                true_scores = np.asarray([float(full_scores[pos]) for pos in finite_positions], dtype=np.float64)
                teacher_beta_eff = np.asarray([float(teacher_beta[true_cols.index(pos)]) for pos in finite_positions], dtype=np.float64)
                teacher_raw_eff = np.asarray([float(teacher_raw[true_cols.index(pos)]) for pos in finite_positions], dtype=np.float64)
                scored = score_benchmark_query(
                    method_true_scores=true_scores,
                    teacher_beta_block=teacher_beta_eff,
                    teacher_raw_block=teacher_raw_eff,
                    full_method_scores=full_scores,
                    support_labels=support_labels,
                    query_label=query_label,
                    ks=eval_ks,
                )
                row_record["methods"][method_id] = {
                    **scored,
                    "profile_true_positions": [int(v) for v in finite_positions],
                    "method_true_scores": [float(v) for v in true_scores.tolist()],
                    "full_method_scores": [float(v) if math.isfinite(float(v)) else None for v in np.asarray(full_scores).tolist()],
                }
                per_method_queries[method_id].append(scored)
                continue
            true_scores = np.asarray([float(full_scores[pos]) for pos in true_cols], dtype=np.float64)
            scored = score_benchmark_query(
                method_true_scores=true_scores,
                teacher_beta_block=teacher_beta,
                teacher_raw_block=teacher_raw,
                full_method_scores=full_scores,
                support_labels=support_labels,
                query_label=query_label,
                ks=eval_ks,
            )
            row_record["methods"][method_id] = {
                **scored,
                "method_true_scores": [float(v) for v in true_scores.tolist()],
                "full_method_scores": [float(v) for v in np.asarray(full_scores).tolist()],
            }
            per_method_queries[method_id].append(scored)
        rows.append(row_record)
    seed_summary: dict[str, Any] = {
        "seed": int(seed),
        "split_id": split.split_id,
        "holdout_ids": list(split.holdout_ids),
        "support_count": int(split.support_features.shape[0]),
        "support_ids": list(split.support_ids),
        "support_labels": [int(v) for v in np.asarray(split.support_labels).tolist()],
        "query_count": int(split.query_features.shape[0]),
        "pseudo_query_positions": [int(v) for v in pseudo_positions],
        "pseudo_query_ids": list(pseudo_query_ids),
        "pseudo_support_ids": list(pseudo_support_ids),
        "methods": {m: _summarize_queries(per_method_queries[m]) for m in METHOD_IDS},
        "query_count_scored": len(rows),
    }
    fingerprints = {
        "student": _fingerprint_params(student_params),
        "supcon": _json_ready(supcon_params),
        "mlp": _fingerprint_params(mlp_params),
        "pca": _json_ready(
            {
                "components_shape": [int(v) for v in np.asarray(pca_projector["components"]).shape],
                "n_components": int(pca_projector["n_components"]),
                "mean_sha256": sha256(np.ascontiguousarray(pca_projector["mean"], dtype=np.float64).tobytes()).hexdigest(),
            }
        ),
    }
    runtime = {"runtime": dict(train_extraction.runtime), "decoder_readout_api": train_extraction.decoder_readout_api,
               "raw_score_hook_path": train_extraction.raw_score_hook_path}
    return seed_summary, rows, {"fingerprints": fingerprints, "runtime": runtime}


def derive_seed_for_split(method_cfg: Mapping[str, Any], dataset_id: str, seed: int) -> int:
    from ..models import derive_seed

    return derive_seed(int(method_cfg["pseudo_query_seed"]), dataset_id, int(seed))


def _check_true_positions(
    pseudo_support_count: int, pseudo_query_count: int, true_positions: Sequence[Sequence[int]]
) -> list[list[int]]:
    """Validate the frozen per-query true-class column maps."""

    checked = [[int(v) for v in row] for row in true_positions]
    if len(checked) != int(pseudo_query_count):
        raise RunnerError("true-class column maps do not cover every pseudo-query")
    for row in checked:
        if len(row) < 2 or len(set(row)) != len(row) or any(v < 0 or v >= int(pseudo_support_count) for v in row):
            raise RunnerError("true-class column maps are invalid")
    return checked


def _summarize_queries(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RunnerError("no eval queries were scored")
    ks = sorted({key for record in records for key in record["teacher_topk_recall_mean"]}, key=int)
    summary: dict[str, Any] = {
        "query_count": len(records),
        "teacher_topk_recall_mean": {},
        "spearman_vs_beta_mean": 0.0,
        "spearman_vs_raw_mean": 0.0,
        "ndcg_at_10_mean": 0.0,
        "teacher_top1_rank_mean": 0.0,
        "label_hit_at_1_rate": 0.0,
        "label_hit_at_5_rate": 0.0,
    }
    for key in ks:
        values = [_require_finite(record["teacher_topk_recall_mean"][key], name="recall@" + key) for record in records]
        summary["teacher_topk_recall_mean"][key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    for field in ("spearman_vs_beta", "spearman_vs_raw", "ndcg_at_10", "teacher_top1_rank",
                  "label_hit_at_1", "label_hit_at_5"):
        values = [_require_finite(record[field], name=field) for record in records]
        out_key = field + "_mean" if field in ("spearman_vs_beta", "spearman_vs_raw", "ndcg_at_10", "teacher_top1_rank") else field + "_rate"
        summary[out_key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return summary


def _median(values: Sequence[float], *, name: str) -> float:
    array = np.asarray([_require_finite(v, name=name) for v in values], dtype=np.float64)
    if array.size == 0:
        raise RunnerError(name + " has no values")
    return float(np.median(array))


def run_benchmark(
    *,
    mode: str,
    dataset_manifest_path: str | Path,
    dataset_cache_dir: str | Path,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> RunResult:
    """Run a smoke or full frozen benchmark without downloading data."""

    manifest, manifest_hash = load_manifest(dataset_manifest_path)
    benchmark_provenance = _load_benchmark_provenance()
    if manifest.get("manifest_id") != BENCHMARK_MANIFEST_ID:
        raise RunnerError("dataset manifest is not the frozen unified benchmark manifest")
    if manifest_hash != str(benchmark_provenance.get("benchmark_manifest_sha256", "")).lower():
        raise RunnerError("dataset manifest hash differs from the frozen benchmark provenance record")
    frozen_teacher = manifest.get("teacher")
    if not isinstance(frozen_teacher, Mapping):
        raise RunnerError("frozen manifest lacks Teacher settings")
    if int(frozen_teacher.get("n_estimators", 0)) != int(n_estimators):
        raise RunnerError("n_estimators differs from the frozen manifest value")
    if model_cache_dir != frozen_teacher.get("model_cache_dir"):
        raise RunnerError("model_cache_dir differs from the frozen manifest cache")
    seeds, split_args = _split_config(manifest, mode=mode)
    method_cfg = _method_config(manifest, mode=mode)
    if mode == "full":
        selected_ids: list[str] | None = None
    else:
        selected_ids = [str(manifest["smoke"]["dataset_id"])]
    datasets = materialize_manifest(
        manifest, cache_dir=dataset_cache_dir, allow_download=False, dataset_ids=selected_ids
    )
    if mode == "smoke" and len(datasets) != 1:
        raise RunnerError("smoke mode must execute exactly the frozen smoke dataset")
    expected_by_id = {str(item["dataset_id"]): str(item.get("expected_content_sha256", "")) for item in manifest["datasets"]}
    for dataset in datasets:
        if str(dataset.metadata.get("content_sha256", "")) != expected_by_id[dataset.dataset_id]:
            raise RunnerError(dataset.dataset_id + ": content hash differs from its frozen pin")
    dataset_lock = verify_dataset_lock(manifest, manifest_hash, datasets, cache_dir=dataset_cache_dir) if mode == "full" else None
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    decoder_api: str | None = None
    raw_hook: str | None = None
    method_fingerprints: dict[str, Any] = {}
    for dataset in datasets:
        seed_records: list[dict[str, Any]] = []
        for seed in seeds:
            seed_summary, seed_rows, artifacts = _run_seed(
                dataset=dataset,
                seed=int(seed),
                split_args=split_args,
                method_cfg=method_cfg,
                manifest=manifest,
                n_estimators=n_estimators,
                model_cache_dir=model_cache_dir,
                week01_source_root=week01_source_root,
            )
            seed_records.append(seed_summary)
            rows.extend(seed_rows)
            key = dataset.dataset_id + ":seed-" + str(seed)
            method_fingerprints[key] = {
                "methods": dict(artifacts["fingerprints"]),
                "pseudo_query_positions": [int(v) for v in seed_summary["pseudo_query_positions"]],
            }
            runtime = artifacts["runtime"]["runtime"]
            decoder_api = artifacts["runtime"]["decoder_readout_api"]
            raw_hook = artifacts["runtime"]["raw_score_hook_path"]
        dataset_gate = _dataset_summary(dataset.dataset_id, dataset.metadata, seed_records)
        results.append(dataset_gate)
    if runtime is None or decoder_api is None or raw_hook is None:
        raise RunnerError("Teacher was never invoked")
    if mode == "full":
        scientific_gate = _global_gate(results, manifest)
    else:
        scientific_gate = {"decision": "not_evaluated_in_smoke", "recommendation": "SMOKE_ONLY_NO_SCIENTIFIC_CLAIM"}
    provenance = {
        "schema_version": 1,
        "dataset_manifest_id": manifest["manifest_id"],
        "dataset_manifest_sha256": manifest_hash,
        "selected_dataset_ids": [dataset.dataset_id for dataset in datasets],
        "datasets": [dataset.metadata for dataset in datasets],
    }
    evidence = _json_ready(
        {
            "schema_version": 1,
            "completed_at_utc": _utc_now(),
            "mode": mode,
            "teacher_source_sha256": WEEK01_SOURCE_SHA256,
            "benchmark_provenance": benchmark_provenance,
            "dataset_manifest_id": manifest["manifest_id"],
            "dataset_manifest_sha256": manifest_hash,
            "required_dataset_ids": [str(item["dataset_id"]) for item in manifest["datasets"]],
            "executed_dataset_ids": [dataset.dataset_id for dataset in datasets],
            "executed_method_ids": list(METHOD_IDS),
            "split_specification": dict(manifest["split"] if mode == "full" else manifest["smoke"]),
            "n_estimators": n_estimators,
            "frozen_teacher": dict(frozen_teacher),
            "frozen_methods": {
                "student": {k: method_cfg[k] for k in ("hidden_dims", "learning_rate", "epochs", "lambda_ranking", "seed")},
                "supcon_epochs": method_cfg["supcon_epochs"],
                "mlp_epochs": method_cfg["mlp_epochs"],
                "pca_components": method_cfg["pca_components"],
            },
            "frozen_method_ids": list(METHOD_IDS),
            "runtime": runtime,
            "dataset_results": results,
            "scientific_gate": scientific_gate,
            "method_fingerprints": method_fingerprints,
        }
    )
    return RunResult(
        evidence=dict(evidence),
        eval_rows=[dict(row) for row in _json_ready(rows)],
        dataset_provenance=dict(_json_ready(provenance)),
        dataset_lock=None if dataset_lock is None else dict(_json_ready(dataset_lock)),
    )


def _dataset_summary(
    dataset_id: str,
    metadata: Mapping[str, Any],
    seed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset": dict(metadata),
        "seeds": [dict(record) for record in seed_records],
        "gate": {},
    }
    per_method: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        recalls: dict[str, list[float]] = {}
        spears_beta: list[float] = []
        spears_raw: list[float] = []
        ndcg: list[float] = []
        top1rank: list[float] = []
        hit1: list[float] = []
        hit5: list[float] = []
        for record in seed_records:
            block = record["methods"][method_id]
            for key, value in block["teacher_topk_recall_mean"].items():
                recalls.setdefault(key, []).append(float(value))
            spears_beta.append(float(block["spearman_vs_beta_mean"]))
            spears_raw.append(float(block["spearman_vs_raw_mean"]))
            ndcg.append(float(block["ndcg_at_10_mean"]))
            top1rank.append(float(block["teacher_top1_rank_mean"]))
            hit1.append(float(block["label_hit_at_1_rate"]))
            hit5.append(float(block["label_hit_at_5_rate"]))
        per_method[method_id] = {
            "teacher_topk_recall_median": {key: _median(values, name=dataset_id + "." + method_id + ".recall") for key, values in recalls.items()},
            "spearman_vs_beta_median": _median(spears_beta, name=dataset_id + "." + method_id + ".spearman_beta"),
            "spearman_vs_raw_median": _median(spears_raw, name=dataset_id + "." + method_id + ".spearman_raw"),
            "ndcg_at_10_median": _median(ndcg, name=dataset_id + "." + method_id + ".ndcg"),
            "teacher_top1_rank_median": _median(top1rank, name=dataset_id + "." + method_id + ".top1rank"),
            "label_hit_at_1_median": _median(hit1, name=dataset_id + "." + method_id + ".hit1"),
            "label_hit_at_5_median": _median(hit5, name=dataset_id + "." + method_id + ".hit5"),
        }
    summary["gate"] = {"aggregation": "median over seeds of per-seed query means", "methods": per_method}
    return summary


def _global_gate(
    dataset_results: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    required_ids = [str(spec["dataset_id"]) for spec in manifest["datasets"]]
    if [str(result["dataset_id"]) for result in dataset_results] != required_ids:
        raise RunnerError("global gate requires every frozen dataset in manifest order")
    gate = manifest["gate"]
    summary: dict[str, Any] = {
        "frozen_dataset_count": len(dataset_results),
        "executed_dataset_ids": required_ids,
        "executed_method_ids": list(METHOD_IDS),
        "decision": gate["decision"],
        "recommendation": gate["recommendation"],
    }
    per_method_global: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        recalls: dict[str, list[float]] = {}
        spears: list[float] = []
        for result in dataset_results:
            block = result["gate"]["methods"][method_id]
            for key, value in block["teacher_topk_recall_median"].items():
                recalls.setdefault(key, []).append(float(value))
            spears.append(float(block["spearman_vs_beta_median"]))
        per_method_global[method_id] = {
            "teacher_topk_recall_mean": {key: float(np.mean(np.asarray(v, dtype=np.float64))) for key, v in recalls.items()},
            "spearman_mean": float(np.mean(np.asarray(spears, dtype=np.float64))),
        }
    summary["global_method_means"] = per_method_global
    # Per-dataset winner by spearman_vs_beta_median (tie tolerance 1e-9).
    win_counts = {m: 0 for m in METHOD_IDS}
    winner_rows = []
    for result in dataset_results:
        best: str | None = None
        best_value = float("-inf")
        for method_id in METHOD_IDS:
            value = float(result["gate"]["methods"][method_id]["spearman_vs_beta_median"])
            if value > best_value + 1e-9:
                best_value = value
                best = method_id
        assert best is not None
        win_counts[best] += 1
        winner_rows.append({"dataset_id": str(result["dataset_id"]), "winner": best, "spearman": best_value})
    summary["win_counts"] = win_counts
    summary["per_dataset_winners"] = winner_rows
    return summary


def _report_markdown(evidence: Mapping[str, Any], *, run_id: str) -> str:
    gate = evidence["scientific_gate"]
    methods_cfg = evidence.get("frozen_methods", {})
    lines = [
        "# PredRel - Formal Benchmark on Frozen Real Data (20 datasets x 8 methods)",
        "",
        "Technical status: PASS",
        "Run ID: " + str(run_id),
        "Completed at UTC: " + str(evidence["completed_at_utc"]),
        "Mode: " + str(evidence["mode"]),
        "Week 1 Teacher source SHA-256: " + str(evidence["teacher_source_sha256"]),
        "Dataset manifest SHA-256: " + str(evidence["dataset_manifest_sha256"]),
        "Methods: " + ", ".join(evidence["executed_method_ids"]),
        "Student epochs: " + str(methods_cfg.get("student", {}).get("epochs"))
        + ", SupCon epochs=" + str(methods_cfg.get("supcon_epochs"))
        + ", MLP epochs=" + str(methods_cfg.get("mlp_epochs")),
        "",
        "## Executed methods",
        "",
        ", ".join(evidence["executed_method_ids"]),
        "",
        "## Dataset coverage",
        "",
        "Executed: " + ", ".join(evidence["executed_dataset_ids"]),
        "Frozen full benchmark: " + ", ".join(evidence["required_dataset_ids"]),
        "",
        "## Global method means (teacher-top-K recall / spearman vs beta)",
        "",
        "| Method | R@1/R@5/R@10 | spearman |",
        "| --- | --- | --- |",
    ]
    ranking = []
    for method_id, block in evidence["scientific_gate"].get("global_method_means", {}).items():
        recall = block.get("teacher_topk_recall_mean", {})
        cell = "/".join("%.4f" % float(recall.get(k, float("nan"))) for k in ("1", "5", "10"))
        spear = float(block.get("spearman_mean", float("nan")))
        lines.append("| %s | %s | %.4f |" % (method_id, cell, spear))
        ranking.append((method_id, spear))
    ranking.sort(key=lambda item: item[1], reverse=True)
    lines.extend(
        [
            "",
            "## Method ranking by global spearman (descriptive)",
            "",
            ", ".join("%s (%.4f)" % item for item in ranking),
            "",
            "## Win counts (per-dataset spearman winner, tie tolerance 1e-9)",
            "",
            ", ".join("%s: %d" % (m, gate.get("win_counts", {}).get(m, 0)) for m in evidence["executed_method_ids"]),
            "",
            "## Scientific gate (descriptive completion)",
            "",
            "Decision: " + str(gate.get("decision")),
            "Recommendation: " + str(gate.get("recommendation")),
            "",
            "A technical PASS validates execution and provenance only. Method interpretation belongs to the investigator.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evidence(
    result: RunResult,
    *,
    output_dir: str | Path,
    run_id: str,
    run_input: Mapping[str, Any],
) -> Path:
    """Write a fresh, non-overwriting evidence directory."""

    root = Path(output_dir).resolve()
    if root.exists():
        raise RunnerError("evidence directory already exists: " + str(root))
    root.mkdir(parents=True, exist_ok=False)
    metrics = dict(result.evidence)
    metrics["run_id"] = run_id
    payloads: dict[str, object] = {
        "run_input.json": dict(run_input),
        "metrics.json": metrics,
        "dataset_provenance.json": result.dataset_provenance,
        "runtime_environment.json": metrics["runtime"],
        "benchmark.md": _report_markdown(metrics, run_id=run_id),
    }
    if result.dataset_lock is not None:
        payloads["dataset_lock.json"] = result.dataset_lock
    for name, payload in payloads.items():
        destination = root / name
        if isinstance(payload, str):
            destination.write_text(payload, encoding="utf-8")
        else:
            destination.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_path = root / "eval_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in result.eval_rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "pass",
        "technical_gate": "completed",
        "scientific_gate": metrics["scientific_gate"]["decision"],
        "recommendation": metrics["scientific_gate"]["recommendation"],
        "evidence_files": sorted([path.name for path in root.iterdir() if path.is_file()] + ["run_status.json"]),
    }
    (root / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


__all__ = ["DatasetError", "RunnerError", "TeacherBridgeError", "run_benchmark", "write_evidence"]
