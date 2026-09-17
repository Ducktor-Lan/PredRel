"""Run the frozen Week 9 target/loss/dimension ablation benchmark.

Protocol per (dataset, seed), fixed by provenance/dataset_manifest.json:
  1. Build the frozen stratified split (same contract as Weeks 3-6/8).
  2. Hold out every query row. From the support, pick label-blind
     pseudo-queries (20 percent, 8..64, seed 20260914); the rest is the
     pseudo-support.
  3. One frozen Teacher fit over the train split yields per-pseudo-query
     alpha rows plus beta blocks and the mean-raw true-class block.
  4. Train every frozen grid cell (target/loss/qk_dim/variant) on the train
     split only, plus one matched SupCon baseline on the pseudo-support.
  5. Freeze students. One eval Teacher fit over the FULL support with all
     holdout queries yields per-query teacher true-class beta/raw blocks.
     Every cell scores every holdout query (labels metric-only).

Smoke mode executes a reduced contract: the frozen smoke dataset/seed, the
frozen smoke cell subset (one grid row per loss family at qk_dim 64), the
SupCon baseline at smoke epochs, and no scientific gate.

Leakage boundary: query labels (train or holdout) never enter any support
context or any Teacher fit. Train pseudo-query labels select the
true-class block only; SupCon uses pseudo-support labels only; holdout
labels are released only for metric computation.

This module orchestrates NumPy/Torch training plus Teacher fits; the
evidence writer persists cell metrics, per-cell-query metric rows (no raw
feature matrices, no teacher blocks, no parameter tensors -- only
recomputable fingerprints), provenance, and the frozen grid so the verifier
can re-derive and re-score every number from the manifest plus the Teacher.
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

from ..data import (
    DatasetError,
    RealDataset,
    load_manifest,
    make_stratified_split,
    materialize_manifest,
    verify_dataset_lock,
)
from ..label_anatomy import AnatomyError, decompose_alpha
from ..losses import EPSILON
from ..metrics import score_eval_query
from ..models import StudentError, select_pseudo_queries, student_params_to_record
from ..supcon import SupconError, cosine_similarities, embed_with_params, train_supcon_encoder
from ..teacher_bridge import WEEK01_SOURCE_SHA256, TeacherBridgeError, extract_readout
from ..train_student import cell_spec, train_ablation_cell


class RunnerError(RuntimeError):
    """Raised when Week 9 cannot make technically valid evidence."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """All fresh serializable evidence produced by one successful execution."""

    evidence: dict[str, Any]
    eval_rows: list[dict[str, Any]]
    dataset_provenance: dict[str, Any]
    dataset_lock: dict[str, Any] | None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_ablation_provenance() -> dict[str, Any]:
    """Read the unified ablation provenance (route + student baseline)."""

    path = _project_root() / "provenance" / "provenance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError("provenance record is unreadable: " + str(error)) from error
    if not isinstance(payload, dict):
        raise RunnerError("provenance record must be a JSON object")
    if payload.get("recommended_route") != "Route B: Static Directed with simple context adapter first":
        raise RunnerError("provenance does not authorize the Route B ablation")
    required = ("ablation_manifest_sha256",)
    if any(not isinstance(payload.get(field), str) or not payload[field] for field in required):
        raise RunnerError("provenance record is incomplete")
    baseline = payload.get("student_claim_baseline")
    if not isinstance(baseline, Mapping) or str(baseline.get("cell")) != "beta-KLrank_qk64_static":
        raise RunnerError("provenance lacks the frozen student claim baseline")
    if str(payload.get("ablation_manifest_id", "")) != "predrel-ablation-v1":
        raise RunnerError("provenance lacks the frozen ablation manifest id")
    return payload


_load_week09_dependency = _load_ablation_provenance


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
    """Collapse one trained cell to a recomputable fingerprint (no tensors)."""

    def _arr(name: str) -> np.ndarray:
        value = np.asarray(params[name], dtype=np.float64)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise RunnerError("cell fingerprint needs finite " + name)
        return value

    arrays = {
        "mean": _arr("mean"),
        "std": _arr("std"),
        "w_q": _arr("w_q"),
        "w_k": _arr("w_k"),
        **{"weights_%d" % i: _arr("w") if False else np.asarray(w, dtype=np.float64) for i, w in enumerate(params["weights"])},
        **{"biases_%d" % i: np.asarray(b, dtype=np.float64) for i, b in enumerate(params["biases"])},
    }
    if params.get("a_q") is not None:
        arrays["a_q"] = _arr("a_q")
    if params.get("a_k") is not None:
        arrays["a_k"] = _arr("a_k")
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
        raise RunnerError("cell fingerprint needs a finite loss history")
    return {
        "sha256": digest.hexdigest(),
        "target": str(params.get("target")),
        "loss": str(params.get("loss")),
        "qk_dim": int(params.get("qk_dim")),
        "adapter": bool(params.get("adapter")),
        "layer_dims": [int(v) for v in params.get("layer_dims", [])],
        "seed": int(params.get("seed")),
        "epochs": int(params.get("epochs")),
        "lambda_ranking": float(params.get("lambda_ranking", 0.5)),
        "loss_first": float(history[0]),
        "loss_last": float(history[-1]),
        "loss_min": float(min(history)),
        "loss_history_len": len(history),
        "true_positions": [[int(v) for v in row] for row in params.get("true_positions", [])],
    }


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


def _cells_for_mode(manifest: Mapping[str, Any], *, mode: str) -> list[dict[str, Any]]:
    cells = [dict(cell) for cell in manifest["student"]["grid"]["cells"]]
    if mode == "full":
        return cells
    smoke_ids = list(manifest["smoke"]["cell_ids"])
    selected = [cell for cell in cells if str(cell["cell_id"]) in smoke_ids]
    if len(selected) != len(smoke_ids):
        raise RunnerError("smoke cell subset does not match the frozen grid")
    return selected


def _student_config(manifest: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    student = manifest["student"]
    supcon = manifest["supcon_baseline"]
    smoke_epochs = int(manifest["smoke"].get("smoke_epochs", student["smoke_epochs"]))
    return {
        "hidden_dims": [int(v) for v in student["hidden_dims"]],
        "learning_rate": float(student["learning_rate"]),
        "epochs": int(student["epochs"] if mode == "full" else smoke_epochs),
        "lambda_ranking": float(manifest["losses"]["lambda_ranking"]),
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
    }


def _train_blocks(
    *,
    alpha: np.ndarray,
    raw_scores: np.ndarray,
    support_labels: np.ndarray,
    pseudo_query_labels: np.ndarray,
    true_positions: list[list[int]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Extract per-pseudo-query teacher alpha rows, beta blocks, mean-raw blocks.

    Blocks are ragged across pseudo-queries (true-class widths differ), so
    the function returns per-query lists. The trainer consumes each block
    with its own column map; nothing here stacks or pads them.
    """

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


def _run_seed(
    *,
    dataset: RealDataset,
    seed: int,
    split_args: Mapping[str, Any],
    student_cfg: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
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
        query_fraction=float(student_cfg["pseudo_query_fraction"]),
        max_queries=int(student_cfg["max_pseudo_queries"]),
        min_queries=int(student_cfg["min_pseudo_queries"]),
        seed=int(derive_seed_for_split(student_cfg, dataset.dataset_id, seed)),
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
    alpha_rows, beta_blocks, raw_blocks = _train_blocks(
        alpha=np.asarray(train_extraction.alpha, dtype=np.float64),
        raw_scores=np.asarray(train_extraction.raw_scores, dtype=np.float64),
        support_labels=pseudo_support_labels,
        pseudo_query_labels=pseudo_labels,
        true_positions=true_positions,
    )
    # Score-column contract: training scores stay in pseudo-support order;
    # each pseudo-query's true-class block is addressed by its own column
    # map (``true_positions``), which the trainer consumes and echoes back
    # and the verifier re-derives independently from the pseudo-split IDs.
    train_params: dict[str, Any] = {}
    eval_params: dict[str, Any] = {}
    for cell in cells:
        cell_spec(str(cell["cell_id"]), manifest)  # fail closed on grid drift
        params = train_ablation_cell(
            support_features=np.asarray(split.support_features)[keep],
            pseudo_query_features=np.asarray(split.support_features)[list(pseudo_positions)],
            teacher_alpha_rows=alpha_rows if str(cell["target"]) == "alpha" else None,
            teacher_beta_blocks=beta_blocks if str(cell["target"]) == "beta" else None,
            teacher_raw_blocks=raw_blocks,
            true_positions=true_positions,
            target=str(cell["target"]),
            loss=str(cell["loss"]),
            hidden_dims=[int(v) for v in student_cfg["hidden_dims"]],
            qk_dim=int(cell["qk_dim"]),
            adapter=bool(str(cell["variant"]) == "adapter"),
            learning_rate=float(student_cfg["learning_rate"]),
            epochs=int(student_cfg["epochs"]),
            lambda_ranking=float(student_cfg["lambda_ranking"]),
            seed=int(student_cfg["seed"]),
        )
        if [list(map(int, row)) for row in params["true_positions"]] != [list(map(int, row)) for row in true_positions]:
            raise RunnerError("trainer did not echo the frozen true-class column maps")
        if int(params["qk_dim"]) != int(cell["qk_dim"]) or str(params["target"]) != str(cell["target"]) or str(params["loss"]) != str(cell["loss"]):
            raise RunnerError("trainer did not honor the frozen cell " + str(cell["cell_id"]))
        train_params[str(cell["cell_id"])] = params
        eval_params[str(cell["cell_id"])] = student_params_from_record(student_params_to_record(params))
    try:
        supcon_params = train_supcon_encoder(
            np.asarray(split.support_features)[keep],
            pseudo_support_labels,
            hidden_dims=[int(v) for v in student_cfg["supcon_hidden_dims"]],
            embedding_dim=int(student_cfg["supcon_embedding_dim"]),
            temperature=float(student_cfg["supcon_temperature"]),
            learning_rate=float(student_cfg["supcon_learning_rate"]),
            epochs=int(student_cfg["supcon_epochs"]),
            batch_size=int(student_cfg["supcon_batch_size"]),
            seed=int(student_cfg["supcon_seed"]),
        )
    except SupconError as error:
        raise RunnerError("SupCon baseline failed: " + str(error)) from error

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
    support_labels = np.asarray(split.support_labels)
    supcon_support_emb = embed_with_params(supcon_params, split.support_features)
    supcon_query_emb = embed_with_params(supcon_params, split.query_features)
    supcon_scores = np.asarray(cosine_similarities(supcon_query_emb, supcon_support_emb), dtype=np.float64)
    eval_ks = [int(v) for v in manifest["evaluation"]["ks"]]
    cell_ids = [str(cell["cell_id"]) for cell in cells]
    rows: list[dict[str, Any]] = []
    per_cell_queries: dict[str, list[dict[str, Any]]] = {cell_id: [] for cell_id in cell_ids}
    per_cell_queries["supcon"] = []
    for qi in range(len(split.query_ids)):
        query_label = int(split.query_labels[qi])
        true_cols = [pos for pos in range(len(split.support_ids)) if int(support_labels[pos]) == query_label]
        if len(true_cols) < 2:
            raise RunnerError("an eval query has fewer than two true-class supports")
        teacher_beta = np.asarray(eval_decomposition.beta[qi][true_cols], dtype=np.float64)
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
            "teacher_mean_raw_block": [float(v) for v in np.asarray(eval_mean_raw[qi][true_cols]).tolist()],
            "cells": {},
        }
        for cell_id in cell_ids:
            full_scores = directed_scores_with_support_context(
                eval_params[cell_id], np.asarray(split.query_features[qi]).reshape(1, -1), split.support_features
            )[0]
            true_scores = np.asarray([float(full_scores[pos]) for pos in true_cols], dtype=np.float64)
            scored = score_eval_query(
                student_true_scores=true_scores,
                teacher_beta_block=teacher_beta,
                full_student_scores=full_scores,
                support_labels=support_labels,
                query_label=query_label,
                ks=eval_ks,
            )
            row_record["cells"][cell_id] = {
                **scored,
                "student_true_scores": [float(v) for v in true_scores.tolist()],
                "full_student_scores": [float(v) for v in np.asarray(full_scores).tolist()],
            }
            per_cell_queries[cell_id].append(scored)
        scored_supcon = score_eval_query(
            student_true_scores=np.asarray(supcon_scores[qi][true_cols], dtype=np.float64),
            teacher_beta_block=teacher_beta,
            full_student_scores=np.asarray(supcon_scores[qi], dtype=np.float64),
            support_labels=support_labels,
            query_label=query_label,
            ks=eval_ks,
        )
        row_record["cells"]["supcon"] = {
            **scored_supcon,
            "student_true_scores": [float(v) for v in np.asarray(supcon_scores[qi][true_cols]).tolist()],
            "full_student_scores": [float(v) for v in np.asarray(supcon_scores[qi]).tolist()],
        }
        per_cell_queries["supcon"].append(scored_supcon)
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
        "cells": {cell_id: _summarize_queries(per_cell_queries[cell_id]) for cell_id in [*cell_ids, "supcon"]},
        "query_count_scored": len(rows),
    }
    fingerprints = {cell_id: _fingerprint_params(train_params[cell_id]) for cell_id in cell_ids}
    runtime = {"runtime": dict(train_extraction.runtime), "decoder_readout_api": train_extraction.decoder_readout_api,
               "raw_score_hook_path": train_extraction.raw_score_hook_path}
    return seed_summary, rows, {"fingerprints": fingerprints, "supcon_params": supcon_params, "runtime": runtime}


def derive_seed_for_split(student_cfg: Mapping[str, Any], dataset_id: str, seed: int) -> int:
    from ..models import derive_seed

    return derive_seed(int(student_cfg["pseudo_query_seed"]), dataset_id, int(seed))


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
    ks = sorted({key for record in records for key in record["teacher_topk_recall"]}, key=int)
    summary: dict[str, Any] = {
        "query_count": len(records),
        "teacher_topk_recall_mean": {},
        "spearman_mean": 0.0,
        "label_hit_at_1_rate": 0.0,
        "label_hit_at_5_rate": 0.0,
    }
    for key in ks:
        values = [_require_finite(record["teacher_topk_recall"][key], name="recall@" + key) for record in records]
        summary["teacher_topk_recall_mean"][key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    spears = [_require_finite(record["spearman_student_vs_beta"], name="spearman") for record in records]
    summary["spearman_mean"] = float(np.mean(np.asarray(spears, dtype=np.float64)))
    summary["label_hit_at_1_rate"] = float(np.mean([float(record["label_hit_at_1"]) for record in records]))
    summary["label_hit_at_5_rate"] = float(np.mean([float(record["label_hit_at_5"]) for record in records]))
    return summary


def _median(values: Sequence[float], *, name: str) -> float:
    array = np.asarray([_require_finite(v, name=name) for v in values], dtype=np.float64)
    if array.size == 0:
        raise RunnerError(name + " has no values")
    return float(np.median(array))


def run_ablation(
    *,
    mode: str,
    dataset_manifest_path: str | Path,
    dataset_cache_dir: str | Path,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> RunResult:
    """Run a smoke or full frozen target/loss/dimension ablation without downloading data."""

    manifest, manifest_hash = load_manifest(dataset_manifest_path)
    ablation_provenance = _load_ablation_provenance()
    if manifest.get("manifest_id") != "predrel-ablation-v1":
        raise RunnerError("dataset manifest is not the frozen unified ablation manifest")
    if manifest_hash != str(ablation_provenance.get("ablation_manifest_sha256", "")).lower():
        raise RunnerError("dataset manifest hash differs from the frozen provenance record")
    cells = _cells_for_mode(manifest, mode=mode)
    frozen_teacher = manifest.get("teacher")
    if not isinstance(frozen_teacher, Mapping):
        raise RunnerError("frozen manifest lacks Teacher settings")
    if int(frozen_teacher.get("n_estimators", 0)) != int(n_estimators):
        raise RunnerError("n_estimators differs from the frozen manifest value")
    if model_cache_dir != frozen_teacher.get("model_cache_dir"):
        raise RunnerError("model_cache_dir differs from the frozen manifest cache")
    seeds, split_args = _split_config(manifest, mode=mode)
    student_cfg = _student_config(manifest, mode=mode)
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
    cell_fingerprints: dict[str, Any] = {}
    for dataset in datasets:
        seed_records: list[dict[str, Any]] = []
        for seed in seeds:
            seed_summary, seed_rows, artifacts = _run_seed(
                dataset=dataset,
                seed=int(seed),
                split_args=split_args,
                student_cfg=student_cfg,
                cells=cells,
                manifest=manifest,
                n_estimators=n_estimators,
                model_cache_dir=model_cache_dir,
                week01_source_root=week01_source_root,
            )
            seed_records.append(seed_summary)
            rows.extend(seed_rows)
            key = dataset.dataset_id + ":seed-" + str(seed)
            first = artifacts["fingerprints"][str(cells[0]["cell_id"])]["true_positions"]
            for cell_id, fingerprint in artifacts["fingerprints"].items():
                if [list(map(int, row)) for row in fingerprint["true_positions"]] != [list(map(int, row)) for row in first]:
                    raise RunnerError("cell column maps disagree on " + key + "/" + str(cell_id))
            cell_fingerprints[key] = {
                "cells": {cell_id: fingerprint for cell_id, fingerprint in artifacts["fingerprints"].items()},
                "supcon": _json_ready(artifacts["supcon_params"]),
                "pseudo_query_positions": [int(v) for v in seed_summary["pseudo_query_positions"]],
                "true_positions": [list(map(int, row)) for row in first],
            }
            runtime = artifacts["runtime"]["runtime"]
            decoder_api = artifacts["runtime"]["decoder_readout_api"]
            raw_hook = artifacts["runtime"]["raw_score_hook_path"]
        dataset_gate = _dataset_summary(dataset.dataset_id, dataset.metadata, seed_records, [str(c["cell_id"]) for c in cells])
        results.append(dataset_gate)
    if runtime is None or decoder_api is None or raw_hook is None:
        raise RunnerError("Teacher was never invoked")
    if mode == "full":
        scientific_gate = _global_gate(results, manifest, [str(c["cell_id"]) for c in cells])
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
            "ablation_provenance": ablation_provenance,
            "dataset_manifest_id": manifest["manifest_id"],
            "dataset_manifest_sha256": manifest_hash,
            "required_dataset_ids": [str(item["dataset_id"]) for item in manifest["datasets"]],
            "executed_dataset_ids": [dataset.dataset_id for dataset in datasets],
            "executed_cell_ids": [str(c["cell_id"]) for c in cells],
            "split_specification": dict(manifest["split"] if mode == "full" else manifest["smoke"]),
            "n_estimators": n_estimators,
            "frozen_teacher": dict(frozen_teacher),
            "frozen_student": {
                "hidden_dims": student_cfg["hidden_dims"],
                "learning_rate": student_cfg["learning_rate"],
                "epochs": student_cfg["epochs"],
                "lambda_ranking": student_cfg["lambda_ranking"],
                "seed": student_cfg["seed"],
                "adapter": "mean-pool additive (no DeepSets)",
            },
            "frozen_grid": [dict(cell) for cell in cells],
            "runtime": runtime,
            "dataset_results": results,
            "scientific_gate": scientific_gate,
            "cell_fingerprints": cell_fingerprints,
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
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset": dict(metadata),
        "seeds": [dict(record) for record in seed_records],
        "gate": {},
    }
    per_cell: dict[str, Any] = {}
    for cell_id in [*cell_ids, "supcon"]:
        recalls: dict[str, list[float]] = {}
        spears: list[float] = []
        hit1: list[float] = []
        hit5: list[float] = []
        for record in seed_records:
            block = record["cells"][cell_id]
            for key, value in block["teacher_topk_recall_mean"].items():
                recalls.setdefault(key, []).append(float(value))
            spears.append(float(block["spearman_mean"]))
            hit1.append(float(block["label_hit_at_1_rate"]))
            hit5.append(float(block["label_hit_at_5_rate"]))
        per_cell[cell_id] = {
            "teacher_topk_recall_median": {key: _median(values, name=dataset_id + "." + cell_id + ".recall") for key, values in recalls.items()},
            "spearman_median": _median(spears, name=dataset_id + "." + cell_id + ".spearman"),
            "label_hit_at_1_median": _median(hit1, name=dataset_id + "." + cell_id + ".hit1"),
            "label_hit_at_5_median": _median(hit5, name=dataset_id + "." + cell_id + ".hit5"),
        }
    summary["gate"] = {"aggregation": "median over seeds of per-seed query means", "cells": per_cell}
    return summary


def _global_gate(
    dataset_results: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    required_ids = [str(spec["dataset_id"]) for spec in manifest["datasets"]]
    if [str(result["dataset_id"]) for result in dataset_results] != required_ids:
        raise RunnerError("global gate requires every frozen dataset in manifest order")
    gate = manifest["gate"]
    summary: dict[str, Any] = {
        "frozen_dataset_count": len(dataset_results),
        "executed_dataset_ids": required_ids,
        "executed_cell_ids": list(cell_ids),
        "decision": gate["decision"],
        "recommendation": gate["recommendation"],
    }
    per_cell_global: dict[str, Any] = {}
    for cell_id in [*cell_ids, "supcon"]:
        recalls: dict[str, list[float]] = {}
        spears: list[float] = []
        for result in dataset_results:
            block = result["gate"]["cells"][cell_id]
            for key, value in block["teacher_topk_recall_median"].items():
                recalls.setdefault(key, []).append(float(value))
            spears.append(float(block["spearman_median"]))
        per_cell_global[cell_id] = {
            "teacher_topk_recall_mean": {key: float(np.mean(np.asarray(v, dtype=np.float64))) for key, v in recalls.items()},
            "spearman_mean": float(np.mean(np.asarray(spears, dtype=np.float64))),
        }
    summary["global_cell_means"] = per_cell_global
    return summary


def _report_markdown(evidence: Mapping[str, Any], *, run_id: str) -> str:
    gate = evidence["scientific_gate"]
    student = evidence.get("frozen_student", {})
    lines = [
        "# Week 09 - Target/Loss/Dimension Ablation on Frozen Real Data",
        "",
        "Technical status: PASS",
        "Run ID: " + str(run_id),
        "Completed at UTC: " + str(evidence["completed_at_utc"]),
        "Mode: " + str(evidence["mode"]),
        "Week 1 Teacher source SHA-256: " + str(evidence["teacher_source_sha256"]),
        "Dataset manifest SHA-256: " + str(evidence["dataset_manifest_sha256"]),
        "Student: hidden=" + str(student.get("hidden_dims"))
        + ", epochs=" + str(student.get("epochs")) + ", lr=" + str(student.get("learning_rate"))
        + ", lambda_ranking=" + str(student.get("lambda_ranking")),
        "",
        "## Executed cells",
        "",
        ", ".join(evidence["executed_cell_ids"]),
        "",
        "## Dataset coverage",
        "",
        "Executed: " + ", ".join(evidence["executed_dataset_ids"]),
        "Frozen full benchmark: " + ", ".join(evidence["required_dataset_ids"]),
        "",
        "## Global cell means (teacher-top-K recall / spearman)",
        "",
        "| Cell | R@1/R@5/R@10 | spearman |",
        "| --- | --- | --- |",
    ]
    ranking = []
    for cell_id, block in evidence["scientific_gate"].get("global_cell_means", {}).items():
        recall = block.get("teacher_topk_recall_mean", {})
        cell = "/".join("%.4f" % float(recall.get(k, float("nan"))) for k in ("1", "5", "10"))
        spear = float(block.get("spearman_mean", float("nan")))
        lines.append("| %s | %s | %.4f |" % (cell_id, cell, spear))
        ranking.append((cell_id, spear))
    ranking.sort(key=lambda item: item[1], reverse=True)
    lines.extend(
        [
            "",
            "## Cell ranking by global spearman (descriptive)",
            "",
            ", ".join("%s (%.4f)" % item for item in ranking),
            "",
            "## Scientific gate (descriptive completion)",
            "",
            "Decision: " + str(gate.get("decision")),
            "Recommendation: " + str(gate.get("recommendation")),
            "",
            "A technical PASS validates execution and provenance only. Target/loss/dimension interpretation belongs to the investigator.",
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
    """Write a fresh, non-overwriting Week 9 evidence directory."""

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
        "week09_ablation.md": _report_markdown(metrics, run_id=run_id),
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


__all__ = ["DatasetError", "RunnerError", "TeacherBridgeError", "run_ablation", "write_evidence"]
