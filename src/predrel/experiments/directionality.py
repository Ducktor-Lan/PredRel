"""Experiment orchestration and evidence writing for Week 5 directionality."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data import (
    DatasetError,
    RealDataset,
    load_manifest,
    make_stratified_split,
    materialize_manifest,
    verify_dataset_lock,
)
from ..directionality import (
    DirectionalityError,
    check_row_stochastic,
    retrieval_overlap,
    select_probe_pool,
    summarize_baseline_matrix,
    summarize_directed_matrix,
)
from ..label_anatomy import AnatomyError, decompose_alpha
from ..removal import (
    RemovalError,
    build_removal_sets,
    class_distribution,
    paired_comparison,
    removal_effect,
    select_removal_queries,
    summarize_effects,
)
from ..supcon import (
    SupconError,
    cosine_similarities,
    embed_with_params,
    train_supcon_encoder,
)
from ..teacher_bridge import WEEK01_SOURCE_SHA256, TeacherBridgeError, extract_readout


class RunnerError(RuntimeError):
    """Raised when a Week 5 execution plan cannot make valid evidence."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """All serializable result material from a completed technical execution."""

    evidence: dict[str, Any]
    directionality_profiles: list[dict[str, Any]]
    extension_removal_profiles: list[dict[str, Any]]
    dataset_provenance: dict[str, Any]
    dataset_lock: dict[str, Any] | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _metric_median(seed_summaries: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for summary in seed_summaries:
        current: Any = summary
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise RunnerError("missing metric path " + ".".join(path))
            current = current[key]
        try:
            values.append(float(current))
        except (TypeError, ValueError) as error:
            raise RunnerError("non-numeric metric path " + ".".join(path)) from error
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise RunnerError("metric path " + ".".join(path) + " is empty or non-finite")
    return float(np.median(arr))


def _directionality_dataset_gate(
    seed_summaries: list[dict[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    dgate = gate["directionality"]
    primary_k = str(dgate["primary_k"])
    median_af = _metric_median(seed_summaries, ("summary", "asymmetry", "A_F"))
    median_recip = _metric_median(
        seed_summaries, ("summary", "reciprocity", primary_k, "mean")
    )
    median_reversal = _metric_median(
        seed_summaries, ("summary", "rank_agreement", "mean_rank_reversal_rate")
    )
    strong_af = float(dgate["strong_af"])
    weak_af = float(dgate["weak_af"])
    strong_recip = float(dgate["strong_reciprocity_at_primary_k"])
    weak_recip = float(dgate["weak_reciprocity_at_primary_k"])
    strong_rev = float(dgate["strong_rank_reversal"])
    weak_rev = float(dgate["weak_rank_reversal"])
    strong = bool(
        median_af >= strong_af
        or (median_recip <= strong_recip and median_reversal >= strong_rev)
    )
    weak = bool(
        median_af <= weak_af and median_recip >= weak_recip and median_reversal <= weak_rev
    )
    if strong and weak:
        raise RunnerError("frozen weak/strong boundaries overlap on observed medians")
    regime = "strong" if strong else ("weak" if weak else "moderate")
    return {
        "seed_aggregation": gate.get("seed_aggregation", "median of per-seed means"),
        "primary_k": int(dgate["primary_k"]),
        "median_A_F": median_af,
        "median_reciprocity_at_primary_k": median_recip,
        "median_rank_reversal_rate": median_reversal,
        "thresholds": {
            "strong_af": strong_af,
            "weak_af": weak_af,
            "strong_reciprocity_at_primary_k": strong_recip,
            "weak_reciprocity_at_primary_k": weak_recip,
            "strong_rank_reversal": strong_rev,
            "weak_rank_reversal": weak_rev,
        },
        "regime": regime,
        "strong": strong,
        "weak": weak,
    }


def _global_gate(
    dataset_results: list[dict[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    count = len(dataset_results)
    if count != len(manifest["datasets"]):
        raise RunnerError("the scientific gate requires every frozen dataset")
    gate = manifest["gate"]
    extension_ids = [str(v) for v in gate["extension_dataset_ids"]]
    strong_ids = [r["dataset_id"] for r in dataset_results if r["gate"]["strong"]]
    weak_ids = [r["dataset_id"] for r in dataset_results if r["gate"]["weak"]]
    ext_results = [r for r in dataset_results if r["dataset_id"] in extension_ids]
    if len(ext_results) != len(extension_ids):
        raise RunnerError("extension-arm results are incomplete")
    beyond_ids = [r["dataset_id"] for r in ext_results if bool(r["gate"].get("beyond_supcon"))]
    strong_majority = len(strong_ids) > count / 2.0
    weak_majority = len(weak_ids) > count / 2.0
    if strong_majority:
        return {
            "frozen_dataset_count": count,
            "strong_count": len(strong_ids),
            "strong_dataset_ids": strong_ids,
            "weak_count": len(weak_ids),
            "weak_dataset_ids": weak_ids,
            "extension_dataset_ids": extension_ids,
            "extension_beyond_supcon_count": len(beyond_ids),
            "extension_beyond_supcon_ids": beyond_ids,
            "majority_definition": gate["continue_when_strong"],
            "recommendation": gate["continue_recommendation_dual_space"],
            "decision": "continue_to_week06_context",
            "dual_space_candidate": True,
            "stop_reason": None,
        }
    if weak_majority and len(beyond_ids) == 0:
        reason = str(gate["stop_reason_weak_no_gain"])
        if reason not in gate["stop_reason_values"]:
            raise RunnerError("stop reason is outside the frozen gate contract")
        return {
            "frozen_dataset_count": count,
            "strong_count": len(strong_ids),
            "strong_dataset_ids": strong_ids,
            "weak_count": len(weak_ids),
            "weak_dataset_ids": weak_ids,
            "extension_dataset_ids": extension_ids,
            "extension_beyond_supcon_count": 0,
            "extension_beyond_supcon_ids": [],
            "majority_definition": gate["stop_when"],
            "recommendation": gate["stop_recommendation"],
            "decision": "stop_readout2rep_route",
            "dual_space_candidate": False,
            "stop_reason": reason,
        }
    return {
        "frozen_dataset_count": count,
        "strong_count": len(strong_ids),
        "strong_dataset_ids": strong_ids,
        "weak_count": len(weak_ids),
        "weak_dataset_ids": weak_ids,
        "extension_dataset_ids": extension_ids,
        "extension_beyond_supcon_count": len(beyond_ids),
        "extension_beyond_supcon_ids": beyond_ids,
        "majority_definition": gate["continue_when_moderate"],
        "recommendation": gate["continue_recommendation"],
        "decision": "continue_to_week06_context",
        "dual_space_candidate": False,
        "stop_reason": None,
    }


def _split_args(manifest: Mapping[str, Any], *, mode: str) -> tuple[list[int], dict[str, Any]]:
    if mode == "full":
        split = manifest["split"]
        return list(split["seeds"]), {
            "max_support_size": int(split["max_support_size"]),
            "max_query_size": int(split["max_query_size"]),
            "train_fraction": float(split["train_fraction"]),
            "min_support_per_class": int(split["min_support_per_class"]),
            "min_query_per_class": int(split["min_query_per_class"]),
        }
    smoke = manifest["smoke"]
    split = manifest["split"]
    return [int(smoke["seed"])], {
        "max_support_size": int(smoke["max_support_size"]),
        "max_query_size": int(smoke["max_query_size"]),
        "train_fraction": float(split["train_fraction"]),
        "min_support_per_class": int(split["min_support_per_class"]),
        "min_query_per_class": int(split["min_query_per_class"]),
    }


def _directionality_plan(manifest: Mapping[str, Any], *, mode: str) -> tuple[int, list[int], int]:
    d = manifest["directionality"]
    supcon = manifest["supcon"]
    if mode == "full":
        return (int(d["probe_pool_size"]), [int(v) for v in d["ks"]], int(supcon["epochs"]))
    return (int(d["smoke_probe_pool_size"]), [int(v) for v in d["ks"]], int(supcon["smoke_epochs"]))


def _directed_row(
    *,
    pool_features: np.ndarray,
    pool_labels: np.ndarray,
    pool_ids: tuple[str, ...],
    held_out: int,
    seed: int,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> np.ndarray:
    n = len(pool_ids)
    keep = [i for i in range(n) if i != held_out]
    kept_ids = tuple(pool_ids[i] for i in keep)
    extraction = extract_readout(
        support_features=np.asarray(pool_features)[keep],
        support_labels=np.asarray(pool_labels)[keep],
        support_ids=kept_ids,
        query_features=np.asarray(pool_features[held_out]).reshape(1, -1),
        query_ids=(pool_ids[held_out],),
        seed=seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    if extraction.support_ids != kept_ids or extraction.query_ids != (pool_ids[held_out],):
        raise RunnerError("directed Teacher output IDs are not aligned to the probe pool")
    alpha = np.asarray(extraction.alpha, dtype=np.float64).reshape(-1)
    if alpha.shape != (n - 1,):
        raise RunnerError("directed alpha row has an unexpected width")
    row = np.zeros(n, dtype=np.float64)
    row[keep] = alpha
    return row


def _symmetric_reference(
    pool_features: np.ndarray, *, supcon_params: Mapping[str, Any]
) -> np.ndarray:
    emb = embed_with_params(supcon_params, np.asarray(pool_features, dtype=np.float64))
    cos = np.asarray(cosine_similarities(emb, emb), dtype=np.float64)
    if cos.shape[0] != cos.shape[1] or cos.shape[0] != np.asarray(pool_features).shape[0]:
        raise RunnerError("SupCon similarity matrix has an unexpected shape")
    sym = (cos + 1.0) / 2.0
    sym = (sym + sym.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    if not np.all(np.isfinite(sym)) or np.any(sym < 0.0):
        raise RunnerError("symmetric reference matrix is not finite non-negative")
    return np.ascontiguousarray(sym, dtype=np.float64)


def _run_dataset(
    dataset: RealDataset,
    *,
    seeds: list[int],
    split_args: Mapping[str, Any],
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
    manifest: Mapping[str, Any],
    max_pool: int,
    ks: list[int],
    supcon_epochs: int,
    is_extension: bool,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dspec = manifest["directionality"]
    supcon_spec = manifest["supcon"]
    gate = manifest["gate"]
    probe_seed = int(dspec["probe_seed"])
    stochastic_atol = float(dspec["stochastic_atol"])
    seed_records: list[dict[str, Any]] = []
    dir_profiles: list[dict[str, Any]] = []
    ext_profiles: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    decoder_readout_api: str | None = None
    raw_score_hook_path: str | None = None
    for seed in seeds:
        split = make_stratified_split(dataset, seed=seed, **dict(split_args))
        pool_positions = select_probe_pool(
            np.asarray(split.support_labels), max_pool=max_pool, seed=probe_seed
        )
        pool_features = np.asarray(split.support_features)[pool_positions]
        pool_labels = np.asarray(split.support_labels)[pool_positions]
        pool_ids = tuple(split.support_ids[i] for i in pool_positions)
        n = len(pool_ids)
        rows: list[np.ndarray] = []
        for held_out in range(n):
            rows.append(
                _directed_row(
                    pool_features=pool_features,
                    pool_labels=pool_labels,
                    pool_ids=pool_ids,
                    held_out=held_out,
                    seed=seed,
                    n_estimators=n_estimators,
                    model_cache_dir=model_cache_dir,
                    week01_source_root=week01_source_root,
                )
            )
        teacher_r = np.ascontiguousarray(np.stack(rows), dtype=np.float64)
        stochastic = check_row_stochastic(teacher_r, atol=stochastic_atol)
        if not stochastic["within_tolerance"]:
            raise RunnerError(
                "directed rows violate the alpha simplex (max_dev="
                + str(stochastic["max_abs_deviation"]) + ")"
            )
        try:
            supcon_params = train_supcon_encoder(
                split.support_features,
                split.support_labels,
                hidden_dims=[int(v) for v in supcon_spec["hidden_dims"]],
                embedding_dim=int(supcon_spec["embedding_dim"]),
                temperature=float(supcon_spec["temperature"]),
                learning_rate=float(supcon_spec["learning_rate"]),
                epochs=supcon_epochs,
                batch_size=int(supcon_spec["batch_size"]),
                seed=int(supcon_spec["seed"]),
            )
        except SupconError as error:
            raise RunnerError("SupCon baseline failed: " + str(error)) from error
        baseline_s = _symmetric_reference(pool_features, supcon_params=supcon_params)
        try:
            r_summary = summarize_directed_matrix(teacher_r, ks=ks)
            s_summary = summarize_baseline_matrix(baseline_s, ks=ks)
            cross = retrieval_overlap(teacher_r, baseline_s, ks=ks)
        except DirectionalityError as error:
            raise RunnerError("directionality summary failed: " + str(error)) from error
        for i in range(n):
            dir_profiles.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "seed": seed,
                    "split_id": split.split_id,
                    "probe_pool_ids": list(pool_ids),
                    "probe_pool_size": n,
                    "held_out_position": int(i),
                    "held_out_id": pool_ids[i],
                    "directed_row": [float(v) for v in teacher_r[i].tolist()],
                    "baseline_row": [float(v) for v in baseline_s[i].tolist()],
                }
            )
        seed_summary: dict[str, Any] = {
            "seed": seed,
            "split_id": split.split_id,
            "holdout_ids": list(split.holdout_ids),
            "holdout_count": len(split.holdout_ids),
            "support_ids": list(split.support_ids),
            "support_count": int(split.support_features.shape[0]),
            "query_ids": list(split.query_ids),
            "query_count": int(split.query_features.shape[0]),
            "probe_pool_size": n,
            "probe_pool_ids": list(pool_ids),
            "stochastic": stochastic,
            "summary": {
                "asymmetry": r_summary["asymmetry"],
                "reciprocity": r_summary["reciprocity"],
                "rank_agreement": r_summary["rank_agreement"],
                "transpose_agreement": r_summary["transpose_agreement"],
                "heatmap": r_summary["heatmap"],
                "baseline_asymmetry": s_summary["asymmetry"],
                "baseline_reciprocity": s_summary["reciprocity"],
                "retrieval_R_vs_S": cross,
            },
        }
        if is_extension and mode == "full":
            recheck = _extension_removal_recheck(
                split=split,
                seed=seed,
                manifest=manifest,
                supcon_params=supcon_params,
                n_estimators=n_estimators,
                model_cache_dir=model_cache_dir,
                week01_source_root=week01_source_root,
                dataset_id=dataset.dataset_id,
                split_id=split.split_id,
                ext_profiles=ext_profiles,
            )
            seed_summary["extension_removal_recheck"] = recheck
        seed_records.append(seed_summary)
        probe = extract_readout(
            support_features=np.asarray(pool_features)[1:],
            support_labels=np.asarray(pool_labels)[1:],
            support_ids=pool_ids[1:],
            query_features=np.asarray(pool_features[:1]),
            query_ids=(pool_ids[0],),
            seed=seed,
            n_estimators=n_estimators,
            model_cache_dir=model_cache_dir,
            week01_source_root=week01_source_root,
        )
        runtime = probe.runtime
        decoder_readout_api = probe.decoder_readout_api
        raw_score_hook_path = probe.raw_score_hook_path
    if runtime is None or decoder_readout_api is None or raw_score_hook_path is None:
        raise RunnerError(dataset.dataset_id + ": Teacher was not invoked")
    dataset_gate = _directionality_dataset_gate(seed_records, gate)
    if is_extension and mode == "full":
        diffs: list[float] = []
        for record in seed_records:
            recheck = record.get("extension_removal_recheck", {})
            paired = ((recheck.get("per_k") or {}).get("3") or {}).get("paired_top_vs_supcon") or {}
            if "mean_diff" in paired:
                diffs.append(float(paired["mean_diff"]))
        margin = float(gate["removal_recheck"]["tv_diff_margin"])
        if diffs:
            median_diff = float(np.median(np.asarray(diffs, dtype=np.float64)))
        else:
            median_diff = float("nan")
        beyond = bool(median_diff > margin)
        dataset_gate["extension_removal_recheck"] = {
            "k": 3,
            "median_tv_diff_top_minus_supcon": median_diff,
            "tv_diff_margin": margin,
        }
        dataset_gate["beyond_supcon"] = beyond
    else:
        dataset_gate["beyond_supcon"] = None
    return (
        {
            "dataset_id": dataset.dataset_id,
            "dataset": dataset.metadata,
            "extension_arm": bool(is_extension),
            "seeds": seed_records,
            "gate": dataset_gate,
            "teacher": {
                "teacher_source_sha256": WEEK01_SOURCE_SHA256,
                "decoder_readout_api": decoder_readout_api,
                "raw_score_hook_path": raw_score_hook_path,
            },
        },
        dir_profiles,
        ext_profiles,
        runtime,
    )


def _extension_removal_recheck(
    *,
    split: Any,
    seed: int,
    manifest: Mapping[str, Any],
    supcon_params: Mapping[str, Any],
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
    dataset_id: str,
    split_id: str,
    ext_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    removal = manifest["removal_recheck"]
    supcon_spec = manifest["supcon"]
    metrics = manifest["metrics"]
    k = int(removal["k"])
    random_repeats = int(removal["random_repeats"])
    max_queries = int(removal["max_removal_queries"])
    seed_base = int(removal["random_seed_base"])
    epsilon = float(metrics["epsilon"])
    extraction = extract_readout(
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
    try:
        decomposition = decompose_alpha(extraction.alpha, split.support_labels)
    except AnatomyError as error:
        raise RunnerError("extension recheck decomposition failed: " + str(error)) from error
    classes_full = np.asarray(decomposition.classes)
    mass_full = np.asarray(decomposition.class_mass, dtype=np.float64)
    beta = np.asarray(decomposition.beta, dtype=np.float64)
    support_emb = embed_with_params(supcon_params, split.support_features)
    query_emb = embed_with_params(supcon_params, split.query_features)
    similarities = cosine_similarities(query_emb, support_emb)
    rng = np.random.default_rng([seed_base, int(seed)])
    positions = select_removal_queries(split.query_labels, max_queries=max_queries)
    support_labels = np.asarray(split.support_labels)
    top_tv: dict[int, float] = {}
    supcon_tv: dict[int, float] = {}
    effects: dict[str, list[Any]] = {"top_beta": [], "supcon_top": []}
    for qp in positions:
        qid = split.query_ids[qp]
        qlabel = int(split.query_labels[qp])
        try:
            sets = build_removal_sets(
                beta[qp], support_labels, qlabel, k=k, rng=rng,
                random_repeats=random_repeats, supcon_scores=similarities[qp],
            )
        except RemovalError as error:
            raise RunnerError("extension recheck set build failed: " + str(error)) from error
        full_dist = mass_full[qp]
        for condition in ("top_beta", "supcon_top"):
            removed = list(sets[condition] or [])
            removed_mass = float(np.asarray(extraction.alpha[qp])[removed].sum())
            keep = np.delete(np.arange(len(split.support_ids)), np.asarray(removed, dtype=np.int64))
            kept_ids = tuple(split.support_ids[int(i)] for i in keep.tolist())
            abl = extract_readout(
                support_features=np.asarray(split.support_features)[keep],
                support_labels=np.asarray(support_labels)[keep],
                support_ids=kept_ids,
                query_features=np.asarray(split.query_features[qp]).reshape(1, -1),
                query_ids=(qid,),
                seed=seed,
                n_estimators=n_estimators,
                model_cache_dir=model_cache_dir,
                week01_source_root=week01_source_root,
            )
            aclasses, amass = class_distribution(abl.alpha, np.asarray(support_labels)[keep])
            try:
                effect = removal_effect(
                    full_dist, classes_full, np.asarray(amass[0]), aclasses,
                    removed_alpha_mass=removed_mass, actual_k=sets["actual_k"], epsilon=epsilon,
                )
            except RemovalError as error:
                raise RunnerError("extension recheck scoring failed: " + str(error)) from error
            ext_profiles.append(
                {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "split_id": split_id,
                    "query_id": qid,
                    "query_index": int(qp),
                    "query_true_label": qlabel,
                    "k": int(k),
                    "condition": condition,
                    "actual_k": int(sets["actual_k"]),
                    "removed_support_ids": [split.support_ids[int(i)] for i in removed],
                    "tv": effect["tv"],
                }
            )
            if condition == "top_beta":
                top_tv[qp] = float(effect["tv"])
            else:
                supcon_tv[qp] = float(effect["tv"])
            effects[condition].append(effect)
    try:
        paired = paired_comparison(top_tv, supcon_tv)
        summary = {c: summarize_effects(effects[c]) for c in ("top_beta", "supcon_top")}
    except RemovalError as error:
        raise RunnerError("extension recheck summary failed: " + str(error)) from error
    return {"per_k": {"3": {"paired_top_vs_supcon": paired, "summary": summary}}}


def run_directionality(
    *,
    mode: str,
    dataset_manifest_path: str | Path,
    dataset_cache_dir: str | Path,
    allow_download: bool,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> RunResult:
    if mode not in {"smoke", "full"}:
        raise RunnerError("mode must be smoke or full")
    if mode == "full" and allow_download:
        raise RunnerError("full mode never downloads data; seed the frozen dataset cache first")
    manifest, manifest_hash = load_manifest(dataset_manifest_path)
    if n_estimators < 1:
        raise RunnerError("n_estimators must be positive")
    frozen_teacher = manifest.get("teacher")
    if not isinstance(frozen_teacher, dict) or not isinstance(frozen_teacher.get("n_estimators"), int):
        raise RunnerError("dataset manifest lacks a frozen Teacher estimator count")
    if n_estimators != int(frozen_teacher["n_estimators"]):
        raise RunnerError("n_estimators differs from the frozen manifest value")
    frozen_cache_dir = frozen_teacher.get("model_cache_dir")
    if model_cache_dir != frozen_cache_dir:
        raise RunnerError("model_cache_dir differs from the frozen manifest cache")
    if mode == "full":
        selected_ids = None
    else:
        selected_ids = [str(manifest["smoke"]["dataset_id"])]
    datasets = materialize_manifest(
        manifest,
        cache_dir=dataset_cache_dir,
        allow_download=allow_download if mode == "smoke" else False,
        dataset_ids=selected_ids,
    )
    if mode == "smoke" and len(datasets) != 1:
        raise RunnerError("smoke mode must execute exactly one frozen dataset")
    expected_by_id = {
        str(item["dataset_id"]): str(item.get("expected_content_sha256", ""))
        for item in manifest["datasets"]
        if isinstance(item, dict)
    }
    for dataset in datasets:
        expected = expected_by_id.get(dataset.dataset_id, "")
        observed = str(dataset.metadata.get("content_sha256", ""))
        if not expected or observed != expected:
            raise RunnerError(dataset.dataset_id + ": content hash differs from the frozen lock")
    dataset_lock = (
        verify_dataset_lock(manifest, manifest_hash, datasets, cache_dir=dataset_cache_dir)
        if mode == "full"
        else None
    )
    seeds, split_args = _split_args(manifest, mode=mode)
    max_pool, ks, supcon_epochs = _directionality_plan(manifest, mode=mode)
    gate = manifest["gate"]
    extension_ids = set(str(v) for v in gate["extension_dataset_ids"])
    directionality_plan = {
        "probe_pool_size": max_pool,
        "ks": ks,
        "supcon_epochs": supcon_epochs,
        "probe_seed": int(manifest["directionality"]["probe_seed"]),
        "primary_k": int(manifest["directionality"]["primary_k"]),
    }
    results: list[dict[str, Any]] = []
    all_dir: list[dict[str, Any]] = []
    all_ext: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    for dataset in datasets:
        result, dir_p, ext_p, observed_runtime = _run_dataset(
            dataset,
            seeds=seeds,
            split_args=split_args,
            n_estimators=n_estimators,
            model_cache_dir=model_cache_dir,
            week01_source_root=week01_source_root,
            manifest=manifest,
            max_pool=max_pool,
            ks=ks,
            supcon_epochs=supcon_epochs,
            is_extension=dataset.dataset_id in extension_ids,
            mode=mode,
        )
        results.append(result)
        all_dir.extend(dir_p)
        all_ext.extend(ext_p)
        runtime = observed_runtime
    if runtime is None:
        raise RunnerError("Teacher was not invoked")
    if mode == "full":
        scientific_gate = _global_gate(results, manifest)
    else:
        scientific_gate = {
            "decision": "not_evaluated_in_smoke",
            "recommendation": "SMOKE_ONLY_NO_SCIENTIFIC_CLAIM",
        }
    provenance = {
        "schema_version": 1,
        "dataset_manifest_id": manifest["manifest_id"],
        "dataset_manifest_sha256": manifest_hash,
        "selected_dataset_ids": [d.dataset_id for d in datasets],
        "datasets": [d.metadata for d in datasets],
    }
    evidence = _json_ready(
        {
            "schema_version": 1,
            "completed_at_utc": _utc_now(),
            "mode": mode,
            "teacher_source_sha256": WEEK01_SOURCE_SHA256,
            "dataset_manifest_id": manifest["manifest_id"],
            "dataset_manifest_sha256": manifest_hash,
            "required_dataset_ids": [str(i["dataset_id"]) for i in manifest["datasets"]],
            "executed_dataset_ids": [d.dataset_id for d in datasets],
            "split_specification": dict(manifest["split"] if mode == "full" else manifest["smoke"]),
            "n_estimators": n_estimators,
            "frozen_teacher": dict(frozen_teacher),
            "directionality_plan": directionality_plan,
            "runtime": runtime,
            "dataset_results": results,
            "scientific_gate": scientific_gate,
        }
    )
    return RunResult(
        evidence=dict(evidence),
        directionality_profiles=[dict(p) for p in _json_ready(all_dir)],
        extension_removal_profiles=[dict(p) for p in _json_ready(all_ext)],
        dataset_provenance=dict(_json_ready(provenance)),
        dataset_lock=None if dataset_lock is None else dict(_json_ready(dataset_lock)),
    )


def _report_markdown(evidence: Mapping[str, Any], *, run_id: str) -> str:
    gate = evidence["scientific_gate"]
    plan = evidence.get("directionality_plan", {})
    lines = [
        "# Week 05 - Directionality on Frozen Real Data",
        "",
        "Technical status: PASS",
        "Run ID: " + str(run_id),
        "Completed at UTC: " + str(evidence["completed_at_utc"]),
        "Mode: " + str(evidence["mode"]),
        "Week 1 Teacher source SHA-256: " + str(evidence["teacher_source_sha256"]),
        "Dataset manifest SHA-256: " + str(evidence["dataset_manifest_sha256"]),
        "Directionality plan: pool=" + str(plan.get("probe_pool_size"))
        + ", ks=" + str(plan.get("ks"))
        + ", supcon_epochs=" + str(plan.get("supcon_epochs")),
        "",
        "## Dataset coverage",
        "",
        "Executed: " + ", ".join(evidence["executed_dataset_ids"]),
        "Frozen full benchmark: " + ", ".join(evidence["required_dataset_ids"]),
        "",
        "## Dataset-level directionality summaries",
        "",
    ]
    for result in evidence["dataset_results"]:
        g = result["gate"]
        tag = " (extension)" if result.get("extension_arm") else ""
        try:
            af = "%.6f" % float(g.get("median_A_F", float("nan")))
        except (TypeError, ValueError):
            af = "nan"
        try:
            rc = "%.6f" % float(g.get("median_reciprocity_at_primary_k", float("nan")))
        except (TypeError, ValueError):
            rc = "nan"
        try:
            rv = "%.6f" % float(g.get("median_rank_reversal_rate", float("nan")))
        except (TypeError, ValueError):
            rv = "nan"
        lines.extend(
            [
                "### " + str(result["dataset_id"]) + tag,
                "",
                "Regime: " + str(g.get("regime")),
                "median A_F: " + af,
                "median reciprocity at primary k: " + rc,
                "median rank-reversal: " + rv,
                "extension beyond_supcon: " + str(g.get("beyond_supcon")),
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific gate",
            "",
            "Decision: " + str(gate.get("decision")),
            "Recommendation: " + str(gate.get("recommendation")),
            "Stop reason: " + str(gate.get("stop_reason")),
            "Strong datasets: " + str(gate.get("strong_dataset_ids")),
            "Weak datasets: " + str(gate.get("weak_dataset_ids")),
            "Extension beyond-SupCon: " + str(gate.get("extension_beyond_supcon_ids")),
            "",
            "A technical PASS proves execution/provenance only. Preserve this result.",
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
    root = Path(output_dir).resolve()
    if root.exists():
        raise RunnerError("evidence directory already exists: " + str(root))
    root.mkdir(parents=True, exist_ok=False)
    metrics_payload = dict(result.evidence)
    metrics_payload["run_id"] = run_id
    report = _report_markdown(metrics_payload, run_id=run_id)
    payloads: dict[str, object] = {
        "run_input.json": dict(run_input),
        "metrics.json": metrics_payload,
        "dataset_provenance.json": result.dataset_provenance,
        "runtime_environment.json": metrics_payload["runtime"],
        "week05_directionality.md": report,
    }
    if result.dataset_lock is not None:
        payloads["dataset_lock.json"] = result.dataset_lock
    for name, payload in payloads.items():
        destination = root / name
        if isinstance(payload, str):
            destination.write_text(payload, encoding="utf-8")
        else:
            destination.write_text(
                json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    dir_path = root / "directionality_profiles.jsonl"
    with dir_path.open("x", encoding="utf-8", newline="\n") as handle:
        for profile in result.directionality_profiles:
            handle.write(json.dumps(_json_ready(profile), ensure_ascii=False, sort_keys=True) + "\n")
    ext_path = root / "extension_removal_profiles.jsonl"
    with ext_path.open("x", encoding="utf-8", newline="\n") as handle:
        for profile in result.extension_removal_profiles:
            handle.write(json.dumps(_json_ready(profile), ensure_ascii=False, sort_keys=True) + "\n")
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "pass",
        "technical_gate": "completed",
        "scientific_gate": metrics_payload["scientific_gate"]["decision"],
        "recommendation": metrics_payload["scientific_gate"]["recommendation"],
        "evidence_files": sorted(
            [p.name for p in root.iterdir() if p.is_file()] + ["run_status.json"]
        ),
    }
    (root / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root


__all__ = [
    "DatasetError",
    "RunnerError",
    "TeacherBridgeError",
    "AnatomyError",
    "RemovalError",
    "DirectionalityError",
    "SupconError",
    "run_directionality",
    "write_evidence",
]
