"""Experiment orchestration and evidence writing for Week 4."""

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
from ..label_anatomy import AnatomyError, decompose_alpha
from ..removal import (
    RemovalError,
    build_removal_sets,
    class_distribution,
    paired_comparison,
    removal_effect,
    select_removal_queries,
    summarize_effects,
    teacher_topk_recall,
)
from ..supcon import (
    SupconError,
    cosine_similarities,
    embed_with_params,
    rank_supports_by_similarity,
    train_supcon_encoder,
)
from ..teacher_bridge import WEEK01_SOURCE_SHA256, TeacherBridgeError, extract_readout


class RunnerError(RuntimeError):
    """Raised when a Week 4 execution plan cannot make valid evidence."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """All serializable result material from a completed technical execution."""

    evidence: dict[str, Any]
    profiles: list[dict[str, Any]]
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
                raise RunnerError(f"missing metric path {'.'.join(path)}")
            current = current[key]
        try:
            values.append(float(current))
        except (TypeError, ValueError) as error:
            raise RunnerError(f"non-numeric metric path {'.'.join(path)}") from error
    if not values or not np.isfinite(values).all():
        raise RunnerError(f"metric path {'.'.join(path)} is empty or non-finite")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _dataset_gate(
    seed_summaries: list[dict[str, Any]],
    gate: Mapping[str, Any],
    *,
    primary_k: int,
) -> dict[str, Any]:
    """Apply the frozen faithfulness criteria to one dataset, after seed aggregation."""

    key = str(primary_k)
    median_diff_random = _metric_median(
        seed_summaries, ("summary", "per_k", key, "paired_top_vs_random", "mean_diff")
    )
    median_win_random = _metric_median(
        seed_summaries, ("summary", "per_k", key, "paired_top_vs_random", "win_rate")
    )
    median_diff_supcon = _metric_median(
        seed_summaries, ("summary", "per_k", key, "paired_top_vs_supcon", "mean_diff")
    )
    margin = float(gate["tv_diff_margin"])
    win_floor = float(gate["win_rate_floor"])
    faithful = bool(median_diff_random > margin and median_win_random > win_floor)
    beyond = bool(median_diff_supcon > margin)
    return {
        "seed_aggregation": gate.get("seed_aggregation", "median of per-seed means"),
        "primary_k": primary_k,
        "median_tv_diff_top_minus_random": median_diff_random,
        "median_win_rate_top_vs_random": median_win_random,
        "median_tv_diff_top_minus_supcon": median_diff_supcon,
        "thresholds": {"tv_diff_margin": margin, "win_rate_floor": win_floor},
        "criteria": {"faithful": faithful, "beyond_supcon": beyond},
        "faithful": faithful,
        "beyond_supcon": beyond,
    }


def _global_gate(dataset_results: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    faithful_ids = [result["dataset_id"] for result in dataset_results if result["gate"]["faithful"]]
    beyond_ids = [result["dataset_id"] for result in dataset_results if result["gate"]["beyond_supcon"]]
    count = len(dataset_results)
    if count != len(manifest["datasets"]):
        raise RunnerError("the scientific gate requires every frozen dataset")
    faithful_majority = len(faithful_ids) > count / 2.0
    beyond_majority = len(beyond_ids) > count / 2.0
    gate = manifest["gate"]
    if faithful_majority and beyond_majority:
        return {
            "frozen_dataset_count": count,
            "faithful_count": len(faithful_ids),
            "faithful_dataset_ids": faithful_ids,
            "beyond_supcon_count": len(beyond_ids),
            "beyond_supcon_dataset_ids": beyond_ids,
            "majority_definition": gate["continue_when"],
            "recommendation": gate["continue_recommendation"],
            "decision": "continue_to_week05",
            "stop_reason": None,
        }
    if not faithful_majority and not beyond_majority:
        reason = "both"
    elif not faithful_majority:
        reason = "removal_not_faithful"
    else:
        reason = "readout_captured_by_supcon"
    if reason not in gate["stop_reason_values"]:
        raise RunnerError(f"stop reason {reason!r} is outside the frozen gate contract")
    return {
        "frozen_dataset_count": count,
        "faithful_count": len(faithful_ids),
        "faithful_dataset_ids": faithful_ids,
        "beyond_supcon_count": len(beyond_ids),
        "beyond_supcon_dataset_ids": beyond_ids,
        "majority_definition": gate["stop_when"],
        "recommendation": gate["stop_recommendation"],
        "decision": "stop_readout2rep_route",
        "stop_reason": reason,
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


def _removal_plan(manifest: Mapping[str, Any], *, mode: str) -> tuple[list[int], int, int, int]:
    removal = manifest["removal"]
    supcon = manifest["supcon"]
    if mode == "full":
        return (
            [int(value) for value in removal["ks"]],
            int(removal["random_repeats"]),
            int(removal["max_removal_queries"]),
            int(supcon["epochs"]),
        )
    return (
        [int(value) for value in removal["smoke_ks"]],
        int(removal["smoke_random_repeats"]),
        int(removal["max_removal_queries"]),
        int(supcon["smoke_epochs"]),
    )
def _ablated_extraction(
    *,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    support_ids: tuple[str, ...],
    query_feature: np.ndarray,
    query_id: str,
    seed: int,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
    remove_positions: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the Teacher with matched removal positions excluded."""

    keep = np.delete(np.arange(len(support_ids)), np.asarray(remove_positions, dtype=np.int64))
    if keep.size == 0:
        raise RunnerError("ablated support is empty")
    kept_ids = tuple(support_ids[int(position)] for position in keep.tolist())
    extraction = extract_readout(
        support_features=np.asarray(support_features)[keep],
        support_labels=np.asarray(support_labels)[keep],
        support_ids=kept_ids,
        query_features=np.asarray(query_feature).reshape(1, -1),
        query_ids=(query_id,),
        seed=seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    if extraction.support_ids != kept_ids or extraction.query_ids != (query_id,):
        raise RunnerError("ablated Teacher output IDs are not aligned to the requested split")
    classes, mass = class_distribution(extraction.alpha, np.asarray(support_labels)[keep])
    return classes, np.asarray(mass[0], dtype=np.float64)


def _run_dataset(
    dataset: RealDataset,
    *,
    seeds: list[int],
    split_args: Mapping[str, Any],
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
    manifest: Mapping[str, Any],
    ks: list[int],
    random_repeats: int,
    max_removal_queries: int,
    supcon_epochs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    removal = manifest["removal"]
    supcon_spec = manifest["supcon"]
    metrics = manifest["metrics"]
    gate = manifest["gate"]
    primary_k = int(metrics["primary_k"])
    epsilon = float(metrics["epsilon"])
    recall_at = int(supcon_spec["recall_at"])
    seed_base = int(removal["random_seed_base"])
    seed_records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    decoder_readout_api: str | None = None
    raw_score_hook_path: str | None = None
    for seed in seeds:
        split = make_stratified_split(dataset, seed=seed, **dict(split_args))
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
        if extraction.support_ids != split.support_ids or extraction.query_ids != split.query_ids:
            raise RunnerError(f"{split.split_id}: Teacher output IDs are not aligned to the requested split")
        try:
            decomposition = decompose_alpha(extraction.alpha, split.support_labels)
        except AnatomyError as error:
            raise RunnerError(f"{split.split_id}: cannot decompose full-support readout: {error}") from error
        classes_full = np.asarray(decomposition.classes)
        mass_full = np.asarray(decomposition.class_mass, dtype=np.float64)
        beta = np.asarray(decomposition.beta, dtype=np.float64)
        removal_positions = select_removal_queries(split.query_labels, max_queries=max_removal_queries)
        try:
            supcon_params = train_supcon_encoder(
                split.support_features,
                split.support_labels,
                hidden_dims=[int(value) for value in supcon_spec["hidden_dims"]],
                embedding_dim=int(supcon_spec["embedding_dim"]),
                temperature=float(supcon_spec["temperature"]),
                learning_rate=float(supcon_spec["learning_rate"]),
                epochs=supcon_epochs,
                batch_size=int(supcon_spec["batch_size"]),
                seed=int(supcon_spec["seed"]),
            )
        except SupconError as error:
            raise RunnerError(f"{split.split_id}: SupCon baseline failed: {error}") from error
        support_emb = embed_with_params(supcon_params, split.support_features)
        query_emb = embed_with_params(supcon_params, split.query_features)
        similarities = cosine_similarities(query_emb, support_emb)
        rankings = rank_supports_by_similarity(similarities)
        rng = np.random.default_rng([seed_base, int(seed)])
        per_k: dict[str, Any] = {}
        support_ids = split.support_ids
        support_labels = np.asarray(split.support_labels)
        for k in ks:
            top_tv: dict[int, float] = {}
            random_tv: dict[int, float] = {}
            supcon_tv: dict[int, float] = {}
            condition_effects: dict[str, list[dict[str, float | int]]] = {
                "top_beta": [],
                "random_same_class": [],
                "bottom_beta": [],
                "supcon_top": [],
            }
            recall_values: list[float] = []
            for query_position in removal_positions:
                query_id = split.query_ids[query_position]
                query_label = int(split.query_labels[query_position])
                try:
                    sets = build_removal_sets(
                        beta[query_position],
                        support_labels,
                        query_label,
                        k=k,
                        rng=rng,
                        random_repeats=random_repeats,
                        supcon_scores=similarities[query_position],
                    )
                except RemovalError as error:
                    raise RunnerError(f"{split.split_id} {query_id}: {error}") from error
                full_distribution = mass_full[query_position]
                tasks: list[tuple[str, int, list[int]]] = [
                    ("top_beta", 0, sets["top_beta"]),
                    ("bottom_beta", 0, sets["bottom_beta"]),
                    ("supcon_top", 0, list(sets["supcon_top"] or [])),
                ]
                tasks.extend(
                    ("random_same_class", repeat, draw)
                    for repeat, draw in enumerate(sets["random_same_class"])
                )
                random_query_effects: list[dict[str, float | int]] = []
                for condition, repeat, positions in tasks:
                    removed_mass = float(np.asarray(extraction.alpha[query_position])[positions].sum())
                    try:
                        ablated_classes, ablated_distribution = _ablated_extraction(
                            support_features=np.asarray(split.support_features),
                            support_labels=support_labels,
                            support_ids=support_ids,
                            query_feature=np.asarray(split.query_features[query_position]),
                            query_id=query_id,
                            seed=seed,
                            n_estimators=n_estimators,
                            model_cache_dir=model_cache_dir,
                            week01_source_root=week01_source_root,
                            remove_positions=positions,
                        )
                        effect = removal_effect(
                            full_distribution,
                            classes_full,
                            ablated_distribution,
                            ablated_classes,
                            removed_alpha_mass=removed_mass,
                            actual_k=sets["actual_k"],
                            epsilon=epsilon,
                        )
                    except (RemovalError, TeacherBridgeError) as error:
                        raise RunnerError(f"{split.split_id} {query_id} {condition}: {error}") from error
                    removed_ids = [support_ids[int(position)] for position in positions]
                    profiles.append(
                        {
                            "dataset_id": dataset.dataset_id,
                            "seed": seed,
                            "split_id": split.split_id,
                            "query_id": query_id,
                            "query_index": int(query_position),
                            "query_true_label": query_label,
                            "k": int(k),
                            "condition": condition,
                            "repeat": int(repeat),
                            "actual_k": int(sets["actual_k"]),
                            "removed_support_ids": removed_ids,
                            "tv": effect["tv"],
                            "l1": effect["l1"],
                            "flip": effect["flip"],
                            "delta_logprob": effect["delta_logprob"],
                            "removed_alpha_mass": effect["removed_alpha_mass"],
                        }
                    )
                    if condition == "random_same_class":
                        random_query_effects.append(effect)
                    elif condition == "top_beta":
                        top_tv[query_position] = float(effect["tv"])
                        condition_effects[condition].append(effect)
                    elif condition == "bottom_beta":
                        condition_effects[condition].append(effect)
                    else:
                        supcon_tv[query_position] = float(effect["tv"])
                        condition_effects[condition].append(effect)
                averaged_random = {
                    "tv": float(np.mean([float(item["tv"]) for item in random_query_effects])),
                    "l1": float(np.mean([float(item["l1"]) for item in random_query_effects])),
                    "flip": float(np.mean([float(item["flip"]) for item in random_query_effects])),
                    "delta_logprob": float(
                        np.mean([float(item["delta_logprob"]) for item in random_query_effects])
                    ),
                    "removed_alpha_mass": float(
                        np.mean([float(item["removed_alpha_mass"]) for item in random_query_effects])
                    ),
                    "actual_k": int(sets["actual_k"]),
                }
                random_tv[query_position] = averaged_random["tv"]
                condition_effects["random_same_class"].append(averaged_random)
                recall_values.append(
                    teacher_topk_recall(sets["top_beta"], rankings[query_position], at=recall_at)
                )
            try:
                summary = {
                    condition: summarize_effects(condition_effects[condition])
                    for condition in ("top_beta", "random_same_class", "bottom_beta", "supcon_top")
                }
                summary["paired_top_vs_random"] = paired_comparison(top_tv, random_tv)
                summary["paired_top_vs_supcon"] = paired_comparison(top_tv, supcon_tv)
                summary[f"mean_teacher_topk_in_supcon_top{recall_at}"] = float(np.mean(recall_values))
            except RemovalError as error:
                raise RunnerError(f"{split.split_id} k={k}: {error}") from error
            per_k[str(k)] = summary
        for profile in profiles:
            if profile["dataset_id"] == dataset.dataset_id and profile["seed"] == seed:
                profile["removal_query_count"] = len(removal_positions)
        seed_records.append(
            {
                "seed": seed,
                "split_id": split.split_id,
                "holdout_count": len(split.holdout_ids),
                "support_count": int(split.support_features.shape[0]),
                "query_count": int(split.query_features.shape[0]),
                "holdout_ids": list(split.holdout_ids),
                "support_ids": list(split.support_ids),
                "query_ids": list(split.query_ids),
                "support_class_counts": {
                    str(label): int((split.support_labels == label).sum())
                    for label in np.unique(split.support_labels).tolist()
                },
                "query_class_counts": {
                    str(label): int((split.query_labels == label).sum())
                    for label in np.unique(split.query_labels).tolist()
                },
                "query_labels_by_id": {
                    sample_id: int(label)
                    for sample_id, label in zip(split.query_ids, split.query_labels.tolist(), strict=True)
                },
                "removal_query_ids": [split.query_ids[int(position)] for position in removal_positions],
                "summary": {
                    "removal_query_count": len(removal_positions),
                    "per_k": per_k,
                },
            }
        )
        runtime = extraction.runtime
        decoder_readout_api = extraction.decoder_readout_api
        raw_score_hook_path = extraction.raw_score_hook_path
    if runtime is None or decoder_readout_api is None or raw_score_hook_path is None:
        raise RunnerError(f"{dataset.dataset_id}: Teacher was not invoked")
    return (
        {
            "dataset_id": dataset.dataset_id,
            "dataset": dataset.metadata,
            "seeds": seed_records,
            "gate": _dataset_gate(seed_records, gate, primary_k=primary_k),
            "teacher": {
                "teacher_source_sha256": WEEK01_SOURCE_SHA256,
                "decoder_readout_api": decoder_readout_api,
                "raw_score_hook_path": raw_score_hook_path,
            },
        },
        profiles,
        runtime,
    )


def run_removal_faithfulness(
    *,
    mode: str,
    dataset_manifest_path: str | Path,
    dataset_cache_dir: str | Path,
    allow_download: bool,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> RunResult:
    """Run a smoke path or all frozen data, returning auditable evidence."""

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
        raise RunnerError(
            f"n_estimators={n_estimators} differs from frozen manifest value {frozen_teacher['n_estimators']}"
        )
    frozen_cache_dir = frozen_teacher.get("model_cache_dir")
    if not isinstance(frozen_cache_dir, str) or not frozen_cache_dir:
        raise RunnerError("dataset manifest lacks a frozen Teacher model cache directory")
    if model_cache_dir != frozen_cache_dir:
        raise RunnerError(
            f"model_cache_dir={model_cache_dir!r} differs from frozen manifest cache {frozen_cache_dir!r}"
        )
    selected_ids = None if mode == "full" else [str(manifest["smoke"]["dataset_id"])]
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
            raise RunnerError(
                f"{dataset.dataset_id}: content hash differs from the frozen Week 3 lock; refuse experiment"
            )
    dataset_lock = (
        verify_dataset_lock(manifest, manifest_hash, datasets, cache_dir=dataset_cache_dir)
        if mode == "full"
        else None
    )
    seeds, split_args = _split_args(manifest, mode=mode)
    ks, random_repeats, max_removal_queries, supcon_epochs = _removal_plan(manifest, mode=mode)
    removal_plan = {
        "ks": ks,
        "random_repeats": random_repeats,
        "max_removal_queries": max_removal_queries,
        "supcon_epochs": supcon_epochs,
        "primary_k": int(manifest["metrics"]["primary_k"]),
    }
    results: list[dict[str, Any]] = []
    all_profiles: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    for dataset in datasets:
        result, profiles, observed_runtime = _run_dataset(
            dataset,
            seeds=seeds,
            split_args=split_args,
            n_estimators=n_estimators,
            model_cache_dir=model_cache_dir,
            week01_source_root=week01_source_root,
            manifest=manifest,
            ks=ks,
            random_repeats=random_repeats,
            max_removal_queries=max_removal_queries,
            supcon_epochs=supcon_epochs,
        )
        results.append(result)
        all_profiles.extend(profiles)
        runtime = observed_runtime
    if runtime is None:
        raise RunnerError("Teacher was not invoked")
    scientific_gate = _global_gate(results, manifest) if mode == "full" else {
        "decision": "not_evaluated_in_smoke",
        "recommendation": "SMOKE_ONLY_NO_SCIENTIFIC_CLAIM",
    }
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
            "dataset_manifest_id": manifest["manifest_id"],
            "dataset_manifest_sha256": manifest_hash,
            "required_dataset_ids": [str(item["dataset_id"]) for item in manifest["datasets"]],
            "executed_dataset_ids": [dataset.dataset_id for dataset in datasets],
            "split_specification": dict(manifest["split"] if mode == "full" else manifest["smoke"]),
            "n_estimators": n_estimators,
            "frozen_teacher": dict(frozen_teacher),
            "removal_plan": removal_plan,
            "runtime": runtime,
            "dataset_results": results,
            "scientific_gate": scientific_gate,
        }
    )
    return RunResult(
        evidence=dict(evidence),
        profiles=[dict(profile) for profile in _json_ready(all_profiles)],
        dataset_provenance=dict(_json_ready(provenance)),
        dataset_lock=None if dataset_lock is None else dict(_json_ready(dataset_lock)),
    )
def _report_markdown(evidence: Mapping[str, Any], *, run_id: str) -> str:
    gate = evidence["scientific_gate"]
    plan = evidence.get("removal_plan", {})
    lines = [
        "# Week 04 — Same-Class Removal Faithfulness on Frozen Real Data",
        "",
        "Technical status: PASS",
        f"Run ID: {run_id}",
        f"Completed at UTC: {evidence['completed_at_utc']}",
        f"Mode: {evidence['mode']}",
        f"Week 1 Teacher source SHA-256: {evidence['teacher_source_sha256']}",
        f"Dataset manifest SHA-256: {evidence['dataset_manifest_sha256']}",
        f"Removal plan: K={plan.get('ks')}, random_repeats={plan.get('random_repeats')}, "
        f"max_removal_queries={plan.get('max_removal_queries')}, supcon_epochs={plan.get('supcon_epochs')}",
        "",
        "## Dataset coverage",
        "",
        f"Executed: {', '.join(evidence['executed_dataset_ids'])}",
        f"Frozen full benchmark: {', '.join(evidence['required_dataset_ids'])}",
        "",
        "## Dataset-level gate summaries",
        "",
    ]
    for result in evidence["dataset_results"]:
        dataset_gate = result["gate"]
        lines.extend(
            [
                f"### {result['dataset_id']}",
                "",
                f"Faithful (primary K={dataset_gate['primary_k']}): {dataset_gate['faithful']}",
                f"Beyond SupCon: {dataset_gate['beyond_supcon']}",
                f"Median TV(top minus random): {dataset_gate['median_tv_diff_top_minus_random']:.8f}",
                f"Median win rate (top vs random): {dataset_gate['median_win_rate_top_vs_random']:.8f}",
                f"Median TV(top minus supcon): {dataset_gate['median_tv_diff_top_minus_supcon']:.8f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific gate",
            "",
            f"Decision: {gate['decision']}",
            f"Recommendation: {gate['recommendation']}",
            f"Stop reason: {gate.get('stop_reason')}",
            f"Faithful datasets: {gate.get('faithful_dataset_ids')}",
            f"Beyond-SupCon datasets: {gate.get('beyond_supcon_dataset_ids')}",
            "",
            "A technical PASS proves execution/provenance only. Preserve this result and interpret the frozen gate without dropping adverse datasets.",
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
    """Create a fresh evidence directory; never overwrite a prior run."""

    root = Path(output_dir).resolve()
    if root.exists():
        raise RunnerError(f"evidence directory already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    metrics_payload = dict(result.evidence)
    metrics_payload["run_id"] = run_id
    payloads: dict[str, object] = {
        "run_input.json": dict(run_input),
        "metrics.json": metrics_payload,
        "dataset_provenance.json": result.dataset_provenance,
        "runtime_environment.json": metrics_payload["runtime"],
        "week04_beyond_supcon.md": _report_markdown(metrics_payload, run_id=run_id),
    }
    if result.dataset_lock is not None:
        payloads["dataset_lock.json"] = result.dataset_lock
    for name, payload in payloads.items():
        destination = root / name
        if isinstance(payload, str):
            destination.write_text(payload, encoding="utf-8")
        else:
            destination.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    profiles_path = root / "removal_profiles.jsonl"
    with profiles_path.open("x", encoding="utf-8", newline="\n") as handle:
        for profile in result.profiles:
            handle.write(json.dumps(_json_ready(profile), ensure_ascii=False, sort_keys=True) + "\n")
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "pass",
        "technical_gate": "completed",
        "scientific_gate": metrics_payload["scientific_gate"]["decision"],
        "recommendation": metrics_payload["scientific_gate"]["recommendation"],
        "evidence_files": sorted([path.name for path in root.iterdir() if path.is_file()] + ["run_status.json"]),
    }
    (root / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


__all__ = [
    "DatasetError",
    "RunnerError",
    "TeacherBridgeError",
    "AnatomyError",
    "RemovalError",
    "SupconError",
    "run_removal_faithfulness",
    "write_evidence",
]