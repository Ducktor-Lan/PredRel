"""Experiment orchestration and evidence writing for Week 3."""

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
from ..label_anatomy import AnatomyError, analyze_beyond_labels
from ..teacher_bridge import WEEK01_SOURCE_SHA256, TeacherBridgeError, extract_readout


class RunnerError(RuntimeError):
    """Raised when a Week 3 execution plan cannot make valid evidence."""


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


def _dataset_gate(seed_summaries: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Apply frozen uniformity criteria to one dataset, after seed aggregation."""

    median_entropy = _metric_median(
        seed_summaries, ("summary", "true_class_normalized_beta_entropy", "mean")
    )
    median_js = _metric_median(seed_summaries, ("summary", "js_alpha_vs_class_only_null", "mean"))
    median_excess = _metric_median(
        seed_summaries, ("summary", "true_class_top_beta_excess_over_uniform", "mean")
    )
    entropy_floor = float(thresholds["uniform_entropy_floor"])
    js_ceiling = float(thresholds["low_js_threshold_nats"])
    excess_ceiling = float(thresholds["top_beta_excess_threshold"])
    checks = {
        "entropy_at_or_above_floor": median_entropy >= entropy_floor,
        "js_at_or_below_ceiling": median_js <= js_ceiling,
        "top_beta_excess_at_or_below_ceiling": median_excess <= excess_ceiling,
    }
    return {
        "seed_aggregation": "median of per-seed mean metrics",
        "median_true_class_normalized_beta_entropy": median_entropy,
        "median_js_alpha_vs_class_only_null": median_js,
        "median_true_class_top_beta_excess_over_uniform": median_excess,
        "thresholds": {
            "uniform_entropy_floor": entropy_floor,
            "low_js_threshold_nats": js_ceiling,
            "top_beta_excess_threshold": excess_ceiling,
        },
        "criteria": checks,
        "approximately_uniform": bool(all(checks.values())),
    }


def _global_gate(dataset_results: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    uniform_ids = [result["dataset_id"] for result in dataset_results if result["gate"]["approximately_uniform"]]
    count = len(dataset_results)
    if count != len(manifest["datasets"]):
        raise RunnerError("the scientific gate requires every frozen dataset")
    stop = len(uniform_ids) > count / 2.0
    gate = manifest["gate"]
    return {
        "frozen_dataset_count": count,
        "approximately_uniform_count": len(uniform_ids),
        "approximately_uniform_dataset_ids": uniform_ids,
        "majority_definition": gate["stop_when"],
        "recommendation": gate["stop_recommendation"] if stop else gate["continue_recommendation"],
        "decision": "stop_readout2rep_route" if stop else "continue_to_week04",
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


def _run_dataset(
    dataset: RealDataset,
    *,
    seeds: list[int],
    split_args: Mapping[str, Any],
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
    metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
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
        summary, seed_profiles = analyze_beyond_labels(
            extraction.alpha,
            support_labels=split.support_labels,
            query_labels=split.query_labels,
            support_ids=split.support_ids,
            query_ids=split.query_ids,
            top_k=int(metrics["top_k"]),
            epsilon=float(metrics["epsilon"]),
        )
        for profile in seed_profiles:
            profile.update({"dataset_id": dataset.dataset_id, "seed": seed, "split_id": split.split_id})
        profiles.extend(seed_profiles)
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
                "summary": summary,
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
            "gate": _dataset_gate(seed_records, metrics),
            "teacher": {
                "teacher_source_sha256": WEEK01_SOURCE_SHA256,
                "decoder_readout_api": decoder_readout_api,
                "raw_score_hook_path": raw_score_hook_path,
            },
        },
        profiles,
        runtime,
    )


def run_beyond_labels(
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
        raise RunnerError("full mode never downloads data; explicitly bootstrap and lock the frozen dataset cache first")
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
    dataset_lock = (
        verify_dataset_lock(manifest, manifest_hash, datasets, cache_dir=dataset_cache_dir)
        if mode == "full"
        else None
    )
    seeds, split_args = _split_args(manifest, mode=mode)
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
            metrics=manifest["metrics"],
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
    lines = [
        "# Week 03 — Beyond Labels on Frozen Real Data",
        "",
        "Technical status: PASS",
        f"Run ID: {run_id}",
        f"Completed at UTC: {evidence['completed_at_utc']}",
        f"Mode: {evidence['mode']}",
        f"Week 1 Teacher source SHA-256: {evidence['teacher_source_sha256']}",
        f"Dataset manifest SHA-256: {evidence['dataset_manifest_sha256']}",
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
                f"Approximately uniform: {dataset_gate['approximately_uniform']}",
                f"Median JS(alpha, class-only null): {dataset_gate['median_js_alpha_vs_class_only_null']:.8f}",
                f"Median true-class beta entropy: {dataset_gate['median_true_class_normalized_beta_entropy']:.8f}",
                f"Median Top-beta excess over uniform: {dataset_gate['median_true_class_top_beta_excess_over_uniform']:.8f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific gate",
            "",
            f"Decision: {gate['decision']}",
            f"Recommendation: {gate['recommendation']}",
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
        "week03_beyond_labels.md": _report_markdown(metrics_payload, run_id=run_id),
    }
    if result.dataset_lock is not None:
        payloads["dataset_lock.json"] = result.dataset_lock
    for name, payload in payloads.items():
        destination = root / name
        if isinstance(payload, str):
            destination.write_text(payload, encoding="utf-8")
        else:
            destination.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    profiles_path = root / "class_residual_profiles.jsonl"
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
    "run_beyond_labels",
    "write_evidence",
]
