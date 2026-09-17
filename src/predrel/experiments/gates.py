"""Frozen gate recomputation for Week 7.

Each function mirrors the corresponding week's own evidence verifier exactly:
same median definition, same seed field paths, same threshold comparisons.
Week 7 introduces no new thresholds and no reinterpretation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("cannot take the median of an empty or non-finite set")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _finite(value: object, *, name: str) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(name + " is not numeric") from error
    if not math.isfinite(numeric):
        raise ValueError(name + " is non-finite")
    return numeric


def week3_dataset_uniformity(
    seed_records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Mirror verify_week03_evidence._validate_dataset_gate (seed-mean median)."""
    paths = {
        "median_true_class_normalized_beta_entropy": "true_class_normalized_beta_entropy",
        "median_js_alpha_vs_class_only_null": "js_alpha_vs_class_only_null",
        "median_true_class_top_beta_excess_over_uniform": "true_class_top_beta_excess_over_uniform",
    }
    recomputed: dict[str, float] = {}
    for gate_name, summary_name in paths.items():
        values = []
        for seed in seed_records:
            summary = seed.get("summary")
            if not isinstance(summary, Mapping):
                raise ValueError("seed lacks summary for gate reconstruction")
            metric = summary.get(summary_name)
            if not isinstance(metric, Mapping):
                raise ValueError("seed lacks " + summary_name)
            values.append(_finite(metric.get("mean"), name="seed " + summary_name + " mean"))
        recomputed[gate_name] = median(values)
    criteria = {
        "entropy_at_or_above_floor": recomputed["median_true_class_normalized_beta_entropy"]
        >= float(thresholds["uniform_entropy_floor"]),
        "js_at_or_below_ceiling": recomputed["median_js_alpha_vs_class_only_null"]
        <= float(thresholds["low_js_threshold_nats"]),
        "top_beta_excess_at_or_below_ceiling": recomputed[
            "median_true_class_top_beta_excess_over_uniform"
        ]
        <= float(thresholds["top_beta_excess_threshold"]),
    }
    return {"metrics": recomputed, "approximately_uniform": bool(all(criteria.values()))}


def week4_dataset_gate(
    seed_records: Sequence[Mapping[str, Any]],
    *,
    primary_k: str,
    tv_diff_margin: float,
    win_rate_floor: float,
) -> dict[str, Any]:
    """Mirror verify_week04_evidence._validate_dataset_gate (paired-diff medians)."""
    reconstructions: dict[str, list[float]] = {
        "median_tv_diff_top_minus_random": [],
        "median_win_rate_top_vs_random": [],
        "median_tv_diff_top_minus_supcon": [],
    }
    for seed in seed_records:
        summary = seed.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("seed lacks summary for gate reconstruction")
        per_k = summary.get("per_k")
        if not isinstance(per_k, Mapping) or primary_k not in per_k:
            raise ValueError("seed lacks primary-K removal summary")
        block = per_k[primary_k]
        if not isinstance(block, Mapping):
            raise ValueError("seed lacks primary-K removal block")
        for gate_name, group, field in (
            ("median_tv_diff_top_minus_random", "paired_top_vs_random", "mean_diff"),
            ("median_win_rate_top_vs_random", "paired_top_vs_random", "win_rate"),
            ("median_tv_diff_top_minus_supcon", "paired_top_vs_supcon", "mean_diff"),
        ):
            paired = block.get(group)
            if not isinstance(paired, Mapping):
                raise ValueError("seed lacks " + group)
            reconstructions[gate_name].append(_finite(paired.get(field), name=field))
    diff_random = median(reconstructions["median_tv_diff_top_minus_random"])
    win_random = median(reconstructions["median_win_rate_top_vs_random"])
    diff_supcon = median(reconstructions["median_tv_diff_top_minus_supcon"])
    return {
        "median_tv_diff_top_minus_random": diff_random,
        "median_win_rate_top_vs_random": win_random,
        "median_tv_diff_top_minus_supcon": diff_supcon,
        "faithful": bool(diff_random > tv_diff_margin and win_random > win_rate_floor),
        "beyond_supcon": bool(diff_supcon > tv_diff_margin),
    }


def week5_dataset_regime(
    seed_records: Sequence[Mapping[str, Any]],
    *,
    primary_k: str,
    strong_af: float,
    weak_af: float,
    strong_reciprocity: float,
    weak_reciprocity: float,
    strong_reversal: float,
    weak_reversal: float,
) -> dict[str, Any]:
    """Mirror verify_week05_evidence._recompute_dataset_gate (median of per-seed means)."""
    af = median([float(record["summary"]["asymmetry"]["A_F"]) for record in seed_records])
    rc = median([float(record["summary"]["reciprocity"][primary_k]["mean"]) for record in seed_records])
    rv = median(
        [float(record["summary"]["rank_agreement"]["mean_rank_reversal_rate"]) for record in seed_records]
    )
    strong = bool(af >= strong_af or (rc <= strong_reciprocity and rv >= strong_reversal))
    weak = bool(af <= weak_af and rc >= weak_reciprocity and rv <= weak_reversal)
    regime = "strong" if strong else ("weak" if weak else "moderate")
    return {"regime": regime, "strong": strong, "weak": weak, "af": af, "rc": rc, "rv": rv}


def week6_primary_median(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mirror verify_week06_evidence._median_metrics over seed primary metrics."""
    if not records:
        raise ValueError("no metric records")
    fields = tuple(records[0].keys())
    if not fields or any(tuple(record.keys()) != fields for record in records):
        raise ValueError("inconsistent metric keys")
    return {field: median([_finite(record[field], name=field) for record in records]) for field in fields}


def classify_context_regime(
    primary: Mapping[str, float], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Mirror ``predrel.context_sensitivity.classify_context_regime``."""
    weak_spec = gate["weak"]
    strong_spec = gate["strong"]
    tau = _finite(primary.get("mean_pair_kendall_tau"), name="kendall")
    spearman = _finite(primary.get("mean_pair_spearman"), name="spearman")
    jaccard = _finite(primary.get("mean_pair_top_k_jaccard"), name="jaccard")
    flip = _finite(primary.get("mean_preference_flip_rate_vs_first"), name="flip")
    weak_conditions = {
        "raw_kendall": tau >= float(weak_spec["min_raw_kendall_tau"]),
        "raw_spearman": spearman >= float(weak_spec["min_raw_spearman"]),
        "raw_top_k_jaccard": jaccard >= float(weak_spec["min_raw_top_k_jaccard"]),
        "raw_preference_flip": flip <= float(weak_spec["max_raw_preference_flip_rate"]),
    }
    strong_conditions = {
        "raw_kendall": tau <= float(strong_spec["max_raw_kendall_tau"]),
        "raw_spearman": spearman <= float(strong_spec["max_raw_spearman"]),
        "raw_top_k_jaccard": jaccard <= float(strong_spec["max_raw_top_k_jaccard"]),
        "raw_preference_flip": flip >= float(strong_spec["min_raw_preference_flip_rate"]),
    }
    strong_trigger_count = int(sum(1 for value in strong_conditions.values() if value))
    weak = bool(all(weak_conditions.values()))
    strong = bool(strong_trigger_count >= int(strong_spec["minimum_trigger_count"]))
    if weak and strong:
        raise ValueError("frozen weak and strong context rules overlap")
    regime = "weak" if weak else ("strong" if strong else "moderate")
    return {
        "regime": regime,
        "weak": weak,
        "strong": strong,
        "raw_metric_finite": True,
        "primary_raw_metrics": dict(primary),
        "weak_conditions": weak_conditions,
        "strong_conditions": strong_conditions,
        "strong_trigger_count": strong_trigger_count,
    }
