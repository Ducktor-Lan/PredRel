"""Run the frozen real-data context-sensitivity gate.

This runner deliberately has no query-label argument at any Teacher call. It
never treats alpha movement as evidence of a raw context shift. The resulting
evidence contains both, along with complete fixed-anchor raw-score surfaces,
composition controls, and a recomputable model-route gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..context_sensitivity import (
    AnchorObservation,
    ContextDefinition,
    ContextSensitivityError,
    classify_context_regime,
    derive_seed,
    kendall_tau,
    observe_anchors,
    preference_flip_rate,
    sample_contexts,
    select_anchor_positions,
    select_query_positions,
    spearman_correlation,
    summarize_contexts,
    top_k_jaccard,
)
from ..data import (
    DatasetError,
    ExperimentSplit,
    RealDataset,
    load_manifest,
    make_stratified_split,
    materialize_manifest,
    verify_dataset_lock,
)
from ..teacher_bridge import WEEK01_SOURCE_SHA256, ReadoutExtraction, TeacherBridgeError, extract_readout


class RunnerError(RuntimeError):
    """Raised when Week 6 cannot make technically valid evidence."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """All fresh serializable evidence produced by one successful execution."""

    evidence: dict[str, Any]
    context_profiles: list[dict[str, Any]]
    dataset_provenance: dict[str, Any]
    dataset_lock: dict[str, Any] | None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_context_authorization() -> dict[str, Any]:
    """Read the unified context-stage authorization."""

    path = _project_root() / "provenance" / "provenance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError("provenance record is unreadable: " + str(error)) from error
    if not isinstance(payload, dict):
        raise RunnerError("provenance record must be a JSON object")
    if payload.get("context_authorized") is not True:
        raise RunnerError("provenance does not authorize the context gate")
    required = ("week05_source_sha256", "week05_full_evidence_run_id", "week05_metrics_sha256")
    if any(not isinstance(payload.get(field), str) or not payload[field] for field in required):
        raise RunnerError("Week 5 dependency record is incomplete")
    return payload


V1_SUPERSEDED_MANIFEST_SHA256 = "e9dcfb18b0086e16494475136ba2b5cb01bee2157b5a0a318f9956e191bde056"


def _load_order_control_revision() -> dict[str, Any]:
    """Read the versioned justification for the v2 relation-level order control."""

    path = _project_root() / "provenance" / "order_control_revision.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError("order-control revision record is unreadable: " + str(error)) from error
    if not isinstance(payload, dict):
        raise RunnerError("order-control revision record must be a JSON object")
    if payload.get("superseded_dataset_manifest_sha256") != V1_SUPERSEDED_MANIFEST_SHA256:
        raise RunnerError("order-control revision does not bind the superseded v1 manifest hash")
    if payload.get("revision_id") != "week06-order-control-v4-relation-level":
        raise RunnerError("order-control revision has an unexpected revision id")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or not evidence.get("order_audit_run_id"):
        raise RunnerError("order-control revision lacks its order-audit evidence binding")
    required = ("reason", "unchanged", "changed", "authorized_by")
    if any(not payload.get(field) for field in required):
        raise RunnerError("order-control revision record is incomplete")
    return payload


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


def _median_metrics(records: Sequence[Mapping[str, float]], *, label: str) -> dict[str, float]:
    if not records:
        raise RunnerError(label + " has no records to aggregate")
    fields = tuple(records[0].keys())
    if not fields or any(tuple(record.keys()) != fields for record in records):
        raise RunnerError(label + " has inconsistent metric keys")
    output: dict[str, float] = {}
    for field in fields:
        values = [_require_finite(record[field], name=label + "." + field) for record in records]
        output[field] = float(np.median(np.asarray(values, dtype=np.float64)))
    return output


def _primary_raw_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    try:
        metrics = summary["headwise_raw_score_stability"]["macro_mean"]
    except (KeyError, TypeError) as error:
        raise RunnerError("context summary lacks headwise raw-score macro metrics") from error
    fields = (
        "mean_anchor_score_std",
        "mean_pairwise_score_difference_std",
        "mean_pair_top_k_jaccard",
        "mean_pair_kendall_tau",
        "mean_pair_spearman",
        "mean_preference_flip_rate_vs_first",
    )
    return {field: _require_finite(metrics.get(field), name="primary_raw." + field) for field in fields}


def _summary_for_gate(primary_metrics: Mapping[str, float]) -> dict[str, Any]:
    return {"headwise_raw_score_stability": {"macro_mean": dict(primary_metrics)}}


def _context_config(manifest: Mapping[str, Any], *, mode: str) -> dict[str, int]:
    if mode not in {"smoke", "full"}:
        raise RunnerError("mode must be smoke or full")
    try:
        context = manifest["context"]
        output = {
            "query_count": int(context["query_count"] if mode == "full" else context["smoke_query_count"]),
            "anchor_count": int(context["anchor_count"]),
            "context_count": int(context["context_count"] if mode == "full" else context["smoke_context_count"]),
            "context_size": int(context["context_size"] if mode == "full" else context["smoke_context_size"]),
            "anchor_top_k": int(context["anchor_top_k"]),
            "sampling_seed": int(context["sampling_seed"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerError("frozen manifest lacks a valid context contract") from error
    if output["query_count"] < 1 or output["anchor_count"] < 3 or output["context_count"] < 2:
        raise RunnerError("frozen context contract has unsafe counts")
    return output


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


def _extract_context(
    *,
    split: ExperimentSplit,
    definition: ContextDefinition,
    query_position: int,
    seed: int,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> tuple[ReadoutExtraction, AnchorObservation, tuple[str, ...]]:
    positions = list(definition.support_positions)
    support_ids = tuple(split.support_ids[position] for position in positions)
    anchor_ids = tuple(split.support_ids[position] for position in definition.anchor_positions)
    extraction = extract_readout(
        support_features=np.asarray(split.support_features)[positions],
        support_labels=np.asarray(split.support_labels)[positions],
        support_ids=support_ids,
        query_features=np.asarray(split.query_features[query_position]).reshape(1, -1),
        query_ids=(split.query_ids[query_position],),
        seed=seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    if extraction.support_ids != support_ids or extraction.query_ids != (split.query_ids[query_position],):
        raise RunnerError("Teacher output IDs do not align with its explicit context/query input")
    observation = observe_anchors(
        alpha=extraction.alpha,
        raw_scores=extraction.raw_scores,
        support_ids=support_ids,
        anchor_ids=anchor_ids,
    )
    return extraction, observation, anchor_ids


def _max_abs_difference(left: ArrayLike, right: ArrayLike, *, name: str) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RunnerError(name + " cannot be compared safely")
    return float(np.max(np.abs(a - b)))


def _control_record(
    reference: AnchorObservation,
    candidate: AnchorObservation,
    *,
    alpha_atol: float,
    raw_atol: float,
    kind: str,
) -> dict[str, Any]:
    alpha_error = _max_abs_difference(reference.anchor_alpha, candidate.anchor_alpha, name=kind + ".alpha")
    raw_error = _max_abs_difference(reference.anchor_raw_scores, candidate.anchor_raw_scores, name=kind + ".raw")
    passed = bool(alpha_error <= alpha_atol and raw_error <= raw_atol)
    return {
        "kind": kind,
        "max_abs_anchor_alpha_difference": alpha_error,
        "max_abs_anchor_raw_score_difference": raw_error,
        "alpha_atol": alpha_atol,
        "raw_atol": raw_atol,
        "passed": passed,
    }


def _context_profile(
    *,
    dataset_id: str,
    seed: int,
    split: ExperimentSplit,
    query_position: int,
    definition: ContextDefinition,
    extraction: ReadoutExtraction,
    observation: AnchorObservation,
    anchor_ids: tuple[str, ...],
) -> dict[str, Any]:
    positions = list(definition.support_positions)
    support_ids = tuple(split.support_ids[position] for position in positions)
    background_ids = tuple(split.support_ids[position] for position in definition.background_positions)
    pairs = [[anchor_ids[first], anchor_ids[second]] for first, second in observation.pair_indices]
    # Removing the singleton query axis keeps JSON evidence compact while
    # retaining every estimator/head and support column of the raw score.
    raw = np.asarray(extraction.raw_scores, dtype=np.float64)[:, :, 0, :]
    return _json_ready(
        {
            "dataset_id": dataset_id,
            "seed": seed,
            "split_id": split.split_id,
            "query_id": split.query_ids[query_position],
            "context_index": definition.context_index,
            "support_ids": list(support_ids),
            "anchor_ids": list(anchor_ids),
            "background_ids": list(background_ids),
            "background_class_counts": definition.background_class_counts,
            "alpha": np.asarray(extraction.alpha, dtype=np.float64)[0],
            "raw_scores": raw,
            "raw_score_shape": list(raw.shape),
            "anchor_alpha": observation.anchor_alpha,
            "anchor_alpha_total": observation.anchor_alpha_total,
            "anchor_alpha_normalized": observation.anchor_alpha_normalized,
            "anchor_raw_scores": observation.anchor_raw_scores,
            "anchor_raw_conditional": observation.anchor_raw_conditional,
            "anchor_mean_raw_scores": observation.anchor_mean_raw_scores,
            "anchor_pair_ids": pairs,
            "pairwise_raw_score_differences": observation.pairwise_raw_score_differences,
            "pairwise_mean_raw_score_differences": observation.pairwise_mean_raw_score_differences,
            "raw_score_ranking": list(observation.raw_score_ranking),
            "anchor_raw_score_ranking": list(observation.anchor_raw_score_ranking),
        }
    )


def _run_query_contexts(
    *,
    dataset_id: str,
    split: ExperimentSplit,
    query_position: int,
    definitions: Sequence[ContextDefinition],
    context_contract: Mapping[str, int],
    controls: Mapping[str, Any],
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], ReadoutExtraction]:
    observations: list[AnchorObservation] = []
    profiles: list[dict[str, Any]] = []
    initial_extraction: ReadoutExtraction | None = None
    anchor_ids: tuple[str, ...] | None = None
    for definition in definitions:
        extraction, observation, observed_anchor_ids = _extract_context(
            split=split,
            definition=definition,
            query_position=query_position,
            seed=split.seed,
            n_estimators=n_estimators,
            model_cache_dir=model_cache_dir,
            week01_source_root=week01_source_root,
        )
        if initial_extraction is None:
            initial_extraction = extraction
            anchor_ids = observed_anchor_ids
        elif anchor_ids != observed_anchor_ids:
            raise RunnerError("anchor IDs changed across supposedly fixed contexts")
        observations.append(observation)
        profiles.append(
            _context_profile(
                dataset_id=dataset_id,
                seed=split.seed,
                split=split,
                query_position=query_position,
                definition=definition,
                extraction=extraction,
                observation=observation,
                anchor_ids=observed_anchor_ids,
            )
        )
    if initial_extraction is None or anchor_ids is None:
        raise RunnerError("no context extraction was performed")
    alpha = np.vstack([observation.anchor_alpha for observation in observations])
    raw = np.stack([observation.anchor_raw_scores for observation in observations], axis=0)
    summary = summarize_contexts(alpha, raw, top_k=int(context_contract["anchor_top_k"]))
    query_gate = classify_context_regime(summary, {"weak": controls["gate"]["weak"], "strong": controls["gate"]["strong"]})
    first_definition = definitions[int(controls["repeat_context_index"])]
    _repeat_extraction, repeat_observation, _ = _extract_context(
        split=split,
        definition=first_definition,
        query_position=query_position,
        seed=split.seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    repeat = _control_record(
        observations[int(controls["repeat_context_index"])],
        repeat_observation,
        alpha_atol=float(controls["alpha_atol"]),
        raw_atol=float(controls["raw_atol"]),
        kind="same_context_repeat",
    )
    if not repeat["passed"]:
        raise RunnerError("same-context repeat control exceeded its frozen tolerance")
    return (
        {
            "query_id": split.query_ids[query_position],
            "anchor_ids": list(anchor_ids),
            "context_count": len(definitions),
            "context_size": len(definitions[0].support_positions),
            "summary": summary,
            "primary_raw_metrics": _primary_raw_metrics(summary),
            "query_gate": query_gate,
            "controls": {"same_context_repeat": repeat},
        },
        profiles,
        initial_extraction,
    )


def _relation_level_control_record(
    reference: AnchorObservation,
    candidate: AnchorObservation,
    *,
    alpha_atol: float,
    raw_scale_guardrail: float,
    top_k: int,
    share_flip_epsilon: float,
    decisive_share_gap: float,
    alpha_total_relative_max: float,
    alpha_abs_max: float,
    kind: str,
) -> dict[str, Any]:
    """Version 2 support-order control on ID-aligned anchor outputs.

    The v1 control compared anchor raw scores elementwise against an absolute
    tolerance of 1e-05.  The diagnostic order audit measured that this
    threshold sits below the environment's cross-order float-accumulation
    noise floor, so v2 evaluates relation-level invariants on the invariant
    anchor universe instead, and keeps elementwise arithmetic only as
    gross-error guardrails and for the same-context repeat control.
    """

    alpha_error = _max_abs_difference(reference.anchor_alpha, candidate.anchor_alpha, name=kind + ".alpha")
    raw_error = _max_abs_difference(reference.anchor_raw_scores, candidate.anchor_raw_scores, name=kind + ".raw")
    raw_magnitude = float(np.max(np.abs(np.asarray(reference.anchor_raw_scores, dtype=np.float64))))
    if not math.isfinite(raw_magnitude) or raw_magnitude <= 0.0:
        raise RunnerError(kind + " reference raw-score magnitude is not positive; scale-relative guardrail is undefined")
    scale_relative_error = raw_error / raw_magnitude
    reference_raw = np.asarray(reference.anchor_raw_scores, dtype=np.float64)
    candidate_raw = np.asarray(candidate.anchor_raw_scores, dtype=np.float64)
    if reference_raw.ndim != 3 or reference_raw.shape != candidate_raw.shape:
        raise RunnerError(kind + " anchor raw layouts must match on [estimator, head, anchors]")
    surfaces: list[dict[str, Any]] = []
    nan_surfaces = 0
    for estimator in range(reference_raw.shape[0]):
        for head in range(reference_raw.shape[1]):
            reference_surface = reference_raw[estimator, head, :]
            candidate_surface = candidate_raw[estimator, head, :]
            kendall = kendall_tau(reference_surface, candidate_surface)
            spearman = spearman_correlation(reference_surface, candidate_surface)
            jaccard = top_k_jaccard(reference_surface, candidate_surface, k=int(top_k))
            flip_rate = preference_flip_rate(reference_surface, candidate_surface)
            vacuous = not math.isfinite(kendall) or not math.isfinite(spearman) or not math.isfinite(flip_rate)
            if vacuous:
                nan_surfaces += 1
            surfaces.append(
                {
                    "estimator_index": estimator,
                    "head_index": head,
                    "kendall_tau": kendall,
                    "spearman": spearman,
                    "top_k_jaccard": jaccard,
                    "preference_flip_rate": flip_rate,
                    "vacuous_all_ties": bool(vacuous),
                }
            )
    comparable_kendall = [surface["kendall_tau"] for surface in surfaces if math.isfinite(surface["kendall_tau"])]
    comparable_spearman = [surface["spearman"] for surface in surfaces if math.isfinite(surface["spearman"])]
    comparable_jaccard = [surface["top_k_jaccard"] for surface in surfaces if math.isfinite(surface["top_k_jaccard"])]
    comparable_flips = [surface["preference_flip_rate"] for surface in surfaces if math.isfinite(surface["preference_flip_rate"])]
    if not comparable_kendall or not comparable_jaccard or not comparable_flips:
        raise RunnerError(kind + " has no comparable anchor pairs on any surface; relation-level control is undefined")
    relation_summary = {
        "surface_count": len(surfaces),
        "vacuous_all_ties_surface_count": nan_surfaces,
        "kendall_tau_min": float(np.min(comparable_kendall)),
        "spearman_min": float(np.min(comparable_spearman)) if comparable_spearman else float("nan"),
        "top_k_jaccard_min": float(np.min(comparable_jaccard)),
        "max_preference_flip_rate": float(np.max(comparable_flips)),
        "anchor_argmax_flip": bool(
            reference.anchor_raw_score_ranking[:1] != candidate.anchor_raw_score_ranking[:1]
        ),
    }
    checks = {
        "pairwise_preference_flip_rate_max": float(relation_summary["max_preference_flip_rate"]),
        "per_surface_kendall_tau_min": float(relation_summary["kendall_tau_min"]),
        "per_surface_top_k_jaccard_min": float(relation_summary["top_k_jaccard_min"]),
        "anchor_argmax_flip": relation_summary["anchor_argmax_flip"],
    }
    alpha_passed = bool(alpha_error <= float(alpha_atol))
    relations_passed = bool(
        checks["pairwise_preference_flip_rate_max"] <= 0.0
        and checks["per_surface_kendall_tau_min"] >= 1.0
        and checks["per_surface_top_k_jaccard_min"] >= 1.0
        and checks["anchor_argmax_flip"] is False
    )
    guardrail_passed = bool(scale_relative_error <= float(raw_scale_guardrail))

    reference_alpha_total = float(np.sum(np.asarray(reference.anchor_alpha, dtype=np.float64)))
    candidate_alpha_total = float(np.sum(np.asarray(candidate.anchor_alpha, dtype=np.float64)))
    if reference_alpha_total <= 0.0:
        raise RunnerError(kind + " reference anchor alpha total is not positive")
    alpha_total_relative_change = abs(candidate_alpha_total - reference_alpha_total) / reference_alpha_total
    reference_shares = np.asarray(reference.anchor_alpha_normalized, dtype=np.float64)
    candidate_shares = np.asarray(candidate.anchor_alpha_normalized, dtype=np.float64)
    share_flip_rate = preference_flip_rate(reference_shares, candidate_shares, epsilon=float(share_flip_epsilon))
    ranked_reference = tuple(np.argsort(-reference_shares, kind="stable").tolist())
    ranked_candidate = tuple(np.argsort(-candidate_shares, kind="stable").tolist())
    decisive_gap = float(np.sort(reference_shares)[::-1][0] - np.sort(reference_shares)[::-1][1])
    argmax_decisive = bool(decisive_gap > float(decisive_share_gap))
    argmax_flip = bool(ranked_reference[:1] != ranked_candidate[:1])
    alpha_abs_error = _max_abs_difference(reference.anchor_alpha, candidate.anchor_alpha, name=kind + ".alpha")
    alpha_checks = {
        "normalized_share_preference_flip_rate": float(share_flip_rate) if math.isfinite(share_flip_rate) else None,
        "normalized_share_flip_epsilon": float(share_flip_epsilon),
        "normalized_share_vacuous_all_ties": not math.isfinite(share_flip_rate),
        "anchor_alpha_argmax_flip": argmax_flip,
        "anchor_alpha_argmax_decisive": argmax_decisive,
        "reference_decisive_share_gap": decisive_gap,
        "anchor_alpha_total_relative_change": alpha_total_relative_change,
        "max_abs_anchor_alpha_difference": alpha_abs_error,
    }
    alpha_relations_passed = bool(
        (not math.isfinite(share_flip_rate) or share_flip_rate <= 0.0)
        and (not argmax_decisive or not argmax_flip)
        and alpha_total_relative_change <= float(alpha_total_relative_max)
        and alpha_abs_error <= float(alpha_abs_max)
    )
    return {
        "kind": kind,
        "reference": "canonical_context_zero",
        "candidate": "reversed_support_order_refit",
        "max_abs_anchor_alpha_difference": alpha_error,
        "alpha_atol": float(alpha_atol),
        "alpha_checks_recorded": _json_ready(alpha_checks),
        "alpha_relations_passed": alpha_relations_passed,
        "relation_summary": _json_ready(relation_summary),
        "relation_checks_recorded": _json_ready(checks),
        "numeric_guardrail": _json_ready(
            {
                "max_abs_anchor_raw_difference": raw_error,
                "max_abs_over_reference_scale": scale_relative_error,
                "raw_scale_guardrail": float(raw_scale_guardrail),
                "reference_max_abs_raw_value": raw_magnitude,
            }
        ),
        "passed": bool(alpha_relations_passed and relations_passed and guardrail_passed),
        "note": "v2 relation-level support-order control; the v1 elementwise raw tolerance is not used here",
    }


def _order_control(
    *,
    split: ExperimentSplit,
    definition: ContextDefinition,
    query_position: int,
    reference: AnchorObservation,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
    controls: Mapping[str, Any],
    top_k: int,
) -> dict[str, Any]:
    reversed_definition = ContextDefinition(
        context_index=definition.context_index,
        support_positions=tuple(reversed(definition.support_positions)),
        anchor_positions=definition.anchor_positions,
        background_positions=definition.background_positions,
        background_class_counts=dict(definition.background_class_counts),
    )
    _extraction, observation, _anchors = _extract_context(
        split=split,
        definition=reversed_definition,
        query_position=query_position,
        seed=split.seed,
        n_estimators=n_estimators,
        model_cache_dir=model_cache_dir,
        week01_source_root=week01_source_root,
    )
    guardrails = controls["order_numeric_guardrails"]
    alpha_checks_config = controls["order_alpha_checks"]
    record = _relation_level_control_record(
        reference,
        observation,
        alpha_atol=float(controls["alpha_atol"]),
        raw_scale_guardrail=float(guardrails["raw_scale_relative_max"]),
        top_k=int(top_k),
        share_flip_epsilon=float(alpha_checks_config["normalized_share_flip_epsilon"]),
        decisive_share_gap=float(alpha_checks_config["decisive_share_gap"]),
        alpha_total_relative_max=float(alpha_checks_config["anchor_alpha_total_relative_max"]),
        alpha_abs_max=float(alpha_checks_config["anchor_alpha_abs_max"]),
        kind="reversed_support_order",
    )
    if not record["passed"]:
        raise RunnerError("support-order control exceeded its frozen relation-level criteria")
    return record


def _dataset_gate(seed_records: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    seed_metrics = [record["primary_raw_metrics"] for record in seed_records]
    aggregated = _median_metrics(seed_metrics, label="dataset primary raw metrics")
    outcome = classify_context_regime(_summary_for_gate(aggregated), gate)
    return {
        "aggregation": "median across seed-level medians of fixed-query summaries",
        **outcome,
        # Keep the full raw diagnostics (including score-difference variance),
        # while ``outcome`` supplies the regime and its four gate conditions.
        "primary_raw_metrics": aggregated,
    }


def _global_gate(dataset_results: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    required_ids = [str(spec["dataset_id"]) for spec in manifest["datasets"]]
    if [str(result["dataset_id"]) for result in dataset_results] != required_ids:
        raise RunnerError("global route requires every frozen dataset in manifest order")
    weak_ids = [str(result["dataset_id"]) for result in dataset_results if bool(result["gate"]["weak"])]
    strong_ids = [str(result["dataset_id"]) for result in dataset_results if bool(result["gate"]["strong"])]
    moderate_ids = [
        str(result["dataset_id"])
        for result in dataset_results
        if str(result["gate"]["regime"]) == "moderate"
    ]
    count = len(dataset_results)
    route = manifest["gate"]
    if len(strong_ids) > count / 2.0:
        decision = "strong_context_conditioned_candidate"
        recommendation = str(route["strong_route"])
    elif len(weak_ids) > count / 2.0:
        decision = "weak_context_static_candidate"
        recommendation = str(route["weak_route"])
    else:
        decision = "moderate_context_adapter_candidate"
        recommendation = str(route["moderate_route"])
    if decision not in route["global_decision_values"]:
        raise RunnerError("scientific route is outside the frozen manifest values")
    return {
        "frozen_dataset_count": count,
        "weak_count": len(weak_ids),
        "weak_dataset_ids": weak_ids,
        "moderate_count": len(moderate_ids),
        "moderate_dataset_ids": moderate_ids,
        "strong_count": len(strong_ids),
        "strong_dataset_ids": strong_ids,
        "decision": decision,
        "recommendation": recommendation,
    }


def _run_dataset(
    dataset: RealDataset,
    *,
    seeds: Sequence[int],
    split_args: Mapping[str, Any],
    manifest: Mapping[str, Any],
    context_contract: Mapping[str, int],
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    controls = manifest["context"]["controls"]
    if int(controls["repeat_context_index"]) >= int(context_contract["context_count"]):
        raise RunnerError("repeat context index is outside the frozen context count")
    seed_records: list[dict[str, Any]] = []
    all_profiles: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    decoder_api: str | None = None
    raw_hook: str | None = None
    for seed in seeds:
        split = make_stratified_split(dataset, seed=int(seed), **dict(split_args))
        anchor_positions = select_anchor_positions(
            split.support_labels, anchor_count=int(context_contract["anchor_count"])
        )
        definitions = sample_contexts(
            split.support_labels,
            anchor_positions=anchor_positions,
            context_size=int(context_contract["context_size"]),
            context_count=int(context_contract["context_count"]),
            seed=derive_seed(int(context_contract["sampling_seed"]), dataset.dataset_id, seed),
        )
        query_positions = select_query_positions(
            int(context_contract["query_count"]),
            available=len(split.query_ids),
            seed=derive_seed(int(context_contract["sampling_seed"]), dataset.dataset_id, seed, "queries"),
        )
        query_records: list[dict[str, Any]] = []
        first_observation: AnchorObservation | None = None
        for query_offset, query_position in enumerate(query_positions):
            query_record, profiles, extraction = _run_query_contexts(
                dataset_id=dataset.dataset_id,
                split=split,
                query_position=query_position,
                definitions=definitions,
                context_contract=context_contract,
                controls={"repeat_context_index": controls["repeat_context_index"], "alpha_atol": controls["alpha_atol"], "raw_atol": controls["raw_atol"], "gate": manifest["gate"]},
                n_estimators=n_estimators,
                model_cache_dir=model_cache_dir,
                week01_source_root=week01_source_root,
            )
            query_records.append(query_record)
            all_profiles.extend(profiles)
            runtime = extraction.runtime
            decoder_api = extraction.decoder_readout_api
            raw_hook = extraction.raw_score_hook_path
            if query_offset == 0:
                first_observation = observe_anchors(
                    alpha=np.asarray(extraction.alpha),
                    raw_scores=np.asarray(extraction.raw_scores),
                    support_ids=tuple(split.support_ids[position] for position in definitions[0].support_positions),
                    anchor_ids=tuple(split.support_ids[position] for position in definitions[0].anchor_positions),
                )
        if first_observation is None:
            raise RunnerError("label-blind query selection produced no query for order control")
        order = _order_control(
            split=split,
            definition=definitions[0],
            query_position=query_positions[0],
            reference=first_observation,
            n_estimators=n_estimators,
            model_cache_dir=model_cache_dir,
            week01_source_root=week01_source_root,
            controls=controls,
            top_k=context_contract["anchor_top_k"],
        )
        primary = _median_metrics(
            [record["primary_raw_metrics"] for record in query_records],
            label=dataset.dataset_id + ".seed-" + str(seed) + " query metrics",
        )
        seed_gate = classify_context_regime(_summary_for_gate(primary), manifest["gate"])
        seed_records.append(
            {
                "seed": int(seed),
                "split_id": split.split_id,
                "holdout_ids": list(split.holdout_ids),
                "support_ids": list(split.support_ids),
                "support_count": len(split.support_ids),
                "query_count_available": len(split.query_ids),
                "selected_query_ids": [split.query_ids[position] for position in query_positions],
                "anchor_ids": [split.support_ids[position] for position in anchor_positions],
                "context_support_ids": [list(split.support_ids[position] for position in item.support_positions) for item in definitions],
                "background_class_counts": definitions[0].background_class_counts,
                "query_summaries": query_records,
                "primary_raw_metrics": primary,
                "seed_gate": seed_gate,
                "controls": {"reversed_support_order": order},
            }
        )
    if runtime is None or decoder_api is None or raw_hook is None:
        raise RunnerError(dataset.dataset_id + ": Teacher was not invoked")
    return (
        {
            "dataset_id": dataset.dataset_id,
            "dataset": dataset.metadata,
            "seeds": seed_records,
            "gate": _dataset_gate(seed_records, manifest["gate"]),
            "teacher": {
                "teacher_source_sha256": WEEK01_SOURCE_SHA256,
                "decoder_readout_api": decoder_api,
                "raw_score_hook_path": raw_hook,
            },
        },
        all_profiles,
        runtime,
    )


def run_context_sensitivity(
    *,
    mode: str,
    dataset_manifest_path: str | Path,
    dataset_cache_dir: str | Path,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
) -> RunResult:
    """Run a smoke or full frozen context experiment without downloading data."""

    manifest, manifest_hash = load_manifest(dataset_manifest_path)
    context_authorization = _load_context_authorization()
    order_control_revision = _load_order_control_revision()
    frozen_teacher = manifest.get("teacher")
    if not isinstance(frozen_teacher, Mapping):
        raise RunnerError("frozen manifest lacks Teacher settings")
    if int(frozen_teacher.get("n_estimators", 0)) != int(n_estimators):
        raise RunnerError("n_estimators differs from the frozen manifest value")
    if model_cache_dir != frozen_teacher.get("model_cache_dir"):
        raise RunnerError("model_cache_dir differs from the frozen manifest cache")
    context_contract = _context_config(manifest, mode=mode)
    seeds, split_args = _split_config(manifest, mode=mode)
    if mode == "full":
        selected_ids: list[str] | None = None
    else:
        selected_ids = [str(manifest["smoke"]["dataset_id"])]
    # Week 6 cannot bootstrap or redownload anything. Prepare-Server must have
    # seeded a distinct cache from the verified Week 5 cache first.
    datasets = materialize_manifest(
        manifest,
        cache_dir=dataset_cache_dir,
        allow_download=False,
        dataset_ids=selected_ids,
    )
    if mode == "smoke" and len(datasets) != 1:
        raise RunnerError("smoke mode must execute exactly the frozen smoke dataset")
    expected_by_id = {str(item["dataset_id"]): str(item.get("expected_content_sha256", "")) for item in manifest["datasets"]}
    for dataset in datasets:
        if str(dataset.metadata.get("content_sha256", "")) != expected_by_id[dataset.dataset_id]:
            raise RunnerError(dataset.dataset_id + ": content hash differs from its frozen pin")
    dataset_lock = verify_dataset_lock(manifest, manifest_hash, datasets, cache_dir=dataset_cache_dir) if mode == "full" else None
    results: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    for dataset in datasets:
        result, dataset_profiles, observed_runtime = _run_dataset(
            dataset,
            seeds=seeds,
            split_args=split_args,
            manifest=manifest,
            context_contract=context_contract,
            n_estimators=n_estimators,
            model_cache_dir=model_cache_dir,
            week01_source_root=week01_source_root,
        )
        results.append(result)
        profiles.extend(dataset_profiles)
        runtime = observed_runtime
    if runtime is None:
        raise RunnerError("Teacher was never invoked")
    scientific_gate = (
        _global_gate(results, manifest)
        if mode == "full"
        else {"decision": "not_evaluated_in_smoke", "recommendation": "SMOKE_ONLY_NO_SCIENTIFIC_CLAIM"}
    )
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
            "context_authorization": context_authorization,
            "dataset_manifest_id": manifest["manifest_id"],
            "dataset_manifest_sha256": manifest_hash,
            "required_dataset_ids": [str(item["dataset_id"]) for item in manifest["datasets"]],
            "executed_dataset_ids": [dataset.dataset_id for dataset in datasets],
            "split_specification": dict(manifest["split"] if mode == "full" else manifest["smoke"]),
            "n_estimators": n_estimators,
            "frozen_teacher": dict(frozen_teacher),
            "context_plan": context_contract,
            "context_controls": dict(manifest["context"]["controls"]),
            "order_control_revision": order_control_revision,
            "runtime": runtime,
            "dataset_results": results,
            "scientific_gate": scientific_gate,
        }
    )
    return RunResult(
        evidence=dict(evidence),
        context_profiles=[dict(profile) for profile in profiles],
        dataset_provenance=dict(_json_ready(provenance)),
        dataset_lock=None if dataset_lock is None else dict(_json_ready(dataset_lock)),
    )


def _report_markdown(evidence: Mapping[str, Any], *, run_id: str) -> str:
    plan = evidence["context_plan"]
    gate = evidence["scientific_gate"]
    lines = [
        "# Week 06 - Context Sensitivity on Frozen Real Data",
        "",
        "Technical status: PASS",
        "Run ID: " + str(run_id),
        "Completed at UTC: " + str(evidence["completed_at_utc"]),
        "Mode: " + str(evidence["mode"]),
        "Week 1 Teacher source SHA-256: " + str(evidence["teacher_source_sha256"]),
        "Dataset manifest SHA-256: " + str(evidence["dataset_manifest_sha256"]),
        "",
        "## Frozen protocol",
        "",
        "Fixed queries per seed: " + str(plan["query_count"]),
        "Fixed anchors: " + str(plan["anchor_count"]),
        "Contexts per query: " + str(plan["context_count"]),
        "Support rows per context: " + str(plan["context_size"]),
        "Anchor Top-K: " + str(plan["anchor_top_k"]),
        "",
        "Only fixed-anchor raw-score ranks and pairwise raw-score differences drive the scientific route. Alpha-only movement is reported separately as softmax competition evidence.",
        "",
        "## Dataset-level raw-score context summaries",
        "",
        "| Dataset | Regime | raw Kendall | raw Spearman | raw Top-K Jaccard | raw flip rate | pairwise raw-difference std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in evidence["dataset_results"]:
        outcome = result["gate"]
        primary = outcome["primary_raw_metrics"]
        lines.append(
            "| {dataset} | {regime} | {kendall:.6f} | {spearman:.6f} | {jaccard:.6f} | {flips:.6f} | {variance:.6f} |".format(
                dataset=result["dataset_id"],
                regime=outcome["regime"],
                kendall=float(primary["mean_pair_kendall_tau"]),
                spearman=float(primary["mean_pair_spearman"]),
                jaccard=float(primary["mean_pair_top_k_jaccard"]),
                flips=float(primary["mean_preference_flip_rate_vs_first"]),
                variance=float(primary["mean_pairwise_score_difference_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Scientific route",
            "",
            "Decision: " + str(gate.get("decision")),
            "Recommendation: " + str(gate.get("recommendation")),
            "Weak datasets: " + str(gate.get("weak_dataset_ids")),
            "Moderate datasets: " + str(gate.get("moderate_dataset_ids")),
            "Strong datasets: " + str(gate.get("strong_dataset_ids")),
            "",
            "A technical PASS validates execution and provenance only. Preserve the evidence whether it supports a static, adapter, or context-conditioned candidate.",
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
    """Write a fresh, non-overwriting Week 6 evidence directory."""

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
        "context.md": _report_markdown(metrics, run_id=run_id),
    }
    if result.dataset_lock is not None:
        payloads["dataset_lock.json"] = result.dataset_lock
    for name, payload in payloads.items():
        destination = root / name
        if isinstance(payload, str):
            destination.write_text(payload, encoding="utf-8")
        else:
            destination.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    profiles_path = root / "context_profiles.jsonl"
    with profiles_path.open("x", encoding="utf-8", newline="\n") as handle:
        for profile in result.context_profiles:
            handle.write(json.dumps(_json_ready(profile), ensure_ascii=False, sort_keys=True) + "\n")
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


__all__ = ["DatasetError", "RunnerError", "TeacherBridgeError", "run_context_sensitivity", "write_evidence"]
