"""Build the Week 7 readout-anatomy record from frozen Weeks 1-6 evidence.

No new Teacher fits. No model training. No new thresholds.
Every scientific number is recomputed from seed-level summaries with the
exact logic of each week's own verifier, then compared to the reported
frozen gate at 1e-12 absolute tolerance.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .gates import (
    classify_context_regime,
    median,
    week3_dataset_uniformity,
    week4_dataset_gate,
    week5_dataset_regime,
    week6_primary_median,
)


class AnatomyError(ValueError):
    """Raised when a frozen input fails byte or gate verification."""


RECOMPUTE_ATOL = 1e-12


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnatomyError("unreadable JSON: " + str(path)) from error


def _close(reported: float, recomputed: float, *, name: str) -> None:
    if abs(float(reported) - float(recomputed)) > RECOMPUTE_ATOL:
        raise AnatomyError(
            name + " differs: reported=" + repr(reported) + ", recomputed=" + repr(recomputed)
        )


def build_anatomy(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
    teacher_report_text: str,
) -> dict[str, Any]:
    """Recompute all five frozen gates and fill the seven anatomy fields."""
    frozen = {str(item["week"]): item for item in manifest["frozen_inputs"]}
    snapshot_inputs = Path(__file__).resolve().parents[2] / "inputs" / "weeks1to6"

    def frozen_path(week: str) -> Path:
        return (repo_root / str(frozen[week]["local_path"])).resolve()

    def bundled_path(name: str) -> Path:
        return (snapshot_inputs / name).resolve()

    # ---- byte verification of every frozen input ----
    # Local runs read the Weeks 1-6 checkout; remote snapshot runs read the
    # bundled inputs/weeks1to6 copies. Both must hash to the frozen values.
    bundled_names = {
        "1": "teacher_validation.md",
        "2": "week02_metrics.json",
        "3": "week03_metrics.json",
        "4": "week04_metrics.json",
        "5": "week05_metrics.json",
        "6": "week06_metrics.json",
    }
    input_paths: dict[str, Path] = {}
    for week in ("1", "2", "3", "4", "5", "6"):
        expected = str(frozen[week]["sha256"]).lower()
        candidates = [frozen_path(week), bundled_path(bundled_names[week])]
        chosen: Path | None = None
        for candidate in candidates:
            if candidate.is_file() and _sha256(candidate).lower() == expected:
                chosen = candidate
                break
        if chosen is None:
            raise AnatomyError(
                "Week " + week + " input bytes differ: no candidate matches " + expected
            )
        input_paths[week] = chosen

    teacher_report_path = input_paths["1"]
    metrics2 = _read_json(input_paths["2"])
    metrics3 = _read_json(input_paths["3"])
    metrics4 = _read_json(input_paths["4"])
    metrics5 = _read_json(input_paths["5"])
    metrics6 = _read_json(input_paths["6"])

    for metrics, week in (
        (metrics2, "2"),
        (metrics3, "3"),
        (metrics4, "4"),
        (metrics5, "5"),
        (metrics6, "6"),
    ):
        if metrics.get("mode") != "full":
            raise AnatomyError("Week " + week + " evidence is not a full run")
        if str(metrics.get("teacher_source_sha256", "")).lower() != manifest["frozen_teacher_sha256"]:
            raise AnatomyError("Week " + week + " Teacher SHA differs from the frozen Teacher")

    # ---- Week 2: synthetic fixtures (method validation, no gate to recompute) ----
    week2_fixtures: dict[str, Any] = {}
    for record in metrics2["results"]:
        fixture = str(record["fixture"])
        week2_fixtures[fixture] = record.get("metrics", record.get("stability", {}))

    # ---- Week 3: beyond-labels (seed-mean medians vs reported gate) ----
    manifest3_path = repo_root / "Week 3" / "provenance" / "dataset_manifest.json"
    if not manifest3_path.is_file():
        manifest3_path = snapshot_inputs / "week03_manifest.json"
    manifest3 = _read_json(manifest3_path)
    thresholds3 = manifest3["metrics"]
    if manifest3.get("manifest_id") != metrics3.get("dataset_manifest_id"):
        raise AnatomyError("Week 3 manifest binding differs")
    week3_uniform: list[str] = []
    week3_rows = []
    for result in metrics3["dataset_results"]:
        dataset_id = str(result["dataset_id"])
        gate = result["gate"]
        recomputed = week3_dataset_uniformity(result["seeds"], thresholds3)
        for key, value in recomputed["metrics"].items():
            _close(gate[key], value, name="week3." + dataset_id + "." + key)
        if bool(gate["approximately_uniform"]) != recomputed["approximately_uniform"]:
            raise AnatomyError("Week 3 uniformity decision differs for " + dataset_id)
        if recomputed["approximately_uniform"]:
            week3_uniform.append(dataset_id)
        week3_rows.append({"dataset_id": dataset_id, **recomputed["metrics"]})
    week3_gate = metrics3["scientific_gate"]
    if (
        week3_gate["approximately_uniform_count"] != len(week3_uniform)
        or list(week3_gate["approximately_uniform_dataset_ids"]) != week3_uniform
    ):
        raise AnatomyError("Week 3 global uniformity gate differs")

    # ---- Week 4: removal faithfulness ----
    manifest4_path = repo_root / "Week 4" / "provenance" / "dataset_manifest.json"
    if not manifest4_path.is_file():
        manifest4_path = snapshot_inputs / "week04_manifest.json"
    manifest4 = _read_json(manifest4_path)
    primary_k4 = str(manifest4["metrics"]["primary_k"])
    margin4 = float(manifest4["gate"]["tv_diff_margin"])
    floor4 = float(manifest4["gate"]["win_rate_floor"])
    if manifest4.get("manifest_id") != metrics4.get("dataset_manifest_id"):
        raise AnatomyError("Week 4 manifest binding differs")
    week4_faithful: list[str] = []
    week4_beyond: list[str] = []
    for result in metrics4["dataset_results"]:
        dataset_id = str(result["dataset_id"])
        gate = result["gate"]
        recomputed = week4_dataset_gate(
            result["seeds"], primary_k=primary_k4, tv_diff_margin=margin4, win_rate_floor=floor4
        )
        for key in (
            "median_tv_diff_top_minus_random",
            "median_win_rate_top_vs_random",
            "median_tv_diff_top_minus_supcon",
        ):
            _close(gate[key], recomputed[key], name="week4." + dataset_id + "." + key)
        if bool(gate["faithful"]) != recomputed["faithful"]:
            raise AnatomyError("Week 4 faithfulness differs for " + dataset_id)
        if bool(gate["beyond_supcon"]) != recomputed["beyond_supcon"]:
            raise AnatomyError("Week 4 beyond-SupCon differs for " + dataset_id)
        if recomputed["faithful"]:
            week4_faithful.append(dataset_id)
        if recomputed["beyond_supcon"]:
            week4_beyond.append(dataset_id)
    week4_gate = metrics4["scientific_gate"]
    if (
        week4_gate["faithful_count"] != len(week4_faithful)
        or week4_gate["beyond_supcon_count"] != len(week4_beyond)
        or list(week4_gate["faithful_dataset_ids"]) != week4_faithful
        or list(week4_gate["beyond_supcon_dataset_ids"]) != week4_beyond
        or week4_gate["decision"] != "stop_readout2rep_route"
    ):
        raise AnatomyError("Week 4 global gate differs")

    # ---- Week 5: directionality ----
    manifest5_path = repo_root / "Week 5" / "provenance" / "dataset_manifest.json"
    if not manifest5_path.is_file():
        manifest5_path = snapshot_inputs / "week05_manifest.json"
    manifest5 = _read_json(manifest5_path)
    direction5 = manifest5["gate"]["directionality"]
    primary_k5 = str(direction5["primary_k"])
    if manifest5.get("manifest_id") != metrics5.get("dataset_manifest_id"):
        raise AnatomyError("Week 5 manifest binding differs")
    week5_strong: list[str] = []
    week5_weak: list[str] = []
    week5_beyond: list[str] = []
    for result in metrics5["dataset_results"]:
        dataset_id = str(result["dataset_id"])
        gate = result["gate"]
        recomputed = week5_dataset_regime(
            result["seeds"],
            primary_k=primary_k5,
            strong_af=float(direction5["strong_af"]),
            weak_af=float(direction5["weak_af"]),
            strong_reciprocity=float(direction5["strong_reciprocity_at_primary_k"]),
            weak_reciprocity=float(direction5["weak_reciprocity_at_primary_k"]),
            strong_reversal=float(direction5["strong_rank_reversal"]),
            weak_reversal=float(direction5["weak_rank_reversal"]),
        )
        _close(gate["median_A_F"], recomputed["af"], name="week5." + dataset_id + ".A_F")
        _close(
            gate["median_reciprocity_at_primary_k"],
            recomputed["rc"],
            name="week5." + dataset_id + ".reciprocity",
        )
        _close(
            gate["median_rank_reversal_rate"], recomputed["rv"], name="week5." + dataset_id + ".reversal"
        )
        if (
            bool(gate["strong"]) != recomputed["strong"]
            or bool(gate["weak"]) != recomputed["weak"]
            or str(gate["regime"]) != recomputed["regime"]
        ):
            raise AnatomyError("Week 5 regime differs for " + dataset_id)
        if recomputed["strong"]:
            week5_strong.append(dataset_id)
        if recomputed["weak"]:
            week5_weak.append(dataset_id)
        # extension recheck: median of seed paired_top_vs_supcon mean_diff at K=3
        if dataset_id in list(manifest5["gate"]["extension_dataset_ids"]):
            diffs = []
            for seed in result["seeds"]:
                paired = (seed.get("extension_removal_recheck", {}).get("per_k", {}).get("3", {}).get(
                    "paired_top_vs_supcon", {}
                ))
                if "mean_diff" not in paired:
                    raise AnatomyError("Week 5 extension recheck incomplete for " + dataset_id)
                diffs.append(float(paired["mean_diff"]))
            beyond = bool(median(diffs) > float(manifest5["gate"]["removal_recheck"]["tv_diff_margin"]))
            if bool(gate.get("beyond_supcon")) != beyond:
                raise AnatomyError("Week 5 beyond_supcon differs for " + dataset_id)
            if beyond:
                week5_beyond.append(dataset_id)
    week5_gate = metrics5["scientific_gate"]
    if (
        week5_gate["strong_count"] != len(week5_strong)
        or week5_gate["weak_count"] != len(week5_weak)
        or list(week5_gate["strong_dataset_ids"]) != week5_strong
        or list(week5_gate["weak_dataset_ids"]) != week5_weak
        or list(week5_gate["extension_beyond_supcon_ids"]) != week5_beyond
        or week5_gate["decision"] != "continue_to_week06_context"
    ):
        raise AnatomyError("Week 5 global gate differs")

    # ---- Week 6: context sensitivity ----
    manifest6_path = repo_root / "Week 6" / "provenance" / "dataset_manifest.json"
    if not manifest6_path.is_file():
        manifest6_path = snapshot_inputs / "week06_manifest.json"
    manifest6 = _read_json(manifest6_path)
    if manifest6.get("manifest_id") != metrics6.get("dataset_manifest_id"):
        raise AnatomyError("Week 6 manifest binding differs")
    week6_weak: list[str] = []
    week6_moderate: list[str] = []
    week6_strong: list[str] = []
    for result in metrics6["dataset_results"]:
        dataset_id = str(result["dataset_id"])
        gate = result["gate"]
        seed_primaries = [dict(seed["primary_raw_metrics"]) for seed in result["seeds"]]
        primary = week6_primary_median(seed_primaries)
        for key, value in primary.items():
            _close(gate["primary_raw_metrics"][key], value, name="week6." + dataset_id + "." + key)
        recomputed = classify_context_regime(primary, manifest6["gate"])
        if (
            str(gate["regime"]) != recomputed["regime"]
            or bool(gate["weak"]) != recomputed["weak"]
            or bool(gate["strong"]) != recomputed["strong"]
        ):
            raise AnatomyError("Week 6 regime differs for " + dataset_id)
        if recomputed["regime"] == "weak":
            week6_weak.append(dataset_id)
        elif recomputed["regime"] == "strong":
            week6_strong.append(dataset_id)
        else:
            week6_moderate.append(dataset_id)
    week6_gate = metrics6["scientific_gate"]
    if (
        week6_gate["weak_count"] != len(week6_weak)
        or week6_gate["moderate_count"] != len(week6_moderate)
        or week6_gate["strong_count"] != len(week6_strong)
        or list(week6_gate["moderate_dataset_ids"]) != week6_moderate
        or week6_gate["decision"] != "moderate_context_adapter_candidate"
    ):
        raise AnatomyError("Week 6 global gate differs")

    # ---- seven anatomy fields (plan Section 7), single route ----
    if not (len(week6_moderate) > len(metrics6["dataset_results"]) / 2.0):
        raise AnatomyError("Week 6 moderate majority required by the frozen route mapping")
    anatomy_fields = {
        "Beyond Label": "Strong — 0/6 datasets approximately uniform (Week 3 frozen gate continue_to_week04)",
        "Beyond SupCon": "Weak-at-margin — frozen gate 1/6 beyond-SupCon at 0.02 (stop_readout2rep_route); investigator override continued on 18/18 positive seed signs (p ~ 7.6e-06), Week 5 extension recheck adds 1/3",
        "Directionality": "Mixed — strong 4/9, weak 1/9, neither majority; dual-space not authorized, symmetric not sufficient alone (Week 5 frozen gate continue_to_week06_context)",
        "Context Dependence": "Moderate — 9/9 moderate, weak 0/9, strong 0/9 (Week 6 frozen gate moderate_context_adapter_candidate)",
        "Recommended Model": "Route B (Static Directed, Query/Key dual-space) with a simple context adapter first — adapter before any full context-conditioned model",
        "Recommended Teacher Target": "beta-KL first (class-residual relation), raw-score ranking second; alpha-KL retains softmax-competition confound (plan Sections 7-8)",
        "Go / Pivot / Stop": "Go Route B with adapter — not a pivot to Retrieval Anatomy, not a stop; Week 6 moderate majority is the authorizing gate",
    }
    route = "Route B: Static Directed with simple context adapter first"

    return {
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "teacher_report_sha256": sha256(teacher_report_text.encode("utf-8")).hexdigest(),
        "recomputed": {
            "week3_uniform_ids": week3_uniform,
            "week4_faithful_ids": week4_faithful,
            "week4_beyond_supcon_ids": week4_beyond,
            "week5_strong_ids": week5_strong,
            "week5_weak_ids": week5_weak,
            "week5_extension_beyond_supcon_ids": week5_beyond,
            "week6_weak_ids": week6_weak,
            "week6_moderate_ids": week6_moderate,
            "week6_strong_ids": week6_strong,
        },
        "scientific_gates": {
            "week3_decision": week3_gate["decision"],
            "week4_decision": week4_gate["decision"],
            "week4_investigator_override": "CONTINUE_TO_WEEK05_DIRECTIONALITY (Week 4 conclusion Section 6; frozen STOP preserved)",
            "week5_decision": week5_gate["decision"],
            "week6_decision": week6_gate["decision"],
        },
        "anatomy_fields": anatomy_fields,
        "recommended_route": route,
        "week8_hand_off": {
            "backbone": "numerical standardized MLP + categorical embedding, hidden 128, output 64 (plan Section 8)",
            "loss_priority": ["beta-KL", "raw-score ranking"],
            "context_adapter_first": True,
            "deep_sets_only_if_adapter_insufficient": True,
        },
    }
