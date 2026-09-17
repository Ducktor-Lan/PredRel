"""Run one fresh smoke/full unified benchmark (20 datasets x 8 methods)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-cache-dir", required=True)
    parser.add_argument(
        "--dataset-manifest",
        default=str(_project_root() / "provenance" / "dataset_manifest_benchmark.json"),
    )
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--n-estimators", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum() or len(args.run_id) > 128:
        print("RUN_ERROR: run_id contains unsupported characters", file=sys.stderr)
        return 2
    sys.path.insert(0, str(_project_root() / "src"))
    from predrel.data import DatasetError
    from predrel.experiments.benchmark import RunnerError, run_benchmark, write_evidence
    from predrel.teacher_bridge import TeacherBridgeError

    try:
        result = run_benchmark(
            mode=args.mode,
            dataset_manifest_path=args.dataset_manifest,
            dataset_cache_dir=args.dataset_cache_dir,
            n_estimators=args.n_estimators,
            model_cache_dir=args.model_cache_dir,
        )
        destination = write_evidence(
            result,
            output_dir=args.output_dir,
            run_id=args.run_id,
            run_input={
                "schema_version": 1,
                "mode": args.mode,
                "run_id": args.run_id,
                "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
                "dataset_cache_dir": str(Path(args.dataset_cache_dir).resolve()),
                "model_cache_dir": str(Path(args.model_cache_dir).resolve()),
                "n_estimators": args.n_estimators,
                "allow_download": False,
            },
        )
    except (DatasetError, RunnerError, TeacherBridgeError, OSError, ValueError) as error:
        print(f"BENCHMARK_EXPERIMENT_ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "evidence_dir": str(destination),
                "scientific_gate": result.evidence["scientific_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
