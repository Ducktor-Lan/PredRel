"""Fail closed on the unified server stack and Teacher (no legacy snapshots)."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys


EXPECTED = {
    "numpy": "2.5.3",
    "scikit-learn": "1.9.1",
    "torch": "2.6.0+cu126",
    "tabpfn": "8.5.0",
    "tabpfn-extensions": "0.6.2",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", default=str(_project_root() / "provenance" / "dataset_manifest_benchmark.json"))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(_project_root() / "src"))
    from predrel.data import DatasetError, load_manifest
    from predrel.teacher_bridge import TeacherBridgeError, runtime_record

    destination = Path(args.output).resolve()
    if destination.exists():
        print("ENVIRONMENT_ERROR: refusing to overwrite output: " + str(destination), file=sys.stderr)
        return 2
    try:
        versions = {name: _version(name) for name in EXPECTED}
        mismatches = {
            name: {"expected": expected, "observed": versions[name]}
            for name, expected in EXPECTED.items()
            if versions[name] != expected
        }
        if mismatches:
            raise RuntimeError("pinned package versions differ: " + str(mismatches))
        manifest, manifest_hash = load_manifest(args.dataset_manifest)
        runtime = runtime_record()
        if not runtime["cuda_available"]:
            raise RuntimeError("CUDA is not available to the selected server environment")
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("nvidia-smi preflight failed: " + str(error)) from error
        payload = {
            "schema_version": 1,
            "status": "pass",
            "python": sys.version.replace("\n", " "),
            "packages": versions,
            "runtime": runtime,
            "gpu": gpu,
            "dataset_manifest_id": manifest["manifest_id"],
            "dataset_manifest_sha256": manifest_hash,
        }
    except (DatasetError, TeacherBridgeError, RuntimeError, OSError, ValueError) as error:
        print("ENVIRONMENT_ERROR: " + str(error), file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
