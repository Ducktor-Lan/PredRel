"""Run the unified benchmark from a verified immutable remote source snapshot."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import sys


SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RemoteRunError(ValueError):
    """Raised when an immutable remote execution request is unsafe."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--dataset-cache-dir", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-estimators", type=int, default=1)
    return parser.parse_args()


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_request(args: argparse.Namespace) -> tuple[Path, Path]:
    snapshot = Path(args.snapshot_path).resolve()
    run_root = Path(args.run_root).resolve()
    if not snapshot.is_dir():
        raise RemoteRunError("SNAPSHOT_MISSING: " + str(snapshot))
    if not SHA256_PATTERN.fullmatch(args.source_sha.lower()):
        raise RemoteRunError("source SHA must be a canonical SHA-256")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise RemoteRunError("run ID contains unsupported characters")
    if args.n_estimators < 1:
        raise RemoteRunError("n_estimators must be positive")
    if run_root.exists():
        raise RemoteRunError("RUN_DIRECTORY_EXISTS: " + str(run_root))
    if _inside(run_root, snapshot):
        raise RemoteRunError("run output must be outside the immutable source snapshot")
    return snapshot, run_root


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _write_record(destination: Path, record: dict[str, object]) -> None:
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = _parse_args()
    try:
        snapshot, run_root = _validate_request(args)
        sys.path.insert(0, str(snapshot / "scripts"))
        from source_manifest import ManifestError, verify_manifest

        try:
            manifest = verify_manifest(snapshot)
        except ManifestError as error:
            raise RemoteRunError("SNAPSHOT_MANIFEST_FAILED: " + str(error)) from error
        observed_source_sha = str(manifest["source_sha256"]).lower()
        if observed_source_sha != args.source_sha.lower():
            raise RemoteRunError(
                "SNAPSHOT_SOURCE_SHA_MISMATCH: expected "
                + args.source_sha.lower()
                + ", observed "
                + observed_source_sha
            )
        run_root.mkdir(parents=True, exist_ok=False)
        evidence_root = run_root / "evidence"
        environment_output = run_root / "environment_preflight.json"
        _run(
            [
                sys.executable,
                "-B",
                "scripts/check_server_environment.py",
                "--output",
                str(environment_output),
            ],
            cwd=snapshot,
        )
        _run(
            [
                sys.executable,
                "-B",
                "scripts/run_benchmark.py",
                "--mode",
                args.mode,
                "--run-id",
                args.run_id,
                "--output-dir",
                str(evidence_root),
                "--dataset-cache-dir",
                args.dataset_cache_dir,
                "--model-cache-dir",
                args.model_cache_dir,
                "--n-estimators",
                str(args.n_estimators),
            ],
            cwd=snapshot,
        )
        record: dict[str, object] = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha256": observed_source_sha,
            "mode": args.mode,
            "snapshot_path": str(snapshot),
            "run_root": str(run_root),
            "evidence_path": str(evidence_root),
            "dataset_cache_dir": str(Path(args.dataset_cache_dir).resolve()),
            "model_cache_dir": str(Path(args.model_cache_dir).resolve()),
            "allow_download": False,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_record(run_root / "runner.json", record)
        _write_record(evidence_root / "runner.json", record)
        print("__PREDREL_RUN_JSON_BEGIN__")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print("__PREDREL_RUN_JSON_END__")
        return 0
    except (RemoteRunError, OSError, subprocess.CalledProcessError) as error:
        print("RUNNER_ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
