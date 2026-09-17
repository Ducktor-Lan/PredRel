"""Create, verify, and archive a credential-safe immutable PredRel source tree."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


SCHEMA_VERSION = 1
MANIFEST_RELATIVE = PurePosixPath("provenance/source_manifest.json")
ALLOWED_EXACT = {
    PurePosixPath("README.md"),
    PurePosixPath("LICENSE"),
    PurePosixPath(".gitignore"),
    PurePosixPath(".gitattributes"),
    PurePosixPath("pyproject.toml"),
    PurePosixPath("provenance/README.md"),
        PurePosixPath("provenance/provenance.json"),
    PurePosixPath("provenance/dataset_manifest_benchmark.json"),
    PurePosixPath("provenance/dataset_manifest_student.json"),
    PurePosixPath("provenance/dataset_manifest_ablation.json"),
    PurePosixPath("provenance/dataset_manifest_context.json"),
    PurePosixPath("provenance/MODULE_SOURCES.md"),
    PurePosixPath("reports/benchmark.template.md"),
}
ALLOWED_PREFIXES = ("configs", "scripts", "src", "tests", "docs")
EXCLUDED_DIRECTORY_NAMES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_PREFIXES = ("reports/evidence", "cache", "checkpoints", "runs", "uploads", "evidence", "history", "diagnosis")
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".zip", ".npz")
CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:https?|ssh)://[^/\s:@]+:[^/\s@]+@"),
    re.compile(
        r"(?im)^\s*(?:TABPFN_TOKEN|(?:API|ACCESS|AUTH)[_-]?TOKEN|(?:API|SECRET)[_-]?KEY|PASSWORD)\s*(?:=|:)\s*"
        r"(?!\$\{?[A-Za-z_][A-Za-z0-9_]*\}?\s*$|<[^>]+>\s*$|REDACTED\s*$|YOUR_[A-Z0-9_]*\s*$|TABPFN_TOKEN\s*$)\S+"
    ),
)


class ManifestError(ValueError):
    """Raised when a source tree is not safe to snapshot."""


def _root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ManifestError(f"source root is not a directory: {root}")
    return root


def _relative(root: Path, path: Path) -> PurePosixPath:
    try:
        return PurePosixPath(path.resolve().relative_to(root).as_posix())
    except ValueError as error:
        raise ManifestError(f"path escapes source root: {path}") from error


def _excluded(relative: PurePosixPath) -> bool:
    text = relative.as_posix()
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
        return True
    if relative == MANIFEST_RELATIVE or text == ".env" or text.startswith(".env."):
        return True
    if any(text == prefix or text.startswith(prefix + "/") for prefix in EXCLUDED_PREFIXES):
        return True
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return relative.suffix.lower() in {".pem", ".key"} or relative.name.lower().startswith("id_rsa")


def _allowed(relative: PurePosixPath) -> bool:
    return relative in ALLOWED_EXACT or bool(relative.parts) and relative.parts[0] in ALLOWED_PREFIXES


def _assert_safe_text(path: Path, relative: PurePosixPath) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ManifestError(f"source file is not strict UTF-8: {relative}") from error
    for pattern in CREDENTIAL_PATTERNS:
        if pattern.search(text):
            raise ManifestError(f"refusing to include likely credential material: {relative}")


def _record(root: Path, path: Path) -> dict[str, object]:
    relative = _relative(root, path)
    if path.is_symlink():
        raise ManifestError(f"refusing symbolic-link source file: {relative}")
    _assert_safe_text(path, relative)
    payload = path.read_bytes()
    return {"path": relative.as_posix(), "sha256": sha256(payload).hexdigest(), "bytes": len(payload)}


def _sort_key(record: MappingLike) -> tuple[str, str]:
    path = str(record["path"])
    return (path.lower(), path)


class MappingLike(dict[str, object]):
    """Narrow helper type used only to keep manifest sorting explicit."""


def _source_hash(records: list[dict[str, object]]) -> str:
    digest = sha256()
    for record in sorted(records, key=_sort_key):
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).lower().encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(record["bytes"])).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(root: str | Path) -> dict[str, object]:
    source_root = _root(root)
    records: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file():
            continue
        relative = _relative(source_root, path)
        if _excluded(relative):
            continue
        if not _allowed(relative):
            raise ManifestError(f"source path is outside the PredRel allowlist: {relative}")
        records.append(_record(source_root, path))
    records.sort(key=_sort_key)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_root_name": source_root.name,
        "git_commit": None,
        "source_sha256": _source_hash(records),
        "file_count": len(records),
        "excluded_patterns": [
            "generated evidence",
            "datasets and caches",
            "checkpoints",
            "virtual environments",
            "credentials",
            "source manifest",
        ],
        "files": records,
    }


def _manifest_path(root: Path) -> Path:
    return root / Path(*MANIFEST_RELATIVE.parts)


def write_manifest(root: str | Path) -> dict[str, object]:
    source_root = _root(root)
    manifest = build_manifest(source_root)
    destination = _manifest_path(source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _read_manifest(root: Path) -> dict[str, object]:
    path = _manifest_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"source manifest is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"source manifest is invalid JSON: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("source manifest has an unsupported schema")
    if not isinstance(payload.get("files"), list) or not isinstance(payload.get("source_sha256"), str):
        raise ManifestError("source manifest lacks files or source_sha256")
    return payload


def verify_manifest(root: str | Path) -> dict[str, object]:
    source_root = _root(root)
    persisted = _read_manifest(source_root)
    rebuilt = build_manifest(source_root)
    if persisted.get("file_count") != len(persisted["files"]):
        raise ManifestError("manifest file_count does not match its entries")
    if persisted["files"] != rebuilt["files"]:
        raise ManifestError("manifest file list differs from current source tree")
    if persisted.get("source_sha256") != rebuilt["source_sha256"]:
        raise ManifestError("manifest source hash differs from current source tree")
    return {
        "valid": True,
        "source_sha256": rebuilt["source_sha256"],
        "file_count": rebuilt["file_count"],
        "manifest_path": str(_manifest_path(source_root)),
    }


def archive_manifest(root: str | Path, archive: str | Path) -> dict[str, object]:
    source_root = _root(root)
    verification = verify_manifest(source_root)
    destination = Path(archive).resolve()
    if destination.exists():
        raise ManifestError(f"refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(source_root)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for record in manifest["files"]:
            relative = PurePosixPath(str(record["path"]))
            handle.write(source_root / Path(*relative.parts), arcname=relative.as_posix())
        handle.write(_manifest_path(source_root), arcname=MANIFEST_RELATIVE.as_posix())
    return {**verification, "archive_path": str(destination)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", required=True)
    archive = subparsers.add_parser("archive")
    archive.add_argument("--root", required=True)
    archive.add_argument("--archive", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "write":
            result = write_manifest(args.root)
        elif args.command == "verify":
            result = verify_manifest(args.root)
        else:
            result = archive_manifest(args.root, args.archive)
    except ManifestError as error:
        print(f"SOURCE_MANIFEST_ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

