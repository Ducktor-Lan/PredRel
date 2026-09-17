"""Deterministic source manifests and runtime provenance records.

No Git repository is required for Week 1.  A source manifest hashes the local
source tree deterministically, while :class:`RunProvenance` records the exact
runtime and every model role used by an extraction.  The latter matters because
TabPFN's readout and train-embedding APIs can require separate fitted models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from hashlib import sha256
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import json
import platform
import re


SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# Keep this list and the canonical hash below byte-for-byte compatible with
# scripts/New-SourceManifest.ps1.  That script is the normal snapshot entry
# point; the Python helpers are intentionally an equivalent implementation.
DEFAULT_EXCLUDE_PATTERNS = (
    ".git/**",
    ".venv/**",
    "**/.venv/**",
    "venv/**",
    "**/venv/**",
    "__pycache__/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    ".mypy_cache/**",
    "**/.mypy_cache/**",
    "reports/evidence/**",
    "reports/week01_teacher_validation.md",
    "provenance/source_manifest.json",
    "provenance/server/**",
    "cache/**",
    "checkpoints/**",
    "runs/**",
    "uploads/**",
    "*.pyc",
    "*.pyo",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
)

# A source snapshot is code and reviewed configuration, not an arbitrary copy
# of the working directory.  An allowlist is deliberately stronger than a
# growing denylist of credential file names: a newly dropped ``secrets/foo``
# file cannot silently enter an SCP archive just because its name was not
# anticipated.
SOURCE_ALLOW_PATTERNS = (
    "README.md",
    "pyproject.toml",
    "configs/**",
    "provenance/.gitkeep",
    "provenance/README.md",
    "reports/week01_teacher_validation.template.md",
    "scripts/**",
    "src/**",
    "tests/**",
)

_CREDENTIAL_CONTENT_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:https?|ssh)://[^/\s:@]+:[^/\s@]+@"),
    re.compile(
        r"(?im)^\s*(?:TABPFN_TOKEN|(?:API|ACCESS|AUTH)[_-]?TOKEN|"
        r"(?:API|SECRET)[_-]?KEY|PASSWORD)\s*(?:=|:)\s*"
        r"(?!\$\{?[A-Za-z_][A-Za-z0-9_]*\}?\s*$|<[^>]+>\s*$|"
        r"REDACTED\s*$|YOUR_[A-Z0-9_]*\s*$|TABPFN_TOKEN\s*$)\S+"
    ),
)


def utc_now_iso() -> str:
    """Return an unambiguous UTC timestamp suitable for provenance JSON."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalised = value.lower()
    if not _SHA256_PATTERN.fullmatch(normalised):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex digest")
    return normalised


def _source_sort_key(relative_path: str) -> tuple[str, str]:
    """Use the same ASCII-safe case-insensitive path order as PowerShell.

    A source manifest is generated on Windows PowerShell but verified by both
    PowerShell and Python.  Sorting merely with Python's default ordinal order
    puts ``README`` before ``configs`` while the Windows shell does the
    opposite, changing an otherwise identical tree digest.
    """

    return (relative_path.lower(), relative_path)


def _matches_patterns(relative_path: PurePosixPath, patterns: Iterable[str]) -> bool:
    """Match source paths case-insensitively, like Windows PowerShell ``-like``.

    The normal workflow generates on Windows but re-verifies in Python.  Folding
    both operands prevents a path whose casing differs only by platform from
    falling through the exclusion/allowlist boundary.
    """

    candidate = relative_path.as_posix().casefold()
    return any(fnmatchcase(candidate, str(pattern).casefold()) for pattern in patterns)


def _is_allowed_source_path(relative_path: PurePosixPath) -> bool:
    return _matches_patterns(relative_path, SOURCE_ALLOW_PATTERNS)


def _assert_no_inline_credentials(path: Path, relative_path: PurePosixPath) -> None:
    """Fail closed when a snapshot-eligible text file contains a likely secret."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Snapshot source file is not UTF-8 text: {relative_path}") from error
    if any(pattern.search(text) for pattern in _CREDENTIAL_CONTENT_PATTERNS):
        raise ValueError(
            "Refusing to include a likely credential in the source snapshot: "
            f"{relative_path}"
        )


def _freeze_mapping(values: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) and key for key in values):
        raise ValueError(f"{field_name} keys must be non-empty strings")
    return MappingProxyType(dict(values))


def sha256_file(path: str | Path, *, chunk_size: int = 1_048_576) -> str:
    """Hash a file's bytes without loading it all into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash non-file path: {file_path}")
    digest = sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFileDigest:
    """Digest and byte size of one source file, relative to manifest root."""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if not self.relative_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("SourceFileDigest.relative_path must be a safe relative POSIX path")
        object.__setattr__(self, "relative_path", path.as_posix())
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256, field_name="sha256"))
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("SourceFileDigest.size_bytes must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Content-addressed description of the locally runnable source tree."""

    root: str
    source_sha256: str
    files: tuple[SourceFileDigest, ...]
    generated_at_utc: str = field(default_factory=utc_now_iso)
    schema_version: int = SCHEMA_VERSION
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("SourceManifest.root must not be empty")
        object.__setattr__(self, "source_sha256", _validate_sha256(self.source_sha256, field_name="source_sha256"))
        files = tuple(self.files)
        paths = tuple(file.relative_path for file in files)
        if paths != tuple(sorted(paths, key=_source_sort_key)):
            raise ValueError("SourceManifest.files must be sorted by relative_path")
        if len(paths) != len(set(paths)):
            raise ValueError("SourceManifest.files must not contain duplicate paths")
        object.__setattr__(self, "files", files)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported source manifest schema_version: {self.schema_version}")
        object.__setattr__(self, "exclude_patterns", tuple(self.exclude_patterns))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "source_root_name": Path(self.root).name,
            "git_commit": None,
            "source_sha256": self.source_sha256,
            "file_count": len(self.files),
            "excluded_patterns": sorted(set(self.exclude_patterns)),
            "files": [file.to_dict() for file in self.files],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def _is_excluded(relative_path: PurePosixPath, patterns: tuple[str, ...]) -> bool:
    return _matches_patterns(relative_path, patterns)


def build_source_manifest(
    root: str | Path, *, extra_exclude_patterns: Iterable[str] = ()
) -> SourceManifest:
    """Build a deterministic source manifest rooted at ``root``.

    Its emitted schema and canonical hash exactly match
    ``scripts/New-SourceManifest.ps1``: for each sorted file, UTF-8 bytes of
    ``path + NUL + lowercase-sha256 + NUL + decimal-bytes + newline`` are fed
    to SHA-256.  This lets either platform produce the same snapshot directory.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Source manifest root must be a directory: {root_path}")
    patterns = tuple(DEFAULT_EXCLUDE_PATTERNS) + tuple(extra_exclude_patterns)
    candidates: list[tuple[PurePosixPath, Path]] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing to snapshot a symbolic-link file: {path}")
        relative_path = PurePosixPath(path.relative_to(root_path).as_posix())
        if _is_excluded(relative_path, patterns):
            continue
        if not _is_allowed_source_path(relative_path):
            raise ValueError(
                "Refusing to snapshot a path outside the Week 1 source allowlist: "
                f"{relative_path.as_posix()}"
            )
        _assert_no_inline_credentials(path, relative_path)
        candidates.append((relative_path, path))

    files = tuple(
        SourceFileDigest(
            relative_path=relative_path.as_posix(),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for relative_path, path in sorted(
            candidates, key=lambda item: _source_sort_key(item[0].as_posix())
        )
    )

    tree_digest = sha256()
    for file in files:
        tree_digest.update(file.relative_path.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(file.sha256.encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(str(file.size_bytes).encode("ascii"))
        tree_digest.update(b"\n")
    return SourceManifest(
        root=str(root_path),
        source_sha256=tree_digest.hexdigest(),
        files=files,
        exclude_patterns=patterns,
    )


def write_source_manifest(
    root: str | Path,
    destination: str | Path | None = None,
    *,
    extra_exclude_patterns: Iterable[str] = (),
) -> SourceManifest:
    """Build and persist a source manifest, returning the in-memory record."""

    root_path = Path(root).resolve()
    output_path = (
        Path(destination)
        if destination is not None
        else root_path / "provenance" / "source_manifest.json"
    )
    output_path = output_path.resolve()
    patterns = tuple(extra_exclude_patterns)
    try:
        manifest_relative_path = output_path.relative_to(root_path).as_posix()
    except ValueError:
        # Like the PowerShell entry point, an external manifest is valid but is
        # not part of the source snapshot in the first place.
        pass
    else:
        if manifest_relative_path not in DEFAULT_EXCLUDE_PATTERNS + patterns:
            patterns += (manifest_relative_path,)
    manifest = build_source_manifest(root_path, extra_exclude_patterns=patterns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def read_source_manifest(
    root: str | Path,
    source: str | Path | None = None,
    *,
    verify_contents: bool = True,
) -> SourceManifest:
    """Load a persisted manifest without modifying its source snapshot.

    Validation runs execute from an immutable, content-addressed snapshot, so
    they must consume the manifest that was packaged with that snapshot rather
    than regenerate and overwrite it.  When ``verify_contents`` is true, the
    source tree is re-hashed before a model is touched.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Source manifest root must be a directory: {root_path}")
    source_path = (
        Path(source) if source is not None else root_path / "provenance" / "source_manifest.json"
    ).resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Source manifest is missing: {source_path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Source manifest is invalid JSON: {source_path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported source manifest schema at {source_path}")
    files_payload = payload.get("files")
    patterns_payload = payload.get("excluded_patterns")
    if not isinstance(files_payload, list) or not isinstance(patterns_payload, list):
        raise ValueError(f"Source manifest is missing files or excluded_patterns: {source_path}")
    try:
        files = tuple(
            SourceFileDigest(
                relative_path=str(item["path"]),
                sha256=str(item["sha256"]),
                size_bytes=int(item["bytes"]),
            )
            for item in files_payload
        )
        manifest = SourceManifest(
            root=str(root_path),
            source_sha256=str(payload["source_sha256"]),
            files=files,
            generated_at_utc=str(payload.get("generated_at_utc", "")),
            schema_version=int(payload["schema_version"]),
            exclude_patterns=tuple(str(pattern) for pattern in patterns_payload),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Source manifest has invalid fields: {source_path}") from error
    if int(payload.get("file_count", -1)) != len(manifest.files):
        raise ValueError(f"Source manifest file_count does not match files: {source_path}")
    if payload.get("git_commit") is not None:
        raise ValueError("Week 1 source manifests must record git_commit as null")

    if verify_contents:
        rebuilt = build_source_manifest(
            root_path,
            extra_exclude_patterns=manifest.exclude_patterns,
        )
        if rebuilt.source_sha256 != manifest.source_sha256 or rebuilt.files != manifest.files:
            raise ValueError(
                "Source manifest does not match the current source tree; regenerate the manifest "
                "and create a new immutable snapshot before running validation"
            )
    return manifest


@dataclass(frozen=True, slots=True)
class ModelRoleProvenance:
    """Configuration and identity of one model used during an extraction.

    Typical roles are ``readout`` and ``embedding``.  Keeping them separate
    makes a shadow embedding model auditable when the readout model's fit mode
    cannot expose train embeddings.
    """

    role: str
    fit_mode: str | None = None
    model_version: str | None = None
    backend: str | None = None
    device: str | None = None
    precision: str | None = None
    n_estimators: int | None = None
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("ModelRoleProvenance.role must be a non-empty string")
        if self.n_estimators is not None and (
            not isinstance(self.n_estimators, int) or self.n_estimators < 1
        ):
            raise ValueError("ModelRoleProvenance.n_estimators must be a positive integer when set")
        if self.checkpoint_sha256 is not None:
            object.__setattr__(
                self,
                "checkpoint_sha256",
                _validate_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256"),
            )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, field_name="metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fit_mode": self.fit_mode,
            "model_version": self.model_version,
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "n_estimators": self.n_estimators,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Complete runtime record attached to a :class:`TeacherOutput`.

    ``git_commit`` defaults to ``None`` by design: this project uses a source
    tree SHA-256 manifest as the version identity.  ``model_roles`` allows a
    readout model and a shadow embedding model to be recorded independently.
    """

    source_sha256: str
    source_manifest_path: str | None = None
    git_commit: str | None = None
    tabpfn_version: str | None = None
    tabpfn_extensions_version: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu: str | None = None
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    run_id: str | None = None
    seed: int | None = None
    model_roles: Mapping[str, ModelRoleProvenance] = field(default_factory=dict)
    extraction_metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sha256", _validate_sha256(self.source_sha256, field_name="source_sha256"))
        if self.git_commit is not None:
            raise ValueError(
                "git_commit must be None for Week 1; use source_sha256/source_manifest_path instead"
            )
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("RunProvenance.seed must be an integer when set")
        if self.checkpoint_sha256 is not None:
            object.__setattr__(
                self,
                "checkpoint_sha256",
                _validate_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256"),
            )

        roles = _freeze_mapping(self.model_roles, field_name="model_roles")
        for name, role in roles.items():
            if not isinstance(role, ModelRoleProvenance):
                raise TypeError("RunProvenance.model_roles values must be ModelRoleProvenance instances")
            if role.role != name:
                raise ValueError(
                    "RunProvenance.model_roles mapping key must match ModelRoleProvenance.role "
                    f"({name!r} != {role.role!r})"
                )
        object.__setattr__(self, "model_roles", roles)
        object.__setattr__(
            self,
            "extraction_metadata",
            _freeze_mapping(self.extraction_metadata, field_name="extraction_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "source_manifest_path": self.source_manifest_path,
            "git_commit": self.git_commit,
            "tabpfn_version": self.tabpfn_version,
            "tabpfn_extensions_version": self.tabpfn_extensions_version,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "gpu": self.gpu,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "run_id": self.run_id,
            "seed": self.seed,
            "model_roles": {name: role.to_dict() for name, role in self.model_roles.items()},
            "extraction_metadata": dict(self.extraction_metadata),
            "timestamp_utc": self.timestamp_utc,
        }


def collect_runtime_provenance(
    *,
    source_manifest: SourceManifest,
    run_id: str | None = None,
    seed: int | None = None,
    checkpoint_id: str | None = None,
    checkpoint_sha256: str | None = None,
    model_roles: Mapping[str, ModelRoleProvenance] | None = None,
    extraction_metadata: Mapping[str, Any] | None = None,
) -> RunProvenance:
    """Collect import-safe runtime facts without requiring TabPFN or PyTorch."""

    def package_version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    torch_version: str | None = None
    cuda_version: str | None = None
    gpu: str | None = None
    try:
        import torch  # type: ignore[import-not-found]

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return RunProvenance(
        source_sha256=source_manifest.source_sha256,
        source_manifest_path="provenance/source_manifest.json",
        tabpfn_version=package_version("tabpfn"),
        tabpfn_extensions_version=package_version("tabpfn-extensions"),
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        gpu=gpu,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        run_id=run_id,
        seed=seed,
        model_roles={} if model_roles is None else model_roles,
        extraction_metadata={} if extraction_metadata is None else extraction_metadata,
    )
