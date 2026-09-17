"""Frozen real-dataset loading, locking, and deterministic split construction.

Dataset caps live in the manifest (``5 <= n_datasets <= 30``); the same
contract serves all real-data stages (6/9/20 datasets).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


class DatasetError(RuntimeError):
    """Raised when a frozen dataset cannot be safely materialized or split."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the complete benchmark contract independently of file whitespace."""

    return sha256(_canonical_json(dict(payload))).hexdigest()


def _safe_dataset_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
        raise DatasetError(f"dataset_id has unsupported characters: {value!r}")
    return value


def load_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"dataset manifest is missing: {source}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"dataset manifest is not valid JSON: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DatasetError("dataset manifest has an unsupported schema")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not 5 <= len(datasets) <= 30:
        raise DatasetError("dataset manifest must freeze between 5 and 30 datasets")
    ids = [_safe_dataset_id(str(item.get("dataset_id", ""))) for item in datasets if isinstance(item, dict)]
    if len(ids) != len(datasets) or len(set(ids)) != len(ids):
        raise DatasetError("dataset manifest has missing or duplicate dataset IDs")
    split = payload.get("split")
    if not isinstance(split, dict) or not isinstance(split.get("seeds"), list) or not split["seeds"]:
        raise DatasetError("dataset manifest lacks a non-empty split seed list")
    if not all(isinstance(seed, int) and seed >= 0 for seed in split["seeds"]):
        raise DatasetError("split seeds must be non-negative integers")
    if len(set(split["seeds"])) != len(split["seeds"]):
        raise DatasetError("split seeds must be unique")
    for name in ("max_support_size", "max_query_size", "min_support_per_class", "min_query_per_class"):
        if not isinstance(split.get(name), int) or split[name] < 1:
            raise DatasetError(f"split {name} must be a positive integer")
    train_fraction = split.get("train_fraction")
    if not isinstance(train_fraction, (float, int)) or not 0.0 < float(train_fraction) < 1.0:
        raise DatasetError("split train_fraction must be strictly between zero and one")
    return payload, manifest_sha256(payload)


@dataclass(frozen=True, slots=True)
class RealDataset:
    dataset_id: str
    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    sample_ids: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[0] == 0:
            raise DatasetError("features must be a non-empty 2-D matrix")
        if self.labels.shape != (self.features.shape[0],):
            raise DatasetError("labels do not align with features")
        if len(self.sample_ids) != self.features.shape[0] or len(set(self.sample_ids)) != len(self.sample_ids):
            raise DatasetError("sample IDs do not uniquely align with features")
        if not np.isfinite(self.features).all():
            raise DatasetError("features must be finite")
        if np.unique(self.labels).size < 2:
            raise DatasetError("a real classification dataset needs at least two classes")


@dataclass(frozen=True, slots=True)
class ExperimentSplit:
    dataset_id: str
    seed: int
    holdout_ids: tuple[str, ...]
    support_features: NDArray[np.float64]
    support_labels: NDArray[np.int64]
    support_ids: tuple[str, ...]
    query_features: NDArray[np.float64]
    query_labels: NDArray[np.int64]
    query_ids: tuple[str, ...]

    @property
    def split_id(self) -> str:
        return f"{self.dataset_id}:seed-{self.seed}:support-{len(self.support_ids)}:query-{len(self.query_ids)}"


def _encode_labels(values: object) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise DatasetError("target must be one-dimensional")
    labels = [str(value) for value in raw.tolist()]
    if any(not value or value.lower() in {"nan", "none", "?"} for value in labels):
        raise DatasetError("target contains missing labels")
    mapping = {value: position for position, value in enumerate(sorted(set(labels)))}
    return np.asarray([mapping[value] for value in labels], dtype=np.int64)


def _content_sha256(features: NDArray[np.float64], labels: NDArray[np.int64], dataset_id: str) -> str:
    digest = sha256()
    digest.update(dataset_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(features, dtype=np.float64).tobytes(order="C"))
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes(order="C"))
    return digest.hexdigest()


def _load_source(spec: Mapping[str, Any], *, cache_dir: Path, allow_download: bool) -> tuple[object, object, dict[str, Any]]:
    source = spec.get("source")
    if not isinstance(source, dict):
        raise DatasetError(f"{spec.get('dataset_id')}: source must be an object")
    kind = source.get("kind")
    if kind == "sklearn_builtin":
        try:
            from sklearn import datasets as sklearn_datasets
        except ImportError as error:
            raise DatasetError("scikit-learn is required to load frozen built-in datasets") from error
        loader_name = str(source.get("loader", ""))
        loaders = {
            "load_breast_cancer": sklearn_datasets.load_breast_cancer,
            "load_wine": sklearn_datasets.load_wine,
            "load_digits": sklearn_datasets.load_digits,
        }
        if loader_name not in loaders:
            raise DatasetError(f"{spec.get('dataset_id')}: unsupported built-in loader {loader_name!r}")
        bundle = loaders[loader_name]()
        return bundle.data, bundle.target, {"kind": kind, "loader": loader_name, "source_version": source.get("source_version")}
    if kind == "openml":
        if not allow_download:
            raise DatasetError(
                f"{spec.get('dataset_id')}: OpenML cache is absent. Run the explicit dataset-cache bootstrap first."
            )
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as error:
            raise DatasetError("scikit-learn is required to fetch frozen OpenML datasets") from error
        data_id = source.get("data_id")
        if not isinstance(data_id, int) or data_id <= 0:
            raise DatasetError(f"{spec.get('dataset_id')}: OpenML data_id is invalid")
        download_home = cache_dir / "sklearn-openml-downloads"
        bundle = fetch_openml(data_id=data_id, data_home=str(download_home), as_frame=False)
        details = getattr(bundle, "details", None)
        if not isinstance(details, Mapping):
            raise DatasetError(f"{spec.get('dataset_id')}: fetch_openml did not expose immutable dataset metadata")
        observed_id = details.get("id", details.get("data_id"))
        observed_name = details.get("name")
        observed_version = details.get("version")
        observed_md5 = details.get("md5_checksum")
        expected_values = {
            "data_id": data_id,
            "name": source.get("name"),
            "version": source.get("version"),
            "openml_md5": source.get("openml_md5"),
        }
        observed_values = {
            "data_id": observed_id,
            "name": observed_name,
            "version": observed_version,
            "openml_md5": observed_md5,
        }
        mismatches = {
            key: {"expected": expected_values[key], "observed": observed_values[key]}
            for key in expected_values
            if str(observed_values[key]) != str(expected_values[key])
        }
        if mismatches:
            raise DatasetError(f"{spec.get('dataset_id')}: OpenML immutable metadata mismatch: {mismatches}")
        return (
            bundle.data,
            bundle.target,
            {
                "kind": kind,
                "data_id": data_id,
                "name": source.get("name"),
                "version": source.get("version"),
                "openml_md5": source.get("openml_md5"),
                "observed_openml_details": {
                    "data_id": observed_id,
                    "name": observed_name,
                    "version": observed_version,
                    "openml_md5": observed_md5,
                },
            },
        )
    raise DatasetError(f"{spec.get('dataset_id')}: unsupported source kind {kind!r}")


def _validate_and_normalize(
    spec: Mapping[str, Any], features: object, target: object, source_metadata: Mapping[str, Any]
) -> RealDataset:
    dataset_id = _safe_dataset_id(str(spec.get("dataset_id", "")))
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{dataset_id}: numeric-only feature conversion failed") from error
    if matrix.ndim != 2:
        raise DatasetError(f"{dataset_id}: source feature matrix is not two-dimensional")
    labels = _encode_labels(target)
    if not np.isfinite(matrix).all():
        raise DatasetError(f"{dataset_id}: source contains non-finite features; frozen policy forbids imputation")
    expected_rows = int(spec.get("expected_rows", -1))
    expected_features = int(spec.get("expected_features", -1))
    expected_classes = int(spec.get("expected_classes", -1))
    observed_classes = int(np.unique(labels).size)
    if matrix.shape != (expected_rows, expected_features):
        raise DatasetError(
            f"{dataset_id}: frozen shape mismatch; expected {(expected_rows, expected_features)}, observed {matrix.shape}"
        )
    if observed_classes != expected_classes:
        raise DatasetError(
            f"{dataset_id}: frozen class-count mismatch; expected {expected_classes}, observed {observed_classes}"
        )
    canonical_features = np.ascontiguousarray(matrix, dtype=np.float64)
    canonical_labels = np.ascontiguousarray(labels, dtype=np.int64)
    sample_ids = tuple(f"{dataset_id}:row:{index}" for index in range(canonical_features.shape[0]))
    content_hash = _content_sha256(canonical_features, canonical_labels, dataset_id)
    metadata = {
        "dataset_id": dataset_id,
        "source": dict(source_metadata),
        "rows": int(canonical_features.shape[0]),
        "features": int(canonical_features.shape[1]),
        "classes": observed_classes,
        "content_sha256": content_hash,
    }
    return RealDataset(dataset_id, canonical_features, canonical_labels, sample_ids, metadata)


def _cache_paths(cache_dir: Path, dataset_id: str) -> tuple[Path, Path]:
    return cache_dir / f"{dataset_id}.npz", cache_dir / f"{dataset_id}.metadata.json"


def _load_cached(spec: Mapping[str, Any], cache_dir: Path) -> RealDataset | None:
    dataset_id = _safe_dataset_id(str(spec.get("dataset_id", "")))
    archive, metadata_path = _cache_paths(cache_dir, dataset_id)
    if not archive.exists() and not metadata_path.exists():
        return None
    if not archive.is_file() or not metadata_path.is_file():
        raise DatasetError(f"{dataset_id}: incomplete cache entry")
    try:
        with np.load(archive, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float64)
            labels = np.asarray(payload["labels"], dtype=np.int64)
            ids = tuple(str(value) for value in payload["sample_ids"].tolist())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise DatasetError(f"{dataset_id}: cached data is unreadable") from error
    source_metadata = metadata.get("source")
    if not isinstance(source_metadata, dict):
        raise DatasetError(f"{dataset_id}: cached metadata lacks source provenance")
    declared_source = spec.get("source")
    if not isinstance(declared_source, dict):
        raise DatasetError(f"{dataset_id}: frozen manifest source is invalid")
    mismatches = {
        key: {"expected": value, "observed": source_metadata.get(key)}
        for key, value in declared_source.items()
        if source_metadata.get(key) != value
    }
    if mismatches:
        raise DatasetError(f"{dataset_id}: cached source provenance differs from frozen manifest: {mismatches}")
    loaded = _validate_and_normalize(spec, features, labels, source_metadata)
    if metadata.get("content_sha256") != loaded.metadata["content_sha256"]:
        raise DatasetError(f"{dataset_id}: cached metadata content hash does not match its arrays")
    if ids != loaded.sample_ids:
        raise DatasetError(f"{dataset_id}: cached sample IDs no longer match frozen row order")
    return loaded


def _write_cache(dataset: RealDataset, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive, metadata_path = _cache_paths(cache_dir, dataset.dataset_id)
    if archive.exists() or metadata_path.exists():
        raise DatasetError(f"{dataset.dataset_id}: refuse to overwrite a cache entry")
    np.savez_compressed(
        archive,
        features=dataset.features,
        labels=dataset.labels,
        sample_ids=np.asarray(dataset.sample_ids, dtype=np.str_),
    )
    metadata_path.write_text(json.dumps(dataset.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize_dataset(spec: Mapping[str, Any], *, cache_dir: str | Path, allow_download: bool) -> RealDataset:
    """Load one exact cached dataset or explicitly retrieve and lock it."""

    root = Path(cache_dir).resolve()
    cached = _load_cached(spec, root)
    if cached is not None:
        return cached
    features, target, source_metadata = _load_source(spec, cache_dir=root, allow_download=allow_download)
    dataset = _validate_and_normalize(spec, features, target, source_metadata)
    _write_cache(dataset, root)
    return dataset


def materialize_manifest(
    manifest: Mapping[str, Any], *, cache_dir: str | Path, allow_download: bool, dataset_ids: Iterable[str] | None = None
) -> list[RealDataset]:
    """Materialize exactly the selected frozen IDs in manifest order."""

    selected = None if dataset_ids is None else set(dataset_ids)
    all_ids = [str(item["dataset_id"]) for item in manifest["datasets"]]
    if selected is not None and not selected.issubset(set(all_ids)):
        raise DatasetError(f"requested dataset IDs are outside the frozen manifest: {sorted(selected.difference(all_ids))}")
    result = [
        materialize_dataset(item, cache_dir=cache_dir, allow_download=allow_download)
        for item in manifest["datasets"]
        if selected is None or str(item["dataset_id"]) in selected
    ]
    if not result:
        raise DatasetError("no frozen datasets were selected")
    return result


def _lock_payload(manifest: Mapping[str, Any], manifest_hash: str, datasets: Iterable[RealDataset]) -> dict[str, Any]:
    values = list(datasets)
    return {
        "schema_version": 1,
        "manifest_id": manifest.get("manifest_id"),
        "dataset_manifest_sha256": manifest_hash,
        "datasets": [dataset.metadata for dataset in values],
    }


def ensure_dataset_lock(
    manifest: Mapping[str, Any], manifest_hash: str, datasets: Iterable[RealDataset], *, cache_dir: str | Path
) -> dict[str, Any]:
    """Create or verify a non-mutating data lock for a complete frozen manifest."""

    expected_ids = [str(item["dataset_id"]) for item in manifest["datasets"]]
    values = list(datasets)
    if [dataset.dataset_id for dataset in values] != expected_ids:
        raise DatasetError("a dataset lock can only be made for the complete manifest in manifest order")
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "dataset_lock.json"
    proposed = _lock_payload(manifest, manifest_hash, values)
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"dataset lock is unreadable: {destination}") from error
        if existing != proposed:
            raise DatasetError("dataset lock differs from the frozen manifest or cached content; refuse mutation")
        return existing
    destination.write_text(json.dumps(proposed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return proposed


def verify_dataset_lock(
    manifest: Mapping[str, Any], manifest_hash: str, datasets: Iterable[RealDataset], *, cache_dir: str | Path
) -> dict[str, Any]:
    """Require a pre-existing lock that exactly matches every cached dataset."""

    expected_ids = [str(item["dataset_id"]) for item in manifest["datasets"]]
    values = list(datasets)
    if [dataset.dataset_id for dataset in values] != expected_ids:
        raise DatasetError("a full run requires the complete manifest in manifest order")
    destination = Path(cache_dir).resolve() / "dataset_lock.json"
    if not destination.is_file():
        raise DatasetError("full experiment requires a pre-existing dataset lock; run explicit dataset bootstrap first")
    proposed = _lock_payload(manifest, manifest_hash, values)
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"dataset lock is unreadable: {destination}") from error
    if existing != proposed:
        raise DatasetError("dataset lock differs from frozen manifest or cached content; refuse experiment")
    return existing


def _allocate(
    labels: NDArray[np.int64], total: int, capacity: NDArray[np.int64], *, minimum_per_class: int
) -> dict[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    if minimum_per_class < 1:
        raise DatasetError("minimum_per_class must be positive")
    if np.any(capacity < minimum_per_class):
        raise DatasetError("a frozen split cannot satisfy the per-class minimum")
    if total < len(classes) * minimum_per_class:
        raise DatasetError("requested split is too small to satisfy every class minimum")
    if total > int(capacity.sum()):
        raise DatasetError("requested split exceeds available samples")
    proportions = capacity.astype(np.float64) / float(capacity.sum())
    raw = proportions * float(total)
    allocated = np.floor(raw).astype(np.int64)
    allocated = np.maximum(allocated, minimum_per_class)
    allocated = np.minimum(allocated, capacity)
    while int(allocated.sum()) < total:
        candidates = [index for index in range(len(classes)) if allocated[index] < capacity[index]]
        if not candidates:
            raise DatasetError("cannot allocate requested split across classes")
        index = max(candidates, key=lambda item: (raw[item] - allocated[item], int(capacity[item] - allocated[item]), -int(classes[item])))
        allocated[index] += 1
    while int(allocated.sum()) > total:
        candidates = [index for index in range(len(classes)) if allocated[index] > minimum_per_class]
        if not candidates:
            raise DatasetError("cannot reduce split without dropping a class")
        index = min(candidates, key=lambda item: (raw[item] - allocated[item], int(classes[item])))
        allocated[index] -= 1
    return {int(label): int(value) for label, value in zip(classes.tolist(), allocated.tolist(), strict=True)}


def make_stratified_split(
    dataset: RealDataset,
    *,
    seed: int,
    max_support_size: int,
    max_query_size: int,
    train_fraction: float,
    min_support_per_class: int,
    min_query_per_class: int,
) -> ExperimentSplit:
    """Create a deterministic stratified holdout, then draw query/train subsets.

    The full holdout is fixed before the query cap is applied.  This matters on
    larger datasets: unused holdout rows must stay held out rather than being
    recycled into support simply because the query sample is capped.
    """

    if (
        seed < 0
        or max_support_size < 1
        or max_query_size < 1
        or min_support_per_class < 1
        or min_query_per_class < 1
        or not 0.0 < train_fraction < 1.0
    ):
        raise DatasetError("split arguments are invalid")
    labels = dataset.labels
    classes, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    holdout_indices: list[int] = []
    support_indices: list[int] = []
    query_indices: list[int] = []
    holdout_capacity = counts - min_support_per_class
    if np.any(holdout_capacity < min_query_per_class):
        raise DatasetError("dataset cannot satisfy frozen train/holdout class minima")
    desired_holdout = max(len(labels) - int(np.floor(len(labels) * train_fraction)), len(classes) * min_query_per_class)
    holdout_quota = _allocate(
        labels,
        desired_holdout,
        holdout_capacity,
        minimum_per_class=min_query_per_class,
    )
    holdout_counts = np.asarray([holdout_quota[int(label)] for label in classes], dtype=np.int64)
    query_total = min(max_query_size, int(holdout_counts.sum()))
    query_quota = _allocate(
        labels,
        query_total,
        holdout_counts,
        minimum_per_class=min_query_per_class,
    )
    train_counts = counts - holdout_counts
    support_total = min(max_support_size, int(train_counts.sum()))
    support_quota = _allocate(
        labels,
        support_total,
        train_counts,
        minimum_per_class=min_support_per_class,
    )
    for class_label in classes.tolist():
        members = np.flatnonzero(labels == class_label)
        shuffled = rng.permutation(members)
        full_holdout_count = holdout_quota[int(class_label)]
        query_count = query_quota[int(class_label)]
        support_count = support_quota[int(class_label)]
        holdout_indices.extend(shuffled[:full_holdout_count].tolist())
        query_indices.extend(shuffled[:query_count].tolist())
        support_indices.extend(shuffled[full_holdout_count : full_holdout_count + support_count].tolist())
    support = np.asarray(support_indices, dtype=np.int64)
    query = np.asarray(query_indices, dtype=np.int64)
    holdout = np.asarray(holdout_indices, dtype=np.int64)
    if np.intersect1d(support, holdout).size:
        raise DatasetError("support selection overlaps the frozen holdout")
    if np.setdiff1d(query, holdout).size:
        raise DatasetError("query selection is outside the frozen holdout")
    return ExperimentSplit(
        dataset_id=dataset.dataset_id,
        seed=seed,
        holdout_ids=tuple(dataset.sample_ids[index] for index in holdout.tolist()),
        support_features=np.ascontiguousarray(dataset.features[support]),
        support_labels=np.ascontiguousarray(dataset.labels[support]),
        support_ids=tuple(dataset.sample_ids[index] for index in support.tolist()),
        query_features=np.ascontiguousarray(dataset.features[query]),
        query_labels=np.ascontiguousarray(dataset.labels[query]),
        query_ids=tuple(dataset.sample_ids[index] for index in query.tolist()),
    )

