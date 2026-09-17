"""Small deterministic fixtures for teacher-contract validation.

Query labels live beside, rather than inside, ``QuerySet``.  This preserves the
same no-leakage interface used by a real teacher call while still allowing test
code to evaluate predictions afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .contracts import ContractError, QuerySet, SupportSet, validate_support_query


def _readonly_labels(values: NDArray[np.generic]) -> NDArray[np.generic]:
    copied = np.ascontiguousarray(values).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class FixedFixture:
    """A fixed teacher input plus held-out labels kept outside ``QuerySet``."""

    dataset_id: str
    support: SupportSet
    query: QuerySet
    query_labels: NDArray[np.generic]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_support_query(self.support, self.query)
        labels = np.asarray(self.query_labels)
        if labels.ndim != 1 or labels.shape[0] != self.query.n_samples:
            raise ContractError(
                "FixedFixture.query_labels must be a 1-D held-out label vector matching QuerySet rows"
            )
        if labels.dtype.kind in {"f", "c"} and not np.isfinite(labels).all():
            raise ContractError("FixedFixture.query_labels must contain finite numeric values")
        if labels.dtype.kind == "O" and any(value is None for value in labels.tolist()):
            raise ContractError("FixedFixture.query_labels must not contain None")
        object.__setattr__(self, "query_labels", _readonly_labels(labels))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _stratified_fixed_split(
    labels: NDArray[np.int64], *, support_per_class: int, query_per_class: int, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    support_indices: list[int] = []
    query_indices: list[int] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        required = support_per_class + query_per_class
        if class_indices.size < required:
            raise ContractError(f"Class {label!r} has {class_indices.size} rows, but {required} are required")
        selected = rng.permutation(class_indices)[:required]
        support_indices.extend(selected[:support_per_class].tolist())
        query_indices.extend(selected[support_per_class:].tolist())
    return (
        rng.permutation(np.asarray(support_indices, dtype=np.int64)),
        rng.permutation(np.asarray(query_indices, dtype=np.int64)),
    )


def make_synthetic_binary_fixture(seed: int = 17) -> FixedFixture:
    """Return a deterministic 48x6 binary fixture with 32 support / 16 query rows."""

    rng = np.random.default_rng(seed)
    n_per_class = 24
    centers = np.asarray(
        [
            [-1.25, -0.85, -0.55, -0.25, -0.75, -0.45],
            [1.25, 0.85, 0.55, 0.25, 0.75, 0.45],
        ],
        dtype=np.float64,
    )
    covariance_scale = np.asarray([0.55, 0.80, 0.65, 0.95, 0.60, 0.75], dtype=np.float64)
    class_zero = rng.normal(loc=centers[0], scale=covariance_scale, size=(n_per_class, 6))
    class_one = rng.normal(loc=centers[1], scale=covariance_scale, size=(n_per_class, 6))
    features = np.vstack((class_zero, class_one))
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), n_per_class)
    support_indices, query_indices = _stratified_fixed_split(
        labels, support_per_class=16, query_per_class=8, rng=rng
    )
    sample_ids = tuple(f"synthetic-{index:03d}" for index in range(features.shape[0]))
    return FixedFixture(
        dataset_id="synthetic_binary_v1",
        support=SupportSet(
            features=features[support_indices],
            labels=labels[support_indices],
            sample_ids=tuple(sample_ids[index] for index in support_indices),
        ),
        query=QuerySet(
            features=features[query_indices],
            sample_ids=tuple(sample_ids[index] for index in query_indices),
        ),
        query_labels=labels[query_indices],
        metadata=MappingProxyType(
            {
                "seed": seed,
                "n_total": 48,
                "n_features": 6,
                "n_support": 32,
                "n_query": 16,
                "split": "stratified_16_support_8_query_per_class",
            }
        ),
    )


def make_breast_cancer_fixture(seed: int = 17) -> FixedFixture:
    """Return a fixed stratified 128-support / 32-query Breast Cancer fixture.

    ``scikit-learn`` is imported lazily so contracts and provenance can run in a
    minimal environment.  The split is deterministic for a fixed seed.
    """

    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import StratifiedShuffleSplit
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "make_breast_cancer_fixture requires scikit-learn; install project dependencies first"
        ) from error

    dataset = load_breast_cancer()
    features = np.asarray(dataset.data, dtype=np.float64)
    labels = np.asarray(dataset.target, dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=128,
        test_size=32,
        random_state=seed,
    )
    support_indices, query_indices = next(splitter.split(features, labels))
    sample_ids = tuple(f"breast-cancer-{index:03d}" for index in range(features.shape[0]))
    return FixedFixture(
        dataset_id="sklearn_breast_cancer_v1",
        support=SupportSet(
            features=features[support_indices],
            labels=labels[support_indices],
            sample_ids=tuple(sample_ids[index] for index in support_indices),
        ),
        query=QuerySet(
            features=features[query_indices],
            sample_ids=tuple(sample_ids[index] for index in query_indices),
        ),
        query_labels=labels[query_indices],
        metadata=MappingProxyType(
            {
                "seed": seed,
                "dataset_name": "sklearn.datasets.load_breast_cancer",
                "n_total": int(features.shape[0]),
                "n_features": int(features.shape[1]),
                "n_support": 128,
                "n_query": 32,
                "split": "StratifiedShuffleSplit(train_size=128, test_size=32)",
            }
        ),
    )
