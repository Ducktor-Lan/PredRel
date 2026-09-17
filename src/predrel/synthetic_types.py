"""Synthetic fixture contracts (Stage 2; query labels eval-only).

Query labels are retained solely for evaluation.  The Teacher bridge receives
only ``query_features`` and ``query_ids`` and therefore cannot leak labels into
the support context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SyntheticContractError(ValueError):
    """Raised when a generated synthetic fixture is internally inconsistent."""


def _features(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise SyntheticContractError(f"{name} must be a non-empty 2-D numeric matrix")
    if not np.isfinite(array).all():
        raise SyntheticContractError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _labels(values: ArrayLike, *, n_rows: int, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise SyntheticContractError(f"{name} must be a length-{n_rows} vector")
    if not np.issubdtype(array.dtype, np.integer):
        raise SyntheticContractError(f"{name} must use an integer dtype")
    return np.ascontiguousarray(array, dtype=np.int64)


def _ids(values: tuple[str, ...], *, n_rows: int, name: str) -> tuple[str, ...]:
    if len(values) != n_rows or any(not isinstance(value, str) or not value for value in values):
        raise SyntheticContractError(f"{name} must contain {n_rows} non-empty string IDs")
    if len(set(values)) != len(values):
        raise SyntheticContractError(f"{name} must be unique")
    return tuple(values)


def _groups(values: Mapping[str, ArrayLike], *, n_rows: int, name: str) -> dict[str, NDArray[np.generic]]:
    normalised: dict[str, NDArray[np.generic]] = {}
    for key, raw in values.items():
        if not isinstance(key, str) or not key:
            raise SyntheticContractError(f"{name} group keys must be non-empty strings")
        array = np.asarray(raw)
        if array.ndim != 1 or array.shape[0] != n_rows:
            raise SyntheticContractError(f"{name}[{key!r}] must be a length-{n_rows} vector")
        normalised[key] = np.ascontiguousarray(array)
    return normalised


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    """One support/query task plus latent annotations for post-hoc evaluation."""

    name: str
    description: str
    support_features: ArrayLike
    support_labels: ArrayLike
    support_ids: tuple[str, ...]
    query_features: ArrayLike
    query_labels: ArrayLike
    query_ids: tuple[str, ...]
    support_groups: Mapping[str, ArrayLike] = field(default_factory=dict)
    query_groups: Mapping[str, ArrayLike] = field(default_factory=dict)
    expectations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SyntheticContractError("fixture name must be a non-empty string")
        support_features = _features(self.support_features, name="support_features")
        query_features = _features(self.query_features, name="query_features")
        if support_features.shape[1] != query_features.shape[1]:
            raise SyntheticContractError("support and query feature dimensions must match")
        support_labels = _labels(self.support_labels, n_rows=support_features.shape[0], name="support_labels")
        query_labels = _labels(self.query_labels, n_rows=query_features.shape[0], name="query_labels")
        support_ids = _ids(self.support_ids, n_rows=support_features.shape[0], name="support_ids")
        query_ids = _ids(self.query_ids, n_rows=query_features.shape[0], name="query_ids")
        overlap = set(support_ids).intersection(query_ids)
        if overlap:
            raise SyntheticContractError("support_ids and query_ids must not overlap")
        object.__setattr__(self, "support_features", support_features)
        object.__setattr__(self, "support_labels", support_labels)
        object.__setattr__(self, "query_features", query_features)
        object.__setattr__(self, "query_labels", query_labels)
        object.__setattr__(self, "support_ids", support_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "support_groups", _groups(self.support_groups, n_rows=support_features.shape[0], name="support_groups"))
        object.__setattr__(self, "query_groups", _groups(self.query_groups, n_rows=query_features.shape[0], name="query_groups"))
        object.__setattr__(self, "expectations", dict(self.expectations))


@dataclass(frozen=True, slots=True)
class ContextCondition:
    """One context variant with the same query and the same named anchors."""

    condition_id: str
    support_features: ArrayLike
    support_labels: ArrayLike
    support_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        features = _features(self.support_features, name="context support_features")
        labels = _labels(self.support_labels, n_rows=features.shape[0], name="context support_labels")
        ids = _ids(self.support_ids, n_rows=features.shape[0], name="context support_ids")
        anchors = tuple(self.anchor_ids)
        if not anchors or len(set(anchors)) != len(anchors) or not set(anchors).issubset(ids):
            raise SyntheticContractError("anchor_ids must be unique non-empty support IDs")
        object.__setattr__(self, "support_features", features)
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "support_ids", ids)
        object.__setattr__(self, "anchor_ids", anchors)


@dataclass(frozen=True, slots=True)
class ContextCompetitionFixture:
    """Fixed query/anchors with a declared set of changed support contexts."""

    name: str
    description: str
    query_features: ArrayLike
    query_labels: ArrayLike
    query_ids: tuple[str, ...]
    conditions: tuple[ContextCondition, ...]
    expectations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query_features = _features(self.query_features, name="context query_features")
        query_labels = _labels(self.query_labels, n_rows=query_features.shape[0], name="context query_labels")
        query_ids = _ids(self.query_ids, n_rows=query_features.shape[0], name="context query_ids")
        conditions = tuple(self.conditions)
        if len(conditions) < 2:
            raise SyntheticContractError("context competition needs at least two conditions")
        feature_counts = {condition.support_features.shape[1] for condition in conditions}
        if feature_counts != {query_features.shape[1]}:
            raise SyntheticContractError("all context supports must match query feature dimensions")
        anchor_sets = {condition.anchor_ids for condition in conditions}
        if len(anchor_sets) != 1:
            raise SyntheticContractError("each context condition must retain exactly the same anchor IDs")
        object.__setattr__(self, "query_features", query_features)
        object.__setattr__(self, "query_labels", query_labels)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "expectations", dict(self.expectations))

