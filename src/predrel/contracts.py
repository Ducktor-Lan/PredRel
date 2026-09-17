"""Strict, backend-independent contracts for Week 1 teacher extraction.

The public ``alpha`` convention is deliberately simple: rows follow
``query_ids`` and columns follow ``support_ids``.  Adapters around a model API
must align any API-specific decoder row indices into this convention before
constructing :class:`TeacherOutput`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .provenance import RunProvenance


SampleId = Hashable


class ContractError(ValueError):
    """Raised when a teacher input or output violates its documented contract."""


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.ascontiguousarray(values).copy()
    copied.setflags(write=False)
    return copied


def _normalise_features(features: ArrayLike, *, owner: str) -> NDArray[np.float64]:
    try:
        values = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{owner}.features must be a numeric 2-D array") from error

    if values.ndim != 2:
        raise ContractError(f"{owner}.features must be 2-D, got shape {values.shape!r}")
    if values.shape[0] == 0:
        raise ContractError(f"{owner}.features must contain at least one row")
    if values.shape[1] == 0:
        raise ContractError(f"{owner}.features must contain at least one feature")
    if not np.isfinite(values).all():
        raise ContractError(f"{owner}.features must contain only finite values")
    return _readonly_copy(values)


def _normalise_labels(labels: ArrayLike, *, n_samples: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ContractError(f"SupportSet.labels must be 1-D, got shape {values.shape!r}")
    if values.shape[0] != n_samples:
        raise ContractError(
            "SupportSet.labels length must equal SupportSet.features rows "
            f"({n_samples}), got {values.shape[0]}"
        )
    if values.dtype.kind in {"f", "c"} and not np.isfinite(values).all():
        raise ContractError("SupportSet.labels must contain only finite numeric values")
    if values.dtype.kind == "O" and any(value is None for value in values.tolist()):
        raise ContractError("SupportSet.labels must not contain None")
    return _readonly_copy(values)


def _normalise_sample_ids(sample_ids: Sequence[SampleId], *, n_samples: int, owner: str) -> tuple[SampleId, ...]:
    if isinstance(sample_ids, (str, bytes)):
        raise ContractError(f"{owner}.sample_ids must be a sequence of IDs, not one string")
    try:
        ids = tuple(sample_ids)
    except TypeError as error:
        raise ContractError(f"{owner}.sample_ids must be an iterable of hashable IDs") from error

    if len(ids) != n_samples:
        raise ContractError(
            f"{owner}.sample_ids length must equal feature rows ({n_samples}), got {len(ids)}"
        )
    if any(sample_id is None for sample_id in ids):
        raise ContractError(f"{owner}.sample_ids must not contain None")
    try:
        unique_count = len(set(ids))
    except TypeError as error:
        raise ContractError(f"{owner}.sample_ids must all be hashable") from error
    if unique_count != len(ids):
        raise ContractError(f"{owner}.sample_ids must be unique")
    return ids


def _normalise_tensor(values: ArrayLike, *, name: str, min_ndim: int = 1) -> NDArray[np.float64]:
    try:
        tensor = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must be a numeric array") from error
    if tensor.ndim < min_ndim:
        raise ContractError(f"{name} must have at least {min_ndim} dimensions, got {tensor.ndim}")
    if 0 in tensor.shape:
        raise ContractError(f"{name} must not have an empty dimension, got shape {tensor.shape!r}")
    if not np.isfinite(tensor).all():
        raise ContractError(f"{name} must contain only finite values")
    return _readonly_copy(tensor)


@dataclass(frozen=True, slots=True)
class SupportSet:
    """Labeled examples that form the only allowed prediction context."""

    features: ArrayLike
    labels: ArrayLike
    sample_ids: Sequence[SampleId]

    def __post_init__(self) -> None:
        features = _normalise_features(self.features, owner="SupportSet")
        labels = _normalise_labels(self.labels, n_samples=features.shape[0])
        sample_ids = _normalise_sample_ids(
            self.sample_ids, n_samples=features.shape[0], owner="SupportSet"
        )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sample_ids", sample_ids)

    @property
    def n_samples(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


@dataclass(frozen=True, slots=True)
class QuerySet:
    """Unlabeled examples to predict.

    This class intentionally has no ``labels`` field.  Passing a query label to
    its constructor therefore fails immediately instead of silently leaking it
    into the support context.
    """

    features: ArrayLike
    sample_ids: Sequence[SampleId]

    def __post_init__(self) -> None:
        features = _normalise_features(self.features, owner="QuerySet")
        sample_ids = _normalise_sample_ids(
            self.sample_ids, n_samples=features.shape[0], owner="QuerySet"
        )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "sample_ids", sample_ids)

    @property
    def n_samples(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


def validate_support_query(support: SupportSet, query: QuerySet) -> None:
    """Validate the cross-set invariants before invoking a teacher."""

    if support.n_features != query.n_features:
        raise ContractError(
            "SupportSet and QuerySet must have the same feature count "
            f"({support.n_features} != {query.n_features})"
        )
    overlap = set(support.sample_ids).intersection(query.sample_ids)
    if overlap:
        preview = ", ".join(repr(value) for value in sorted(overlap, key=repr)[:3])
        raise ContractError(f"SupportSet and QuerySet sample_ids overlap: {preview}")


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    """A validated support/query pair passed to one teacher prediction call."""

    support: SupportSet
    query: QuerySet

    def __post_init__(self) -> None:
        validate_support_query(self.support, self.query)


@dataclass(frozen=True, slots=True)
class EmbeddingBundle:
    """Native embedding tensor plus an explicitly documented aggregation.

    ``sample_axis`` identifies the sample dimension in ``native``.  This keeps
    model-native shapes intact while still making row/ID alignment checkable.
    When an aggregate is supplied it must be a ``[samples, embedding_dim]``
    matrix and ``aggregation`` records how it was made.
    """

    native: ArrayLike
    sample_axis: int
    aggregated: ArrayLike | None = None
    aggregation: str | None = None

    def __post_init__(self) -> None:
        native = _normalise_tensor(self.native, name="EmbeddingBundle.native", min_ndim=2)
        sample_axis = self.sample_axis
        if not isinstance(sample_axis, (int, np.integer)):
            raise ContractError("EmbeddingBundle.sample_axis must be an integer")
        if not -native.ndim <= int(sample_axis) < native.ndim:
            raise ContractError(
                "EmbeddingBundle.sample_axis is outside native tensor dimensions "
                f"for shape {native.shape!r}"
            )
        sample_axis = int(sample_axis) % native.ndim

        aggregated = self.aggregated
        aggregation = self.aggregation
        if aggregated is None and aggregation is not None:
            raise ContractError("EmbeddingBundle.aggregation requires aggregated embeddings")
        if aggregated is not None:
            if not isinstance(aggregation, str) or not aggregation.strip():
                raise ContractError(
                    "EmbeddingBundle.aggregation must name the operation used to aggregate native embeddings"
                )
            aggregated = _normalise_tensor(
                aggregated, name="EmbeddingBundle.aggregated", min_ndim=2
            )
            if aggregated.ndim != 2:
                raise ContractError(
                    "EmbeddingBundle.aggregated must have shape [samples, embedding_dim]"
                )

        object.__setattr__(self, "native", native)
        object.__setattr__(self, "sample_axis", sample_axis)
        object.__setattr__(self, "aggregated", aggregated)
        object.__setattr__(self, "aggregation", aggregation)

    @property
    def n_samples(self) -> int:
        return self.native.shape[self.sample_axis]

    def validate_for_samples(self, *, n_samples: int, name: str) -> None:
        if self.n_samples != n_samples:
            raise ContractError(
                f"{name}.native sample axis has {self.n_samples} rows; expected {n_samples}"
            )
        if self.aggregated is not None and self.aggregated.shape[0] != n_samples:
            raise ContractError(
                f"{name}.aggregated has {self.aggregated.shape[0]} rows; expected {n_samples}"
            )


@dataclass(frozen=True, slots=True)
class TeacherCapabilities:
    """Truthful record of which optional extraction surfaces were available."""

    alpha_available: bool = True
    hidden_support_available: bool = False
    hidden_query_available: bool = False
    raw_scores_available: bool = False
    decoder_readout_api: str | None = None
    raw_score_hook_path: str | None = None
    notes: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in (
            "alpha_available",
            "hidden_support_available",
            "hidden_query_available",
            "raw_scores_available",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractError(f"TeacherCapabilities.{field_name} must be bool")
        for field_name in ("decoder_readout_api", "raw_score_hook_path"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractError(f"TeacherCapabilities.{field_name} must be a non-empty string when set")
        notes = tuple(self.notes)
        if not all(isinstance(note, str) and note.strip() for note in notes):
            raise ContractError("TeacherCapabilities.notes must contain non-empty strings")
        object.__setattr__(self, "notes", notes)


def _normalise_predictions(predictions: ArrayLike, *, n_queries: int) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim != 1:
        raise ContractError(f"TeacherOutput.predictions must be 1-D, got shape {values.shape!r}")
    if values.shape[0] != n_queries:
        raise ContractError(
            "TeacherOutput.predictions length must equal query_ids length "
            f"({n_queries}), got {values.shape[0]}"
        )
    if values.dtype.kind in {"f", "c"} and not np.isfinite(values).all():
        raise ContractError("TeacherOutput.predictions must contain finite numeric values")
    return _readonly_copy(values)


def _normalise_training_indices(indices: ArrayLike | None, *, n_support: int) -> NDArray[np.int64] | None:
    if indices is None:
        return None
    values = np.asarray(indices)
    if values.ndim != 1 or values.shape[0] != n_support:
        raise ContractError(
            "TeacherOutput.decoder_training_row_indices must have shape "
            f"[{n_support}]"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ContractError("TeacherOutput.decoder_training_row_indices must be integer-valued")
    normalised = values.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(normalised), np.arange(n_support, dtype=np.int64)):
        raise ContractError(
            "TeacherOutput.decoder_training_row_indices must be a permutation of 0..N-1"
        )
    return _readonly_copy(normalised)


def _normalise_probabilities(probabilities: ArrayLike | None, *, n_queries: int) -> NDArray[np.float64] | None:
    if probabilities is None:
        return None
    values = _normalise_tensor(
        probabilities, name="TeacherOutput.prediction_probabilities", min_ndim=2
    )
    if values.ndim != 2 or values.shape[0] != n_queries:
        raise ContractError(
            "TeacherOutput.prediction_probabilities must have shape [Q, C] with Q equal to query_ids length"
        )
    if np.any(values < -1e-12):
        raise ContractError("TeacherOutput.prediction_probabilities must be non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ContractError("TeacherOutput.prediction_probabilities rows must sum to 1 within atol=1e-6")
    return values


def aggregate_raw_scores_to_alpha(raw_scores: ArrayLike) -> NDArray[np.float64]:
    """Return ``mean_{estimator, head}(softmax(raw_scores))``.

    Scores must be in the canonical ``[E, H, Q, N]`` layout.  In particular,
    this intentionally does *not* compute ``softmax(mean(raw_scores))``.
    """

    scores = _normalise_tensor(raw_scores, name="raw_scores", min_ndim=4)
    if scores.ndim != 4:
        raise ContractError("raw_scores must have canonical shape [E, H, Q, N]")
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    return probabilities.mean(axis=(0, 1))


def align_decoder_columns(
    values: ArrayLike, decoder_training_row_indices: ArrayLike, *, support_size: int
) -> NDArray[np.float64]:
    """Reindex a decoder's last axis to canonical support-row order.

    ``decoder_training_row_indices[j]`` is the support row represented by source
    column ``j``.  The returned array places that source column at the matching
    canonical support position, so its final axis always follows ``support_ids``.
    """

    tensor = _normalise_tensor(values, name="values", min_ndim=1)
    if tensor.shape[-1] != support_size:
        raise ContractError(
            f"values last axis must have support_size {support_size}, got {tensor.shape[-1]}"
        )
    indices = _normalise_training_indices(decoder_training_row_indices, n_support=support_size)
    assert indices is not None  # The helper always receives indices.
    aligned = np.empty_like(tensor)
    aligned[..., indices] = tensor
    aligned.setflags(write=False)
    return aligned


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    """Canonical result of one TabPFN teacher extraction call.

    ``alpha[q, n]`` and ``raw_scores[..., q, n]`` use exactly the support and
    query ID order stored alongside them.  ``decoder_training_row_indices`` is
    retained as audit evidence of the original decoder ordering, while tensors
    stored here have already been aligned to canonical support order.
    """

    predictions: ArrayLike
    alpha: ArrayLike
    support_ids: Sequence[SampleId]
    query_ids: Sequence[SampleId]
    provenance: RunProvenance | Mapping[str, Any]
    raw_scores: ArrayLike | None = None
    hidden_support: EmbeddingBundle | None = None
    hidden_query: EmbeddingBundle | None = None
    prediction_probabilities: ArrayLike | None = None
    decoder_training_row_indices: ArrayLike | None = None
    capabilities: TeacherCapabilities = field(default_factory=TeacherCapabilities)

    def __post_init__(self) -> None:
        support_ids = _normalise_sample_ids(
            self.support_ids, n_samples=len(self.support_ids), owner="TeacherOutput.support_ids"
        )
        query_ids = _normalise_sample_ids(
            self.query_ids, n_samples=len(self.query_ids), owner="TeacherOutput.query_ids"
        )
        overlap = set(support_ids).intersection(query_ids)
        if overlap:
            preview = ", ".join(repr(value) for value in sorted(overlap, key=repr)[:3])
            raise ContractError(f"TeacherOutput support_ids and query_ids overlap: {preview}")

        alpha = _normalise_tensor(self.alpha, name="TeacherOutput.alpha", min_ndim=2)
        expected_alpha_shape = (len(query_ids), len(support_ids))
        if alpha.ndim != 2 or alpha.shape != expected_alpha_shape:
            raise ContractError(
                "TeacherOutput.alpha must have shape [Q, N] equal to query_ids x support_ids; "
                f"expected {expected_alpha_shape}, got {alpha.shape!r}"
            )
        if np.any(alpha < -1e-12):
            raise ContractError("TeacherOutput.alpha must be non-negative")
        if not np.allclose(alpha.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise ContractError("TeacherOutput.alpha rows must sum to 1 within atol=1e-6")

        predictions = _normalise_predictions(self.predictions, n_queries=len(query_ids))
        prediction_probabilities = _normalise_probabilities(
            self.prediction_probabilities, n_queries=len(query_ids)
        )
        raw_scores = self.raw_scores
        if raw_scores is not None:
            raw_scores = _normalise_tensor(
                raw_scores, name="TeacherOutput.raw_scores", min_ndim=4
            )
            expected_tail = (len(query_ids), len(support_ids))
            if raw_scores.ndim != 4 or raw_scores.shape[2:] != expected_tail:
                raise ContractError(
                    "TeacherOutput.raw_scores must have shape [E, H, Q, N] with canonical Q/N axes; "
                    f"got {raw_scores.shape!r}"
                )

        hidden_support = self.hidden_support
        hidden_query = self.hidden_query
        if hidden_support is not None:
            hidden_support.validate_for_samples(n_samples=len(support_ids), name="hidden_support")
        if hidden_query is not None:
            hidden_query.validate_for_samples(n_samples=len(query_ids), name="hidden_query")

        capabilities = self.capabilities
        if not capabilities.alpha_available:
            raise ContractError("TeacherOutput always contains alpha, so alpha_available must be True")
        if (raw_scores is not None) != capabilities.raw_scores_available:
            raise ContractError(
                "TeacherCapabilities.raw_scores_available must exactly match raw_scores presence"
            )
        if (hidden_support is not None) != capabilities.hidden_support_available:
            raise ContractError(
                "TeacherCapabilities.hidden_support_available must exactly match hidden_support presence"
            )
        if (hidden_query is not None) != capabilities.hidden_query_available:
            raise ContractError(
                "TeacherCapabilities.hidden_query_available must exactly match hidden_query presence"
            )

        provenance = self.provenance
        if isinstance(provenance, Mapping):
            provenance = MappingProxyType(dict(provenance))
        elif not isinstance(provenance, RunProvenance):
            raise ContractError("TeacherOutput.provenance must be RunProvenance or a mapping")

        decoder_training_row_indices = _normalise_training_indices(
            self.decoder_training_row_indices, n_support=len(support_ids)
        )
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "support_ids", support_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "raw_scores", raw_scores)
        object.__setattr__(self, "hidden_support", hidden_support)
        object.__setattr__(self, "hidden_query", hidden_query)
        object.__setattr__(self, "prediction_probabilities", prediction_probabilities)
        object.__setattr__(self, "decoder_training_row_indices", decoder_training_row_indices)

    @property
    def hidden_support_native(self) -> NDArray[np.float64] | None:
        """Convenience view of support-native embeddings, preserving native shape."""

        return None if self.hidden_support is None else self.hidden_support.native

    @property
    def hidden_query_native(self) -> NDArray[np.float64] | None:
        """Convenience view of query-native embeddings, preserving native shape."""

        return None if self.hidden_query is None else self.hidden_query.native

    @property
    def hidden_support_aggregated(self) -> NDArray[np.float64] | None:
        """Explicitly aggregated support embeddings, if extraction requested them."""

        return None if self.hidden_support is None else self.hidden_support.aggregated

    @property
    def hidden_query_aggregated(self) -> NDArray[np.float64] | None:
        """Explicitly aggregated query embeddings, if extraction requested them."""

        return None if self.hidden_query is None else self.hidden_query.aggregated

    @property
    def model_roles(self) -> Mapping[str, Any]:
        """Model-role provenance (for example, readout and shadow embedding models)."""

        if isinstance(self.provenance, RunProvenance):
            return self.provenance.model_roles
        return self.provenance.get("model_roles", {})

    @property
    def extraction_metadata(self) -> Mapping[str, Any]:
        """Extraction-path metadata, including independently recorded fit modes."""

        if isinstance(self.provenance, RunProvenance):
            return self.provenance.extraction_metadata
        return self.provenance.get("extraction_metadata", {})
