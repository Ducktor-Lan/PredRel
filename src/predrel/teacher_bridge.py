"""Readout+hidden bridge to the frozen Teacher (``predrel.teacher``).

The Teacher code lives in this package (``predrel.teacher``); the extraction
API has no query-label argument. ``include_hidden=True`` (default) also
returns the shadow-model aggregated hidden embeddings (benchmark method);
``include_hidden=False`` is the readout-only path.

``week01_source_root`` / ``expected_week01_source_sha256`` are retained only
as deprecated aliases: provenance is now recorded in
``provenance/MODULE_SOURCES.md`` and the package's own manifest. Passing a
source root still triggers the legacy snapshot check for audit purposes,
but normal runs should omit it.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


WEEK01_SOURCE_SHA256 = "b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4"


class TeacherBridgeError(RuntimeError):
    """Raised when the frozen Teacher dependency cannot safely be reused."""


@dataclass(frozen=True, slots=True)
class ReadoutExtraction:
    """Aligned decoder outputs used by the Week 10 benchmark training and evaluation."""

    alpha: NDArray[np.float64]
    raw_scores: NDArray[np.float64]
    support_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    teacher_source_sha256: str
    decoder_readout_api: str
    raw_score_hook_path: str
    runtime: dict[str, object]
    hidden_support_aggregated: NDArray[np.float64] | None = None
    hidden_query_aggregated: NDArray[np.float64] | None = None

    @property
    def mean_raw_scores(self) -> NDArray[np.float64]:
        """Average exact pre-softmax scores over estimator and attention head."""

        return self.raw_scores.mean(axis=(0, 1))


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if not root.is_dir():
        raise TeacherBridgeError(f"Week 1 source snapshot is not a directory: {root}")
    return root


def _load_week01(source_root: Path | None = None) -> dict[str, Any]:
    """Return Teacher callables from this package (``predrel.teacher``).

    The ``source_root`` argument is a deprecated audit hook: when given, the
    legacy snapshot manifest hash is still checked before binding, so old
    provenance records stay verifiable. Normal runs pass ``None`` and bind
    directly to the in-package Teacher.
    """

    if source_root is not None:
        root = _safe_root(source_root)
        manifest_path = root / "provenance" / "source_manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise TeacherBridgeError(f"Week 1 source manifest is missing: {manifest_path}") from error
        except json.JSONDecodeError as error:
            raise TeacherBridgeError(f"Week 1 source manifest is invalid JSON: {manifest_path}") from error
        observed = str(payload.get("source_sha256", "")).lower()
        if observed != WEEK01_SOURCE_SHA256.lower():
            raise TeacherBridgeError(
                f"Week 1 source hash mismatch: expected {WEEK01_SOURCE_SHA256}, observed {observed or '<missing>'}"
            )
    try:
        from . import contracts as contracts
        from . import provenance as provenance
        from .teacher import config as config
        from .teacher import decoder as decoder
        from .teacher import tabpfn_teacher as teacher_module
    except ImportError as error:
        raise TeacherBridgeError("cannot import the in-package Teacher modules") from error
    return {
        "SupportSet": contracts.SupportSet,
        "QuerySet": contracts.QuerySet,
        "TeacherRequest": contracts.TeacherRequest,
        "align_decoder_columns": contracts.align_decoder_columns,
        "aggregate_raw_scores_to_alpha": contracts.aggregate_raw_scores_to_alpha,
        "read_source_manifest": provenance.read_source_manifest,
        "TeacherConfig": config.TeacherConfig,
        "official_decoder_readout": decoder.official_decoder_readout,
        "capture_raw_scores": decoder.capture_raw_scores,
        "assert_raw_score_parity": decoder.assert_raw_score_parity,
        "TabPFNTeacher": teacher_module.TabPFNTeacher,
        "model_cache_environment": teacher_module._model_cache_environment,
        "seed_runtime": teacher_module._seed_runtime,
    }


def verify_week01_snapshot(
    source_root: str | Path, *, expected_source_sha256: str = WEEK01_SOURCE_SHA256
) -> Path:
    """Deprecated: validate a legacy Week 1 snapshot directory (audit only)."""

    root = _safe_root(source_root)
    manifest_path = root / "provenance" / "source_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TeacherBridgeError(f"Week 1 source manifest is missing: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise TeacherBridgeError(f"Week 1 source manifest is invalid JSON: {manifest_path}") from error
    observed = str(payload.get("source_sha256", "")).lower()
    if observed != expected_source_sha256.lower():
        raise TeacherBridgeError(
            f"Week 1 source hash mismatch: expected {expected_source_sha256}, observed {observed or '<missing>'}"
        )
    return root


def verify_week01_snapshot_contents(
    source_root: str | Path, *, expected_source_sha256: str = WEEK01_SOURCE_SHA256
) -> Path:
    """Deprecated: legacy snapshot check (audit only)."""

    return verify_week01_snapshot(source_root, expected_source_sha256=expected_source_sha256)


def runtime_record() -> dict[str, object]:
    """Capture a redacted runtime identity after the Teacher import boundary."""

    try:
        import torch
    except ImportError as error:
        raise TeacherBridgeError("PyTorch is absent from the selected server environment") from error
    return {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "tabpfn": _distribution_version("tabpfn"),
        "tabpfn_extensions": _distribution_version("tabpfn-extensions"),
    }


def extract_readout(
    *,
    support_features: ArrayLike,
    support_labels: ArrayLike,
    support_ids: tuple[str, ...],
    query_features: ArrayLike,
    query_ids: tuple[str, ...],
    seed: int,
    n_estimators: int,
    model_cache_dir: str | None,
    week01_source_root: str | Path | None = None,
    expected_week01_source_sha256: str = WEEK01_SOURCE_SHA256,
    include_hidden: bool = True,
    estimator_role: str = "benchmark",
) -> ReadoutExtraction:
    """Fit one Teacher readout and return alpha plus raw decoder scores.

    Query labels are intentionally absent. The in-package contracts enforce
    alpha row/column alignment and raw-score parity before return.
    ``include_hidden=True`` (default) also returns shadow-model aggregated
    hidden embeddings; ``include_hidden=False`` is the readout-only path.
    ``week01_source_root`` is a deprecated audit hook (legacy snapshot hash
    check); normal runs omit it.
    """

    modules = _load_week01(
        Path(week01_source_root).resolve() if week01_source_root is not None else None
    )
    teacher_source_sha = "in-package:predrel.teacher"
    support = modules["SupportSet"](
        features=np.asarray(support_features, dtype=np.float64),
        labels=np.asarray(support_labels, dtype=np.int64),
        sample_ids=tuple(support_ids),
    )
    query = modules["QuerySet"](
        features=np.asarray(query_features, dtype=np.float64),
        sample_ids=tuple(query_ids),
    )
    request = modules["TeacherRequest"](support=support, query=query)
    config = modules["TeacherConfig"](
        seed=seed,
        n_estimators=n_estimators,
        device="cuda",
        inference_precision="float32",
        readout_fit_mode="fit_with_cache",
        embedding_fit_mode="fit_preprocessors",
        kv_cache_precision="auto",
        n_preprocessing_jobs=1,
        model_cache_dir=model_cache_dir,
        raw_score_atol=1.0e-6,
        raw_score_rtol=1.0e-5,
    )
    teacher = modules["TabPFNTeacher"](config=config, source_manifest=None)
    with modules["model_cache_environment"](config.model_cache_dir):
        teacher._validate_environment()
        with modules["seed_runtime"](config.seed):
            model = teacher._make_estimator(config.readout_fit_mode)
            model.fit(request.support.features, request.support.labels)
            teacher._validate_fitted_estimator(model, role=str(estimator_role))
            reference = modules["official_decoder_readout"](model, request.query.features)
            capture = modules["capture_raw_scores"](model, request.query.features)
            teacher._require_estimator_axis(reference.weights_per_estimator, surface="official decoder readout")
            teacher._require_estimator_axis(capture.scores, surface="raw-score capture")
            modules["assert_raw_score_parity"](
                capture.scores,
                reference.weights_per_estimator,
                rtol=config.raw_score_rtol,
                atol=config.raw_score_atol,
            )
            alpha_by_estimator = modules["align_decoder_columns"](
                reference.weights_per_estimator,
                reference.training_row_indices,
                support_size=request.support.n_samples,
            )
            raw_scores = modules["align_decoder_columns"](
                capture.scores,
                reference.training_row_indices,
                support_size=request.support.n_samples,
            )
            alpha = np.asarray(alpha_by_estimator.mean(axis=0), dtype=np.float64)
            reconstructed = modules["aggregate_raw_scores_to_alpha"](raw_scores)
            if not np.allclose(
                reconstructed,
                alpha,
                rtol=config.raw_score_rtol,
                atol=config.raw_score_atol,
            ):
                maximum_error = float(np.max(np.abs(reconstructed - alpha)))
                raise TeacherBridgeError(
                    f"aligned raw scores no longer reconstruct alpha (max_abs_error={maximum_error:.3e})"
                )
    hidden_support_agg: NDArray[np.float64] | None = None
    hidden_query_agg: NDArray[np.float64] | None = None
    if include_hidden:
        embedding_model = teacher._make_estimator(config.embedding_fit_mode)
        embedding_model.fit(request.support.features, request.support.labels)
        teacher._validate_fitted_estimator(embedding_model, role="embedding")
        hidden_support_agg = np.asarray(
            teacher._extract_embedding_bundle(
                embedding_model,
                np.asarray(request.support.features),
                data_source="train",
                n_samples=request.support.n_samples,
            ).aggregated,
            dtype=np.float64,
        )
        hidden_query_agg = np.asarray(
            teacher._extract_embedding_bundle(
                embedding_model,
                np.asarray(request.query.features),
                data_source="test",
                n_samples=request.query.n_samples,
                rowwise=True,
            ).aggregated,
            dtype=np.float64,
        )
        if hidden_support_agg.shape[0] != request.support.n_samples:
            raise TeacherBridgeError("hidden support embeddings do not align to support IDs")
        if hidden_query_agg.shape[0] != request.query.n_samples:
            raise TeacherBridgeError("hidden query embeddings do not align to query IDs")
    return ReadoutExtraction(
        alpha=np.ascontiguousarray(alpha),
        raw_scores=np.ascontiguousarray(raw_scores, dtype=np.float64),
        support_ids=tuple(request.support.sample_ids),
        query_ids=tuple(request.query.sample_ids),
        teacher_source_sha256=teacher_source_sha,
        decoder_readout_api=reference.api_path,
        raw_score_hook_path=capture.hook_path,
        runtime=runtime_record(),
        hidden_support_aggregated=None if hidden_support_agg is None else np.ascontiguousarray(hidden_support_agg),
        hidden_query_aggregated=None if hidden_query_agg is None else np.ascontiguousarray(hidden_query_agg),
    )

