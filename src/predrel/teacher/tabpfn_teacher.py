"""A provenance-first TabPFN-v3 teacher for Week 1.

The class deliberately has one narrow job: turn a labeled support set and an
unlabeled query set into an auditable decoder relation.  It does not implement
``beta``, anatomy metrics, or student training; those belong to later weeks.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
import os
from pathlib import Path
import random
from typing import Any, Callable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..contracts import (
    ContractError,
    EmbeddingBundle,
    TeacherCapabilities,
    TeacherOutput,
    TeacherRequest,
    aggregate_raw_scores_to_alpha,
    align_decoder_columns,
)
from ..provenance import (
    ModelRoleProvenance,
    SourceManifest,
    build_source_manifest,
    collect_runtime_provenance,
    sha256_file,
)
from .config import TeacherConfig
from .decoder import (
    RawScoreCapture,
    assert_raw_score_parity,
    capture_raw_scores,
    official_decoder_readout,
)


class TeacherEnvironmentError(RuntimeError):
    """Raised when the server cannot provide the frozen Week 1 teacher stack."""


class TeacherExtractionError(RuntimeError):
    """Raised when a fitted teacher violates the Week 1 extraction contract."""


EstimatorFactory = Callable[[str], Any]
EmbeddingExtractor = Callable[[Any, ArrayLike, str], ArrayLike]


def _distribution_version(name: str) -> str | None:
    try:
        value = metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    return None if value in {None, "None"} else value


@contextmanager
def _model_cache_environment(path: str | None) -> Iterator[None]:
    """Temporarily direct TabPFN's downloaded checkpoints into the run cache."""

    if path is None:
        yield
        return
    cache_path = Path(path)
    previous = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache_path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TABPFN_MODEL_CACHE_DIR", None)
        else:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = previous


@contextmanager
def _seed_runtime(seed: int) -> Iterator[None]:
    """Temporarily seed Python/NumPy/Torch in addition to TabPFN's RNG argument.

    TabPFN receives ``random_state`` separately, but saving/restoring the process
    RNGs makes the hook and shadow-model work reproducible without permanently
    perturbing a host process that runs more than one fixture.
    """

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch: Any | None = None
    torch_cpu_state: Any | None = None
    torch_cuda_states: Any | None = None
    try:
        import torch as torch_module

        torch = torch_module
        torch_cpu_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_states = torch.cuda.get_rng_state_all()
    except ImportError:
        # The normal extraction path has already validated torch, but keeping
        # this context import-safe makes the environment gate truthful.
        torch = None

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_cpu_state is not None:
            torch.random.set_rng_state(torch_cpu_state)
            if torch_cuda_states is not None:
                torch.cuda.set_rng_state_all(torch_cuda_states)


def _default_source_root() -> Path:
    # .../Week 1/src/predrel_week1/teacher/tabpfn_teacher.py -> Week 1
    return Path(__file__).resolve().parents[3]


class TabPFNTeacher:
    """Extract a canonical v3 decoder relation plus shadow-model embeddings.

    A cache-enabled estimator is used only for prediction/readout/raw scores;
    a same-config ``fit_preprocessors`` shadow estimator supplies train/test
    hidden embeddings.  The two roles are always preserved in provenance.
    """

    def __init__(
        self,
        config: TeacherConfig | None = None,
        *,
        source_manifest: SourceManifest | None = None,
        source_root: str | Path | None = None,
        estimator_factory: EstimatorFactory | None = None,
        embedding_extractor: EmbeddingExtractor | None = None,
    ) -> None:
        self.config = TeacherConfig() if config is None else config
        if source_manifest is not None and source_root is not None:
            raise ValueError("Pass source_manifest or source_root, not both")
        self._source_manifest = source_manifest
        self._source_root = None if source_root is None else Path(source_root)
        self._estimator_factory = estimator_factory
        self._embedding_extractor = embedding_extractor

    def extract(self, request: TeacherRequest) -> TeacherOutput:
        """Fit the two model roles and return one strictly aligned TeacherOutput."""

        manifest = self._manifest()
        with _model_cache_environment(self.config.model_cache_dir):
            # TabPFN reads TABPFN_MODEL_CACHE_DIR while it is imported.  Keep
            # this guard outside the environment check so even the first v3
            # checkpoint download cannot fall back to a user-profile cache.
            self._validate_environment()
            with _seed_runtime(self.config.seed):
                readout_model = self._make_estimator(self.config.readout_fit_mode)
                readout_model.fit(request.support.features, request.support.labels)
                self._validate_fitted_estimator(readout_model, role="readout")

                reference = official_decoder_readout(readout_model, request.query.features)
                raw_capture = capture_raw_scores(readout_model, request.query.features)
                self._require_estimator_axis(
                    reference.weights_per_estimator,
                    surface="official decoder readout",
                )
                self._require_estimator_axis(raw_capture.scores, surface="raw-score capture")
                assert_raw_score_parity(
                    raw_capture.scores,
                    reference.weights_per_estimator,
                    rtol=self.config.raw_score_rtol,
                    atol=self.config.raw_score_atol,
                )

                aligned_reference = align_decoder_columns(
                    reference.weights_per_estimator,
                    reference.training_row_indices,
                    support_size=request.support.n_samples,
                )
                aligned_raw_scores = align_decoder_columns(
                    raw_capture.scores,
                    reference.training_row_indices,
                    support_size=request.support.n_samples,
                )
                alpha = aligned_reference.mean(axis=0)
                raw_alpha = aggregate_raw_scores_to_alpha(aligned_raw_scores)
                if not np.allclose(
                    raw_alpha,
                    alpha,
                    rtol=self.config.raw_score_rtol,
                    atol=self.config.raw_score_atol,
                ):
                    maximum_error = float(np.max(np.abs(raw_alpha - alpha)))
                    raise TeacherExtractionError(
                        "Aligned raw-score alpha does not reproduce the official readout "
                        f"(max_abs_error={maximum_error:.3e})"
                    )
                prediction_probabilities = np.asarray(
                    readout_model.predict_proba(request.query.features), dtype=np.float64
                )
                predictions = np.asarray(readout_model.predict(request.query.features))

                embedding_model = self._make_estimator(self.config.embedding_fit_mode)
                embedding_model.fit(request.support.features, request.support.labels)
                self._validate_fitted_estimator(embedding_model, role="embedding")
                hidden_support = self._extract_embedding_bundle(
                    embedding_model,
                    request.support.features,
                    data_source="train",
                    n_samples=request.support.n_samples,
                )
                hidden_query = self._extract_embedding_bundle(
                    embedding_model,
                    request.query.features,
                    data_source="test",
                    n_samples=request.query.n_samples,
                    rowwise=True,
                )

        provenance = self._provenance(
            manifest,
            readout_model=readout_model,
            embedding_model=embedding_model,
            reference_api=reference.api_path,
            raw_hook_path=raw_capture.hook_path,
            raw_capture=raw_capture,
            decoder_training_row_indices=reference.training_row_indices,
        )
        return TeacherOutput(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            alpha=alpha,
            raw_scores=aligned_raw_scores,
            support_ids=request.support.sample_ids,
            query_ids=request.query.sample_ids,
            hidden_support=hidden_support,
            hidden_query=hidden_query,
            decoder_training_row_indices=reference.training_row_indices,
            capabilities=TeacherCapabilities(
                alpha_available=True,
                hidden_support_available=True,
                hidden_query_available=True,
                raw_scores_available=True,
                decoder_readout_api=reference.api_path,
                raw_score_hook_path=raw_capture.hook_path,
                notes=(
                    "alpha is official ManyClassDecoder readout aligned to support_ids",
                    "raw scores are reconstructed from decoder forward inputs and parity-checked",
                    "hidden embeddings come from the separately fitted fit_preprocessors shadow model",
                ),
            ),
            provenance=provenance,
        )

    def _validate_environment(self) -> None:
        """Fail before model download when the frozen local-only stack is absent."""

        try:
            import torch
            from tabpfn.constants import ModelVersion
        except ImportError as error:  # pragma: no cover - server dependent
            raise TeacherEnvironmentError(
                "Week 1 requires local PyTorch and tabpfn==8.5.0 on the experiment server"
            ) from error
        if not torch.cuda.is_available():
            raise TeacherEnvironmentError("Week 1 requires CUDA, but torch.cuda.is_available() is False")
        if not hasattr(ModelVersion, "V3"):
            raise TeacherEnvironmentError("Installed tabpfn does not expose ModelVersion.V3")
        tabpfn_version = _distribution_version("tabpfn")
        if tabpfn_version != "8.5.0":
            raise TeacherEnvironmentError(
                f"Frozen Week 1 stack requires tabpfn==8.5.0, found {tabpfn_version}"
            )
        extensions_version = _distribution_version("tabpfn-extensions")
        if extensions_version != "0.6.2":
            raise TeacherEnvironmentError(
                "Frozen Week 1 stack requires tabpfn-extensions[interpretability]==0.6.2, "
                f"found {extensions_version!r}"
            )
        try:
            from tabpfn_extensions.interpretability import get_decoder_readout  # noqa: F401
        except ImportError as error:  # pragma: no cover - server dependent
            raise TeacherEnvironmentError(
                "tabpfn-extensions is installed without its required interpretability surface"
            ) from error

    def _make_estimator(self, fit_mode: str) -> Any:
        if self._estimator_factory is not None:
            return self._estimator_factory(fit_mode)
        try:
            import torch
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as error:  # pragma: no cover - server dependent
            raise TeacherEnvironmentError("Unable to import the local TabPFN classifier") from error
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3,
            n_estimators=self.config.n_estimators,
            auto_scale_n_estimators=False,
            softmax_temperature=1.0,
            balance_probabilities=False,
            average_before_softmax=False,
            device=self.config.device,
            inference_precision=torch.float32,
            fit_mode=fit_mode,
            keep_cache_on_device=True,
            kv_cache_precision=self.config.kv_cache_precision,
            random_state=self.config.seed,
            n_preprocessing_jobs=self.config.n_preprocessing_jobs,
            inference_config={
                "SUBSAMPLE_SAMPLES": None,
                # Week 1 pins FINGERPRINT_FEATURE=False: the fingerprint column's
                # salt depends on the fitted row count and its hash amplifies
                # tiny batch/row-order numeric differences, which breaks both
                # support-row permutation equivariance and batch-vs-single
                # parity (Phase A/B diagnostics, 2026-09-11).
                "FINGERPRINT_FEATURE": False,
            },
        )

    def _validate_fitted_estimator(self, estimator: Any, *, role: str) -> None:
        if getattr(estimator, "auto_scale_n_estimators", False):
            raise TeacherExtractionError(f"{role} model enabled estimator auto-scaling")
        actual_estimators = getattr(estimator, "n_estimators_", self.config.n_estimators)
        if actual_estimators != self.config.n_estimators:
            raise TeacherExtractionError(
                f"{role} estimator count changed after fit ({actual_estimators} != {self.config.n_estimators})"
            )
        inference_config = getattr(estimator, "inference_config_", None)
        if isinstance(inference_config, dict):
            subsample = inference_config.get("SUBSAMPLE_SAMPLES")
            fingerprint = inference_config.get("FINGERPRINT_FEATURE")
        else:
            subsample = getattr(inference_config, "SUBSAMPLE_SAMPLES", None)
            fingerprint = getattr(inference_config, "FINGERPRINT_FEATURE", None)
        if subsample is not None:
            raise TeacherExtractionError(
                f"{role} model enabled unsupported row subsampling: {subsample!r}"
            )
        if fingerprint:
            raise TeacherExtractionError(
                f"{role} model enabled FINGERPRINT_FEATURE; Week 1 fixes it to False "
                "so alpha and prediction probabilities keep row-order equivariance"
            )

    def _require_estimator_axis(self, values: ArrayLike, *, surface: str) -> None:
        """Reject an upstream ABI change rather than averaging an ambiguous axis."""

        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.config.n_estimators:
            observed = None if array.ndim < 1 else int(array.shape[0])
            raise TeacherExtractionError(
                f"{surface} has estimator axis {observed}; expected exactly "
                f"{self.config.n_estimators}"
            )

    def _extract_embedding_bundle(
        self,
        estimator: Any,
        features: ArrayLike,
        *,
        data_source: str,
        n_samples: int,
        rowwise: bool = False,
    ) -> EmbeddingBundle:
        if self._embedding_extractor is None:
            try:
                from tabpfn.base import get_embeddings
            except ImportError as error:  # pragma: no cover - server dependent
                raise TeacherEnvironmentError("Installed tabpfn does not expose get_embeddings") from error
            if rowwise and n_samples > 1:
                # Query embeddings are extracted one row per call and stacked
                # back together.  TabPFN's batched extraction is not bitwise
                # identical to the single-row path at fp32 (~1e-5), which broke
                # batch-vs-single parity; extracting row-wise removes that
                # batch-shape error by construction.
                pieces = []
                for row in np.asarray(features):
                    native = np.asarray(
                        get_embeddings(
                            estimator,
                            np.asarray(row)[np.newaxis, :],
                            data_source=data_source,
                        ),
                        dtype=np.float64,
                    )
                    if native.ndim == 2:
                        # One-row calls squeeze the sample axis: [E,D] -> [E,1,D].
                        native = native[:, np.newaxis, :]
                    elif native.ndim != 3:
                        raise TeacherExtractionError(
                            "TabPFN get_embeddings returned an unexpected single-row shape "
                            f"{native.shape!r}"
                        )
                    pieces.append(native)
                native = np.concatenate(pieces, axis=1)
            else:
                native = get_embeddings(estimator, np.asarray(features), data_source=data_source)
        else:
            native = self._embedding_extractor(estimator, np.asarray(features), data_source)
        return _embedding_bundle(native, n_samples=n_samples)

    def _manifest(self) -> SourceManifest:
        if self._source_manifest is not None:
            return self._source_manifest
        return build_source_manifest(
            _default_source_root() if self._source_root is None else self._source_root
        )

    def _provenance(
        self,
        manifest: SourceManifest,
        *,
        readout_model: Any,
        embedding_model: Any,
        reference_api: str,
        raw_hook_path: str,
        raw_capture: RawScoreCapture,
        decoder_training_row_indices: NDArray[np.int64],
    ) -> Any:
        readout_checkpoint_id, readout_checkpoint_hash = _checkpoint_identity(readout_model)
        embedding_checkpoint_id, embedding_checkpoint_hash = _checkpoint_identity(embedding_model)
        if readout_checkpoint_hash is None or embedding_checkpoint_hash is None:
            raise TeacherExtractionError(
                "Week 1 requires a readable local checkpoint path and SHA-256 for both model roles"
            )
        common_metadata = {
            "random_state": self.config.seed,
            "softmax_temperature": 1.0,
            "balance_probabilities": False,
            "average_before_softmax": False,
            "subsample_samples": None,
            "fingerprint_feature": False,
            "auto_scale_n_estimators": False,
            "n_preprocessing_jobs": self.config.n_preprocessing_jobs,
            "process_rng_seeded": True,
        }
        roles = {
            "readout": ModelRoleProvenance(
                role="readout",
                fit_mode=self.config.readout_fit_mode,
                model_version="v3",
                backend="local_pytorch",
                device=self.config.device,
                precision=self.config.inference_precision,
                n_estimators=self.config.n_estimators,
                checkpoint_id=readout_checkpoint_id,
                checkpoint_sha256=readout_checkpoint_hash,
                metadata={**common_metadata, "kv_cache_precision": self.config.kv_cache_precision},
            ),
            "embedding": ModelRoleProvenance(
                role="embedding",
                fit_mode=self.config.embedding_fit_mode,
                model_version="v3",
                backend="local_pytorch",
                device=self.config.device,
                precision=self.config.inference_precision,
                n_estimators=self.config.n_estimators,
                checkpoint_id=embedding_checkpoint_id,
                checkpoint_sha256=embedding_checkpoint_hash,
                metadata=common_metadata,
            ),
        }
        return collect_runtime_provenance(
            source_manifest=manifest,
            run_id=self.config.run_id,
            seed=self.config.seed,
            checkpoint_id=readout_checkpoint_id,
            checkpoint_sha256=readout_checkpoint_hash,
            model_roles=roles,
            extraction_metadata={
                "teacher_target": "v3_many_class_decoder_readout",
                "fingerprint_feature": False,
                "query_embedding_extraction": "rowwise_one_row_per_call_then_concatenated",
                "alpha_layout": "[Q,N] aligned_to_support_ids",
                "raw_scores_layout": "[E,H,Q,N] aligned_to_support_ids",
                "raw_score_reconstruction": "mean_E(mean_H(softmax(score, support_axis)))",
                "decoder_readout_api": reference_api,
                "raw_score_hook_path": raw_hook_path,
                "raw_score_hook_abi": {
                    "decoder_type": "ManyClassDecoder",
                    "num_heads": raw_capture.num_heads,
                    "head_dim": raw_capture.head_dim,
                    "softmax_scaling_applied": raw_capture.softmax_scaling_applied,
                    "captured_estimators": int(raw_capture.scores.shape[0]),
                },
                "decoder_training_row_indices": decoder_training_row_indices.tolist(),
                "class_order": np.asarray(readout_model.classes_).tolist(),
            },
        )


def _embedding_bundle(native: ArrayLike, *, n_samples: int) -> EmbeddingBundle:
    """Preserve native shape except for a documented squeezed singleton axis."""

    array = np.asarray(native, dtype=np.float64)
    if n_samples == 1 and array.ndim == 2:
        # TabPFN 8.5's get_embeddings uses Tensor.squeeze() internally.  For
        # a one-row query that removes its sample axis and leaves [E,D].
        # Restore only that documented singleton axis so the batch-vs-single
        # parity test compares the same semantic [E,N,D] layout.
        array = array[:, np.newaxis, :]
    if array.ndim < 2 or not np.isfinite(array).all():
        raise TeacherExtractionError("TabPFN get_embeddings returned an invalid native tensor")
    candidates = [axis for axis, size in enumerate(array.shape) if size == n_samples]
    if not candidates:
        raise TeacherExtractionError(
            f"Cannot identify the sample axis of embedding shape {array.shape!r} for {n_samples} rows"
        )
    # The documented/common shapes are [N,D] and [E,N,D].  Prefer the
    # penultimate axis for the latter, then the leading sample axis.
    sample_axis = array.ndim - 2 if array.ndim >= 3 and array.shape[-2] == n_samples else candidates[0]
    aggregated: NDArray[np.float64] | None = None
    aggregation: str | None = None
    if array.ndim == 2 and sample_axis == 0:
        aggregated = array
        aggregation = "identity"
    elif array.ndim == 3 and sample_axis == 1:
        aggregated = array.mean(axis=0)
        aggregation = "mean_over_estimator_axis_0"
    return EmbeddingBundle(
        native=array,
        sample_axis=sample_axis,
        aggregated=aggregated,
        aggregation=aggregation,
    )


def _checkpoint_identity(estimator: Any) -> tuple[str | None, str | None]:
    """Return a stable checkpoint path/hash when the fitted estimator exposes one."""

    candidate = getattr(estimator, "model_path", None)
    if isinstance(candidate, (list, tuple)):
        candidate = candidate[0] if len(candidate) == 1 else None
    if candidate is None:
        return None, None
    path = Path(candidate)
    if not path.is_file():
        return str(path), None
    try:
        return str(path.resolve()), sha256_file(path)
    except OSError:
        return str(path), None
