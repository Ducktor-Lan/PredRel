"""Decoder-readout helpers for the TabPFN-v3 ``ManyClassDecoder``.

``ManyClassDecoder.forward`` normally uses a fused attention kernel and does
not materialise example-level scores.  The helpers below attach a short-lived
forward-pre-hook to the decoder, reconstruct the exact scaled QK score at the
last observable point before softmax, and validate it against Prior Labs'
official readout implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..contracts import ContractError


class DecoderExtractionError(RuntimeError):
    """Raised when the installed TabPFN decoder cannot satisfy the v3 contract."""


@dataclass(frozen=True, slots=True)
class DecoderReadoutReference:
    """Official decoder weights before their columns are aligned to support IDs."""

    weights_per_estimator: NDArray[np.float64]
    training_row_indices: NDArray[np.int64]
    api_path: str = "tabpfn_extensions.interpretability.get_decoder_readout"


@dataclass(frozen=True, slots=True)
class RawScoreCapture:
    """Pre-softmax scaled decoder scores in canonical ``[E,H,Q,N]`` layout."""

    scores: NDArray[np.float64]
    hook_path: str
    num_heads: int
    head_dim: int
    softmax_scaling_applied: bool


def _as_float64(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise DecoderExtractionError(f"{name} contains non-finite values")
    return array


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on server environment
        raise DecoderExtractionError("Raw decoder-score extraction requires PyTorch") from error
    return torch


def _get_fitted_model(estimator: Any) -> Any:
    try:
        return estimator.model_
    except (AttributeError, ValueError) as error:
        raise DecoderExtractionError(
            "Expected a fitted local TabPFNClassifier exposing model_; cloud/API backends are unsupported"
        ) from error


def _find_many_class_decoder(model: Any) -> Any:
    decoders = [module for module in model.modules() if type(module).__name__ == "ManyClassDecoder"]
    if len(decoders) != 1:
        if not decoders:
            raise DecoderExtractionError(
                "No ManyClassDecoder was found. This Week 1 extractor supports TabPFN-v3 classification only."
            )
        raise DecoderExtractionError(
            "Expected exactly one ManyClassDecoder, "
            f"but found {len(decoders)}; refusing an ambiguous raw-score hook"
        )
    return decoders[0]


def official_decoder_readout(estimator: Any, query_features: ArrayLike) -> DecoderReadoutReference:
    """Return the official per-estimator v3 decoder weights and row indices.

    The official function is deliberately called with
    ``average_over_estimators=False`` so the independent raw-score parity check
    retains the estimator axis.
    """

    try:
        from tabpfn_extensions.interpretability import get_decoder_readout
    except ImportError as error:  # pragma: no cover - depends on server environment
        raise DecoderExtractionError(
            "Install tabpfn-extensions[interpretability]==0.6.2 to obtain the official decoder readout"
        ) from error

    try:
        weights, training_row_indices = get_decoder_readout(
            estimator,
            np.asarray(query_features),
            average_over_estimators=False,
        )
    except Exception as error:  # The upstream API deliberately raises multiple concrete errors.
        raise DecoderExtractionError(f"Official decoder-readout extraction failed: {error}") from error

    weights_array = _as_float64(weights, name="official decoder weights")
    if weights_array.ndim == 2:
        weights_array = weights_array[None, ...]
    if weights_array.ndim != 3:
        raise DecoderExtractionError(
            "Official decoder weights must have shape [E,Q,N] or [Q,N], "
            f"got {weights_array.shape!r}"
        )
    if np.any(weights_array < -1e-12):
        raise DecoderExtractionError("Official decoder weights contain negative values")
    if not np.allclose(weights_array.sum(axis=-1), 1.0, rtol=0.0, atol=1e-6):
        raise DecoderExtractionError("Official decoder weights do not sum to one along support rows")

    indices = np.asarray(training_row_indices)
    if indices.ndim != 1 or indices.shape[0] != weights_array.shape[-1]:
        raise DecoderExtractionError(
            "Official decoder training-row indices do not match the support dimension"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise DecoderExtractionError("Official decoder training-row indices must be integers")
    indices = indices.astype(np.int64, copy=False)
    return DecoderReadoutReference(
        weights_per_estimator=np.ascontiguousarray(weights_array),
        training_row_indices=np.ascontiguousarray(indices),
    )


def _reconstruct_scaled_scores(decoder: Any, args: tuple[Any, ...]) -> Any:
    """Reconstruct ``[B,H,Q,N]`` scores from the decoder's exact forward inputs."""

    torch = _require_torch()
    if len(args) < 2:
        raise DecoderExtractionError(
            "ManyClassDecoder forward-pre-hook did not receive train keys and test embeddings"
        )
    train_keys, test_embeddings = args[:2]
    project_q = getattr(decoder, "_project_q", None)
    if project_q is None:
        raise DecoderExtractionError(
            "Installed ManyClassDecoder has no _project_q; the v8.5 raw-score hook is incompatible"
        )
    projected = project_q(train_keys, test_embeddings)
    if not isinstance(projected, tuple) or len(projected) != 2:
        raise DecoderExtractionError("ManyClassDecoder._project_q did not return (query, key)")
    query, key = projected
    scaling_layer = getattr(decoder, "softmax_scaling_layer", None)
    if scaling_layer is not None:
        query = scaling_layer(query, key.shape[1])
    head_dim = getattr(decoder, "head_dim", None)
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise DecoderExtractionError("ManyClassDecoder.head_dim is missing or invalid")

    # Q is the test/query axis and N is the support/train axis.  These values
    # are post scaling but immediately pre-softmax, matching attention_weights.
    scores = torch.einsum("bqhd,bnhd->bhqn", query, key).float()
    return scores / math.sqrt(head_dim)


def capture_raw_scores(estimator: Any, query_features: ArrayLike) -> RawScoreCapture:
    """Capture v3 decoder raw scores while triggering exactly one prediction call."""

    torch = _require_torch()
    query_values = np.asarray(query_features)
    model = _get_fitted_model(estimator)
    decoder = _find_many_class_decoder(model)
    num_heads = getattr(decoder, "num_heads", None)
    head_dim = getattr(decoder, "head_dim", None)
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise DecoderExtractionError("ManyClassDecoder.num_heads is missing or invalid")
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise DecoderExtractionError("ManyClassDecoder.head_dim is missing or invalid")
    captured: list[NDArray[np.float64]] = []

    def hook(module: Any, args: tuple[Any, ...]) -> None:
        scores = _reconstruct_scaled_scores(module, args)
        values = scores.detach().to(torch.float32).cpu().numpy()
        captured.append(_as_float64(values, name="captured decoder raw scores"))

    handle = decoder.register_forward_pre_hook(hook)
    try:
        # Calling predict, rather than raw logits, follows the same public
        # inference path used by the official readout helper.
        estimator.predict(query_values)
    except Exception as error:
        raise DecoderExtractionError(f"Raw-score capture prediction failed: {error}") from error
    finally:
        handle.remove()

    if not captured:
        raise DecoderExtractionError("The decoder hook was never invoked during prediction")
    expected_queries = query_values.shape[0]
    for index, piece in enumerate(captured):
        if piece.ndim != 4:
            raise DecoderExtractionError(
                f"Raw-score capture {index} must have shape [B,H,Q,N], got {piece.shape!r}"
            )
        if piece.shape[1] != num_heads:
            raise DecoderExtractionError(
                "Raw-score capture head dimension does not match ManyClassDecoder "
                f"({piece.shape[1]} != {num_heads})"
            )
        if piece.shape[-2] != expected_queries:
            raise DecoderExtractionError(
                "Decoder query chunking is unsupported for Week 1: captured query dimension "
                f"{piece.shape[-2]} != requested {expected_queries}"
            )
    scores = np.concatenate(captured, axis=0)
    if scores.ndim != 4:
        raise DecoderExtractionError(
            f"Concatenated raw scores must have shape [E,H,Q,N], got {scores.shape!r}"
        )
    return RawScoreCapture(
        scores=np.ascontiguousarray(scores, dtype=np.float64),
        hook_path="ManyClassDecoder.forward_pre_hook -> _project_q -> scaled_qk_einsum",
        num_heads=num_heads,
        head_dim=head_dim,
        softmax_scaling_applied=getattr(decoder, "softmax_scaling_layer", None) is not None,
    )


def assert_raw_score_parity(
    raw_scores: ArrayLike,
    official_weights_per_estimator: ArrayLike,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Require each estimator's head-softmaxed scores to match official alpha."""

    raw = _as_float64(raw_scores, name="raw_scores")
    official = _as_float64(official_weights_per_estimator, name="official_weights_per_estimator")
    if raw.ndim != 4:
        raise ContractError(f"raw_scores must have shape [E,H,Q,N], got {raw.shape!r}")
    if official.ndim != 3:
        raise ContractError(
            "official_weights_per_estimator must have shape [E,Q,N], "
            f"got {official.shape!r}"
        )
    reconstructed = _softmax(raw, axis=-1).mean(axis=1)
    if reconstructed.shape != official.shape:
        raise DecoderExtractionError(
            "Raw-score and official readout shapes differ: "
            f"{reconstructed.shape!r} != {official.shape!r}"
        )
    if not np.allclose(reconstructed, official, rtol=rtol, atol=atol):
        maximum_error = float(np.max(np.abs(reconstructed - official)))
        raise DecoderExtractionError(
            "Raw-score parity failed: mean_head(softmax(raw_scores)) does not reproduce "
            f"official decoder weights (max_abs_error={maximum_error:.3e})"
        )


def _softmax(values: NDArray[np.float64], *, axis: int) -> NDArray[np.float64]:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)
