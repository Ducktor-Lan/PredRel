"""Configuration for the Week 1 TabPFN-v3 teacher.

The configuration intentionally separates the model used to obtain decoder
readouts from the shadow model used to obtain hidden embeddings.  TabPFN v3's
cached inference path does not retain train embeddings, so pretending that the
two are produced by one fitted estimator would make the provenance misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    """Immutable, reproducible configuration for one extraction run."""

    seed: int = 17
    n_estimators: int = 1
    device: str = "cuda"
    inference_precision: str = "float32"
    readout_fit_mode: str = "fit_with_cache"
    embedding_fit_mode: str = "fit_preprocessors"
    kv_cache_precision: str = "auto"
    n_preprocessing_jobs: int = 1
    model_cache_dir: str | None = None
    run_id: str | None = None
    raw_score_atol: float = 1e-6
    raw_score_rtol: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.n_estimators not in (1, 8):
            raise ValueError("Week 1 permits only n_estimators=1 (correctness) or 8 (smoke)")
        if self.device != "cuda":
            raise ValueError("Week 1 is intentionally configured for the local CUDA backend")
        if self.inference_precision != "float32":
            raise ValueError("Week 1 validation requires inference_precision='float32'")
        if self.readout_fit_mode != "fit_with_cache":
            raise ValueError("readout_fit_mode must be 'fit_with_cache'")
        if self.embedding_fit_mode != "fit_preprocessors":
            raise ValueError("embedding_fit_mode must be 'fit_preprocessors'")
        if self.kv_cache_precision != "auto":
            raise ValueError("Week 1 validation requires kv_cache_precision='auto'")
        if self.n_preprocessing_jobs != 1:
            raise ValueError("Week 1 validation requires n_preprocessing_jobs=1")
        if self.model_cache_dir is not None:
            object.__setattr__(self, "model_cache_dir", str(Path(self.model_cache_dir)))
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id.strip()):
            raise ValueError("run_id must be a non-empty string when provided")
        for field_name in ("raw_score_atol", "raw_score_rtol"):
            value = getattr(self, field_name)
            if not isinstance(value, (float, int)) or value <= 0:
                raise ValueError(f"{field_name} must be positive")

    @property
    def is_ensemble_smoke(self) -> bool:
        """Whether this configuration exercises the requested 8-estimator smoke run."""

        return self.n_estimators == 8
