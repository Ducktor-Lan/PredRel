from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predrel.data import RealDataset
from predrel.metrics import METHOD_IDS
from predrel.experiments.benchmark import run_benchmark, write_evidence
from predrel.teacher_bridge import WEEK01_SOURCE_SHA256, ReadoutExtraction


ROOT = Path(__file__).resolve().parents[1]


def _fake_dataset() -> RealDataset:
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 60)
    features = np.column_stack((np.arange(labels.size, dtype=np.float64), labels.astype(np.float64)))
    return RealDataset(
        dataset_id="sklearn_breast_cancer",
        features=features,
        labels=labels,
        sample_ids=tuple(f"sklearn_breast_cancer:row:{index}" for index in range(labels.size)),
        metadata={
            "dataset_id": "sklearn_breast_cancer",
            "content_sha256": "17a561dca8f46aa20420dab2e09abad32c3d16e010bbce76d8ddc85a9db250d7",
        },
    )


def _fake_extract_readout(**kwargs: object) -> ReadoutExtraction:
    support_ids = tuple(kwargs["support_ids"])  # type: ignore[arg-type]
    query_ids = tuple(kwargs["query_ids"])  # type: ignore[arg-type]
    n_support = len(support_ids)
    n_query = len(query_ids)
    indices = np.asarray([int(s.rsplit(":", 1)[1].split(":")[0]) for s in support_ids], dtype=np.float64)
    query_index = np.asarray([int(str(s).split(":row:")[1].split(":")[0]) for s in query_ids], dtype=np.float64)
    raw = np.zeros((1, 2, n_query, n_support), dtype=np.float64)
    for head in range(2):
        for qi in range(n_query):
            raw[0, head, qi, :] = 13.0 + indices * 1e-2 + query_index[qi] * 1e-4 + head * 1e-3
    mean = raw.mean(axis=(0, 1))
    shifted = mean - mean.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    alpha = exp / exp.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    hidden_support = np.ascontiguousarray(rng.normal(size=(n_support, 8)))
    hidden_query = np.ascontiguousarray(rng.normal(size=(n_query, 8)))
    return ReadoutExtraction(
        alpha=np.ascontiguousarray(alpha),
        raw_scores=np.ascontiguousarray(raw),
        support_ids=support_ids,
        query_ids=query_ids,
        teacher_source_sha256=WEEK01_SOURCE_SHA256,
        decoder_readout_api="mock.decoder",
        raw_score_hook_path="mock.raw",
        runtime={"python": "mock", "cuda_available": True},
        hidden_support_aggregated=hidden_support,
        hidden_query_aggregated=hidden_query,
    )


class MockPipelineTests(unittest.TestCase):
    def test_smoke_pipeline_end_to_end_with_mock_teacher(self) -> None:
        manifest = json.loads((ROOT / "provenance" / "dataset_manifest_benchmark.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ExitStack() as stack:
                stack.enter_context(mock.patch("predrel.experiments.benchmark.materialize_manifest", return_value=[_fake_dataset()]))
                stack.enter_context(mock.patch("predrel.experiments.benchmark.extract_readout", side_effect=_fake_extract_readout))
                result = run_benchmark(
                    mode="smoke",
                    dataset_manifest_path=ROOT / "provenance" / "dataset_manifest_benchmark.json",
                    dataset_cache_dir=root / "cache",
                    n_estimators=1,
                    model_cache_dir=manifest["teacher"]["model_cache_dir"],
                )
            self.assertEqual(result.evidence["executed_method_ids"], list(METHOD_IDS))
            self.assertEqual(result.evidence["executed_dataset_ids"], [str(manifest["smoke"]["dataset_id"])])
            evidence = write_evidence(
                result,
                output_dir=root / "evidence",
                run_id="week10-smoke-mock",
                run_input={"mode": "smoke", "run_id": "week10-smoke-mock"},
            )
            self.assertNotIn("support_features", (evidence / "eval_rows.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result.evidence["scientific_gate"]["decision"], "not_evaluated_in_smoke")
            self.assertGreater(len(result.eval_rows), 0)

    def test_train_blocks_reject_short_true_class(self) -> None:
        from predrel.experiments.benchmark import _train_blocks

        alpha = np.asarray([[0.5, 0.5]])
        raw = np.zeros((1, 1, 1, 2))
        with self.assertRaises(Exception):
            _train_blocks(
                alpha=alpha,
                raw_scores=raw,
                support_labels=np.asarray([0, 1]),
                pseudo_query_labels=np.asarray([0]),
                true_positions=[[0]],
            )


if __name__ == "__main__":
    unittest.main()
