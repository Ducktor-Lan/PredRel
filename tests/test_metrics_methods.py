from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predrel.metrics import (
    METHOD_IDS,
    ndcg_at_k,
    score_benchmark_query,
    spearman_corr,
    teacher_top1_rank,
    topk_positions,
    topk_recall,
)


class BenchmarkMetricsTests(unittest.TestCase):
    def test_method_ids_frozen(self) -> None:
        self.assertEqual(
            tuple(METHOD_IDS),
            ("student", "supcon", "fusion", "raw", "pca", "mlp", "hidden", "readout_profile"),
        )

    def test_topk_positions_tie_aware(self) -> None:
        order = topk_positions(np.asarray([1.0, 1.0 + 1e-12, 0.0]), 2)
        self.assertEqual(order, [0, 1])

    def test_topk_recall_perfect_and_zero(self) -> None:
        teacher = np.asarray([0.7, 0.2, 0.1])
        self.assertAlmostEqual(topk_recall(teacher, teacher, 1), 1.0)
        self.assertAlmostEqual(topk_recall(teacher, np.asarray([0.1, 0.2, 0.7]), 1), 0.0)

    def test_spearman_perfect_and_reversed(self) -> None:
        x = np.asarray([3.0, 1.0, 0.0])
        self.assertAlmostEqual(spearman_corr(x, x), 1.0, places=12)
        self.assertLess(spearman_corr(x, -x), -0.99)

    def test_ndcg_perfect_is_one(self) -> None:
        beta = np.asarray([0.6, 0.3, 0.1])
        self.assertAlmostEqual(ndcg_at_k(beta, beta, 10), 1.0, places=12)

    def test_ndcg_reversed_below_one(self) -> None:
        beta = np.asarray([0.6, 0.3, 0.1, 0.05])
        self.assertLess(ndcg_at_k(np.asarray([0.0, 0.0, 0.0, 1.0]), beta, 10), 1.0)

    def test_top1_rank_perfect_is_one(self) -> None:
        beta = np.asarray([0.6, 0.3, 0.1])
        self.assertAlmostEqual(teacher_top1_rank(beta, beta), 1.0)

    def test_score_benchmark_query_fields(self) -> None:
        beta = np.asarray([0.6, 0.3, 0.1])
        raw = np.asarray([14.0, 12.0, 11.0])
        scores = np.asarray([1.0, 0.0, -1.0])
        full = np.asarray([1.0, 0.0, -1.0, -2.0, -3.0])
        labels = np.asarray([0, 0, 0, 1, 1])
        out = score_benchmark_query(
            method_true_scores=scores,
            teacher_beta_block=beta,
            teacher_raw_block=raw,
            full_method_scores=full,
            support_labels=labels,
            query_label=0,
            ks=(1, 5, 10),
        )
        for field in ("teacher_topk_recall_mean", "teacher_topk_positions", "method_topk_positions",
                      "spearman_vs_beta", "spearman_vs_raw", "ndcg_at_10", "teacher_top1_rank",
                      "label_hit_at_1", "label_hit_at_5"):
            self.assertIn(field, out)
        self.assertEqual(out["label_hit_at_1"], 1)
        self.assertEqual(out["label_hit_at_5"], 1)


if __name__ == "__main__":
    unittest.main()
