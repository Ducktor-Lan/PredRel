from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predrel.baselines import (
    BaselineError,
    apply_pca,
    fit_pca,
    mlp_penultimate_embeddings,
    mlp_scores,
    negative_euclidean_scores,
    pca_scores,
    raw_scores,
    train_mlp_classifier,
)
from predrel.fusion import average_ranks_descending, fuse_rank_average


torch = None
try:
    import torch as _torch

    torch = _torch
except ImportError:
    torch = None


class BaselinesTests(unittest.TestCase):
    def test_raw_scores_nearest_is_top(self) -> None:
        support = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        query = np.asarray([[0.1, 0.0]])
        scores = raw_scores(support, query)
        self.assertEqual(scores.shape, (1, 3))
        self.assertEqual(int(np.argmax(scores[0])), 0)

    def test_negative_euclidean_rejects_width_mismatch(self) -> None:
        with self.assertRaises(BaselineError):
            negative_euclidean_scores(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_pca_fit_apply_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        support = rng.normal(size=(30, 8))
        mean = support.mean(axis=0)
        std = np.where(support.std(axis=0) > 0, support.std(axis=0), 1.0)
        support_std = (support - mean) / std
        projector = fit_pca(support_std, n_components=4)
        self.assertEqual(projector["n_components"], 4)
        projected = apply_pca(projector, support_std)
        self.assertEqual(projected.shape, (30, 4))
        scores = pca_scores(support, support[:2], n_components=4)
        self.assertEqual(scores.shape, (2, 30))

    def test_pca_caps_components(self) -> None:
        rng = np.random.default_rng(1)
        support = rng.normal(size=(6, 20))
        mean = support.mean(axis=0)
        std = np.where(support.std(axis=0) > 0, support.std(axis=0), 1.0)
        projector = fit_pca((support - mean) / std, n_components=32)
        self.assertEqual(projector["n_components"], 5)

    @unittest.skipIf(torch is None, "torch is required for the MLP baseline test")
    def test_mlp_trains_and_scores(self) -> None:
        rng = np.random.default_rng(2)
        support = np.vstack([rng.normal(loc=-2.0, size=(20, 4)), rng.normal(loc=2.0, size=(20, 4))])
        labels = np.asarray([0] * 20 + [1] * 20, dtype=np.int64)
        params = train_mlp_classifier(support, labels, epochs=3)
        self.assertEqual(len(params["loss_history"]), 3)
        self.assertTrue(all(np.isfinite(params["loss_history"])))
        embeddings = mlp_penultimate_embeddings(params, support)
        self.assertEqual(embeddings.shape, (40, 128))
        scores = mlp_scores(params, support, support[:3])
        self.assertEqual(scores.shape, (3, 40))
        self.assertTrue(np.all(np.isfinite(scores)))


class FusionTests(unittest.TestCase):
    def test_average_ranks_ties_share_mean(self) -> None:
        ranks = average_ranks_descending(np.asarray([1.0, 1.0, 0.0]))
        self.assertAlmostEqual(float(ranks[0]), 1.5)
        self.assertAlmostEqual(float(ranks[1]), 1.5)
        self.assertAlmostEqual(float(ranks[2]), 3.0)

    def test_fuse_agreement_keeps_order(self) -> None:
        student = np.asarray([3.0, 2.0, 1.0, 0.0])
        supcon = np.asarray([3.0, 2.0, 1.0, 0.0])
        fused = fuse_rank_average(student, supcon)
        self.assertEqual(list(np.argsort(-fused)), [0, 1, 2, 3])

    def test_fuse_disagreement_averages(self) -> None:
        student = np.asarray([2.0, 1.0, 0.0])
        supcon = np.asarray([0.0, 1.0, 2.0])
        fused = fuse_rank_average(student, supcon)
        # ranks (1,3),(2,2),(3,1) -> fused ranks (2,2,2): exact three-way tie.
        self.assertTrue(np.allclose(fused, fused[0]))
        # A clear compromise case: ranks (1,2),(2,1),(3,3) fuse to (1.5,1.5,3):
        # rows 0 and 1 tie for best, row 2 is worst.
        fused2 = fuse_rank_average(np.asarray([2.0, 1.0, 0.0]), np.asarray([1.0, 2.0, 0.0]))
        self.assertLess(float(fused2[2]), float(fused2[0]))
        self.assertAlmostEqual(float(fused2[0]), float(fused2[1]))

    def test_fuse_rejects_misalignment(self) -> None:
        with self.assertRaises(Exception):
            fuse_rank_average(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
