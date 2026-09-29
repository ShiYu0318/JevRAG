import math
import unittest

from jevrag import metrics as M


class RankingTest(unittest.TestCase):
    def test_ndcg_hand(self):
        qrels = {"a": 3, "b": 1}
        # ranked b, x, a -> DCG = 1/log2(2) + 7/log2(4) = 1 + 3.5
        dcg = 1 + 7 / 2
        idcg = 7 + 1 / math.log2(3)
        self.assertAlmostEqual(M.ndcg_at_k(["b", "x", "a"], qrels, 10), dcg / idcg)

    def test_mrr_recall(self):
        qrels = {"a": 3}
        self.assertEqual(M.mrr_at_k(["x", "y", "a"], qrels, 10), 1 / 3)
        self.assertEqual(M.mrr_at_k(["x", "y", "a"], qrels, 2), 0.0)
        self.assertEqual(M.recall_at_k(["x", "a"], {"a": 1, "b": 1}, 2), 0.5)


class CalibrationTest(unittest.TestCase):
    def test_perfect_and_worst(self):
        self.assertEqual(M.ece([1.0, 0.0], [1, 0]), 0.0)
        self.assertEqual(M.ece([1.0, 0.0], [0, 1]), 1.0)

    def test_ece_hand(self):
        # two bins (width, B=2): [0.2, 0.4] acc 0.5 conf 0.3; [0.8] acc 1 conf 0.8
        ece = M.ece([0.2, 0.4, 0.8], [1, 0, 1], n_bins=2)
        self.assertAlmostEqual(ece, 2 / 3 * 0.2 + 1 / 3 * 0.2)

    def test_equal_mass_bins_cover_all(self):
        pts = M.reliability([i / 10 for i in range(10)], [1] * 10, n_bins=3, scheme="mass")
        self.assertEqual(sum(n for _, _, n in pts), 10)

    def test_multiclass(self):
        probs = [{"a": 0.7, "b": 0.3}, {"a": 0.4, "b": 0.6}]
        self.assertAlmostEqual(M.brier_multiclass(probs, ["a", "a"], ["a", "b"]),
                               ((0.09 + 0.09) + (0.36 + 0.36)) / 2)
        cw = M.classwise_ece(probs, ["a", "b"], ["a", "b"], n_bins=10)
        self.assertEqual(set(cw), {"a", "b"})
        self.assertAlmostEqual(M.top_label_ece(probs, ["a", "b"], n_bins=10), (0.3 + 0.4) / 2)

    def test_temperature_and_platt_recover_parameters(self):
        import random
        r = random.Random(1)
        scores = [r.gauss(0, 2) for _ in range(4000)]
        labels = [int(r.random() < 1 / (1 + math.exp(-(0.5 * s - 1)))) for s in scores]
        a, b = M.fit_platt(scores, labels)
        self.assertAlmostEqual(a, 0.5, delta=0.05)
        self.assertAlmostEqual(b, -1.0, delta=0.1)
        probs = M.apply_platt(scores, 2.0, 0.0)  # over-confident by a factor of 4 relative to the truth slope
        t = M.fit_temperature(probs, [int(r.random() < p ** 0.25 / (p ** 0.25 + (1 - p) ** 0.25)) for p in probs])
        self.assertAlmostEqual(t, 4.0, delta=0.6)

    def test_auroc(self):
        self.assertEqual(M.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertEqual(M.auroc([0.5, 0.5], [0, 1]), 0.5)

    def test_confident_error(self):
        self.assertEqual(M.confident_error_rate([0.95, 0.95, 0.5], [0, 1, 0]), 1 / 3)


class SelectiveTest(unittest.TestCase):
    def test_curve(self):
        covs, risks = M.risk_coverage([0.9, 0.8, 0.1], [1, 0, 1])
        self.assertEqual(covs, [1 / 3, 2 / 3, 1.0])
        self.assertEqual(risks, [0.0, 0.5, 1 / 3])
        self.assertAlmostEqual(M.aurc([0.9, 0.8, 0.1], [1, 0, 1]), (0 + 0.5 + 1 / 3) / 3)
        self.assertEqual(M.coverage_at_risk([0.9, 0.8, 0.1], [1, 0, 1], 0.05), 1 / 3)
        self.assertEqual(M.risk_at_coverage([0.9, 0.8, 0.1], [1, 0, 1], 0.6), 0.5)


class ClassificationTest(unittest.TestCase):
    def test_f1(self):
        t = ["a", "a", "b", "b"]
        p = ["a", "b", "b", "b"]
        f = M.per_class_f1(t, p)
        self.assertAlmostEqual(f["a"], 2 / 3)
        self.assertAlmostEqual(f["b"], 0.8)
        self.assertAlmostEqual(M.macro_f1(t, p), (2 / 3 + 0.8) / 2)
        self.assertEqual(M.confusion(t, p)["a"]["b"], 1)

    def test_kappa_flip(self):
        self.assertEqual(M.cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(M.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)
        self.assertEqual(M.flip_rate([1, 2, 3], [1, 2, 4]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
