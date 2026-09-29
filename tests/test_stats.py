import unittest

from jevrag import stats


class StatsTest(unittest.TestCase):
    def test_mcnemar(self):
        # b=5, c=0 -> exact two-sided p = 2 * 0.5**5
        r = stats.mcnemar([1] * 5 + [1] * 3, [0] * 5 + [1] * 3)
        self.assertEqual((r["b"], r["c"]), (5, 0))
        self.assertAlmostEqual(r["p"], 2 / 32)
        self.assertEqual(stats.mcnemar([1, 0], [1, 0])["p"], 1.0)

    def test_holm(self):
        out = stats.holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(out[0][0], 0.03)
        self.assertAlmostEqual(out[2][0], 0.06)
        self.assertAlmostEqual(out[1][0], 0.06)
        self.assertEqual([r for _, r in out], [True, False, False])

    def test_bootstrap_deterministic(self):
        xs = [1, 0, 1, 1, 0, 1, 1, 1]
        acc = lambda idx: sum(xs[i] for i in idx) / len(idx)  # noqa: E731
        a = stats.bootstrap_ci(len(xs), acc, n=500, seed=3)
        b = stats.bootstrap_ci(len(xs), acc, n=500, seed=3)
        self.assertEqual(a, b)
        self.assertLessEqual(a[1], a[0])
        self.assertLessEqual(a[0], a[2])

    def test_paired_diff(self):
        a = [1] * 50
        b = [0] * 50
        r = stats.paired_bootstrap_diff(50, lambda i: sum(a[j] for j in i) / len(i),
                                        lambda i: sum(b[j] for j in i) / len(i), n=200)
        self.assertEqual(r["diff"], 1.0)
        self.assertTrue(stats.non_inferior(r["lo"], 0.01))
        self.assertEqual(r["p"], 0.0)


if __name__ == "__main__":
    unittest.main()
