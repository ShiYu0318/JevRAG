import unittest

from jevrag.baselines.nli import hypothesis, verdict


class NLIVerdictTest(unittest.TestCase):
    def test_single_question(self):
        self.assertEqual(verdict([0.9, 0.1], [0.0, 0.1], [], 0.5, 0.5), "sufficient")
        self.assertEqual(verdict([0.9, 0.1], [0.0, 0.8], [], 0.5, 0.5), "conflicting")
        self.assertEqual(verdict([0.2, 0.1], [0.9, 0.9], [], 0.5, 0.5), "insufficient")

    def test_conjunction(self):
        self.assertEqual(verdict([], [], [0.9, 0.8], 0.5, 0.5), "sufficient")
        self.assertEqual(verdict([], [], [0.9, 0.1], 0.5, 0.5), "partial")
        self.assertEqual(verdict([], [], [0.2, 0.1], 0.5, 0.5), "insufficient")

    def test_hypothesis(self):
        self.assertEqual(hypothesis("甲在哪年？", "1962年"), "甲在哪年？答案是1962年。")


if __name__ == "__main__":
    unittest.main()
