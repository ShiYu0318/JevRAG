"""End-to-end plumbing against the in-process mock server."""
import json
import socket
import tempfile
import unittest
from pathlib import Path

from jevrag import ops
from jevrag.backends import BudgetExceeded, SystemOneClient, SystemOneError
from jevrag.cache import CallCache
from jevrag.experiments import e0
from jevrag.mock import serve
from jevrag.runlog import Run
from jevrag.tables import make_tables


def fixture(k, n=8):
    """Tiny S / I-hard instances in the benchmark's JSONL shape."""
    rows = []
    for i in range(n):
        gold = {"pid": f"{i}_0", "text": f"第{i}號建築完成於{1900 + i}年。", "role": "gold", "retrieval_rank": None}
        hard = [{"pid": f"{i}_{j}", "text": f"第{i}號建築的第{j}段介紹。", "role": "hard_negative",
                 "retrieval_rank": j} for j in range(1, k + 1)]
        q = f"第{i}號建築完成於哪一年？"
        for cond, label, ctx in (("S", "sufficient", [gold] + hard[:k - 1]), ("I-hard", "insufficient", hard[:k])):
            rows.append({"id": f"t-{i}-{cond}-k{k}", "condition": cond, "label": label, "question": q,
                         "answers": [f"{1900 + i}年"], "contexts": ctx})
    return rows


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.srv = serve(cls.port, background=True)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def client(self, **kw):
        return SystemOneClient(f"http://127.0.0.1:{self.port}", "mock", name="mock", **kw)

    def test_cache_hit_and_version(self):
        c = self.client(cache=CallCache(self.root / "c1.sqlite"))
        a = c.ask("問題", {"x": {"type": "noul", "instructions": "?"}})
        b = c.ask("問題", {"x": {"type": "noul", "instructions": "?"}})
        self.assertFalse(a.cache_hit)
        self.assertTrue(b.cache_hit)
        self.assertEqual(a["x"].p, b["x"].p)
        self.assertEqual(c.meter.versions, {"mock-0": 2})
        pinned = self.client(cache=CallCache(self.root / "c1.sqlite"), pin_version="other")
        with self.assertRaises(SystemOneError):
            pinned.ask("問題", {"x": {"type": "noul", "instructions": "?"}})

    def test_kev_run_check(self):
        from jevrag.backends import get_backend, loaded_run, run_mismatch
        info = {"models": [{"name": "kev-latest", "run": "jaredpalmer/kev-0.8b"}]}
        self.assertEqual(loaded_run(info), "jaredpalmer/kev-0.8b")
        self.assertEqual(run_mismatch(get_backend("kev08b"), info), "")
        self.assertIn("not kev-4b", run_mismatch(get_backend("kev4b"), info))
        self.assertIn("no server", run_mismatch(get_backend("kev4b"), {}))
        self.assertEqual(run_mismatch(self.client(), {}), "")  # no expected run: nothing to check

    def test_budget(self):
        c = self.client(usd_per_mtok=1e6, budget_usd=0.5)
        c.ask("a", {"x": {"type": "noul", "instructions": "?"}})
        with self.assertRaises(BudgetExceeded):
            c.ask("b", {"x": {"type": "noul", "instructions": "?"}})

    def test_ops(self):
        c = self.client()
        g = ops.grade_passages(c, "中央大學哪年復校？", ["中央大學1962年復校。", "天氣很好。"])
        self.assertEqual(len(g), 2)
        packed = ops.grade_passages(c, "中央大學哪年復校？", ["中央大學1962年復校。", "天氣很好。"], mode="packed")
        self.assertEqual(len(packed), 2)
        v = ops.assess_sufficiency(c, "中央大學哪年復校？", ["中央大學1962年復校。"])
        self.assertIn(ops.decide_action(v), {"answer", "answer_with_both_sides", "answer_partial_and_flag_gap",
                                             "abstain_or_search_web", "correct_premise"})
        self.assertAlmostEqual(ops.aggregate([0.5, 0.5], "noisy_or"), 0.75)
        self.assertTrue(ops.compress(c, "復校", "中央大學1962年復校。天氣很好。", tau=0.0))

    def test_e0_run_and_tables(self):
        for k in (3, 10):
            d = self.root / f"tc-k{k}"
            d.mkdir()
            (d / "dev.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fixture(k)))
        cfg = {"dataset": str(self.root / "tc-k3"), "length_dataset": str(self.root / "tc-k10"),
               "latency": {"n_questions": [1, 4], "state_chars": [64, 256], "reps": 2},
               "length": {"n_passages": [1, 3, 10], "n_items": 6},
               "packed": {"n_items": 4}, "consistency": {"n_items": 5, "repeats": 2}}
        results = self.root / "results"
        with Run("E0", "mock", "dev", cfg, root=results) as run:
            run.attach(self.client(cache=CallCache(self.root / "e0.sqlite")))
            out = e0.run(run.clients[0], run, "dev", cfg)
        self.assertEqual(len(out["latency"]), 4)
        self.assertEqual([r["n_passages"] for r in out["length"]], [1, 3, 10])
        self.assertEqual(out["consistency"]["repeat_flip_rate_max"], 0.0)  # mock is deterministic
        man = json.loads((run.dir / "manifest.json").read_text())
        self.assertEqual(man["calls"], sum(1 for _ in (run.dir / "calls.jsonl").open()))
        self.assertFalse(man["test_read"])
        written = make_tables(results, self.root / "docs")
        self.assertIn("E0 backend profile", written[0].read_text())


if __name__ == "__main__":
    unittest.main()
