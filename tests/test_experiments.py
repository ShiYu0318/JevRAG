"""E1-E4 end to end on tiny fixtures against the mock server."""
import json
import tempfile
import unittest
from pathlib import Path

from jevrag.backends import SystemOneClient
from jevrag.cache import CallCache
from jevrag.experiments import e1, e2, e3, e4
from jevrag.mock import serve
from jevrag.runlog import Run
from jevrag.tables import make_tables

from test_pipeline import free_port

LABELS = {"S": "sufficient", "I-hard": "insufficient", "I-easy": "insufficient", "P": "partial",
          "C": "conflicting", "S-conj": "sufficient", "J": "sufficient"}


def instances(split, n=6):
    rows = []
    for i in range(n):
        gold = {"pid": f"{i}_0", "text": f"第{i}號建築完成於{1900 + i}年。", "role": "gold", "retrieval_rank": None}
        hard = [{"pid": f"{i}_{j}", "text": f"第{i}號建築的第{j}段介紹。", "role": "hard_negative",
                 "retrieval_rank": j} for j in (1, 2, 3)]
        easy = [{"pid": f"9{j}_{i}", "text": f"無關的第{j}段文字。", "role": "easy_negative",
                 "retrieval_rank": None} for j in (1, 2, 3)]
        pert = dict(gold, pid=f"{i}_0_perturbed", text=f"第{i}號建築完成於{1950 + i}年。", role="perturbed_gold")
        ctxs = {"S": [gold] + hard[:2], "I-hard": hard, "I-easy": easy, "P": [gold] + hard[:2],
                "C": [gold, pert, hard[0]], "S-conj": [gold] + hard[:2], "J": [gold, dict(hard[0], role="injection"),
                                                                             hard[1]]}
        for cond, ctx in ctxs.items():
            rows.append({"id": f"{split}-{i}-{cond}", "condition": cond, "label": LABELS[cond],
                         "question": f"第{i}號建築完成於哪一年？", "answers": [f"{1900 + i}年"], "contexts": ctx,
                         "parallel_id": f"{split}-{i}-{cond}-zhHans"})
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class ExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.srv = serve(cls.port, background=True)
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        tcdir = root / "tc"
        for split in ("dev", "training", "test"):
            rows = instances(split)
            write(tcdir / f"{split}.jsonl", rows)
            write(tcdir / f"{split}.zh-Hans.jsonl",
                  [dict(r, id=r["parallel_id"], parallel_id=r["id"]) for r in rows])
        # mini DRCD + pools for E1
        raw = root / "raw"
        raw.mkdir()
        arts = [{"id": str(a), "title": f"t{a}", "paragraphs": [
            {"id": f"{a}-{p}", "context": f"第{a}號建築的第{p}段，完成於{1900 + a * 10 + p}年。",
             "qas": [{"id": f"{a}-{p}-1", "question": f"第{a}號建築第{p}段完成於哪一年？",
                      "answers": [{"id": "1", "text": f"{1900 + a * 10 + p}年", "answer_start": 0}]}]}
            for p in range(3)]} for a in range(3)]
        for s in ("training", "dev", "test"):
            (raw / f"DRCD_{s}.json").write_text(json.dumps({"version": "1", "data": arts if s == "dev" else []},
                                                           ensure_ascii=False))
        pools = [{"qid": f"{a}_{p}_1", "question": f"第{a}號建築第{p}段完成於哪一年？", "answers": ["x"],
                  "qrels": {f"{a}_{p}": 3},
                  "candidates": [{"pid": f"{x}_{y}", "rank": r, "score": 1.0}
                                 for r, (x, y) in enumerate([(b, q) for b in range(3) for q in range(3)], 1)]}
                 for a in range(3) for p in range(3)]
        write(root / "pools" / "dev.jsonl", pools)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def go(self, mod, name, cfg):
        with Run(name, "mock", "dev", cfg, root=self.root / "results") as run:
            c = run.attach(SystemOneClient(f"http://127.0.0.1:{self.port}", "mock", name="mock",
                                           cache=CallCache(self.root / "cache.sqlite")))
            return mod.run(c, run, "dev", cfg)

    def test_e1(self):
        out = self.go(e1, "E1", {"pools": str(self.root / "pools"), "raw": str(self.root / "raw"),
                                 "n_items": 5, "packed_n_items": 2, "bootstrap": 50})
        self.assertIn("rrf", out["methods"])
        self.assertIn("mock:pointwise:noul", out["methods"])
        self.assertIn("vs_reference_ndcg@10", out["methods"]["mock:pointwise:noul"])
        self.assertEqual(out["pool_gold_recall@20"], 1.0)

    def test_e2(self):
        out = self.go(e2, "E2", {"dataset": str(self.root / "tc"), "n_per_condition": 4, "fit_n_per_condition": 4})
        self.assertEqual(set(out["raw"]), {"hard", "easy", "all_passages", "set"})
        self.assertGreater(out["temperature"]["passage"], 0)
        self.assertIn("supported", out["h2"])

    def test_e3(self):
        out = self.go(e3, "E3", {"dataset": str(self.root / "tc"), "n_per_condition": 4, "bootstrap": 50})
        self.assertEqual(out["n"], 20)
        self.assertEqual(set(out["per_passage"]), {"max", "noisy_or", "sum"})
        self.assertIn("macro_f1", out["two_stage"]["max"])
        with Run("E3", "mock", "test", {}, root=self.root / "results2") as run:
            c = run.attach(SystemOneClient(f"http://127.0.0.1:{self.port}", "mock", name="mock"))
            with self.assertRaises(ValueError):
                e3.run(c, run, "test", {"dataset": str(self.root / "tc")})

    def test_e4(self):
        out = self.go(e4, "E4", {"dataset": str(self.root / "tc"), "n_per_condition": 3, "bootstrap": 50})
        self.assertEqual(len(out["settings"]), 4)
        self.assertIn("flip_rate_vs_base", out["settings"]["zh-Hans/en"])

    def test_tables(self):
        self.go(e3, "E3", {"dataset": str(self.root / "tc"), "n_per_condition": 2, "bootstrap": 20})
        written = make_tables(self.root / "results", self.root / "docs")
        self.assertTrue(any(p.name.startswith("E3") for p in written))


if __name__ == "__main__":
    unittest.main()
