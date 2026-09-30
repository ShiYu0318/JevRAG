"""Demo console API against the mock backend."""
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from jevrag.demo import serve
from jevrag.mock import serve as serve_mock

from test_pipeline import free_port


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers.get("content-type"), r.read()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class DemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        run = root / "results" / "E3-mock-dev-20260101T000000"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"run_id": run.name, "experiment": "E3", "backend": "mock",
                                                       "split": "dev", "calls": 3, "cost_usd": 0.0}))
        (run / "E3.json").write_text(json.dumps({"n": 1}))
        live = root / "results" / "E1-mock-dev-20260101T000001"
        live.mkdir()
        (live / "calls.jsonl").write_text("{}\n{}\n")
        data = root / "tc"
        data.mkdir()
        inst = {"id": "x-1", "condition": "S", "label": "sufficient", "question": "甲在哪年成立？",
                "answers": ["1962年"], "contexts": [{"text": "甲在1962年成立。", "role": "gold"},
                                                   {"text": "<script>alert(1)</script>", "role": "hard_negative"}]}
        (data / "dev.jsonl").write_text(json.dumps(inst, ensure_ascii=False) + "\n")
        (root / "milestones.json").write_text(json.dumps([{"title": "M1", "status": "done"}]))
        cls.mock_port = free_port()
        cls.mock = serve_mock(cls.mock_port, background=True)
        os.environ["MOCK_PORT"] = str(cls.mock_port)
        cls.port = free_port()
        cls.srv = serve(cls.port, root / "results", data, root / "milestones.json", cache=None,
                        backends=["mock", "laya-ml"], background=True)
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.mock):
            s.shutdown()
            s.server_close()
        os.environ.pop("MOCK_PORT", None)
        cls.tmp.cleanup()

    def test_static(self):
        status, ctype, body = get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("JevRAG", body.decode())
        self.assertEqual(get(self.base + "/static/app.js")[0], 200)
        with self.assertRaises(urllib.error.HTTPError):
            get(self.base + "/static/../server.py")

    def test_results_and_running(self):
        data = json.loads(get(self.base + "/api/results")[2])
        self.assertIn("mock / dev", data["runs"]["E3"])
        self.assertEqual(data["running"][0]["calls"], 2)

    def test_milestones_instances_backends(self):
        self.assertEqual(json.loads(get(self.base + "/api/milestones")[2])[0]["title"], "M1")
        rows = json.loads(get(self.base + "/api/instances?split=dev&condition=S&n=1")[2])
        self.assertEqual(rows[0]["id"], "x-1")
        self.assertEqual(json.loads(get(self.base + "/api/instances?split=dev&condition=P")[2]), [])
        avail = json.loads(get(self.base + "/api/backends")[2])
        self.assertTrue(avail["mock"]["ok"])
        self.assertFalse(avail["mock"]["remote"])
        self.assertIn("laya-ml", avail)

    def test_judge(self):
        status, out = post(self.base + "/api/judge", {"question": "甲在哪年成立？",
                                                     "passages": ["甲在1962年成立。", "無關"], "backends": ["mock"]})
        self.assertEqual(status, 200)
        self.assertEqual(len(out[0]["passages"]), 2)
        self.assertAlmostEqual(sum(out[0]["verdict"].values()), 1.0, places=2)
        self.assertIn(out[0]["action"], {"answer", "answer_with_both_sides", "answer_partial_and_flag_gap",
                                         "abstain_or_search_web", "correct_premise"})

    def test_judge_rejects_bad_requests(self):
        self.assertEqual(post(self.base + "/api/judge", {"question": "", "passages": ["x"], "backends": ["mock"]})[0], 400)
        self.assertEqual(post(self.base + "/api/judge", {"question": "q", "passages": ["x"], "backends": ["jev"]})[0], 400)


if __name__ == "__main__":
    unittest.main()
