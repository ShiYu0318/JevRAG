# JevRAG

Early research code. The question behind it: can small decision models that answer typed questions with probabilities, instead of generating text, take over the judgment steps inside a RAG pipeline?

A RAG system makes many small calls that are not generation. Is this passage relevant? Is there enough evidence to answer? Does this query need retrieval at all? Is this sentence backed by its source? These are usually handled by a task-specific classifier, by prompting an LLM, or by a fixed rule. "System One" models such as Jev, Kev and Laya return a distribution over a fixed answer space, so if those probabilities are well calibrated, simple threshold policies (answer, retrieve more, abstain) become possible without training anything new.

The focus is Traditional Chinese, where there is very little public evaluation of this.

## Status

Infrastructure stage. No results yet.

## Layout

| Path | |
|---|---|
| `jevrag/backends` | One client for the `/v1/systemone` protocol, presets for Jev, Kev and Laya, on-disk call cache |
| `jevrag/questions` | Versioned question templates |
| `jevrag/ops` | Decision points: passage grading, evidence sufficiency, routing, compression, claim checks |
| `jevrag/retrieval` | Character-bigram BM25, optional bge-m3, rank fusion |
| `jevrag/metrics`, `jevrag/stats` | Ranking, calibration and selective-prediction metrics; bootstrap, McNemar, Holm |
| `jevrag/experiments` | Experiment runners |
| `jevrag/demo` | Local console for browsing results and judging a single question across backends |
| `scripts/` | Mock server and a small run over `data/toy_zh.json` |

The core uses only the Python standard library. OpenCC, sentence-transformers and PyYAML are optional extras.

## Try it

```bash
python scripts/mock_server.py &
python scripts/run_toy.py --backends mock
python -m unittest discover -s tests
```

The mock server fakes probabilities from character overlap. It only checks the plumbing.

`python -m jevrag demo` opens a console on http://127.0.0.1:8900 that shows the latest run of each experiment and lets you send one question with its passages to several backends side by side. It only reads local files and binds to localhost.

## Backends

| Name | Runs on | Setup |
|---|---|---|
| `jev` | OpenRouter | `export OPENROUTER_API_KEY=...` |
| `kev4b`, `kev08b` | Local GPU or Apple Silicon | `python -m kev.serve --run jaredpalmer/kev-4b --port 8009` ([kev](https://github.com/jaredpalmer/kev)) |
| `laya-ml` | Local, CPU is fine | `LAYA_MODELS=multilingual laya-serve` ([laya](https://github.com/NandhaKishorM/laya)) |

For Chinese text, use Laya's multilingual checkpoint.

## Data

Evaluation data is built from [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) (CC BY-SA 3.0). Nothing is released yet.

## License

Code is released under the [Apache License 2.0](LICENSE). Data derived from DRCD will be released under CC BY-SA 4.0, as its share-alike terms require.

If you use this code, please cite it as described in [CITATION.cff](CITATION.cff).

This project is not affiliated with TypeSafe AI.
