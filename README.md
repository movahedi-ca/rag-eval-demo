# RAG evaluation demo (offline mini-benchmark)

A weekend-sized RAG evaluation on Canadian regulatory text, behind the
article [Test Your RAG Pipeline Like a Regulator Is Watching](https://movahedi.ca/insights/test-rag-pipeline-like-regulator-watching/).

## What it does

`run_benchmark.py` builds a tiny RAG pipeline over 14 passages of public
regulatory text (the TBS Directive on Automated Decision-Making and a
professional summary of OSFI Guideline E-23 expectations), answers 12
hand-written test questions with reference answers, and scores three Ragas
0.4.3 metrics (faithfulness, context precision, answer relevancy) on two
variants:

- **baseline:** 400-character chunks, no overlap, top-3 retrieval
- **improved:** 900-character chunks with 150-character overlap, top-6 retrieval

Results (means over 12 questions): faithfulness 0.95 -> 1.00, context
precision 0.71 -> 0.86, answer relevancy 0.29 -> 0.28. Per-question scores
are in `rag_benchmark.json`. Small sample: treat deltas as directional.

## Run it

```bash
pip install -r requirements.txt
python run_benchmark.py
```

The script shells out to a Gemini CLI for generation and judging; set the
`GEMINI` path near the top of the script to your own command. Retrieval uses
sklearn TF-IDF (CPU-only, deterministic). The committed
`rag_benchmark.json` is the run of 2026-09-26.

## Files

- `corpus.json` — 14 passages + 12 test questions with reference answers
- `run_benchmark.py` — the benchmark script
- `rag_benchmark.json` — full results, including per-question scores
