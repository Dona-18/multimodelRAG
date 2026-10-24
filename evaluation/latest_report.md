# Retrieval evaluation report

Run locally on 2026-10-24 using the installed `nomic-embed-text:latest` model,
the included three-question test set, and a temporary ChromaDB. This measures
embedding plus LangChain vector/BM25/RRF retrieval only; answer generation is
excluded.

| Metric | Result |
| --- | ---: |
| Test questions | 3 |
| Recall@3 | 3 / 3 (100%) |
| Mean retrieval latency | 28.3 ms |
| P95 retrieval latency | 27.9 ms |

All three expected source documents were present in the top three fused results.
These figures are a tiny smoke test, not a benchmark. Re-run
`../.venv/bin/python run_retrieval_eval.py` from this folder after changing
models, chunk settings, hardware, or the test data.

## Additional live checks

- End-to-end text QA: the assistant retrieved `workshop_notes.txt`, answered the
  maximum-attendee question as 24, and cited `[workshop_notes.txt]`.
- PDF rendering: Poppler rendered page 1 of the lesson PDF successfully.
- Vision: local `gemma4:e2b-mlx` described that rendered page; a visual
  description was embedded and persisted with modality `image_description` and
  page `1`.
- Multi-query: `ChatOllama` successfully returned structured alternate queries
  for the same attendee-capacity question, without any additional model.