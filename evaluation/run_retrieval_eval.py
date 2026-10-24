"""Run a small local retrieval quality and latency check against sample_data."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ingestion import load_document, make_chunks
from src.ollama_client import LocalOllama
from src.retrieval import hybrid_search
from src.vector_store import VectorStore


def main() -> int:
    test_cases = json.loads((PROJECT_DIR / "evaluation" / "test_set.json").read_text())
    with tempfile.TemporaryDirectory(prefix="local-rag-eval-") as temp_dir:
        store = VectorStore(Path(temp_dir), "retrieval_evaluation")
        ollama = LocalOllama()
        statuses = ollama.check_required_models()
        missing = [model for model, state in statuses.items() if state != "available"]
        if missing:
            print(f"Cannot evaluate: missing local model(s): {', '.join(missing)}")
            return 2
        for document_path in sorted((PROJECT_DIR / "sample_data").iterdir()):
            store.add_chunks(make_chunks(load_document(document_path), 500, 50))

        timings: list[float] = []
        correct = 0
        print("Retrieval evaluation (no answer generation):")
        for case in test_cases:
            start = time.perf_counter()
            result = hybrid_search(case["question"], store, top_k=3, min_similarity=0.0)
            elapsed = (time.perf_counter() - start) * 1_000
            timings.append(elapsed)
            found = any(hit.chunk.source_name == case["expected_source"] for hit in result.hits)
            correct += int(found)
            print(f"- {'PASS' if found else 'FAIL'} {case['question']} ({elapsed:.1f} ms)")

        print(f"Recall@3: {correct}/{len(test_cases)} ({correct / len(test_cases):.0%})")
        print(f"Mean retrieval latency: {sum(timings) / len(timings):.1f} ms")
        print(f"P95 retrieval latency: {sorted(timings)[max(0, int(len(timings) * .95) - 1)]:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
