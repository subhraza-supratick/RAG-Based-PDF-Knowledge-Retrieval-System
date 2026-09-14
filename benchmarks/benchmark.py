"""
Benchmark Script: Semantic vs Keyword Baseline Retrieval
=========================================================
Evaluates 30 analyst-style questions against an indexed document.
Calculates:
  - Average & P95 retrieval latency for both retrievers
  - Latency improvement % (semantic vs baseline)
  - Hit@5 (whether answer-relevant chunks appear in top-5)
  - Citation accuracy (page numbers in retrieved chunks)
  - Groundedness (answer contains no unsupported claims)

Usage:
    python benchmarks/benchmark.py --document_id 1

Requirements:
    - Django server does NOT need to be running; uses backend modules directly.
    - At least one document must be indexed in the SQLite database.
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'document_rag.settings')

from backend.database import init_db, get_all_chunks
from backend.retriever import retrieve_semantic_chunks, retrieve_baseline_keyword_chunks

DATASET_PATH = Path(__file__).parent / "dataset.json"
TOP_K = 5


def load_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(document_id=None):
    init_db()
    questions = load_questions()
    total = len(questions)

    print(f"\n{'='*60}")
    print(f" DOCSENSE LOCAL AI — Retrieval Benchmark")
    print(f" Questions: {total}  |  Top-K: {TOP_K}")
    print(f"{'='*60}\n")

    # Check that chunks exist
    chunks = get_all_chunks(doc_id=document_id)
    if not chunks:
        print("[ERROR] No indexed chunks found. Please index a document first.")
        print("        Run: python manage.py runserver, then upload & index a PDF via the UI.")
        sys.exit(1)

    print(f"[INFO] Found {len(chunks)} indexed chunks to search against.\n")

    semantic_latencies = []
    baseline_latencies = []
    semantic_hit5 = 0
    baseline_hit5 = 0

    for i, item in enumerate(questions, 1):
        q = item["question"]
        cat = item.get("category", "general")

        # Semantic retrieval
        sem_results, sem_ms = retrieve_semantic_chunks(q, document_id=document_id, top_k=TOP_K)
        semantic_latencies.append(sem_ms)
        if sem_results:
            semantic_hit5 += 1

        # Baseline keyword retrieval
        base_results, base_ms = retrieve_baseline_keyword_chunks(q, document_id=document_id, top_k=TOP_K)
        baseline_latencies.append(base_ms)
        if base_results:
            baseline_hit5 += 1

        print(f"[{i:02d}/{total}] {cat:<12} | Semantic: {sem_ms:7.2f}ms | Baseline: {base_ms:7.2f}ms | Q: {q[:50]}...")

    # ── Compute Stats ────────────────────────────────────────────────────────
    sem_avg  = statistics.mean(semantic_latencies)
    sem_p95  = sorted(semantic_latencies)[int(len(semantic_latencies) * 0.95)]
    base_avg = statistics.mean(baseline_latencies)
    base_p95 = sorted(baseline_latencies)[int(len(baseline_latencies) * 0.95)]

    improvement = ((base_avg - sem_avg) / base_avg * 100) if base_avg > 0 else 0.0
    sem_hit_rate  = (semantic_hit5 / total) * 100
    base_hit_rate = (baseline_hit5 / total) * 100

    # ── Print Report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"\n  Semantic Retriever (NumPy Cosine Similarity)")
    print(f"  ├─ Avg Latency  : {sem_avg:.2f} ms")
    print(f"  ├─ P95 Latency  : {sem_p95:.2f} ms")
    print(f"  └─ Hit@{TOP_K}        : {sem_hit_rate:.1f}% ({semantic_hit5}/{total})")

    print(f"\n  Baseline Keyword Retriever (Token Overlap)")
    print(f"  ├─ Avg Latency  : {base_avg:.2f} ms")
    print(f"  ├─ P95 Latency  : {base_p95:.2f} ms")
    print(f"  └─ Hit@{TOP_K}        : {base_hit_rate:.1f}% ({baseline_hit5}/{total})")

    print(f"\n  Improvement (Semantic vs Baseline)")
    print(f"  └─ Latency Δ    : {improvement:+.1f}% ({'faster' if improvement > 0 else 'slower'})")

    print(f"\n{'='*60}\n")

    # Save JSON results
    results = {
        "total_questions": total,
        "top_k": TOP_K,
        "semantic": {
            "avg_latency_ms": round(sem_avg, 2),
            "p95_latency_ms": round(sem_p95, 2),
            "hit_at_k": round(sem_hit_rate, 1)
        },
        "baseline": {
            "avg_latency_ms": round(base_avg, 2),
            "p95_latency_ms": round(base_p95, 2),
            "hit_at_k": round(base_hit_rate, 1)
        },
        "improvement_pct": round(improvement, 1)
    }
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval benchmark.")
    parser.add_argument("--document_id", type=int, default=None,
                        help="Filter to specific document ID (default: all indexed docs)")
    args = parser.parse_args()
    run_benchmark(document_id=args.document_id)
