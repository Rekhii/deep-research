"""
eval/retrieval_eval.py

Scores the retrieval pipeline against the hand-labeled questions in
eval/golden.jsonl.

Reports Recall@k and MRR for two paths over the same questions:
  reranked  - the full pipeline, dense + sparse -> RRF -> cross-encoder
  fusion    - the same candidates in RRF order, cross-encoder skipped

Run:
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --candidates 25 50 100
    python -m eval.retrieval_eval --diffs
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.retriever import (
    close_client,
    dense_search,
    sparse_search,
    rrf_fuse,
    rerank,
)

GOLDEN = Path("eval/golden.jsonl")
KS = (1, 3, 5, 10)   # the cutoffs Recall@k is reported at


def load_golden():
    """Read golden.jsonl into a list of dicts."""
    items = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def ranked_indices(results):
    """
    Pull chunk_index out of a result list, preserving rank order.

    retrieve() returns dicts shaped like
        {"id": ..., "score": ..., "payload": {...}, "rerank_score": ...}
    so the chunk_index we label against lives one level down in payload.
    """
    return [r["payload"]["chunk_index"] for r in results]


def recall_at_k(ranked, gold, k):
    """
    Fraction of gold chunks that appear in the top k.

    Using the fraction rather than a hit/miss flag matters for the multi-chunk
    questions: finding one of two gold chunks scores 0.5, not 1.0. That keeps
    a partially-correct retrieval from looking perfect.
    """
    if not gold:
        return 0.0
    found = sum(1 for g in gold if g in ranked[:k])
    return found / len(gold)


def reciprocal_rank(ranked, gold):
    """
    1 / rank of the first gold chunk, or 0.0 if none appear at all.

    Recall only asks whether a gold chunk made the cut; MRR also cares where
    it landed. A pipeline that puts gold at rank 1 and one that puts it at
    rank 5 score identically on Recall@5 but differently here, which is
    exactly the difference reranking is supposed to make.
    """
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def run_paths(query, candidates, top_k):
    """
    Run retrieval once and return both the fusion-only and reranked orderings.

    The two paths share the same dense and sparse calls, so this measures the
    cross-encoder's contribution in isolation rather than re-running the whole
    pipeline twice with different random-ish conditions.
    """
    dense = dense_search(query, limit=candidates)
    sparse = sparse_search(query, limit=candidates)
    fused = rrf_fuse([dense, sparse], limit=candidates)

    # Fusion-only path: take the RRF ordering as-is, no cross-encoder.
    fusion_ranked = ranked_indices(fused)

    # Reranked path: cross-encoder reorders the same candidate pool.
    t0 = time.perf_counter()
    reranked = rerank(query, fused, limit=top_k)
    rerank_secs = time.perf_counter() - t0
    reranked_ranked = ranked_indices(reranked)

    return fusion_ranked, reranked_ranked, rerank_secs


def evaluate(items, candidates, top_k):
    """Score every question and return per-path, per-difficulty aggregates."""
    # scores[path][difficulty] -> list of per-question metric dicts
    scores = defaultdict(lambda: defaultdict(list))
    rerank_times = []

    for item in items:
        gold = item["gold_ids"]
        difficulty = item.get("difficulty", "unknown")

        fusion_ranked, reranked_ranked, secs = run_paths(
            item["question"], candidates, top_k
        )
        rerank_times.append(secs)

        for path, ranked in (("fusion", fusion_ranked), ("reranked", reranked_ranked)):
            row = {f"r@{k}": recall_at_k(ranked, gold, k) for k in KS}
            row["mrr"] = reciprocal_rank(ranked, gold)
            scores[path][difficulty].append(row)
            scores[path]["ALL"].append(row)

    return scores, rerank_times


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_table(scores, candidates, top_k, rerank_times):
    """Print the two paths side by side, broken down by difficulty."""
    print()
    print(f"CANDIDATES={candidates}  TOP_K={top_k}")
    print(f"mean rerank time: {mean(rerank_times):.2f}s per query")
    print()

    header = f"{'path':<9} {'difficulty':<12} {'n':>3} " + " ".join(
        f"{'R@' + str(k):>7}" for k in KS
    ) + f"{'MRR':>8}"
    print(header)
    print("-" * len(header))

    # ALL first, then each difficulty tag, so the headline number leads.
    for path in ("fusion", "reranked"):
        keys = ["ALL"] + sorted(k for k in scores[path] if k != "ALL")
        for difficulty in keys:
            rows = scores[path][difficulty]
            cells = " ".join(f"{mean([r[f'r@{k}'] for r in rows]):>7.3f}" for k in KS)
            mrr = mean([r["mrr"] for r in rows])
            print(f"{path:<9} {difficulty:<12} {len(rows):>3} {cells}{mrr:>8.3f}")
        print()


def print_diffs(items, candidates, top_k):
    """
    Show where each gold chunk ranked before and after reranking.

    The aggregate table says the reranker hurts but not why. This prints
    per-question rank movement plus the chunk indices the reranker put on top,
    so you can go read those chunks and judge whether the problem is scoring,
    boilerplate noise in the candidate pool, or a model that is simply weak on
    this corpus.
    """
    print("\nper-question rank of first gold chunk (fusion -> reranked)")
    print("None means the gold chunk did not appear in the top k at all\n")

    for item in items:
        gold = item["gold_ids"]
        fusion_ranked, reranked_ranked, _ = run_paths(
            item["question"], candidates, top_k
        )

        def first_rank(ranked):
            """Position of the earliest gold chunk, 1-indexed, or None."""
            for i, cid in enumerate(ranked, start=1):
                if cid in gold:
                    return i
            return None

        f = first_rank(fusion_ranked)
        r = first_rank(reranked_ranked)

        # Flag the direction of movement so the damaging cases are scannable.
        if f and r and r > f:
            flag = "  WORSE"
        elif f and not r:
            flag = "  LOST"
        elif f and r and r < f:
            flag = "  better"
        else:
            flag = ""

        print(f"{item['qid']:<5} {item.get('difficulty', '?'):<11} "
              f"{str(f):>4} -> {str(r):>4}{flag}")
        print(f"      top-3 reranked: {reranked_ranked[:3]}   gold: {gold}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    # Sweeping candidate counts is the point of this flag: it answers whether
    # CANDIDATES=50 is doing anything that 25 would not.
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[50],
        help="candidate pool size(s) to evaluate; pass several to sweep",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--diffs",
        action="store_true",
        help="print per-question rank movement instead of only aggregates",
    )
    args = parser.parse_args()

    items = load_golden()
    print(f"{len(items)} questions loaded from {GOLDEN}")

    try:
        for candidates in args.candidates:
            scores, rerank_times = evaluate(items, candidates, args.top_k)
            print_table(scores, candidates, args.top_k, rerank_times)

            if args.diffs:
                print_diffs(items, candidates, args.top_k)
    finally:
        # Release the embedded Qdrant lock explicitly. Without this the
        # interpreter can tear down modules before QdrantClient.__del__ runs,
        # producing a noisy ImportError at exit.
        close_client()


if __name__ == "__main__":
    main()