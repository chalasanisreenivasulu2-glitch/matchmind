#!/usr/bin/env python3
"""
Redrob Hackathon -- candidate ranker entrypoint.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs entirely on CPU, makes no network calls, and completes the full
100,000-candidate pool well within the 5-minute / 16GB compute budget --
there is no model to load at rank time; TF-IDF is fit fresh on the pool
each run (a few seconds for 100K short documents) and every other feature
is closed-form arithmetic over the candidate's own JSON fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import semantic
from src.reasoning import generate_reasoning
from src.scorer import score_candidate


def load_candidates(path: str) -> list[dict]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write submission.csv")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"  {len(candidates)} candidates loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    t1 = time.time()
    print("Building candidate documents + TF-IDF semantic similarity ...", file=sys.stderr)
    documents = [semantic.candidate_document(c) for c in candidates]
    sims = semantic.compute_semantic_similarity(documents)
    print(f"  done in {time.time()-t1:.1f}s", file=sys.stderr)

    t2 = time.time()
    print("Scoring all candidates ...", file=sys.stderr)
    results = []
    for c, sim in zip(candidates, sims):
        try:
            r = score_candidate(c, sim)
        except Exception as e:
            print(f"  WARNING: failed to score {c.get('candidate_id')}: {e}", file=sys.stderr)
            continue
        r["_candidate"] = c
        results.append(r)
    print(f"  scored {len(results)} candidates in {time.time()-t2:.1f}s", file=sys.stderr)

    honeypot_count = sum(1 for r in results if r["honeypot"])
    print(f"  flagged {honeypot_count} structural honeypots (excluded from contention)", file=sys.stderr)

    # Round to the same precision we write to the CSV BEFORE sorting, so the
    # tie-break (candidate_id ascending on equal score) is consistent with
    # what the validator actually sees in the file -- sorting on raw
    # unrounded floats can produce two rows that round to an identical
    # displayed score but aren't in candidate_id order.
    for r in results:
        r["score"] = round(r["score"], 4)

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    top100 = results[:100]

    t3 = time.time()
    print("Generating reasoning text for top 100 ...", file=sys.stderr)
    rows = []
    for rank, r in enumerate(top100, start=1):
        reasoning = generate_reasoning(r["_candidate"], r)
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": f"{r['score']:.4f}",
            "reasoning": reasoning,
        })
    print(f"  done in {time.time()-t3:.1f}s", file=sys.stderr)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    total = time.time() - t0
    print(f"\nWrote {len(rows)} rows to {out_path}", file=sys.stderr)
    print(f"Total wall-clock time: {total:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
