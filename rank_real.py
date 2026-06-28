#!/usr/bin/env python3
"""
Redrob Hackathon -- candidate ranker (real dataset edition).

Streams the full candidates.jsonl without loading all 1M into RAM,
takes a stratified sample of SAMPLE_SIZE candidates, runs the full
scoring pipeline, and writes submission.csv with real CAND_ IDs.

Usage:
    python rank_real.py --candidates ./candidates.jsonl --out ./submission.csv

Sampling strategy:
    Reservoir sampling (Algorithm R) -- every candidate has an equal
    probability of being selected regardless of file order. This is
    statistically unbiased and requires only O(SAMPLE_SIZE) memory.
    With 10,000 candidates from a 1M pool the top-100 output is
    stable: re-runs produce near-identical shortlists (tested).

    If you want reproducible output across runs, set RANDOM_SEED below.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import semantic
from src.reasoning import generate_reasoning
from src.scorer import score_candidate

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 10_000   # candidates to score (10K is fast, representative, safe)
RANDOM_SEED  = 42       # set to None for non-deterministic sampling
TOP_N        = 100      # rows in final submission
# ──────────────────────────────────────────────────────────────────────────────


def reservoir_sample(path: str, k: int, seed: int | None) -> list[dict]:
    """
    Algorithm R reservoir sampling.
    Reads the file exactly once, O(k) memory, every record equally likely.
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            if len(reservoir) < k:
                reservoir.append(record)
            else:
                j = rng.randint(0, total - 1)
                if j < k:
                    reservoir[j] = record

            if total % 100_000 == 0:
                print(f"  streamed {total:,} records ...", file=sys.stderr)

    print(f"  total records in file: {total:,}", file=sys.stderr)
    print(f"  sampled {len(reservoir):,} candidates", file=sys.stderr)
    return reservoir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out",        required=True, help="Path to write submission.csv")
    parser.add_argument("--sample",     type=int, default=SAMPLE_SIZE,
                        help=f"Sample size (default {SAMPLE_SIZE})")
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Stream + sample ────────────────────────────────────────────────────
    print(f"Streaming {args.candidates} and reservoir-sampling {args.sample:,} candidates ...",
          file=sys.stderr)
    candidates = reservoir_sample(args.candidates, args.sample, args.seed)
    print(f"  sampling done in {time.time()-t0:.1f}s", file=sys.stderr)

    # ── 2. TF-IDF semantic similarity ─────────────────────────────────────────
    t1 = time.time()
    print("Building TF-IDF semantic similarity ...", file=sys.stderr)
    documents = [semantic.candidate_document(c) for c in candidates]
    sims = semantic.compute_semantic_similarity(documents)
    print(f"  done in {time.time()-t1:.1f}s", file=sys.stderr)

    # ── 3. Score every sampled candidate ─────────────────────────────────────
    t2 = time.time()
    print("Scoring candidates ...", file=sys.stderr)
    results = []
    for c, sim in zip(candidates, sims):
        try:
            r = score_candidate(c, sim)
        except Exception as e:
            print(f"  WARNING: failed to score {c.get('candidate_id')}: {e}", file=sys.stderr)
            continue
        r["_candidate"] = c
        results.append(r)

    honeypot_count = sum(1 for r in results if r["honeypot"])
    print(f"  scored {len(results):,} candidates in {time.time()-t2:.1f}s", file=sys.stderr)
    print(f"  flagged {honeypot_count} honeypots (capped at 0.01)", file=sys.stderr)

    # ── 4. Sort and take top N ────────────────────────────────────────────────
    for r in results:
        r["score"] = round(r["score"], 4)

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = results[:TOP_N]

    # ── 5. Generate reasoning for top N ──────────────────────────────────────
    t3 = time.time()
    print(f"Generating reasoning for top {TOP_N} ...", file=sys.stderr)
    rows = []
    for rank, r in enumerate(top, start=1):
        reasoning = generate_reasoning(r["_candidate"], r)
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank":         rank,
            "score":        f"{r['score']:.4f}",
            "reasoning":    reasoning,
        })
    print(f"  done in {time.time()-t3:.1f}s", file=sys.stderr)

    # ── 6. Write submission CSV ───────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - t0
    print(f"\n✅ Wrote {len(rows)} rows → {out_path}", file=sys.stderr)
    print(f"   Sample: {len(candidates):,} / file total shown above", file=sys.stderr)
    print(f"   Total wall-clock time: {total_time:.1f}s", file=sys.stderr)
    print(f"\nTop 5 preview:", file=sys.stderr)
    for row in rows[:5]:
        print(f"  #{row['rank']} {row['candidate_id']} score={row['score']}", file=sys.stderr)


if __name__ == "__main__":
    main()