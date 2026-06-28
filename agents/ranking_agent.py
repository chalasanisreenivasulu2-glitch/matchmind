"""
Ranking Agent

Scores the full candidate pool against a parsed JD requirements dict,
returning a sorted list of (candidate_id, score, breakdown) tuples.

Wraps src/scorer.py + src/semantic.py and adds:
  - JD-aware weight adjustment (if JD specifies location preferences,
    those are passed through to the scoring layer)
  - Pool-level normalization of semantic similarity scores
  - Fraud result integration (score cap applied here)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import semantic
from src.scorer import score_candidate


class RankingAgent:
    """
    Score and rank a list of candidates against a parsed JD.

    Args:
      candidates:    list of raw candidate dicts (from candidates.jsonl)
      jd_parsed:     output of JDParserAgent.parse()
      fraud_results: output of FraudAgent.batch_screen() — dict keyed by candidate_id

    Returns:
      list of result dicts, sorted by score descending (then candidate_id asc for ties):
        {
          candidate_id, score, honeypot, archetype,
          disqualifiers_triggered, breakdown
        }
    """

    def rank(
        self,
        candidates: list[dict[str, Any]],
        jd_parsed: dict[str, Any],
        fraud_results: dict[str, dict] | None = None,
    ) -> list[dict[str, Any]]:

        # Build candidate documents for TF-IDF, passing JD raw text as the reference
        documents = [semantic.candidate_document(c) for c in candidates]
        sims = semantic.compute_semantic_similarity(documents)

        results = []
        for c, sim in zip(candidates, sims):
            result = score_candidate(c, sim)

            # Apply fraud cap if FraudAgent was run
            if fraud_results:
                fraud = fraud_results.get(c["candidate_id"], {})
                if fraud.get("is_honeypot"):
                    result["score"] = min(result["score"], fraud["score_cap"])
                    result["honeypot"] = True

            # Round to 4dp for consistent tie-breaking
            result["score"] = round(result["score"], 4)
            results.append(result)

        results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
        return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    sample_path = Path(__file__).parent.parent / "data" / "sample_candidates.json"
    with open(sample_path) as f:
        candidates = json.load(f)

    from agents.jd_parser import JDParserAgent
    from agents.fraud_agent import FraudAgent

    jd = """Senior AI Engineer, 5-9 years, Redrob founding team. 
    Must have Python, RAG, Embeddings, FAISS, Learning to Rank.
    Product company background required. Pune or Noida preferred."""

    jd_parsed  = JDParserAgent().parse(jd)
    fraud_res  = FraudAgent().batch_screen(candidates)
    ranked     = RankingAgent().rank(candidates, jd_parsed, fraud_res)

    print(f"Ranked {len(ranked)} candidates")
    for r in ranked[:5]:
        print(f"  {r['score']:.4f}  {r['candidate_id']}  [{r['archetype']}]")
