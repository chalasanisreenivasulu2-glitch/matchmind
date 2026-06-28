"""
Explanation Agent

Generates a concise, grounded, human-readable explanation for why a
candidate was ranked where they were. Every claim in the explanation is
directly verifiable against the candidate's own profile fields — no
hallucination possible because the explanation template only emits
field values, never inferred/guessed text.

This agent wraps src/reasoning.py and adds:
  - Bulk generation over a ranked list
  - A 'verbose' mode that emits the full feature breakdown alongside
    the explanation (useful for recruiter audit / debug views)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reasoning import generate_reasoning


class ExplanationAgent:
    """
    Generate explanations for a list of ranked score results.

    Args:
      candidates_by_id: dict mapping candidate_id -> raw candidate dict
      ranked_results:   output of RankingAgent.rank()
      verbose:          if True, include full feature breakdown in output

    Returns:
      list of dicts:
        { candidate_id, rank, score, reasoning, breakdown? }
    """

    def explain(
        self,
        candidates_by_id: dict[str, dict[str, Any]],
        ranked_results: list[dict[str, Any]],
        top_n: int = 100,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:

        explained = []
        for rank_pos, result in enumerate(ranked_results[:top_n], start=1):
            cid = result["candidate_id"]
            candidate = candidates_by_id.get(cid)
            if candidate is None:
                continue

            reasoning = generate_reasoning(candidate, result)

            row: dict[str, Any] = {
                "candidate_id": cid,
                "rank": rank_pos,
                "score": result["score"],
                "reasoning": reasoning,
            }
            if verbose:
                row["breakdown"] = result.get("breakdown", {})
                row["disqualifiers"] = result.get("disqualifiers_triggered", [])
                row["honeypot"] = result.get("honeypot", False)

            explained.append(row)

        return explained


if __name__ == "__main__":
    import json
    from pathlib import Path
    from agents.jd_parser import JDParserAgent
    from agents.fraud_agent import FraudAgent
    from agents.ranking_agent import RankingAgent

    sample_path = Path(__file__).parent.parent / "data" / "sample_candidates.json"
    with open(sample_path) as f:
        candidates = json.load(f)

    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    jd = """Senior AI Engineer, 5-9 years. Python, RAG, Embeddings, FAISS.
    Product company. Pune or Noida."""

    jd_parsed   = JDParserAgent().parse(jd)
    fraud_res   = FraudAgent().batch_screen(candidates)
    ranked      = RankingAgent().rank(candidates, jd_parsed, fraud_res)
    explained   = ExplanationAgent().explain(candidates_by_id, ranked, top_n=5, verbose=True)

    for row in explained:
        print(f"#{row['rank']}  {row['score']:.4f}  {row['candidate_id']}")
        print(f"   {row['reasoning']}")
        if "disqualifiers" in row and row["disqualifiers"]:
            print(f"   ⚠ Disqualifiers: {row['disqualifiers']}")
