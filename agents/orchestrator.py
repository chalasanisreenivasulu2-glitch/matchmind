"""
Orchestrator

The single public entry point for a MatchMind ranking request.
Coordinates all five agents in the correct order, manages shared context,
and returns a fully explained ranked shortlist.

Architecture:
  Synchronous path (blocks the API response):
    JDParserAgent → FraudAgent (batch) → RankingAgent → ExplanationAgent

  Async path (does NOT block the API response):
    FeedbackAgent (records recruiter actions via event bus)

Usage:
    from agents.orchestrator import Orchestrator

    orch = Orchestrator()

    # One-shot: parse JD, rank, explain — returns top-N explained results
    results = orch.rank_for_jd(jd_text, candidates, top_n=10)

    # Record a recruiter action (async in production; sync here for demo)
    orch.record_feedback("JD_001", "CAND_0000031", action="shortlist",
                         rank_shown=1, score_shown=0.82)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.explanation_agent import ExplanationAgent
from agents.feedback_agent import FeedbackAgent
from agents.fraud_agent import FraudAgent
from agents.jd_parser import JDParserAgent
from agents.ranking_agent import RankingAgent


class Orchestrator:
    """
    Multi-agent coordinator for MatchMind.

    All agents are instantiated once and reused across requests. In a
    production microservices deployment, each agent would be a separate
    service behind a REST endpoint; the Orchestrator would make HTTP calls
    rather than in-process calls. The interface here is identical either way.
    """

    def __init__(self, feedback_store_path: str = "data/feedback.jsonl"):
        self.jd_parser     = JDParserAgent()
        self.fraud_agent   = FraudAgent()
        self.ranking_agent = RankingAgent()
        self.explanation   = ExplanationAgent()
        self.feedback      = FeedbackAgent(store_path=feedback_store_path)

    def rank_for_jd(
        self,
        jd_text: str,
        candidates: list[dict[str, Any]],
        top_n: int = 100,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Full synchronous ranking pipeline for one JD.

        Returns:
          {
            "jd_parsed":   structured JD requirements,
            "results":     list of {candidate_id, rank, score, reasoning, ...},
            "stats":       pool size, honeypots flagged, runtime_seconds,
            "top_n":       number of results returned
          }
        """
        t0 = time.time()

        # Step 1: Parse JD
        jd_parsed = self.jd_parser.parse(jd_text)

        # Step 2: Fraud screening (runs over full pool before ranking)
        candidates_by_id = {c["candidate_id"]: c for c in candidates}
        fraud_results = self.fraud_agent.batch_screen(candidates)
        honeypot_count = sum(1 for v in fraud_results.values() if v["is_honeypot"])

        # Step 3: Rank
        ranked = self.ranking_agent.rank(candidates, jd_parsed, fraud_results)

        # Step 4: Explain top-N
        explained = self.explanation.explain(
            candidates_by_id, ranked, top_n=top_n, verbose=verbose
        )

        return {
            "jd_parsed": jd_parsed,
            "results": explained,
            "stats": {
                "pool_size": len(candidates),
                "honeypots_flagged": honeypot_count,
                "runtime_seconds": round(time.time() - t0, 2),
                "top_n_returned": len(explained),
            },
        }

    def record_feedback(
        self,
        jd_id: str,
        candidate_id: str,
        action: str,
        rank_shown: int,
        score_shown: float,
        recruiter_id: str | None = None,
    ) -> dict:
        """
        Record a recruiter action against a candidate result.
        In production this is called async from the event bus; here it's sync.
        """
        return self.feedback.record(
            jd_id=jd_id,
            candidate_id=candidate_id,
            action=action,
            rank_shown=rank_shown,
            score_shown=score_shown,
            recruiter_id=recruiter_id,
        )

    def feedback_stats(self) -> dict:
        return {
            "total_events": self.feedback.event_count(),
            "distribution": self.feedback.label_distribution(),
            "training_labels_available": len(self.feedback.get_training_labels()),
        }


if __name__ == "__main__":
    import json
    from pathlib import Path

    sample_path = Path(__file__).parent.parent / "data" / "sample_candidates.json"
    with open(sample_path) as f:
        candidates = json.load(f)

    JD = """
    Senior AI Engineer — Founding Team at Redrob AI (Series A).
    5-9 years experience. Own the ranking and retrieval systems.
    Must have: Python, RAG, Embeddings, FAISS or Qdrant, Learning to Rank.
    Experience at product companies (not services/consulting).
    Preferred location: Pune or Noida. Hyderabad/Bangalore welcome.
    """

    orch = Orchestrator(feedback_store_path="/tmp/matchmind_demo_feedback.jsonl")
    output = orch.rank_for_jd(JD, candidates, top_n=5, verbose=True)

    print(f"\n{'='*60}")
    print(f"MatchMind — Ranked {output['stats']['pool_size']} candidates")
    print(f"Honeypots excluded: {output['stats']['honeypots_flagged']}")
    print(f"Runtime: {output['stats']['runtime_seconds']}s")
    print(f"{'='*60}")
    print(f"\nJD parsed requirements:")
    print(f"  Skills: {output['jd_parsed']['required_skills'][:5]}")
    print(f"  Experience: {output['jd_parsed']['min_experience_years']}-{output['jd_parsed']['max_experience_years']} yrs")
    print(f"  Locations: {output['jd_parsed']['preferred_locations']}")

    print(f"\nTop {len(output['results'])} candidates:")
    for r in output["results"]:
        print(f"\n  #{r['rank']}  {r['score']:.4f}  {r['candidate_id']}")
        print(f"  {r['reasoning']}")
        if r.get("disqualifiers"):
            print(f"  ⚠  {r['disqualifiers']}")

    # Simulate recruiter feedback
    if output["results"]:
        top = output["results"][0]
        orch.record_feedback(
            jd_id="JD_DEMO_001",
            candidate_id=top["candidate_id"],
            action="shortlist",
            rank_shown=top["rank"],
            score_shown=top["score"],
        )
        print(f"\nFeedback stats: {orch.feedback_stats()}")
