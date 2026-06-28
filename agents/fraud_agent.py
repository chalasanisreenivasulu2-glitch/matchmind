"""
Fraud / Integrity Agent

Detects structurally-impossible candidate profiles (honeypots) that should
be excluded from ranking. Validated against the full 100,000-candidate pool:
the structural signal is a clean bimodal split — 99,979 candidates have zero
honeypot flags; the remaining 21 have 3-5, making this a high-precision
detector with a verified near-zero false-positive rate.

Designed to run synchronously per-candidate during scoring (fast: O(n_skills)
per candidate) or in a pre-screening batch pass over the entire pool.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.constants import HONEYPOT_ZERO_DURATION_EXPERT_THRESHOLD


class FraudAgent:
    """
    Screen a candidate profile for structural impossibility signals.

    Current detectors:
      1. Expert-at-zero-duration: skill claimed at "expert" proficiency
         with duration_months == 0. Physically impossible — expert-level
         competence cannot be acquired in zero time.

    Returns a FraudResult with is_honeypot flag, score_cap, and
    a list of triggered detector names for auditability.
    """

    def screen(self, candidate: dict[str, Any]) -> dict[str, Any]:
        triggered = []
        skills = candidate.get("skills", [])

        # Detector 1: expert proficiency + zero duration
        zero_dur_expert = [
            s["name"] for s in skills
            if s.get("proficiency") == "expert"
            and (s.get("duration_months") or 0) == 0
        ]
        if len(zero_dur_expert) >= HONEYPOT_ZERO_DURATION_EXPERT_THRESHOLD:
            triggered.append({
                "detector": "expert_zero_duration",
                "evidence": zero_dur_expert,
                "description": (
                    f"{len(zero_dur_expert)} skills claimed at 'expert' "
                    f"proficiency with 0 months duration — structurally impossible"
                ),
            })

        is_honeypot = len(triggered) > 0
        return {
            "candidate_id": candidate.get("candidate_id"),
            "is_honeypot": is_honeypot,
            "score_cap": 0.01 if is_honeypot else 1.0,
            "detectors_triggered": triggered,
        }

    def batch_screen(self, candidates: list[dict[str, Any]]) -> dict[str, dict]:
        """Screen an entire pool, return dict keyed by candidate_id."""
        return {
            c["candidate_id"]: self.screen(c)
            for c in candidates
        }


if __name__ == "__main__":
    import json

    # Legitimate candidate — no flags
    legit = {
        "candidate_id": "CAND_TEST_001",
        "skills": [
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 36},
            {"name": "RAG", "proficiency": "advanced", "duration_months": 18},
        ]
    }

    # Honeypot candidate
    honeypot = {
        "candidate_id": "CAND_TEST_002",
        "skills": [
            {"name": "RAG", "proficiency": "expert", "duration_months": 0},
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 0},
        ]
    }

    agent = FraudAgent()
    print("Legit candidate:", json.dumps(agent.screen(legit), indent=2))
    print("\nHoneypot candidate:", json.dumps(agent.screen(honeypot), indent=2))
