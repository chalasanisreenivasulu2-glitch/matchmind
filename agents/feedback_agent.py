"""
Feedback Agent

Collects recruiter shortlist/reject actions and stores them as labelled
training examples. In production these feed a LightGBM/XGBoost Learning-
to-Rank retrainer that runs async (never on the critical path of a search
response). The retrainer adjusts the scoring formula's feature weights
based on observed recruiter preferences, while the transparent base formula
stays auditable and the adjustment is bounded (max ±20% per weight).

This module covers:
  - FeedbackStore: in-memory + JSON-file persistence for recruiter actions
  - FeedbackAgent: records actions, derives labels, cold-start from rules
  - RerankerTrainer: stub for the LightGBM LTR retrainer (Phase 3)
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

Action = Literal["shortlist", "reject", "view", "message"]


class FeedbackStore:
    """Persist recruiter feedback events to a JSONL file."""

    def __init__(self, store_path: str = "data/feedback.jsonl"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        self._load()

    def _load(self):
        if self.store_path.exists():
            with open(self.store_path) as f:
                self._events = [json.loads(line) for line in f if line.strip()]

    def append(self, event: dict):
        self._events.append(event)
        with open(self.store_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def label_counts(self) -> dict[str, dict[str, int]]:
        """Returns {candidate_id: {shortlist: N, reject: N, view: N}} counts."""
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ev in self._events:
            counts[ev["candidate_id"]][ev["action"]] += 1
        return dict(counts)


class FeedbackAgent:
    """
    Record recruiter actions and expose training labels for the reranker.

    Usage:
        agent = FeedbackAgent()
        agent.record(jd_id="JD_001", candidate_id="CAND_0000031",
                     action="shortlist", rank_shown=1, score_shown=0.82)
        labels = agent.get_training_labels()
    """

    POSITIVE_ACTIONS = {"shortlist", "message"}
    NEGATIVE_ACTIONS = {"reject"}

    def __init__(self, store_path: str = "data/feedback.jsonl"):
        self.store = FeedbackStore(store_path)

    def record(
        self,
        jd_id: str,
        candidate_id: str,
        action: Action,
        rank_shown: int,
        score_shown: float,
        recruiter_id: str | None = None,
    ) -> dict:
        event = {
            "ts": time.time(),
            "jd_id": jd_id,
            "candidate_id": candidate_id,
            "action": action,
            "rank_shown": rank_shown,
            "score_shown": score_shown,
            "recruiter_id": recruiter_id,
        }
        self.store.append(event)
        return event

    def get_training_labels(self) -> list[dict[str, Any]]:
        """
        Convert stored recruiter events into (candidate_id, relevance_label) pairs.

        Labels:
          shortlist / message → relevance = 1
          reject              → relevance = 0
          view (no further action) → not labelled (ambiguous intent)
        """
        labels = []
        for event in self.store.events:
            action = event["action"]
            if action in self.POSITIVE_ACTIONS:
                relevance = 1
            elif action in self.NEGATIVE_ACTIONS:
                relevance = 0
            else:
                continue  # view-only events are ambiguous
            labels.append({
                "jd_id": event["jd_id"],
                "candidate_id": event["candidate_id"],
                "relevance": relevance,
                "rank_shown": event["rank_shown"],
                "score_shown": event["score_shown"],
                "ts": event["ts"],
            })
        return labels

    def event_count(self) -> int:
        return len(self.store.events)

    def label_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = defaultdict(int)
        for ev in self.store.events:
            dist[ev["action"]] += 1
        return dict(dist)


class RerankerTrainer:
    """
    Stub for the Phase 3 LightGBM Learning-to-Rank retrainer.

    In production this would:
      1. Load training labels from FeedbackAgent.get_training_labels()
      2. Load feature vectors from the Candidate Intelligence Store
      3. Train/update a LightGBM ranker with lambda_rank objective
      4. Serialize new model weights and push to the model store
      5. Ranking Agent loads new weights on next cold start (no downtime)

    The base rule-based formula stays fully auditable; the LightGBM layer
    adds a bounded adjustment (±20% per weight max) on top.
    """

    def train(self, labels: list[dict]) -> dict:
        if len(labels) < 50:
            return {
                "status": "skipped",
                "reason": f"only {len(labels)} labels; need ≥50 for reliable training",
                "labels_available": len(labels),
            }
        # Phase 3: implement LightGBM LTR here
        # import lightgbm as lgb
        # model = lgb.train(params, train_set, ...)
        return {
            "status": "pending_phase3",
            "labels_used": len(labels),
            "message": "LightGBM LTR retrainer scheduled for Phase 3 rollout",
        }


if __name__ == "__main__":
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        store_path = tmp.name

    agent = FeedbackAgent(store_path=store_path)
    agent.record("JD_001", "CAND_0000031", "shortlist", rank_shown=1, score_shown=0.82)
    agent.record("JD_001", "CAND_0000014", "reject",    rank_shown=2, score_shown=0.74)
    agent.record("JD_001", "CAND_0000001", "view",      rank_shown=3, score_shown=0.68)
    agent.record("JD_001", "CAND_0000010", "shortlist", rank_shown=4, score_shown=0.61)

    print("Events recorded:", agent.event_count())
    print("Distribution:", agent.label_distribution())
    print("Training labels:", agent.get_training_labels())

    trainer = RerankerTrainer()
    print("\nTrainer result:", trainer.train(agent.get_training_labels()))

    os.unlink(store_path)
