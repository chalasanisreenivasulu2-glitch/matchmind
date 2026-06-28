"""
Final scoring function.

Deliberately a transparent weighted rule system, not a trained model.
There is no ground-truth relevance data available to participants (the
hidden eval set is exactly what's being protected), so a model "trained"
on anything would really be trained on our own heuristics -- which adds
opacity without adding real signal. A hand-justified weighted combination
is more defensible at Stage 4 (manual review) and Stage 5 (defend-your-work
interview): every weight here traces back to an explicit line in the JD.
See README "What we'd do with more time" for the path to a learned ranker
if/when real engagement-outcome labels exist.
"""

from __future__ import annotations

from typing import Any

from . import features as F


def score_candidate(candidate: dict[str, Any], semantic_similarity: float) -> dict[str, Any]:
    skill = F.skill_features(candidate)
    archetype, gate = F.title_archetype(candidate)
    exp_fit = F.experience_fit(candidate)
    career = F.career_features(candidate)
    loc_fit = F.location_fit(candidate)
    disq_mult, disq_triggered = F.disqualifier_multiplier(candidate, career, skill)
    behavioral = F.behavioral_modifier(candidate)
    honeypot = F.is_honeypot(candidate)

    base_combo = (
        0.32 * skill["skill_match_score"]
        + 0.22 * semantic_similarity
        + 0.16 * exp_fit
        + 0.12 * career["product_company_score"]
        + 0.08 * loc_fit
        + 0.10 * career["shipped_system_evidence"]
    )
    base_combo += 0.05 * skill["plain_language_signal"]
    base_combo += 0.05 * career["ai_native_exposure"]
    base_combo = min(1.0, base_combo)

    gated = base_combo * gate
    gated *= disq_mult
    final = gated * behavioral["behavioral_modifier"]

    if honeypot:
        final = min(final, 0.01)

    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(final, 6),
        "honeypot": honeypot,
        "archetype": archetype,
        "disqualifiers_triggered": disq_triggered,
        # raw feature breakdown, used by reasoning.py
        "breakdown": {
            "skill_match_score": skill["skill_match_score"],
            "plain_language_signal": skill["plain_language_signal"],
            "cv_speech_heavy": skill["cv_speech_heavy"],
            "named_evidence": skill["named_evidence"],
            "buzzword_only_flag": skill["buzzword_only_flag"],
            "semantic_similarity": round(semantic_similarity, 4),
            "experience_fit": exp_fit,
            "product_company_score": career["product_company_score"],
            "ai_native_exposure": career["ai_native_exposure"],
            "consulting_only_flag": career["consulting_only_flag"],
            "shipped_system_evidence": career["shipped_system_evidence"],
            "title_chaser_flag": career["title_chaser_flag"],
            "location_fit": loc_fit,
            "behavioral_modifier": behavioral["behavioral_modifier"],
            "recency_days": behavioral["recency_days"],
            "notice_period_days": behavioral["notice_period_days"],
            "recruiter_response_rate": behavioral["recruiter_response_rate"],
        },
    }
