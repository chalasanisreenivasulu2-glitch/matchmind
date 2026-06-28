"""
Grounded reasoning text generation.

Deliberately NOT an LLM call: the ranking step runs with no network access,
and even a local LLM per-candidate would risk inventing facts not in the
profile (the spec's #1 reasoning-quality check is "No hallucination").
Instead, every clause here is built directly from a verified field value
(years_of_experience, current_title, an actual skill name the candidate
holds, a real signal value) -- so a claim that doesn't exist in the
profile is structurally impossible, not just discouraged.

Style is deliberately terse -- a single semicolon-joined sentence, matching
the submission_spec.md example ("Senior AI Engineer with 7 years building
RAG systems at product companies; strong recent engagement and
Bangalore-based.") rather than a multi-sentence paragraph.

Variation across the 100 rows comes from two places: (1) the underlying
facts are genuinely different per candidate, and (2) clause order is chosen
deterministically from a small template pool keyed on candidate_id, so two
structurally-similar candidates still read differently and the same
candidate always gets the same reasoning across reruns (reproducibility).
"""

from __future__ import annotations

import hashlib
from typing import Any


def _skill_list(breakdown: dict[str, Any], n: int) -> str:
    named = breakdown.get("named_evidence", [])
    return ", ".join(name for name, _w in named[:n])


DOMINANT_PHRASES = {
    "plain_language": lambda c, b: (
        f"describes retrieval/ranking work in plain language, not buzzwords "
        f"(holds {_skill_list(b, 2)})"
    ),
    "skill": lambda c, b: (
        f"holds {_skill_list(b, 3)}" if _skill_list(b, 3) else "some relevant tooling exposure"
    ),
    "semantic": lambda c, b: (
        "career narrative closely echoes the JD's own ranking/retrieval language"
    ),
    "experience": lambda c, b: (
        f"{c['profile']['years_of_experience']:.1f} yrs lands in the JD's 5-9 yr target band"
    ),
    "product": lambda c, b: "career almost entirely at product companies, not services",
    "shipped": lambda c, b: "role descriptions describe shipping a ranking/rec system to production",
    "ai_native": lambda c, b: f"current employer ({c['profile']['current_company']}) is AI-native",
}

CONCERN_PHRASES = {
    "consulting_only_career": lambda c, b: "entire career at IT-services/consulting firms",
    "title_chaser_pattern": lambda c, b: "short average tenure across roles reads as title-chasing",
    "cv_speech_without_nlp": lambda c, b: "CV/speech-heavy skills, little NLP/retrieval exposure",
    "buzzword_only_no_depth": lambda c, b: (
        "skills lean buzzword-only (RAG/LangChain-type), no PyTorch/scikit-learn-level depth"
    ),
}

# {fact}; {dominant}{; concern}. -- clause order varies slightly per template
TEMPLATES = [
    "{fact}; {dominant}{concern}.",
    "{dominant_cap}; {fact}{concern}.",
    "{fact}; {dominant}{concern_alt}.",
]


def _dominant_factor(breakdown: dict[str, Any]) -> str:
    if breakdown.get("plain_language_signal") == 1.0:
        return "plain_language"
    contributions = {
        "skill": 0.32 * breakdown["skill_match_score"],
        "semantic": 0.22 * breakdown["semantic_similarity"],
        "experience": 0.16 * breakdown["experience_fit"],
        "product": 0.12 * breakdown["product_company_score"],
        "shipped": 0.10 * breakdown["shipped_system_evidence"],
        "ai_native": 0.05 * breakdown["ai_native_exposure"],
    }
    return max(contributions, key=contributions.get)


LOCATION_CONCERN_VARIANTS = [
    "based outside the JD's preferred India hub cities",
    "location isn't Pune/Noida or one of the JD's welcomed hub cities",
    "city isn't among the JD's preferred or welcomed locations",
]

NOTICE_CONCERN_VARIANTS = [
    "{notice}-day notice, longer than the JD's sub-30-day preference",
    "notice period ({notice} days) runs past the JD's sub-30-day ask",
    "{notice}-day notice period exceeds the JD's preferred window",
]


def _concern(candidate: dict[str, Any], breakdown: dict[str, Any],
             disq_triggered: list[str]) -> str:
    seed = int(hashlib.md5(candidate["candidate_id"].encode()).hexdigest(), 16)

    if disq_triggered:
        phrase_fn = CONCERN_PHRASES.get(disq_triggered[0])
        if phrase_fn:
            return phrase_fn(candidate, breakdown)
    notice = breakdown.get("notice_period_days", 0)
    if notice > 90:
        return NOTICE_CONCERN_VARIANTS[seed % len(NOTICE_CONCERN_VARIANTS)].format(notice=notice)
    recency = breakdown.get("recency_days", 0)
    if recency > 150:
        return f"inactive {recency} days, current availability uncertain"
    if breakdown.get("location_fit", 1.0) < 0.5:
        return LOCATION_CONCERN_VARIANTS[seed % len(LOCATION_CONCERN_VARIANTS)]
    response = breakdown.get("recruiter_response_rate", 1.0)
    if response < 0.3:
        return f"{response:.0%} recruiter response rate, weak engagement"
    return ""


def generate_reasoning(candidate: dict[str, Any], score_result: dict[str, Any]) -> str:
    breakdown = score_result["breakdown"]
    p = candidate["profile"]

    fact = f"{p['current_title']} at {p['current_company']}, {p['years_of_experience']:.1f} yrs"

    dominant_key = _dominant_factor(breakdown)
    dominant_text = DOMINANT_PHRASES[dominant_key](candidate, breakdown)
    dominant_text_cap = dominant_text[0].upper() + dominant_text[1:]

    concern_text = _concern(candidate, breakdown, score_result.get("disqualifiers_triggered", []))
    concern_part = f"; {concern_text}" if concern_text else ""

    seed = int(hashlib.md5(candidate["candidate_id"].encode()).hexdigest(), 16)
    template = TEMPLATES[seed % len(TEMPLATES)]

    text = template.format(
        fact=fact, dominant=dominant_text, dominant_cap=dominant_text_cap,
        concern=concern_part, concern_alt=concern_part,
    )
    return text
