"""
Per-candidate feature engineering.

Every function here takes a raw candidate dict (matching candidate_schema.json)
and returns either a scalar feature or a small dict of related features.
All features are designed to be independently inspectable -- scorer.py
combines them, and reasoning.py explains a ranking by pointing back at
the specific feature values that drove it.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from . import constants as C

# Dataset bundle was generated 2026-06-04 (per file timestamps); recency
# features are computed relative to that date for reproducibility.
REFERENCE_DATE = date(2026, 6, 4)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def skill_features(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Returns:
      skill_match_score: 0-1, weighted coverage of CORE_SKILLS, with
        proficiency- and duration-based trust discounting, plus credit
        for rare plain-language synonyms.
      plain_language_signal: 1.0 if candidate holds >=1 rare-tier synonym
        skill (see constants.RARE_SYNONYMS), else 0.0. Tracked separately
        because this is rare enough (8/100,000 in the dataset) to be worth
        surfacing explicitly in reasoning rather than burying it in the
        aggregate score.
      cv_speech_heavy: fraction of the candidate's skill list that falls in
        CV_SPEECH_SKILLS with no offsetting NLP/IR skill -- feeds the
        cv_speech_only disqualifier.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return {"skill_match_score": 0.0, "plain_language_signal": 0.0,
                "cv_speech_heavy": 0.0, "named_evidence": [], "buzzword_only_flag": 0.0}

    PROFICIENCY_WEIGHT = {"beginner": 0.3, "intermediate": 0.55, "advanced": 0.8, "expert": 1.0}

    matched_weight = 0.0
    max_possible = 0.0
    has_rare_synonym = False
    cv_speech_count = 0
    nlp_ir_count = 0
    depth_skill_count = 0

    # Track best evidence per CORE_SKILLS concept (direct hit or via synonym)
    concept_evidence: dict[str, float] = {}
    # Track the candidate's OWN skill names (never the synonym mapping target)
    # with their effective weight, so reasoning.py can cite real profile facts.
    named_evidence: list[tuple[str, float]] = []

    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0) or 0
        prof_w = PROFICIENCY_WEIGHT.get(prof, 0.3)

        # Trust discount: a high proficiency claim backed by little time
        # used is weighted down. Full trust at >=18 months; linear ramp below.
        trust = min(1.0, duration / 18.0) if duration > 0 else 0.0
        # Even at trust=0 we don't want to fully zero out a skill someone
        # just started -- floor at 0.25 of the proficiency weight so a
        # genuinely new-but-real skill still registers a little.
        effective_w = prof_w * max(trust, 0.25 if duration > 0 else 0.0)

        if name in C.CORE_SKILLS:
            concept_evidence[name] = max(concept_evidence.get(name, 0.0), effective_w)
            named_evidence.append((name, effective_w))
        if name in C.RARE_SYNONYMS:
            has_rare_synonym = True
            for mapped in C.RARE_SYNONYMS[name]:
                concept_evidence[mapped] = max(concept_evidence.get(mapped, 0.0), effective_w)
            named_evidence.append((name, effective_w))
        if name in C.CV_SPEECH_SKILLS:
            cv_speech_count += 1
        if name in C.DEPTH_SKILLS:
            depth_skill_count += 1
        if name in C.CORE_SKILLS and name in {"NLP", "Information Retrieval", "Machine Learning",
                                               "Deep Learning", "Semantic Search", "RAG"}:
            nlp_ir_count += 1

    # Coverage of CORE_SKILLS concept space, weighted by evidence strength.
    n_core = len(C.CORE_SKILLS)
    matched_weight = sum(concept_evidence.values())
    # Normalize against a realistic "strong candidate" ceiling rather than
    # "holds all 41 core skills" (no real person would / should). 10 strong
    # (trust=1.0) core skills is treated as a full-credit ceiling.
    skill_match_score = min(1.0, matched_weight / 10.0)

    total_skills = len(skills)
    cv_speech_heavy = (cv_speech_count / total_skills) if total_skills else 0.0
    if nlp_ir_count > 0:
        cv_speech_heavy *= 0.3  # offsetting NLP/IR exposure neutralizes most of the penalty

    named_evidence.sort(key=lambda x: -x[1])
    # buzzword_only_flag: ml/AI-relevant skill list with zero depth-tier
    # skills. Only meaningful when the candidate actually has buzzword-tier
    # skills to begin with (an empty-skills candidate isn't "buzzword-only,"
    # they're just sparse -- handled separately).
    buzzword_only_flag = 1.0 if (depth_skill_count == 0 and matched_weight > 0) else 0.0

    return {
        "skill_match_score": round(skill_match_score, 4),
        "plain_language_signal": 1.0 if has_rare_synonym else 0.0,
        "cv_speech_heavy": round(cv_speech_heavy, 4),
        "named_evidence": named_evidence[:5],
        "buzzword_only_flag": buzzword_only_flag,
    }


# ---------------------------------------------------------------------------
# Title / archetype
# ---------------------------------------------------------------------------

def title_archetype(candidate: dict[str, Any]) -> tuple[str, float]:
    title = candidate["profile"]["current_title"]
    archetype = C.TITLE_ARCHETYPE.get(title, C._FALLBACK_ARCHETYPE)
    gate = C.ARCHETYPE_GATE[archetype]
    return archetype, gate


# ---------------------------------------------------------------------------
# Experience fit
# ---------------------------------------------------------------------------

def experience_fit(candidate: dict[str, Any]) -> float:
    """
    JD: "5-9 years... roughly where people we've hired into this kind of
    role have landed" with an explicit sweet spot description of "6-8 years
    total ... of which 4-5 are in applied ML/AI roles." We model this as a
    smooth peak at 7 years, full credit across 5-9, decaying outside.
    """
    yoe = candidate["profile"]["years_of_experience"]
    if 5.0 <= yoe <= 9.0:
        # Peak at 7, full 1.0 across the whole band but slightly favor the center
        center_bonus = 1.0 - (abs(yoe - 7.0) / 8.0)
        return round(0.92 + 0.08 * center_bonus, 4)
    if yoe < 5.0:
        # Linear ramp from 0 at yoe=0 to 0.92 at yoe=5
        return round(max(0.0, (yoe / 5.0)) * 0.92, 4)
    # yoe > 9: gentle decay, JD says "we'll seriously consider candidates
    # outside the band if other signals are strong" so don't crater it
    over = yoe - 9.0
    return round(max(0.35, 0.92 - 0.07 * over), 4)


# ---------------------------------------------------------------------------
# Company / career-history derived features
# ---------------------------------------------------------------------------

def career_features(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Returns:
      product_company_score: 0-1, fraction of career (by duration) spent
        outside IT-services/consulting companies.
      ai_native_exposure: 0-1, fraction of career (by duration) at an
        explicitly AI-native company.
      consulting_only_flag: 1.0 if EVERY role has been at an IT-services/
        consulting company (the JD's hard disqualifier), else 0.0.
      shipped_system_evidence: 0-1, keyword-grounded signal that the
        candidate's own description text claims they built/shipped a
        ranking, search, retrieval, or recommendation system in production.
      title_chaser_flag: 1.0 if career_history shows 3+ roles averaging
        under 1.5 years each (the JD's explicit "title-chaser" pattern).
      tenure_stability_years: average months per role / 12, for reasoning text.
    """
    history = candidate.get("career_history", [])
    if not history:
        return {
            "product_company_score": 0.5, "ai_native_exposure": 0.0,
            "consulting_only_flag": 0.0, "shipped_system_evidence": 0.0,
            "title_chaser_flag": 0.0, "tenure_stability_years": 0.0,
        }

    total_months = sum(h.get("duration_months", 0) for h in history) or 1
    services_months = 0
    ai_native_months = 0
    all_services = True

    # Co-occurrence based, not proximity-based: requires a ranking/retrieval
    # term AND a scale/production term to both appear somewhere in the same
    # role description, but doesn't demand they sit within N characters of
    # each other. A tight proximity window turned out to be brittle against
    # genuine plain-language descriptions like "ranking layer for the
    # flagship product... across millions of items, for millions of users"
    # -- which clearly describes a shipped, at-scale ranking system but
    # doesn't use the word "production" near the word "ranking."
    HAS_RANKING_TERM = re.compile(
        r"\b(ranking|retrieval|recommend(ation|er)?s?|search|matching|embeddings?|"
        r"vector (search|database)|semantic search)\b", re.IGNORECASE,
    )
    HAS_SCALE_TERM = re.compile(
        r"\b(production|scale|scaled|real users?|shipped|live|deployed|"
        r"millions? of|billions? of|flagship product)\b", re.IGNORECASE,
    )
    shipped_hits = 0

    for h in history:
        comp = h.get("company", "")
        months = h.get("duration_months", 0)
        arch = C.company_archetype(comp)
        if arch == "it_services_consulting":
            services_months += months
        else:
            all_services = False
        if arch == "ai_native":
            ai_native_months += months
        desc = h.get("description", "") or ""
        if HAS_RANKING_TERM.search(desc) and HAS_SCALE_TERM.search(desc):
            shipped_hits += 1

    product_company_score = round(1.0 - (services_months / total_months), 4)
    ai_native_exposure = round(ai_native_months / total_months, 4)
    consulting_only_flag = 1.0 if all_services else 0.0
    shipped_system_evidence = min(1.0, shipped_hits / 2.0)  # 2+ roles describing it = full credit

    # title-chaser: the JD's specific definition is escalating titles
    # (Senior -> Staff -> Principal) via switching companies every ~1.5
    # years -- NOT simply "had several short jobs," which also describes
    # plenty of strong candidates who moved for legitimate reasons (layoffs,
    # better scope, comp). We require an actual seniority escalation across
    # >=2 distinct company changes, combined with short average tenure.
    def _seniority_level(title: str) -> int:
        t = title.lower()
        if "junior" in t:
            return 0
        if "staff" in t or "lead" in t or "principal" in t:
            return 3
        if "senior" in t:
            return 2
        return 1

    title_chaser_flag = 0.0
    if len(history) >= 3:
        avg_months = total_months / len(history)
        sorted_hist = sorted(history, key=lambda h: h.get("start_date", ""))
        levels = [_seniority_level(h.get("title", "")) for h in sorted_hist]
        companies_seq = [h.get("company", "") for h in sorted_hist]
        escalations = sum(
            1 for i in range(1, len(levels))
            if levels[i] > levels[i - 1] and companies_seq[i] != companies_seq[i - 1]
        )
        non_decreasing = all(levels[i] >= levels[i - 1] for i in range(1, len(levels)))
        if avg_months < 18 and escalations >= 2 and non_decreasing:
            title_chaser_flag = 1.0

    return {
        "product_company_score": product_company_score,
        "ai_native_exposure": ai_native_exposure,
        "consulting_only_flag": consulting_only_flag,
        "shipped_system_evidence": round(shipped_system_evidence, 4),
        "title_chaser_flag": title_chaser_flag,
        "tenure_stability_years": round(total_months / len(history) / 12.0, 2),
    }


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def location_fit(candidate: dict[str, Any]) -> float:
    profile = candidate["profile"]
    if profile.get("country") != "India":
        # JD: "Outside India: case-by-case, but we don't sponsor work visas."
        return 0.15
    city = profile.get("location", "").split(",")[0].strip()
    if city in C.LOCATION_PREFERRED:
        return 1.0
    if city in C.LOCATION_WELCOME:
        return 0.75
    return 0.45  # other India city -- plausible, not penalized hard


# ---------------------------------------------------------------------------
# Disqualifiers (the "what we explicitly do NOT want" section of the JD)
# ---------------------------------------------------------------------------

def disqualifier_multiplier(candidate: dict[str, Any], career: dict[str, float],
                             skill: dict[str, float]) -> tuple[float, list[str]]:
    """
    Returns (multiplier in (0, 1], list of triggered disqualifier names)
    so reasoning.py can surface honest concerns.
    """
    mult = 1.0
    triggered: list[str] = []

    if career["consulting_only_flag"] == 1.0:
        mult *= 0.12
        triggered.append("consulting_only_career")

    if career["title_chaser_flag"] == 1.0:
        mult *= 0.55
        triggered.append("title_chaser_pattern")

    if skill["cv_speech_heavy"] > 0.4:
        mult *= 0.45
        triggered.append("cv_speech_without_nlp")

    if skill.get("buzzword_only_flag") == 1.0:
        # Lighter penalty than the others -- this is a softer, inferred
        # signal (verified against only 14/1,179 ml_core candidates), not
        # a literal JD-named rule like consulting-only.
        mult *= 0.70
        triggered.append("buzzword_only_no_depth")

    return round(mult, 4), triggered


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def is_honeypot(candidate: dict[str, Any]) -> bool:
    skills = candidate.get("skills", [])
    zero_duration_expert = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0
    )
    return zero_duration_expert >= C.HONEYPOT_ZERO_DURATION_EXPERT_THRESHOLD


# ---------------------------------------------------------------------------
# Behavioral signals (redrob_signals)
# ---------------------------------------------------------------------------

def behavioral_modifier(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Returns a modifier in roughly [0.4, 1.15] -- behavioral signals should
    adjust the score, not dominate it, per the JD: "weigh behavioral
    signals... down-weight them appropriately" (not "rank purely by them").
    Also returns the raw recency_days and notice_period for reasoning text.
    """
    sig = candidate["redrob_signals"]

    last_active = date.fromisoformat(sig["last_active_date"])
    recency_days = (REFERENCE_DATE - last_active).days
    # Full credit within 30 days, linear decay to 0.5x by 180 days, floor after
    if recency_days <= 30:
        recency_score = 1.0
    elif recency_days <= 180:
        recency_score = 1.0 - 0.5 * ((recency_days - 30) / 150.0)
    else:
        recency_score = 0.45

    response = sig.get("recruiter_response_rate", 0.0)
    interview_completion = sig.get("interview_completion_rate", 0.0)
    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.6

    notice = sig.get("notice_period_days", 90)
    # JD: "We'd love sub-30-day notice. We can buy out up to 30 days."
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.85
    elif notice <= 90:
        notice_score = 0.65
    else:
        notice_score = 0.45

    demand_signal = min(1.0, sig.get("saved_by_recruiters_30d", 0) / 10.0)

    composite = (
        0.30 * recency_score
        + 0.20 * response
        + 0.15 * interview_completion
        + 0.15 * open_to_work
        + 0.10 * notice_score
        + 0.10 * demand_signal
    )
    # Map composite (0-1) onto a modifier range of [0.55, 1.15]
    modifier = 0.55 + composite * 0.60

    return {
        "behavioral_modifier": round(modifier, 4),
        "recency_days": recency_days,
        "notice_period_days": notice,
        "recruiter_response_rate": response,
    }
