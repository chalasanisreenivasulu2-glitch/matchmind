"""
JD Parser Agent

Converts raw, unstructured job description text into a structured
requirements dict that the Ranking Agent can directly consume.

In production this would call an LLM (Claude/GPT-4) with a structured
output schema. The rule-based fallback implemented here is used when
no API key is available (demo mode) and extracts requirements by
matching against the known closed vocabularies from the candidate pool.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.constants import CORE_SKILLS, DEPTH_SKILLS, LOCATION_PREFERRED, LOCATION_WELCOME


EXPERIENCE_RE = re.compile(r"(\d+)[\s\-–]+(\d+)\s*(?:years?|yrs?)", re.IGNORECASE)
MIN_EXP_RE    = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience", re.IGNORECASE)

# Keywords that hint a role is explicitly NOT services/outsourcing
PRODUCT_SIGNALS = re.compile(
    r"\b(product company|startup|series [a-e]|founding team|scale-up|"
    r"not.*consulting|no.*services|not.*outsourc)\b", re.IGNORECASE
)

# Location mentions
INDIA_CITIES = {
    "pune", "noida", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "gurugram", "gurgaon", "chennai"
}


class JDParserAgent:
    """
    Parse a raw JD string into a structured requirements dictionary.

    Returns:
      {
        "required_skills": list[str],      # from CORE_SKILLS vocabulary
        "min_experience_years": float,
        "max_experience_years": float,
        "preferred_locations": list[str],
        "requires_product_company": bool,
        "raw_text": str                    # original JD preserved for TF-IDF
      }
    """

    def parse(self, jd_text: str) -> dict[str, Any]:
        text_lower = jd_text.lower()

        # --- Skill extraction ---
        required_skills = [
            skill for skill in CORE_SKILLS
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower)
        ]

        # --- Experience range ---
        min_exp, max_exp = 0.0, 15.0
        range_match = EXPERIENCE_RE.search(jd_text)
        if range_match:
            min_exp = float(range_match.group(1))
            max_exp = float(range_match.group(2))
        else:
            single_match = MIN_EXP_RE.search(jd_text)
            if single_match:
                min_exp = float(single_match.group(1))
                max_exp = min_exp + 4.0

        # --- Location ---
        preferred_locations = [
            city.title() for city in INDIA_CITIES
            if city in text_lower
        ]

        # --- Product company preference ---
        requires_product_company = bool(PRODUCT_SIGNALS.search(jd_text))

        return {
            "required_skills": required_skills,
            "min_experience_years": min_exp,
            "max_experience_years": max_exp,
            "preferred_locations": preferred_locations or list(LOCATION_PREFERRED),
            "requires_product_company": requires_product_company,
            "raw_text": jd_text,
        }


if __name__ == "__main__":
    sample_jd = """
    We're looking for a Senior AI Engineer (5-9 years) to join Redrob's founding team.
    You'll own our ranking and retrieval systems. Must have Python, RAG, Embeddings,
    FAISS or Qdrant experience, and Learning to Rank. Product company background required.
    Located in Pune or Noida preferred; Hyderabad/Bangalore welcome.
    """
    agent = JDParserAgent()
    result = agent.parse(sample_jd)
    import json
    print(json.dumps(result, indent=2))
