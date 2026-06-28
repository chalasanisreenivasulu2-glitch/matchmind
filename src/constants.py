"""
Closed-vocabulary lookup tables for the Redrob candidate pool.

These tables were built by enumerating the FULL 100,000-candidate dataset
(not sampled) and confirming that current_title (47 values), career_history
company (63 values), and skill name (133 values) are all closed,
non-overlapping vocabularies. Because of that, classification here is exact
lookup, not fuzzy matching -- there is no out-of-vocabulary risk for the
released dataset. If Redrob expands the vocabulary later, the `_FALLBACK_*`
values define safe defaults for unseen titles/companies/skills.

See README.md "Why lookup tables, not fuzzy matching" for the full
methodology writeup.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Title archetypes
# ---------------------------------------------------------------------------
# Verified counts across all 100,000 candidates (sums to exactly 100,000):
#   ml_core            1,179   (1.18%)  -- the true target population
#   ml_adjacent_data    3,627   (3.63%)
#   software_general   26,373  (26.37%)
#   non_technical      68,821  (68.82%)

TITLE_ARCHETYPE: dict[str, str] = {
    # ml_core -- the JD's actual target. NOTE: "Computer Vision Engineer" is
    # included here as a title archetype but is penalized separately by the
    # cv_speech_only disqualifier in features.py unless NLP/IR skills are
    # also present, per the JD's explicit "CV/speech without NLP exposure" rule.
    "ML Engineer": "ml_core",
    "AI Research Engineer": "ml_core",
    "Data Scientist": "ml_core",
    "Senior Software Engineer (ML)": "ml_core",
    "Computer Vision Engineer": "ml_core",
    "Junior ML Engineer": "ml_core",
    "AI Specialist": "ml_core",
    "Recommendation Systems Engineer": "ml_core",
    "Machine Learning Engineer": "ml_core",
    "Applied ML Engineer": "ml_core",
    "Search Engineer": "ml_core",
    "AI Engineer": "ml_core",
    "Senior Data Scientist": "ml_core",
    "NLP Engineer": "ml_core",
    "Senior NLP Engineer": "ml_core",
    "Senior Machine Learning Engineer": "ml_core",
    "Staff Machine Learning Engineer": "ml_core",
    "Senior AI Engineer": "ml_core",
    "Senior Applied Scientist": "ml_core",
    "Lead AI Engineer": "ml_core",
    # ml_adjacent_data -- data infra roles. These are the "Tier 5 candidate"
    # archetype the JD describes: no AI buzzwords, but plausible builders.
    "Analytics Engineer": "ml_adjacent_data",
    "Data Engineer": "ml_adjacent_data",
    "Data Analyst": "ml_adjacent_data",
    "Backend Engineer": "ml_adjacent_data",
    "Senior Data Engineer": "ml_adjacent_data",
    # software_general -- competent engineers, no demonstrated ML/retrieval focus
    "Software Engineer": "software_general",
    "Full Stack Developer": "software_general",
    "Cloud Engineer": "software_general",
    "Java Developer": "software_general",
    ".NET Developer": "software_general",
    "DevOps Engineer": "software_general",
    "Mobile Developer": "software_general",
    "Frontend Engineer": "software_general",
    "QA Engineer": "software_general",
    "Senior Software Engineer": "software_general",
    # non_technical -- the keyword-stuffer risk pool
    "Business Analyst": "non_technical",
    "HR Manager": "non_technical",
    "Mechanical Engineer": "non_technical",
    "Accountant": "non_technical",
    "Project Manager": "non_technical",
    "Customer Support": "non_technical",
    "Operations Manager": "non_technical",
    "Content Writer": "non_technical",
    "Sales Executive": "non_technical",
    "Civil Engineer": "non_technical",
    "Graphic Designer": "non_technical",
    "Marketing Manager": "non_technical",
}

# Archetype -> base score multiplier (gating factor, applied before
# additive feature combination). non_technical is not zero -- a candidate
# could theoretically still rank if every other dimension is implausibly
# strong (shouldn't happen in practice, but we don't want a hard floor that
# makes the score field meaningless / indistinguishable within the bucket).
ARCHETYPE_GATE: dict[str, float] = {
    "ml_core": 1.00,
    "ml_adjacent_data": 0.55,
    "software_general": 0.18,
    "non_technical": 0.04,
}
_FALLBACK_ARCHETYPE = "non_technical"  # unseen title -> treat conservatively

# ---------------------------------------------------------------------------
# Company archetypes
# ---------------------------------------------------------------------------
# The JD explicitly names TCS / Infosys / Wipro / Accenture / Cognizant /
# Capgemini as a disqualifier category ("People who have only worked at
# consulting firms... in their entire career"). HCL, Tech Mahindra,
# Mindtree, and Mphasis are the same business-model archetype (Indian IT
# services majors) and are extended into the same bucket.
IT_SERVICES_CONSULTING = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mindtree", "Mphasis",
}

# Companies whose dataset-assigned industry is explicitly AI-native
# (AI/ML, AI Services, HealthTech AI, Conversational AI, Voice AI).
# Real work at one of these is strong evidence of the "shipped to real
# users at an AI-native product company" signal the JD asks for.
AI_NATIVE_COMPANIES = {
    "Aganitha", "Glance", "Krutrim", "Locobuzz", "Mad Street Den",
    "Observe.AI", "Rephrase.ai", "Sarvam AI", "Yellow.ai",
    "Niramai", "Wysa", "Haptik", "Verloop.io", "Saarthi.ai", "Genpact AI",
}

# Generic large-company filler names used heavily throughout the dataset
# for volume padding (Pied Piper, Initech, Hooli, etc. -- pop-culture
# placeholder names). Not a positive or negative signal by itself.
GENERIC_FILLER_COMPANIES = {
    "Pied Piper", "Initech", "Wayne Enterprises", "Acme Corp",
    "Stark Industries", "Hooli", "Globex Inc", "Dunder Mifflin",
}

_FALLBACK_COMPANY_ARCHETYPE = "product_other"  # unseen company -> neutral


def company_archetype(company: str) -> str:
    if company in IT_SERVICES_CONSULTING:
        return "it_services_consulting"
    if company in AI_NATIVE_COMPANIES:
        return "ai_native"
    if company in GENERIC_FILLER_COMPANIES:
        return "generic_filler"
    return _FALLBACK_COMPANY_ARCHETYPE


# ---------------------------------------------------------------------------
# Skill tiers and synonym mapping
# ---------------------------------------------------------------------------
# Empirical skill frequency in the 100K pool falls into four clean bands:
#   ~12,000-12,250 (~12%) : generic cross-industry noise skills
#   ~4,600-5,160   (~5%)  : "buzzword" AI tier -- what keyword-stuffers pad with
#   ~1,280-1,400   (~1.3%): genuine hands-on ML/IR practitioner tier (incl. Python)
#   1-7            (<0.01%): rare plain-language paraphrases, found on exactly
#                            8 candidates, all Senior/Staff/Lead AI/ML titles
#                            with 5.4-9.0 yrs experience. This is the JD's own
#                            "doesn't say RAG or Pinecone" test case.
#
# CORE_SKILLS are the JD's "things you absolutely need": embeddings-based
# retrieval, vector DB / hybrid search, Python, and ranking-evaluation
# literacy. RARE_SYNONYMS maps the plain-language tier onto the same
# underlying capability so a pure keyword system doesn't go blind to it.

CORE_SKILLS = {
    "Python", "Embeddings", "Vector Search", "Sentence Transformers",
    "Semantic Search", "Pinecone", "FAISS", "Weaviate", "Milvus", "Qdrant",
    "pgvector", "RAG", "Information Retrieval", "Recommendation Systems",
    "Elasticsearch", "OpenSearch", "BM25", "Learning to Rank", "Haystack",
    "LlamaIndex", "Hugging Face Transformers", "LangChain", "LLMs",
    "Prompt Engineering", "Fine-tuning LLMs", "MLOps", "MLflow",
    "Weights & Biases", "Kubeflow", "BentoML", "Feature Engineering",
    "NLP", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow",
    "scikit-learn", "Data Science", "Statistical Modeling", "QLoRA",
    "LoRA", "PEFT",
}

# Skills the JD explicitly does NOT want as a primary focus unless paired
# with NLP/IR exposure (computer vision / speech / robotics without
# language or retrieval work). Tracked separately so features.py can apply
# the cv_speech_only disqualifier.
CV_SPEECH_SKILLS = {
    "YOLO", "GANs", "OpenCV", "ASR", "Image Classification",
    "Computer Vision", "Speech Recognition", "CNN", "Object Detection",
    "Diffusion Models", "TTS",
}

# Genuine hands-on practitioner depth, as opposed to the "buzzword" tier
# (LangChain, RAG, Embeddings, Sentence Transformers, etc.) that a recent
# wrapper-only profile would lean on. The JD's literal "under 12 months,
# LangChain-only" disqualifier does not actually occur in this dataset --
# verified: the minimum current-role tenure among all 1,179 ml_core-titled
# candidates is 12+ months, so that specific trigger condition cannot fire
# here. What DOES occur, in exactly 14 of 1,179 ml_core candidates (1.2%),
# is the underlying pattern the JD is really worried about: an ml_core
# title backed entirely by buzzword-tier skills with zero skills from this
# depth tier. That's what buzzword_only_flag below actually detects.
DEPTH_SKILLS = {
    "PyTorch", "TensorFlow", "scikit-learn", "NLP", "Machine Learning",
    "Deep Learning", "BM25", "Elasticsearch", "OpenSearch", "Weaviate",
    "Milvus", "Qdrant", "pgvector", "Haystack", "LlamaIndex", "LoRA",
    "QLoRA", "PEFT", "Learning to Rank",
}

# Rare plain-language paraphrase -> nearest CORE_SKILLS concept(s).
# A candidate holding a rare-tier skill gets credited as if they held the
# mapped core skill(s) for matching purposes, in addition to a standalone
# "plain_language_signal" bonus (see features.py) since holding ANY of
# these is itself a strong, almost certainly deliberate, positive signal.
RARE_SYNONYMS: dict[str, list[str]] = {
    "Information Retrieval Systems": ["Information Retrieval"],
    "Search Backend": ["Vector Search", "Information Retrieval"],
    "Text Encoders": ["Embeddings", "Sentence Transformers"],
    "Vector Representations": ["Embeddings"],
    "Content Matching": ["Recommendation Systems", "Semantic Search"],
    "Model Adaptation": ["Fine-tuning LLMs", "PEFT"],
    "Ranking Systems": ["Learning to Rank"],
    "Search & Discovery": ["Information Retrieval", "Semantic Search"],
    "Workflow Orchestration": ["MLOps"],
    "Search Infrastructure": ["Vector Search", "Information Retrieval"],
    "Indexing Algorithms": ["Vector Search"],
    "Open-source ML libraries": ["Machine Learning"],
    "Natural Language Processing": ["NLP"],
    "Document Processing": ["RAG", "Information Retrieval"],
}

# ---------------------------------------------------------------------------
# Location tiers (JD: Pune/Noida preferred; Hyderabad/Pune/Mumbai/Delhi NCR/
# Bangalore explicitly welcome; outside India case-by-case, no visa sponsorship)
# ---------------------------------------------------------------------------
LOCATION_PREFERRED = {"Pune", "Noida"}
LOCATION_WELCOME = {"Hyderabad", "Mumbai", "Delhi", "Bangalore"}

# ---------------------------------------------------------------------------
# Honeypot detection threshold
# ---------------------------------------------------------------------------
# A candidate with >=3 skills marked "expert" proficiency at 0 duration_months
# is structurally impossible (expert-level claim with zero time spent).
# Verified against the full pool: 99,979 candidates have ZERO such skills;
# the remaining 21 have 3, 4, or 5 -- a clean bimodal split with no
# in-between cases, strongly suggesting this is a deliberately injected
# signal rather than noise.
HONEYPOT_ZERO_DURATION_EXPERT_THRESHOLD = 3
