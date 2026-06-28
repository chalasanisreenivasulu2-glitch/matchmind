"""
Lightweight semantic similarity layer.

Why TF-IDF instead of a neural embedding model: the ranking step runs under
a hard 5-minute / 16GB / CPU-only / no-network budget for 100,000
candidates. TF-IDF + cosine similarity is deterministic, has zero model-
download or GPU dependency, and computes over the full pool in seconds --
which trivially satisfies the constraint instead of fighting it. It also
means Stage 3 reproduction has nothing exotic to fail on.

This is a genuine architectural tradeoff, not a default: a sentence-
transformers / BGE-small embedding model would generalize better to truly
novel phrasing, at the cost of a ~100MB model load and materially higher
rank-time compute. Given that the dataset's title/company/skill fields are
closed vocabularies (see constants.py) where exact lookup already does most
of the semantic work, TF-IDF's main job is just to catch free-text phrasing
in `summary` and career_history `description` fields that the lookup
tables can't see -- a job it's well suited for. See README "What we'd do
with more time" for the embedding-model upgrade path.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

JOB_DESCRIPTION_TEXT = """
Senior AI Engineer Founding Team. Own the intelligence layer of Redrob's
product: the ranking, retrieval, and matching systems that decide what
recruiters see when they search for candidates and what candidates see
when they search for roles. Ship a v2 ranking system involving embeddings,
hybrid retrieval, and LLM-based re-ranking. Set up evaluation infrastructure:
offline benchmarks, online A/B testing, NDCG, MRR, MAP, offline-to-online
correlation. Production experience with embeddings-based retrieval systems:
sentence-transformers, OpenAI embeddings, BGE, E5. Production experience
with vector databases or hybrid search infrastructure: Pinecone, Weaviate,
Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Strong Python, code
quality. Hands-on experience designing evaluation frameworks for ranking
systems. Scrappy product-engineering attitude, ship a working ranker
quickly. Has shipped at least one end-to-end ranking, search, or
recommendation system to real users at meaningful scale. Strong opinions
about retrieval, hybrid vs dense, evaluation offline vs online, LLM
integration, when to fine-tune vs prompt. Mentoring engineers, recruiter-
experience product collaboration. Six to eight years total experience,
four to five years in applied ML AI roles at product companies, not pure
services. Located in or willing to relocate to Noida or Pune, India.
Hyderabad Mumbai Delhi NCR Bangalore welcome.
"""


def candidate_document(candidate: dict[str, Any]) -> str:
    profile = candidate["profile"]
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    for h in candidate.get("career_history", []):
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(p for p in parts if p)


def compute_semantic_similarity(documents: list[str]) -> list[float]:
    """
    documents: list of candidate documents, in pool order.
    Returns cosine similarity of each document against JOB_DESCRIPTION_TEXT,
    fit on (documents + [jd]) so the vocabulary reflects the actual pool.
    """
    corpus = documents + [JOB_DESCRIPTION_TEXT]
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(corpus)
    jd_vector = matrix[-1]
    candidate_matrix = matrix[:-1]
    sims = linear_kernel(candidate_matrix, jd_vector).ravel()
    # Normalize to 0-1 (cosine sim with sublinear TF is already roughly in
    # this range, but min-max normalize against the pool's own spread so the
    # score is well-distributed regardless of corpus drift).
    lo, hi = sims.min(), sims.max()
    if hi > lo:
        sims = (sims - lo) / (hi - lo)
    return sims.tolist()
