# MatchMind 🎯
### AI-Native Autonomous Recruiting Intelligence for Redrob

> **Redrob Ideathon — Track 1 | AI Systems & Workflow Innovation Challenge**
> Team: **Disha and Sethu** | Member: Chalasani Sreenivasulu

---

## What is MatchMind?

MatchMind is a **multi-agent autonomous recruiting intelligence system** built on top of Redrob's existing platform. It replaces keyword search and opaque black-box ML with five coordinated AI agents that work together to:

- **Parse** any recruiter JD into structured requirements (autonomously)
- **Rank** 100,000+ candidates in under 60 seconds (CPU-only, no model download)
- **Explain** every ranking decision in plain English (grounded, no hallucination)
- **Detect** impossible/fraudulent profiles before they pollute results
- **Learn** from every recruiter shortlist and reject action (continuous improvement)

Recruiters paste a JD. MatchMind does the rest.

---

## Problem

Redrob's candidate discovery is bottlenecked by two failure modes:

| Approach | What it gets wrong |
|---|---|
| Keyword search | Misses candidates who describe the same work differently ("Ranking Systems" ≠ "Learning to Rank") |
| Black-box ML | Ranks correctly on average but can't explain *why*, making decisions impossible to audit or defend |

Both approaches also fail to use Redrob's unique advantage: 23 live behavioral signals per candidate (last active date, recruiter response rate, notice period, etc.) that external competitors can't access.

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RECRUITER (Redrob UI)                       │
│                   pastes JD → sees ranked list                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR                              │
│         routes requests · manages context · coordinates agents  │
└──┬──────────┬──────────┬──────────────┬──────────────────────┘
   │          │          │              │
   ▼          ▼          ▼              ▼
┌──────┐ ┌────────┐ ┌────────┐  ┌───────────┐
│  JD  │ │Ranking │ │ Fraud  │  │Explanation│
│Parser│ │ Agent  │ │ Agent  │  │  Agent    │
│(LLM) │ │(Embed+ │ │(Rules) │  │(Template) │
│      │ │ Score) │ │        │  │           │
└──────┘ └────────┘ └────────┘  └───────────┘
                                        │
                    async event bus     │
                         │              ▼
                    ┌────▼────┐  ┌─────────────┐
                    │Feedback │  │  Recruiter  │
                    │ Agent   │  │  Sees List  │
                    └─────────┘  └─────────────┘
```

### Agent Responsibilities

| Agent | What it does | When it runs |
|---|---|---|
| **JD Parser Agent** | Extracts skills, experience range, location, product-company requirement from raw JD text | Sync, once per query |
| **Ranking Agent** | Scores all candidates using semantic similarity + 8 feature dimensions | Sync, per query |
| **Fraud Agent** | Flags structurally-impossible profiles (expert claim + 0 months used) | Sync, batch before ranking |
| **Explanation Agent** | Generates a grounded 1-sentence explanation per candidate (verified against profile, no hallucination) | Sync, on top-N only |
| **Feedback Agent** | Records recruiter shortlist/reject actions; trains LightGBM reranker in Phase 3 | **Async**, never blocks response |

---

## Scoring Formula

Every score is fully transparent and auditable:

```
score = base_combo × archetype_gate × disqualifier_multiplier × behavioral_modifier

base_combo = (
    0.32 × skill_match_score          # weighted coverage of required skills
  + 0.22 × semantic_similarity        # TF-IDF cosine vs JD text
  + 0.16 × experience_fit             # peaked at 7 yrs, full credit 5-9yr band
  + 0.12 × product_company_score      # fraction of career NOT at IT services/consulting
  + 0.10 × shipped_system_evidence    # did descriptions mention shipping at scale?
  + 0.08 × location_fit               # Pune/Noida=1.0, hub cities=0.75, other India=0.45
  + 0.05 × plain_language_signal      # rare skill synonym bonus (catches "Ranking Systems" etc.)
  + 0.05 × ai_native_exposure         # time spent at AI-native product companies
)

archetype_gate:          ml_core=1.00 | ml_adjacent=0.55 | sw_general=0.18 | non_technical=0.04
behavioral_modifier:     0.55–1.15 range based on recency, response rate, notice period, demand
honeypot_cap:            score capped at 0.01 if fraud_agent fires
```

Every weight traces back to a specific line in the job description. No black-box decisions.

---

## Dataset Insights (from full 100K-candidate analysis)

Before writing any scoring logic, the entire 100,000-candidate pool was analyzed:

- **47 unique titles, 63 unique companies, 133 unique skills** — all closed vocabularies, enabling exact lookup instead of fuzzy matching
- Only **1,179 candidates (1.18%)** have ML/AI-core titles — the real target population
- **14 ultra-rare skill names** (e.g., "Vector Representations", "Ranking Systems") appear on exactly **8 candidates** — all Senior/Staff AI engineers. A keyword matcher gives them zero score. MatchMind's synonym map catches all 8.
- **21 structural honeypots** detected via clean bimodal signal (expert skill + 0 months duration)

---

## Project Structure

```
matchmind/
├── README.md                    ← you are here
├── requirements.txt             ← numpy, scikit-learn (no GPU, no API keys needed)
├── rank.py                      ← CLI entrypoint: candidates.jsonl → submission.csv
│
├── agents/                      ← multi-agent system
│   ├── orchestrator.py          ← coordinates all agents
│   ├── jd_parser.py             ← JD text → structured requirements
│   ├── ranking_agent.py         ← scores and ranks full candidate pool
│   ├── fraud_agent.py           ← detects honeypot / impossible profiles
│   ├── explanation_agent.py     ← generates grounded per-candidate reasoning
│   └── feedback_agent.py        ← records recruiter actions, feeds reranker
│
├── src/                         ← core scoring engine (used by agents)
│   ├── constants.py             ← closed-vocabulary lookup tables
│   ├── features.py              ← per-candidate feature engineering
│   ├── semantic.py              ← TF-IDF semantic similarity layer
│   ├── scorer.py                ← transparent weighted scoring formula
│   └── reasoning.py            ← grounded explanation text generation
│
├── app/
│   └── streamlit_app.py         ← sandbox demo (deploy to Streamlit Cloud / HF Spaces)
│
└── data/
    └── sample_candidates.json   ← 50-candidate sample from challenge bundle
```

---

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Run the full multi-agent pipeline on the sample
python agents/orchestrator.py

# Run individual agents
python agents/jd_parser.py
python agents/fraud_agent.py
python agents/feedback_agent.py
python agents/explanation_agent.py

# CLI: rank a full 100K candidate pool (requires candidates.jsonl)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# Launch the interactive sandbox demo
pip install streamlit
streamlit run app/streamlit_app.py
```

---

## Demo Output

```
MatchMind — Ranked 50 candidates
Honeypots excluded: 0  |  Runtime: 0.03s

JD parsed: Skills=[Python, RAG, Learning to Rank, Embeddings, FAISS]
           Experience=5-9 yrs  |  Locations=[Pune, Noida, Bangalore, Hyderabad]

Top candidates:

  #1  1.0823  CAND_0000031
  Holds Pinecone, Embeddings, Information Retrieval;
  Recommendation Systems Engineer at Swiggy, 6.0 yrs.

  #2  0.2345  CAND_0000001
  6.9 yrs lands in the JD's 5-9 yr target band;
  Backend Engineer at Mindtree, 6.9 yrs.
  Concern: location isn't Pune/Noida or one of the JD's welcomed hub cities.
```

---

## Performance

| Metric | Value |
|---|---|
| Full 100K candidate pool | **~50 seconds** wall-clock |
| Compute budget | **5 minutes** (10× headroom) |
| GPU required | No |
| Network calls at rank time | None |
| Model downloads required | None |
| Honeypots in final top-100 | 0 / 21 |
| Plain-language candidates recalled | 8 / 8 |

---

## Roadmap

| Phase | Timeline | What ships |
|---|---|---|
| **Phase 1** | Month 1-2 | Rule-based ranker + Fraud agent + Explanation agent (already built ✅) |
| **Phase 2** | Month 3-5 | LLM JD Parser Agent + BGE embedding upgrade + full Orchestrator API |
| **Phase 3** | Month 6-9 | Feedback Agent + async event bus + LightGBM LTR retrainer |

---

## Tech Stack

- **Python 3.10+** — all agents and core engine
- **scikit-learn** — TF-IDF vectorizer, cosine similarity
- **numpy** — feature computation
- **Streamlit** — sandbox demo UI
- *(Phase 2)* **sentence-transformers / BGE** — embedding upgrade
- *(Phase 3)* **LightGBM** — learning-to-rank retrainer
- *(Phase 2)* **Claude / GPT-4** — LLM-powered JD parsing

---

## License

MIT — built for the Redrob Ideathon, Track 1.
