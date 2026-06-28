"""
MatchMind — Interactive Sandbox Demo
Redrob Ideathon | Track 1 | Team: Disha and Sethu

Runs the full 5-agent MatchMind pipeline live in the browser:
  JD Parser Agent → Fraud Agent → Ranking Agent → Explanation Agent
  (Feedback Agent records your shortlist/reject clicks)

No API keys, no GPU, no model downloads required.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import Orchestrator

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MatchMind — AI Recruiting Intelligence",
    page_icon="🎯",
    layout="wide",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:0'>🎯 MatchMind</h1>
<p style='color:#666;font-size:1.1rem;margin-top:4px'>
  AI-Native Autonomous Recruiting Intelligence · Redrob Ideathon Track 1
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline")
    st.markdown("""
**Agents running live:**
1. 📋 **JD Parser Agent** — extracts structured requirements from your JD
2. 🔍 **Fraud Agent** — screens for impossible profiles
3. 🏆 **Ranking Agent** — scores all candidates (semantic + 8 feature dimensions)
4. 💬 **Explanation Agent** — generates grounded reasoning per candidate
5. 📊 **Feedback Agent** — records your shortlist/reject clicks

All coordinated by the **Orchestrator**.
""")
    st.divider()
    st.markdown("""
**Data**
- Default: 50-candidate sample from the official challenge bundle
- Upload your own JSON/JSONL to test with different candidates
""")
    uploaded = st.file_uploader(
        "Upload candidates (JSON or JSONL)",
        type=["json", "jsonl"],
    )
    st.divider()
    top_n = st.slider("Results to show", min_value=5, max_value=50, value=10)

# ── Load candidates ───────────────────────────────────────────────────────────
SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample_candidates.json"

@st.cache_data(show_spinner=False)
def load_sample() -> list[dict]:
    return json.load(open(SAMPLE_PATH))

def load_upload(f) -> list[dict]:
    raw = f.read().decode("utf-8")
    if f.name.endswith(".jsonl"):
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    d = json.loads(raw)
    return d if isinstance(d, list) else [d]

candidates = load_upload(uploaded) if uploaded else load_sample()
source_label = f"📂 {uploaded.name} ({len(candidates)} candidates)" if uploaded \
               else f"📦 Built-in sample ({len(candidates)} candidates)"

# ── JD Input ──────────────────────────────────────────────────────────────────
st.subheader("1. Paste a Job Description")

DEFAULT_JD = """Senior AI Engineer — Founding Team at Redrob AI (Series A, Pune/Noida).

We're looking for someone to own our candidate ranking and retrieval systems.

Must have:
- 5-9 years of experience
- Python, RAG, Embeddings (sentence-transformers / BGE / E5)
- Vector database experience: FAISS, Qdrant, Weaviate, or Pinecone
- Learning to Rank or recommendation systems experience
- Experience at product companies (not IT services / consulting)

Location: Pune or Noida preferred; Hyderabad, Bangalore, Mumbai welcome.
"""

jd_text = st.text_area(
    "Job Description",
    value=DEFAULT_JD,
    height=220,
    label_visibility="collapsed",
)

col_source, col_run = st.columns([3, 1])
col_source.caption(source_label)
run = col_run.button("🚀 Run MatchMind", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run:
    if not jd_text.strip():
        st.warning("Please paste a job description first.")
        st.stop()

    orch = Orchestrator(feedback_store_path="/tmp/matchmind_streamlit_feedback.jsonl")

    with st.spinner("Running 5-agent pipeline..."):
        t0 = time.time()
        output = orch.rank_for_jd(jd_text, candidates, top_n=top_n, verbose=True)
        elapsed = time.time() - t0

    stats = output["stats"]
    jd_parsed = output["jd_parsed"]
    results = output["results"]

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("2. Pipeline Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates scored", stats["pool_size"])
    c2.metric("Honeypots excluded", stats["honeypots_flagged"])
    c3.metric("Results returned", stats["top_n_returned"])
    c4.metric("Runtime", f"{elapsed:.2f}s")

    # ── JD Parsed ─────────────────────────────────────────────────────────────
    with st.expander("📋 JD Parser Agent output", expanded=False):
        col_a, col_b = st.columns(2)
        col_a.markdown("**Required skills extracted:**")
        if jd_parsed["required_skills"]:
            for sk in jd_parsed["required_skills"]:
                col_a.markdown(f"- {sk}")
        else:
            col_a.caption("None matched from closed vocabulary")

        col_b.markdown(f"**Experience range:** {jd_parsed['min_experience_years']:.0f}–{jd_parsed['max_experience_years']:.0f} yrs")
        col_b.markdown(f"**Preferred locations:** {', '.join(jd_parsed['preferred_locations']) or 'Any'}")
        col_b.markdown(f"**Product company required:** {'Yes' if jd_parsed['requires_product_company'] else 'Not specified'}")

    # ── Ranked results ────────────────────────────────────────────────────────
    st.subheader("3. Ranked Candidates")
    st.caption("Click ✅ Shortlist or ❌ Reject to feed the Feedback Agent (recorded live).")

    for r in results:
        score_color = "#2ecc71" if r["score"] > 0.5 else "#e67e22" if r["score"] > 0.2 else "#e74c3c"
        bd = r.get("breakdown", {})

        with st.container(border=True):
            hcol, scol = st.columns([7, 1])
            with hcol:
                flags = ""
                if r.get("honeypot"):
                    flags += " ⚠️ HONEYPOT"
                if r.get("disqualifiers"):
                    flags += f" 🚩 {', '.join(r['disqualifiers'])}"

                st.markdown(
                    f"**#{r['rank']}** &nbsp; `{r['candidate_id']}`"
                    f"<span style='color:{score_color};font-weight:700;margin-left:12px'>"
                    f"score: {r['score']:.4f}</span>"
                    f"<span style='color:#999;font-size:0.85rem;margin-left:8px'>{flags}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"*{r['reasoning']}*")

            with scol:
                st.markdown("<br>", unsafe_allow_html=True)
                fb_col1, fb_col2 = st.columns(2)
                if fb_col1.button("✅", key=f"sl_{r['candidate_id']}", help="Shortlist"):
                    orch.record_feedback("DEMO_JD", r["candidate_id"], "shortlist",
                                         r["rank"], r["score"])
                    st.toast(f"Shortlisted {r['candidate_id']}", icon="✅")
                if fb_col2.button("❌", key=f"rj_{r['candidate_id']}", help="Reject"):
                    orch.record_feedback("DEMO_JD", r["candidate_id"], "reject",
                                         r["rank"], r["score"])
                    st.toast(f"Rejected {r['candidate_id']}", icon="❌")

            if bd:
                with st.expander("Feature breakdown", expanded=False):
                    feat_cols = st.columns(4)
                    feat_cols[0].metric("Skill match", f"{bd.get('skill_match_score',0):.2f}")
                    feat_cols[1].metric("Semantic sim", f"{bd.get('semantic_similarity',0):.2f}")
                    feat_cols[2].metric("Experience fit", f"{bd.get('experience_fit',0):.2f}")
                    feat_cols[3].metric("Product co.", f"{bd.get('product_company_score',0):.2f}")
                    feat_cols2 = st.columns(4)
                    feat_cols2[0].metric("Shipped evidence", f"{bd.get('shipped_system_evidence',0):.2f}")
                    feat_cols2[1].metric("Location fit", f"{bd.get('location_fit',0):.2f}")
                    feat_cols2[2].metric("Behavioral mod.", f"{bd.get('behavioral_modifier',0):.2f}")
                    feat_cols2[3].metric("Notice (days)", bd.get('notice_period_days', '—'))

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    csv_lines = ["candidate_id,rank,score,reasoning"]
    for r in results:
        esc = r["reasoning"].replace('"', '""')
        csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.4f},"{esc}"')

    st.download_button(
        "⬇️ Download ranked list as CSV",
        data="\n".join(csv_lines),
        file_name="matchmind_results.csv",
        mime="text/csv",
    )

    # ── Feedback stats ────────────────────────────────────────────────────────
    fb_stats = orch.feedback_stats()
    if fb_stats["total_events"] > 0:
        st.caption(
            f"📊 Feedback Agent: {fb_stats['total_events']} events recorded — "
            f"{fb_stats['distribution']} | "
            f"{fb_stats['training_labels_available']} training labels available"
        )

else:
    st.info("Paste a JD above and click **Run MatchMind** to start the agent pipeline.")

    st.markdown("""
### How it works

```
Your JD
  └─▶ [JD Parser Agent]     extracts skills, experience, location from raw text
        └─▶ [Fraud Agent]    screens 50 candidates for impossible profiles  
              └─▶ [Ranking Agent]   scores all candidates (8 feature dimensions + semantic)
                    └─▶ [Explanation Agent]  generates grounded 1-sentence reasoning
                          └─▶ Results shown below
                                └─▶ Your ✅/❌ clicks → [Feedback Agent] → model improves
```

All 5 agents coordinate through the **Orchestrator**. No black-box decisions — 
every score is a transparent weighted formula you can inspect in the breakdown panel.
""")
