"""
MatchMind — multi-agent autonomous recruiting intelligence system.

agents/
  orchestrator.py     — coordinates all agents for a single recruiter query
  jd_parser.py        — extracts structured requirements from raw JD text
  ranking_agent.py    — scores and ranks the full candidate pool
  explanation_agent.py — generates grounded per-candidate reasoning
  fraud_agent.py      — detects structurally-impossible (honeypot) profiles
  feedback_agent.py   — ingests recruiter actions to improve rankings

Each agent exposes a single public method and can be tested independently.
The Orchestrator is the only component that knows about all agents.
"""
