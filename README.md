# Retail Analytics Copilot (local hybrid)

Short README: this repository contains a compact local implementation of the assignment "Retail Analytics Copilot (DSPy + LangGraph)". It is intentionally small and deterministic so you can run it locally against the provided Northwind SQLite DB.

Design (2–4 bullets):
- Graph nodes: Router (DSPy-like), Retriever (TF-IDF), Planner (regex heuristics), NL→SQL (rule templates), Executor (SQLite), Synthesizer (format + citations), Repair loop + Checkpointer.
- Retriever stores doc chunk ids as `filename::chunk0` (e.g., `marketing_calendar::chunk0`).
- DSPy optimization demo: Router trained (tiny) classifier improved routing accuracy on a handcrafted eval (see `agent/dspy_signatures.py`). Metric: acc_before -> acc_after.

Which DSPy module was optimized: Router. Demo metric (tiny, local): see `agent/dspy_signatures.py` demo_optimizer() which prints a small JSON with before/after accuracy. Example delta will be placed in this README after running.

Assumptions / trade-offs:
- CostOfGoods approximation: 0.7 * UnitPrice when no explicit cost exists (documented in queries).
- NL→SQL is rule-based and handles the eval questions; not a general NL2SQL model.
- Repair loop tries up to 2 fixes (switching to lowercase view names) before failing.

How to run (example):
1. Ensure `data/northwind.sqlite` or `data/northwind.db` exists (downloaded already).
2. Install dependencies (use a virtualenv):

   pip install -r requirements.txt

3. Run the agent on the sample batch:

   python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl

Files of note:
- `agent/graph_hybrid.py` — orchestration and repair loop
- `agent/dspy_signatures.py` — Router + tiny optimizer demo
- `agent/rag/retrieval.py` — TF-IDF retriever
- `agent/tools/sqlite_tool.py` — DB access and simple introspection
