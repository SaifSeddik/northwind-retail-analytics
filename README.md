
# Northwind Retail Analytics Copilot

Welcome! This project is a local, free AI agent that answers retail analytics questions by combining:

- Retrieval-Augmented Generation (RAG) over local documentation (`docs/`)
- SQL querying over a local Northwind SQLite database

The agent produces typed, auditable answers with citations, and is fully self-contained—no paid APIs or external calls at inference time. It uses DSPy and LangGraph for modular, explainable reasoning and includes a demo of DSPy-based optimization.

## Features

- **Hybrid Reasoning:** Combines document retrieval and SQL execution for robust analytics.
- **DSPy Optimization:** The Router module is optimized using a tiny, local classifier to improve routing accuracy (see `agent/dspy_signatures.py`).
- **Citations:** Every answer includes references to the source docs or database queries used.
- **Repair Loop:** Automatically retries and repairs failed SQL queries for robustness.
- **No Cloud Required:** 100% local, deterministic, and reproducible.

## Project Structure

- `agent/graph_hybrid.py` — Orchestrates the agent, including the repair loop
- `agent/dspy_signatures.py` — Router logic and DSPy optimization demo
- `agent/rag/retrieval.py` — TF-IDF document retriever
- `agent/tools/sqlite_tool.py` — SQLite DB access and schema introspection
- `docs/` — Local documentation corpus (marketing calendar, KPIs, catalog, product policy)
- `data/northwind.sqlite` — Northwind sample database (ensure this exists)
- `sample_questions_hybrid_eval.jsonl` — Example evaluation questions
- `run_agent_hybrid.py` — CLI entrypoint for batch question answering
- `requirements.txt` — Python dependencies

## Setup & Usage

1. **Install dependencies** (recommended: use a virtual environment):

   ```sh
   pip install -r requirements.txt
   ```

2. **Ensure the Northwind database is present:**

   Place `northwind.db` or `northwind.sqlite` in the `data/` directory. (Already included in this repo.)

3. **Run the agent on the sample batch:**

   ```sh
   python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
   ```

4. **Check the output:**

   Results will be saved in `outputs_hybrid.jsonl` with answers, types, and citations.

## Notes & Assumptions

- **CostOfGoods** is approximated as `0.7 * UnitPrice` if not explicitly available (see queries for details).
- **NL→SQL** is rule-based and tailored to the provided evaluation questions (not a general NL2SQL model).
- **Repair loop**: If a query fails, the agent will retry with alternative table/view names before giving up.
- **DSPy Optimization**: The router’s accuracy before/after optimization can be seen by running `demo_optimizer()` in `agent/dspy_signatures.py`.
