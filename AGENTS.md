# Shared Repo Memory

Use this file as the committed team memory for AI assistants working in this
repository. Keep it current when architecture, conventions, or product behavior
changes.

## Project

This repo implements an AI-powered insurance claims intelligence assistant for
fraud detection, claim retrieval, policy validation, and investigation
recommendations.

Primary stack:
- FastAPI for the API layer.
- LangChain and LangGraph for LLM and multi-agent orchestration.
- ChromaDB for persisted vector retrieval.
- BM25 plus semantic retrieval plus cross-encoder reranking.
- DeepEval, Ragas, and LangSmith for evaluation and monitoring.

## Runtime Knowledge

The chatbot's claim-specific context comes from the dataset in
`data/fraud_oracle.csv` after it is ingested into ChromaDB.

Local default:
- `CHROMA_PERSIST_DIR=./data/chroma_db`

Team recommendation:
- Do not commit generated ChromaDB files.
- Everyone should ingest the same dataset version locally, or point
  `CHROMA_PERSIST_DIR` at a shared ChromaDB-compatible volume/service.
- If the dataset changes, document the dataset version and rerun ingestion.

## Main Workflows

Setup:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Ingest:
```bash
python ingestion/ingest.py
```

Run API:
```bash
uvicorn api.main:app --reload --port 8000
```

Evaluate:
```bash
python evaluation/evaluate.py
```

## Code Conventions

- Keep API routes in `api/routes/`.
- Keep orchestration logic in `pipeline/graph.py`.
- Keep retrieval behavior in `retrieval/retriever.py`.
- Keep dataset loading and embedding in `ingestion/ingest.py`.
- Keep agent-specific reasoning in `agents/`.
- Use `config.py` and environment variables for settings.
- Do not commit `.env`, API keys, generated vector databases, Python caches, or
  local virtual environments.

## Expected Chatbot Behavior

The assistant should:
- Ground fraud analysis in retrieved claims when available.
- Explain risk factors clearly.
- Preserve the HITL flow for borderline fraud scores.
- Redact or avoid exposing sensitive personal information.
- Prefer concise investigation recommendations with concrete next steps.

