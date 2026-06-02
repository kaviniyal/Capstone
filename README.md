# AI-Powered Insurance Claims Intelligence Assistant

FDE Training — Final Capstone Project | Prodapt Chennai

---

## Overview

An AI-powered microservice that helps insurance operations teams detect fraud, retrieve historical claims using natural language, and generate explainable investigation recommendations.

**Tech Stack:** LangChain · LangGraph · ChromaDB · FastAPI · DeepEval · Ragas · LangSmith · Docker

---

## Project Structure

```
capstone_project/
├── data/                         # Place fraud_oracle.csv here
├── ingestion/ingest.py           # CSV → embed → ChromaDB
├── retrieval/retriever.py        # Hybrid BM25 + semantic + reranker
├── agents/
│   ├── fraud_retrieval_agent.py  # CRAG-powered claim retrieval
│   ├── risk_scoring_agent.py     # Fraud probability scoring
│   ├── policy_validation_agent.py
│   └── recommendation_agent.py  # Final investigation report
├── pipeline/graph.py             # LangGraph orchestration + HITL
├── guardrails/guards.py          # Input validation + PII redaction
├── api/
│   ├── main.py                   # FastAPI app entry point
│   ├── schemas.py
│   └── routes/                   # ingest · query · analyze
├── evaluation/evaluate.py        # DeepEval + Ragas + LLM-as-Judge
├── monitoring/tracer.py          # LangSmith setup
├── config.py                     # Settings (pydantic-settings)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Setup

### 1. Get the dataset

Download `fraud_oracle.csv` from Kaggle — [Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) and place it in `capstone_project/data/fraud_oracle.csv`.

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (and optionally LANGCHAIN_API_KEY)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 4. Ingest claims data

```bash
python ingestion/ingest.py
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Docker

```bash
# Build and run
docker-compose up --build

# API will be available at http://localhost:8000
```

---

## API Endpoints

### POST `/ingest`
Load and embed claims data into ChromaDB.
```json
{ "reset": false }
```

### POST `/query`
Hybrid semantic + BM25 search over claims.
```json
{
  "query": "vehicle theft claim filed two days after policy start",
  "top_k": 5,
  "filters": { "fraud_label": "Y" }
}
```

### GET `/query/claim/{claim_id}`
Retrieve a specific claim by ID.

### POST `/analyze`
Run the full multi-agent fraud investigation pipeline.
```json
{
  "query": "Customer claiming $80,000 for minor collision, no police report filed"
}
```

**Response includes:**
- `risk_assessment` — fraud probability, risk level, key risk factors
- `policy_validation` — violations, eligibility flags
- `recommendation` — decision (APPROVE/INVESTIGATE/ESCALATE/REJECT), priority, investigation steps
- `awaiting_human` — `true` if HITL is triggered (score in uncertainty band)
- `thread_id` — use this to resume after human review

### POST `/analyze/resume`
Resume a paused HITL analysis.
```json
{
  "thread_id": "abc-123",
  "human_decision": "escalate"
}
```

---

## Sample Fraud Investigation Query

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Policy holder filed a total loss claim for vehicle fire within 3 days of policy start. Claim amount: $72,000. No police report. Customer has 3 prior claims in 2 years."
  }'
```

**Example response:**
```json
{
  "thread_id": "f3a1c2...",
  "awaiting_human": false,
  "risk_assessment": {
    "fraud_probability": 0.87,
    "risk_level": "CRITICAL",
    "key_risk_factors": [
      "Claim filed 3 days after policy inception",
      "No police report for major incident",
      "3 prior claims in 2 years",
      "Claim amount significantly high"
    ]
  },
  "recommendation": {
    "decision": "ESCALATE",
    "priority": "P1",
    "investigation_steps": [
      "Request police/fire department report",
      "Verify vehicle ownership and title history",
      "Review prior claim history for patterns",
      "Conduct recorded statement with claimant"
    ]
  }
}
```

---

## Evaluation

```bash
python evaluation/evaluate.py
```

Runs DeepEval (faithfulness, answer relevancy, contextual precision) and Ragas (faithfulness, answer relevancy, context recall) on sample test cases.

---

## Key Features

| Feature | Description |
|---------|-------------|
| Hybrid Retrieval | BM25 + semantic vector search + cross-encoder reranker |
| Corrective RAG | Auto-refines query when retrieval confidence is low |
| HITL | Pauses pipeline for human review on borderline cases (0.4–0.6 score band) |
| Multi-Agent | 4 specialised agents orchestrated via LangGraph with checkpointing |
| Guardrails | Input length validation, prompt injection detection, PII redaction |
| Evaluation | DeepEval + Ragas + LLM-as-Judge |
| Monitoring | Full LangSmith tracing on all LLM and agent calls |
| Docker | Single `docker-compose up` deployment |
