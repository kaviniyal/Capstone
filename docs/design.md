# Design Document — AI-Powered Insurance Claims Intelligence Assistant

## 1. Vector Database Selection

**Choice: ChromaDB**

| Option | Pros | Cons |
|--------|------|------|
| **ChromaDB** | Zero-ops local deployment; native Python; cosine/L2/IP similarity; persistent storage; metadata filtering | Single-node; not horizontally scalable |
| Pinecone | Managed, scalable, production-grade | Requires cloud account; cost; network latency |
| Weaviate | Multi-modal; GraphQL; horizontal scaling | Heavier infra; Docker/Kubernetes dependency |
| FAISS | Fastest in-memory ANN; battle-tested | No metadata filtering; no persistence layer; no REST API |

**Rationale:** For a capstone demonstrating RAG concepts, ChromaDB provides the ideal balance — it runs entirely on local disk (`./data/chroma_db`), requires zero external infrastructure, supports metadata filtering for claim type/region/fraud label queries, and exposes a simple Python API that integrates cleanly with LangChain. The dataset (15k rows) fits well within ChromaDB's single-node capacity. A production deployment would migrate to a managed vector database like Pinecone or Weaviate behind a proper service layer.

---

## 2. Chunking Strategy for Claim Documents

**Choice: Row-per-document (no chunking)**

Each CSV row becomes exactly one document in ChromaDB. The document is a structured natural-language representation:

```
Claim ID: CLM-001234 | Policy: Comprehensive | Accident: Multi-vehicle Collision
Amount: $18,500 | Region: Northeast | Fraud Label: Y | Status: Settled
Incident Date: 2019-03-14 | Customer History: 2 prior claims
```

**Why not chunk?**

| Approach | Reason Rejected |
|----------|----------------|
| Fixed-size token chunks | Insurance claim records are already atomic — splitting a single row across chunks breaks the claim's coherent context |
| Sliding window | Creates duplicate partial records; confuses BM25 with repeated tokens |
| Sentence splitting | Claim records are semi-structured, not prose; sentence boundaries are arbitrary |

**Trade-off:** Because each record is short (~150–300 tokens), we lose the benefit of fine-grained retrieval. However, for structured tabular data this is the correct approach — we want to retrieve complete claim records, not fragments. If the system were extended to ingest unstructured policy documents or investigator notes, hierarchical chunking (document → section → paragraph) would be appropriate.

---

## 3. Hybrid Search vs Semantic-Only Retrieval

**Choice: Hybrid (BM25 + semantic) with Reciprocal Rank Fusion**

### Why not semantic-only?

Semantic embeddings capture *conceptual* similarity but can miss exact identifier matches (claim IDs, policy numbers, specific dollar amounts). A query like `"policy MX-4421 total loss"` needs BM25 to surface the exact policy match, not a semantically similar but different policy.

### Why not keyword-only?

BM25 fails on paraphrasing and domain synonyms. `"staged collision"` won't retrieve records that say `"fraudulent multi-vehicle accident"` without semantic coverage.

### Fusion strategy: Reciprocal Rank Fusion (RRF)

```python
rrf_score = sum(1 / (60 + rank_i) for rank_i in rankings)
```

RRF was chosen over weighted score fusion because:
- BM25 and cosine similarity scores are on incomparable scales
- RRF is rank-based — robust to score magnitude differences
- The `k=60` constant prevents top-ranked documents from dominating excessively
- No tuning required beyond the constant

### Cross-encoder reranking

After fusion, the top `fetch_k × 3` candidates are reranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`). Cross-encoders jointly encode the query and document, producing more accurate relevance scores than bi-encoder embeddings at the cost of higher latency. We rerank only the fused top-N to keep latency acceptable.

---

## 4. Agent Orchestration Architecture

**Choice: LangGraph StateGraph with TypedDict state**

### Alternative considered: LangChain AgentExecutor with tool calling

| Aspect | LangGraph | AgentExecutor |
|--------|-----------|---------------|
| Control flow | Deterministic (developer-defined graph) | Emergent (LLM decides tool order) |
| HITL | First-class (`interrupt_before`) | Requires custom wrapping |
| Auditability | Full state history via MemorySaver | Limited |
| Parallelism | Explicit via `Send` API | Not supported |
| Debugging | State snapshots per node | Hard to inspect mid-run |

LangGraph was chosen because:
1. The investigation pipeline has a known, fixed structure — non-deterministic agent looping is unnecessary overhead.
2. HITL is a first-class feature via `interrupt_before` + `MemorySaver` checkpointing.
3. Every node's input/output is captured in `ClaimState`, providing a full audit trail.
4. The graph is easy to extend (e.g. adding a new `document_retrieval` node) without changing other nodes.

### Agent specialisation rationale

| Agent | Why separate? |
|-------|---------------|
| Fraud Retrieval | Isolates CRAG logic (confidence check + query refinement) from scoring |
| Correlation | Cross-claim patterns require a different prompt and statistical pre-processing than per-claim scoring |
| Risk Scoring | Fraud probability estimation is a distinct ML-style judgment call |
| Policy Validation | Compliance rules are orthogonal to fraud probability |
| Recommendation | Synthesis requires all upstream outputs simultaneously — placing it last is the correct dependency order |

---

## 5. Fraud Correlation Strategy

**Two-phase approach: statistical pre-filter → LLM analysis**

**Phase 1 — Rule-based statistical checks (fast, cheap)**

| Signal | Detection Method |
|--------|-----------------|
| Region hotspot | Counter on `customer_region` metadata; flag if any region ≥ 2 occurrences |
| Repeat customer | Counter on policy number; flag duplicates |
| Amount clustering | Coefficient of Variation < 0.12 across claim amounts signals suspiciously tight grouping |
| Fraud density | Flag if > 60% of retrieved claims carry `fraud_label = Y` |

These run in pure Python before any LLM call, providing grounding signal to the LLM prompt at zero token cost.

**Phase 2 — LLM semantic analysis**

The statistical signals are injected into the prompt context. The LLM then identifies higher-order patterns (PATTERN_MATCH, TEMPORAL_BURST) that require semantic reasoning across the claim narratives.

**Why this ordering?** Statistical checks are deterministic and reproducible — they anchor the LLM's analysis and reduce hallucination risk. The LLM adds value for patterns that require language understanding, not for counting.

---

## 6. Operational Reliability Guardrails

### Input guardrails (`guardrails/guards.py`)

| Guard | Mechanism | Failure mode handled |
|-------|-----------|---------------------|
| Length validation | Truncate queries > 2000 chars | Token overflow in LLM context |
| Prompt injection | Regex pattern matching (`ignore previous`, `act as`, etc.) | Adversarial query hijacking |
| PII redaction | Microsoft Presidio (primary) + regex fallback | GDPR/data compliance; leaking customer names/SSNs into LLM logs |

**PII redaction strategy:** Presidio provides entity-level detection (PERSON, EMAIL, PHONE, SSN, CREDIT_CARD) with confidence scoring. The regex fallback handles cases where Presidio is not installed, covering common patterns. Redacted entities are replaced with typed placeholders (`[EMAIL]`, `[SSN]`) rather than deleted, preserving document structure for the LLM.

### HITL uncertainty band

```
0.0 ─────── 0.4 ──────── 0.6 ──────── 1.0
   AUTO-APPROVE    HUMAN   AUTO-ESCALATE
```

The band `[0.4, 0.6]` represents genuine model uncertainty. Claims in this range are paused and routed to a human investigator rather than auto-decided. The thresholds are configurable via environment variables (`HITL_LOW_THRESHOLD`, `HITL_HIGH_THRESHOLD`).

### CRAG (Corrective RAG)

If the average cross-encoder confidence score of the top-k retrieved claims falls below 0.30, the system:
1. Asks the LLM to reformulate the query with more insurance-specific terminology.
2. Retries the hybrid retrieval with the refined query.
3. Records `crag_triggered: True` in the response for monitoring.

This prevents the recommendation agent from reasoning over low-quality context.

### LangSmith tracing

All LLM calls are automatically traced when `LANGCHAIN_TRACING_V2=true`. Traces include:
- Full prompt and completion text
- Token usage and latency per node
- Run IDs for cross-referencing with the HITL audit trail

### Evaluation framework

Quality is measured across three complementary frameworks to reduce evaluation blind spots:

| Framework | Metrics | Validates |
|-----------|---------|-----------|
| DeepEval | Faithfulness, AnswerRelevancy, ContextualPrecision | Hallucination, relevance |
| Ragas | Faithfulness, AnswerRelevancy, ContextRecall | Retrieval coverage |
| LLM-as-Judge | 1–5 scores on faithfulness, helpfulness, accuracy | Human-aligned quality |

Thresholds: Faithfulness ≥ 0.7, AnswerRelevancy ≥ 0.7, ContextualPrecision ≥ 0.6.

---

## 7. A2A Communication Design

**Problem:** Upstream agent findings (high fraud probability, policy violations) must influence the downstream recommendation agent without hardcoding cross-agent logic.

**Solution:** Typed message passing via `ClaimState.a2a_messages`

Messages are plain Python dataclasses serialised to dicts for LangGraph MemorySaver compatibility:

```python
@dataclass
class A2AMessage:
    sender: str
    receiver: str
    message_type: str   # ESCALATE | FLAG | APPROVE | INFO
    subject: str
    payload: dict
    timestamp: str
```

**Why not direct function calls?**
Direct calls would create tight coupling — every agent would need to know about every other agent's interface. Message passing decouples producers from consumers; a new agent can emit messages without modifying the recommendation agent.

**Why not a separate message broker (Redis, RabbitMQ)?**
Within a single synchronous LangGraph run, a broker would add network latency and operational complexity without benefit. The in-process list is sufficient and survives MemorySaver checkpointing for HITL resumption.

---

## 8. Deployment Architecture

```
┌─────────────────────────────────────┐
│           Docker Container          │
│                                     │
│  FastAPI (uvicorn, port 8000)       │
│    ├── /ingest                      │
│    ├── /query                       │
│    ├── /analyze  + /analyze/resume  │
│    └── /correlate                   │
│                                     │
│  ChromaDB (./data/chroma_db)        │
│  └── Mounted as Docker volume       │
└─────────────────────────────────────┘
        │
        │ OpenAI-compatible API
        ▼
   LLM Gateway (Arsh Nivlabs / OpenAI)
```

Single-container deployment simplifies the demo. A production deployment would separate the vector store into a dedicated ChromaDB server (or replace with Pinecone) and run multiple FastAPI workers behind a load balancer.
