from typing import Any, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    csv_path: Optional[str] = Field(default=None, description="Absolute path to fraud_oracle.csv. Uses default data/ path if omitted.")
    reset: bool = Field(default=False, description="Drop existing collection and re-ingest from scratch.")


class IngestResponse(BaseModel):
    status: str
    records_ingested: int
    message: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Natural language claim investigation query.")
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[dict] = Field(default=None, description="Metadata filters e.g. {\"fraud_label\": \"Y\"}")
    rerank: bool = Field(default=True)


class QueryResponse(BaseModel):
    query_used: str
    crag_triggered: bool
    results: list[dict]


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=5)
    filters: Optional[dict] = None
    thread_id: Optional[str] = Field(default=None, description="Provide to resume a paused HITL session.")


class AnalyzeResponse(BaseModel):
    thread_id: str
    awaiting_human: bool
    guardrail_flags: list[str]
    crag_triggered: bool
    risk_assessment: dict
    policy_validation: dict
    recommendation: dict


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID of the paused analysis.")
    human_decision: str = Field(..., description="One of: approve | escalate | reject")


class ClaimResponse(BaseModel):
    claim_id: str
    document: str
    metadata: dict


class CorrelationRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Natural language claim investigation query.")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of similar claims to retrieve for correlation.")
    filters: Optional[dict] = Field(default=None, description="Metadata filters e.g. {\"fraud_label\": \"Y\"}")


class CorrelationSignalSchema(BaseModel):
    signal_type: str
    description: str
    affected_claims: list[str]
    severity: str


class CorrelationResponse(BaseModel):
    query_used: str
    crag_triggered: bool
    claims_analysed: int
    signals: list[dict]
    overall_correlation_risk: str
    summary: str
    investigation_flags: list[str]
    statistical_pre_signals: dict
