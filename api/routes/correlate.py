"""
POST /correlate — standalone cross-claim fraud correlation analysis.

Retrieves similar historical claims for the given query and runs the
correlation agent to surface cross-claim patterns (staging rings, region
hotspots, amount clusters, repeat customers, temporal bursts).
"""

from __future__ import annotations
import os
import sys

from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from api.schemas import CorrelationRequest, CorrelationResponse
from agents.fraud_retrieval_agent import run_fraud_retrieval_agent
from agents.correlation_agent import run_correlation_agent

router = APIRouter(prefix="/correlate", tags=["Correlation"])


@router.post("", response_model=CorrelationResponse)
def correlate(req: CorrelationRequest):
    """
    Retrieve similar historical claims and run cross-claim correlation analysis.

    Returns detected fraud patterns such as repeat customers, geographic
    hotspots, amount clustering, and staging ring signals across the
    retrieved claim set.
    """
    try:
        retrieval = run_fraud_retrieval_agent(
            query=req.query,
            top_k=req.top_k,
            filters=req.filters,
        )
        claims = retrieval["claims"]

        correlation = run_correlation_agent(
            query=req.query,
            retrieved_claims=claims,
        )

        return CorrelationResponse(
            query_used=retrieval["query_used"],
            crag_triggered=retrieval["crag_triggered"],
            claims_analysed=len(claims),
            **correlation,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
