import os
import sys
import uuid
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from api.schemas import AnalyzeRequest, AnalyzeResponse, ResumeRequest
from pipeline.graph import run_analysis, resume_analysis

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("", response_model=AnalyzeResponse)
def analyze_claim(req: AnalyzeRequest):
    """
    Run the full multi-agent fraud analysis pipeline on a claim query.

    If the risk score falls in the HITL uncertainty band (configurable thresholds),
    the response will have `awaiting_human=True` and a `thread_id`.
    Call POST /analyze/resume with that thread_id and a human decision to continue.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        state = run_analysis(query=req.query, filters=req.filters, thread_id=thread_id)
        return AnalyzeResponse(
            thread_id=thread_id,
            awaiting_human=state.get("awaiting_human", False),
            guardrail_flags=state.get("guardrail_flags", []),
            crag_triggered=state.get("crag_triggered", False),
            risk_assessment=state.get("risk_assessment", {}),
            policy_validation=state.get("policy_validation", {}),
            recommendation=state.get("recommendation", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume", response_model=AnalyzeResponse)
def resume_hitl(req: ResumeRequest):
    """
    Resume a paused HITL analysis with the human investigator's decision.
    human_decision must be one of: approve | escalate | reject
    """
    if req.human_decision not in ("approve", "escalate", "reject"):
        raise HTTPException(status_code=422, detail="human_decision must be approve | escalate | reject")
    try:
        state = resume_analysis(thread_id=req.thread_id, human_decision=req.human_decision)
        return AnalyzeResponse(
            thread_id=req.thread_id,
            awaiting_human=False,
            guardrail_flags=state.get("guardrail_flags", []),
            crag_triggered=state.get("crag_triggered", False),
            risk_assessment=state.get("risk_assessment", {}),
            policy_validation=state.get("policy_validation", {}),
            recommendation=state.get("recommendation", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
