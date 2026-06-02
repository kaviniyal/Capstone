"""
Risk Scoring Agent — estimates fraud probability and assigns investigation priority.
Uses retrieved similar claims + the original query as context for the LLM scorer.
"""

import os
import sys
import json
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from llm_factory import get_llm_temp


class RiskScore(BaseModel):
    fraud_probability: float = Field(ge=0.0, le=1.0, description="Estimated probability of fraud (0–1)")
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in this assessment")
    key_risk_factors: list[str] = Field(description="Top reasons driving the fraud risk score")
    requires_human_review: bool = Field(description="True if score falls in the HITL uncertainty band")


_parser = PydanticOutputParser(pydantic_object=RiskScore)

SCORE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert insurance fraud analyst. Analyse the submitted claim query "
     "against the retrieved similar historical claims and produce a structured fraud "
     "risk assessment.\n\n"
     "Risk level mapping:\n"
     "  0.0–0.3  → LOW\n"
     "  0.3–0.5  → MEDIUM\n"
     "  0.5–0.75 → HIGH\n"
     "  0.75–1.0 → CRITICAL\n\n"
     "Mark requires_human_review=True when fraud_probability is between "
     "{low_threshold} and {high_threshold} (uncertainty band).\n\n"
     "{format_instructions}"),
    ("human",
     "Claim query: {query}\n\n"
     "Similar historical claims retrieved:\n{similar_claims}"),
])


def run_risk_scoring_agent(query: str, retrieved_claims: list[dict]) -> dict:
    """
    Score fraud risk for the query claim given its nearest historical neighbours.
    Returns a RiskScore dict plus the raw LLM output for auditing.
    """
    llm = get_llm_temp(temperature=0.1)

    claims_text = "\n\n".join(
        f"[{i+1}] (score={c['score']}) {c['document']}"
        for i, c in enumerate(retrieved_claims[:5])
    )

    chain = SCORE_PROMPT | llm | _parser
    risk: RiskScore = chain.invoke({
        "query":              query,
        "similar_claims":     claims_text,
        "low_threshold":      settings.hitl_low_threshold,
        "high_threshold":     settings.hitl_high_threshold,
        "format_instructions": _parser.get_format_instructions(),
    })

    result = risk.model_dump()
    result["requires_human_review"] = (
        settings.hitl_low_threshold <= risk.fraud_probability <= settings.hitl_high_threshold
    )
    return result
