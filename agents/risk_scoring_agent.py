"""
Risk Scoring Agent — estimates fraud probability and assigns investigation priority.
Uses retrieved similar claims + the original query as context for the LLM scorer.
"""

import os
import sys
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
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
     "You are an expert insurance fraud analyst. Assess the fraud risk of the "
     "submitted claim query.\n\n"

     "STEP 1 — Read the query for LEGITIMACY SIGNALS (these LOWER fraud probability):\n"
     "  • 'police report filed' or 'police report' alone        → strong legitimacy signal, -0.3\n"
     "  • 'third party fault' or 'third party at fault'         → strong legitimacy signal, -0.25\n"
     "  • 'witness present' or 'witness'                        → legitimacy signal, -0.15\n"
     "  • 'no previous claims' or 'none past claims'            → legitimacy signal, -0.1\n\n"

     "STEP 2 — Read the query for FRAUD SIGNALS (these RAISE fraud probability):\n"
     "  • 'no police report' or 'police report not filed'       → strong fraud signal, +0.35\n"
     "  • 'missing accident date' or 'days to accident: none'   → strongest fraud signal, +0.4\n"
     "  • 'no witness' or 'witness not present'                 → fraud signal, +0.2\n"
     "  • 'more than 4 past claims' or 'excessive past claims'  → fraud signal, +0.25\n"
     "  • 'policy holder at fault'                              → mild fraud signal, +0.1\n\n"

     "STEP 3 — Use retrieved similar claims as SUPPORTING CONTEXT only.\n"
     "  The retrieved claims inform patterns but DO NOT override what the query explicitly states.\n"
     "  If the query has strong legitimacy signals, score LOW even if retrieved claims are fraudulent.\n\n"

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
     "Similar historical claims (context only — query signals take priority):\n{similar_claims}"),
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
