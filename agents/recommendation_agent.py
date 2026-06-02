"""
Recommendation Agent — synthesises all upstream agent outputs into a final
explainable investigation recommendation report.
"""

import os
import sys
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_factory import get_llm_temp


class InvestigationRecommendation(BaseModel):
    decision: str = Field(description="APPROVE | INVESTIGATE | ESCALATE | REJECT")
    priority: str = Field(description="P1 (urgent) | P2 (high) | P3 (normal) | P4 (low)")
    investigation_steps: list[str] = Field(description="Ordered list of recommended investigation actions")
    evidence_summary: str = Field(description="Summary of evidence supporting this recommendation")
    escalation_reason: str | None = Field(default=None, description="Reason for escalation, if applicable")
    estimated_fraud_savings: str = Field(description="Estimated potential claim amount at risk if fraudulent")


_parser = PydanticOutputParser(pydantic_object=InvestigationRecommendation)

RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior insurance fraud investigation supervisor. Based on the "
     "complete analysis below, produce a clear, actionable investigation recommendation.\n\n"
     "Decision rules:\n"
     "  APPROVE   — fraud_probability < 0.3 and no policy violations\n"
     "  INVESTIGATE — fraud_probability 0.3–0.6 or minor violations\n"
     "  ESCALATE  — fraud_probability > 0.6 or critical violations\n"
     "  REJECT    — clear policy ineligibility\n\n"
     "Priority rules:\n"
     "  P1 — CRITICAL risk or claim_amount > 50000\n"
     "  P2 — HIGH risk\n"
     "  P3 — MEDIUM risk\n"
     "  P4 — LOW risk\n\n"
     "{format_instructions}"),
    ("human",
     "Original claim query: {query}\n\n"
     "Risk Assessment:\n{risk_assessment}\n\n"
     "Policy Validation:\n{policy_validation}\n\n"
     "Top Similar Historical Claims:\n{similar_claims}"),
])


def run_recommendation_agent(
    query: str,
    risk_assessment: dict,
    policy_validation: dict,
    retrieved_claims: list[dict],
) -> dict:
    """
    Generate the final investigation recommendation.
    Returns an InvestigationRecommendation dict.
    """
    llm = get_llm_temp(temperature=0.2)

    similar_text = "\n".join(
        f"- [{c['metadata'].get('fraud_label','?')}] {c['document'][:200]}..."
        for c in retrieved_claims[:3]
    )

    chain = RECOMMEND_PROMPT | llm | _parser
    rec: InvestigationRecommendation = chain.invoke({
        "query":               query,
        "risk_assessment":     str(risk_assessment),
        "policy_validation":   str(policy_validation),
        "similar_claims":      similar_text,
        "format_instructions": _parser.get_format_instructions(),
    })
    return rec.model_dump()
