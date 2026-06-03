"""
Recommendation Agent — synthesises all upstream agent outputs into a final
explainable investigation recommendation report.
"""

import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
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

     "HUMAN DECISION RULE (HIGHEST PRIORITY — overrides everything else):\n"
     "  If human_decision = 'approve'  → decision MUST be APPROVE. Do not escalate.\n"
     "  If human_decision = 'escalate' → decision MUST be ESCALATE.\n"
     "  If human_decision = 'reject'   → decision MUST be REJECT.\n"
     "  Human decision overrides AI risk score, A2A messages, and policy violations.\n\n"

     "AI Decision rules (only apply when human_decision is empty):\n"
     "  APPROVE   — fraud_probability < 0.3 and no policy violations\n"
     "  INVESTIGATE — fraud_probability 0.3–0.6 or minor violations\n"
     "  ESCALATE  — fraud_probability > 0.6 or critical violations or ESCALATE A2A received\n"
     "  REJECT    — clear policy ineligibility\n\n"

     "Priority rules:\n"
     "  P1 — CRITICAL risk or claim_amount > 50000\n"
     "  P2 — HIGH risk\n"
     "  P3 — MEDIUM risk\n"
     "  P4 — LOW risk\n\n"
     "{format_instructions}"),
    ("human",
     "Human Decision (HIGHEST PRIORITY): {human_decision}\n\n"
     "Original claim query: {query}\n\n"
     "Risk Assessment:\n{risk_assessment}\n\n"
     "Policy Validation:\n{policy_validation}\n\n"
     "Correlation Analysis:\n{correlation_signals}\n\n"
     "Agent-to-Agent Messages Received:\n{a2a_messages}\n\n"
     "Top Similar Historical Claims:\n{similar_claims}"),
])


def run_recommendation_agent(
    query: str,
    risk_assessment: dict,
    policy_validation: dict,
    retrieved_claims: list[dict],
    correlation_signals: dict | None = None,
    a2a_messages: list[dict] | None = None,
    human_decision: str = "",
) -> dict:
    """
    Generate the final investigation recommendation.
    Factors in correlation signals and any A2A escalation messages from upstream agents.
    Returns an InvestigationRecommendation dict.
    """
    llm = get_llm_temp(temperature=0.2)

    similar_text = "\n".join(
        f"- [{c['metadata'].get('fraud_label','?')}] {c['document'][:200]}..."
        for c in retrieved_claims[:3]
    )

    corr_text = "No correlation analysis available."
    if correlation_signals:
        corr_text = (
            f"Overall risk: {correlation_signals.get('overall_correlation_risk', 'N/A')}\n"
            f"Summary: {correlation_signals.get('summary', '')}\n"
            f"Flags: {', '.join(correlation_signals.get('investigation_flags', []))}"
        )

    a2a_text = "None."
    if a2a_messages:
        a2a_text = "\n".join(
            f"[{m.get('message_type','?')}] from {m.get('sender','?')}: "
            f"{m.get('subject','')} — {m.get('payload',{})}"
            for m in a2a_messages
        )

    human_dec_text = (
        f"'{human_decision}' — THIS OVERRIDES ALL AI DECISIONS. "
        f"Decision MUST be {human_decision.upper()}."
        if human_decision else "None — use AI decision rules."
    )

    chain = RECOMMEND_PROMPT | llm | _parser
    rec: InvestigationRecommendation = chain.invoke({
        "query":               query,
        "human_decision":      human_dec_text,
        "risk_assessment":     str(risk_assessment),
        "policy_validation":   str(policy_validation),
        "correlation_signals": corr_text,
        "a2a_messages":        a2a_text,
        "similar_claims":      similar_text,
        "format_instructions": _parser.get_format_instructions(),
    })
    return rec.model_dump()
