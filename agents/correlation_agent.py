"""
Fraud Anomaly Correlation Agent — detects cross-claim patterns in the retrieved set.

Identifies staging rings, repeat-customer fraud, geographic hotspots, amount
clustering, and temporal bursts that individual claim scoring alone may miss.
Each signal is graded LOW / MEDIUM / HIGH and aggregated into an overall
correlation risk rating passed downstream to the recommendation agent.
"""

from __future__ import annotations
import os
import sys
from collections import Counter

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_factory import get_llm_temp


# ── Output schema ─────────────────────────────────────────────────────────────

class CorrelationSignal(BaseModel):
    signal_type: str = Field(
        description="REPEAT_CUSTOMER | REGION_HOTSPOT | AMOUNT_CLUSTER | TEMPORAL_BURST | PATTERN_MATCH"
    )
    description: str = Field(description="Human-readable explanation of the detected correlation")
    affected_claims: list[str] = Field(description="Claim indices or IDs involved in this signal")
    severity: str = Field(description="LOW | MEDIUM | HIGH")


class CorrelationResult(BaseModel):
    signals: list[CorrelationSignal] = Field(description="All detected correlation signals")
    overall_correlation_risk: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    summary: str = Field(description="One-paragraph narrative of correlation findings")
    investigation_flags: list[str] = Field(
        description="Concise action flags for the investigation team (e.g. 'Cross-check claims 2 & 4 for staging')"
    )


# ── LLM prompt ────────────────────────────────────────────────────────────────

_parser = PydanticOutputParser(pydantic_object=CorrelationResult)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an insurance fraud analyst specialising in cross-claim pattern detection. "
     "Analyse the set of retrieved historical claims and identify anomalous correlations "
     "that may indicate organised fraud, staging rings, or systematic policy abuse.\n\n"
     "Signal types to look for:\n"
     "  REPEAT_CUSTOMER  — same insured appearing multiple times with fraud labels\n"
     "  REGION_HOTSPOT   — geographic concentration of fraud in one region\n"
     "  AMOUNT_CLUSTER   — suspiciously similar claim amounts across unrelated customers\n"
     "  TEMPORAL_BURST   — multiple claims filed within a very short time window\n"
     "  PATTERN_MATCH    — identical accident type + fraud label suggesting a staging ring\n\n"
     "Statistical pre-analysis results (rule-based, for reference):\n{stat_signals}\n\n"
     "{format_instructions}"),
    ("human",
     "Current investigation query:\n{query}\n\n"
     "Retrieved claims to analyse for correlations:\n{claims_text}"),
])


# ── Statistical pre-filter ─────────────────────────────────────────────────────

def _statistical_signals(claims: list[dict]) -> dict:
    """
    Fast rule-based checks run before the LLM call.
    Results are injected into the prompt as grounding context.
    """
    signals: dict[str, str] = {}

    # Region concentration
    regions = [c.get("metadata", {}).get("customer_region", "") for c in claims]
    rc = Counter(r for r in regions if r)
    if rc:
        top_region, top_count = rc.most_common(1)[0]
        if top_count >= 2:
            signals["region_hotspot"] = f"{top_region!r} appears {top_count}× in this set"

    # Repeat customers
    customers = [
        c.get("metadata", {}).get("insured_or_policy_number", "")
        or c.get("metadata", {}).get("policy_number", "")
        for c in claims
    ]
    cc = Counter(c for c in customers if c)
    repeats = {k: v for k, v in cc.items() if v > 1}
    if repeats:
        signals["repeat_customers"] = str(list(repeats.keys()))

    # Amount clustering — low coefficient of variation signals tight grouping
    amounts: list[float] = []
    for c in claims:
        try:
            amt = float(c.get("metadata", {}).get("claim_amount", 0) or 0)
            if amt > 0:
                amounts.append(amt)
        except (ValueError, TypeError):
            pass
    if len(amounts) >= 3:
        mean = sum(amounts) / len(amounts)
        std = (sum((x - mean) ** 2 for x in amounts) / len(amounts)) ** 0.5
        cv = std / mean if mean else 0
        if cv < 0.12:
            signals["amount_cluster"] = f"CV={cv:.3f}, mean=${mean:,.0f} — unusually tight spread"

    # Fraud label density
    fraud_flags = [
        c.get("metadata", {}).get("fraud_label", "N") for c in claims
    ]
    fraud_count = sum(1 for f in fraud_flags if str(f).upper() in ("Y", "1", "TRUE", "YES"))
    if fraud_count >= len(claims) * 0.6 and len(claims) >= 3:
        signals["high_fraud_density"] = f"{fraud_count}/{len(claims)} retrieved claims are labelled fraudulent"

    return signals


# ── Public entry point ────────────────────────────────────────────────────────

def run_correlation_agent(query: str, retrieved_claims: list[dict]) -> dict:
    """
    Detect cross-claim fraud correlations within the retrieved claim set.
    Returns a CorrelationResult dict including statistical pre-signals.
    """
    if not retrieved_claims:
        return {
            "signals": [],
            "overall_correlation_risk": "LOW",
            "summary": "No claims retrieved — correlation analysis skipped.",
            "investigation_flags": [],
            "statistical_pre_signals": {},
        }

    stat_signals = _statistical_signals(retrieved_claims)

    claims_text = "\n\n".join(
        f"[Claim {i + 1}] (score={c.get('score', 'N/A')}, "
        f"fraud_label={c.get('metadata', {}).get('fraud_label', '?')})\n{c['document']}"
        for i, c in enumerate(retrieved_claims[:10])
    )

    llm = get_llm_temp(temperature=0.1)
    chain = _PROMPT | llm | _parser
    result: CorrelationResult = chain.invoke({
        "query":               query,
        "claims_text":         claims_text,
        "stat_signals":        str(stat_signals) if stat_signals else "None detected by rule-based checks.",
        "format_instructions": _parser.get_format_instructions(),
    })

    output = result.model_dump()
    output["statistical_pre_signals"] = stat_signals
    return output
