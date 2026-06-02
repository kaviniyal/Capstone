"""
Policy Validation Agent — checks whether the claim meets policy eligibility rules
and flags any compliance violations that may indicate fraud.
"""

import os
import sys
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_factory import get_llm


class PolicyValidationResult(BaseModel):
    is_policy_valid: bool = Field(description="Whether the claim appears to comply with policy terms")
    violations: list[str] = Field(description="List of identified policy violations or anomalies")
    eligibility_flags: list[str] = Field(description="Eligibility concerns that need further investigation")
    validation_summary: str = Field(description="One-paragraph plain-English summary of the validation result")


_parser = PydanticOutputParser(pydantic_object=PolicyValidationResult)

POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an insurance policy compliance specialist. Analyse the claim details "
     "and identify any policy violations, eligibility issues, or inconsistencies "
     "that are common fraud indicators.\n\n"
     "Common fraud indicators to check:\n"
     "- Claim submitted very shortly after policy inception\n"
     "- Claim amount significantly exceeds vehicle/property value\n"
     "- Multiple claims from same customer in a short window\n"
     "- Inconsistency between reported incident type and damage description\n"
     "- Missing police report for major incidents\n"
     "- Claim filed in a high-fraud region\n\n"
     "{format_instructions}"),
    ("human",
     "Claim details: {query}\n\n"
     "Supporting historical context:\n{context}"),
])


def run_policy_validation_agent(query: str, retrieved_claims: list[dict]) -> dict:
    """
    Validate policy compliance for the submitted claim.
    Returns a PolicyValidationResult dict.
    """
    llm = get_llm(temperature=0)

    context = "\n".join(
        f"- {c['document']}"
        for c in retrieved_claims[:3]
    )

    chain = POLICY_PROMPT | llm | _parser
    result: PolicyValidationResult = chain.invoke({
        "query":               query,
        "context":             context,
        "format_instructions": _parser.get_format_instructions(),
    })
    return result.model_dump()
