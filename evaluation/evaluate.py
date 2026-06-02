"""
Evaluation suite using DeepEval and Ragas.
Tests the RAG pipeline's faithfulness, answer relevancy, and context recall.
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from llm_factory import get_llm_temp

# ── DeepEval ────────────────────────────────────────────────────────────────
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase
    _deepeval_available = True
except ImportError:
    _deepeval_available = False
    print("DeepEval not installed. Run: pip install deepeval")

# ── Ragas ────────────────────────────────────────────────────────────────────
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset
    _ragas_available = True
except ImportError:
    _ragas_available = False
    print("Ragas not installed. Run: pip install ragas datasets")


SAMPLE_EVAL_CASES = [
    {
        "input":    "Customer filed a claim for vehicle theft two days after policy inception",
        "output":   "This claim shows high fraud indicators: policy inception date is very close to incident date, suggesting possible fraud staging.",
        "expected": "Claims filed shortly after policy inception are a known fraud pattern requiring investigation.",
        "context":  [
            "Claim filed 2 days after policy start. Policy Type: Collision. Fraud Label: Y.",
            "Similar claim: Policy age 5 days, theft reported. Marked fraudulent.",
        ],
    },
    {
        "input":    "Multiple vehicle accidents reported in the same region within 30 days",
        "output":   "Cluster of 4 claims from the same zip code within a month. Possible fraud ring activity.",
        "expected": "Geographically clustered claims within a short period are a strong fraud indicator.",
        "context":  [
            "Claim 1: Region 45056, accident_type: Multi-vehicle. Fraud: Y.",
            "Claim 2: Region 45056, accident_type: Multi-vehicle, 12 days later. Fraud: Y.",
        ],
    },
    {
        "input":    "Claim amount of $85,000 for minor fender bender with no police report",
        "output":   "Claim amount significantly exceeds typical fender bender damage. No police report filed.",
        "expected": "Inflated claim amounts without supporting police reports are a major fraud indicator.",
        "context":  [
            "Total claim amount: 85000. Incident type: Minor collision. Police report: NO. Fraud Label: Y.",
        ],
    },
]


def run_deepeval(cases: Optional[list[dict]] = None) -> dict:
    """Run DeepEval faithfulness, relevancy, and context metrics."""
    if not _deepeval_available:
        return {"error": "deepeval not installed"}

    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    eval_cases = cases or SAMPLE_EVAL_CASES

    test_cases = [
        LLMTestCase(
            input=c["input"],
            actual_output=c["output"],
            expected_output=c["expected"],
            retrieval_context=c["context"],
        )
        for c in eval_cases
    ]

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=settings.llm_model),
        AnswerRelevancyMetric(threshold=0.7, model=settings.llm_model),
        ContextualPrecisionMetric(threshold=0.6, model=settings.llm_model),
    ]

    results = evaluate(test_cases=test_cases, metrics=metrics)
    summary = {
        "total_cases": len(test_cases),
        "passed":      sum(1 for tc in test_cases if all(m.is_successful() for m in metrics)),
        "metrics":     {type(m).__name__: m.score for m in metrics},
    }
    return summary


def run_ragas(cases: Optional[list[dict]] = None) -> dict:
    """Run Ragas evaluation on the RAG pipeline outputs."""
    if not _ragas_available:
        return {"error": "ragas not installed"}

    eval_cases = cases or SAMPLE_EVAL_CASES
    dataset = Dataset.from_list([
        {
            "question":  c["input"],
            "answer":    c["output"],
            "contexts":  c["context"],
            "ground_truth": c["expected"],
        }
        for c in eval_cases
    ])

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()


def run_llm_judge(query: str, answer: str, context: list[str]) -> dict:
    """
    LLM-as-Judge: asks the LLM to score the answer on faithfulness and helpfulness.
    Returns scores 1-5 for each dimension.
    """
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model=settings.llm_model, temperature=0, openai_api_key=settings.openai_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an impartial judge evaluating insurance fraud investigation responses. "
         "Score the answer on:\n"
         "1. Faithfulness (1-5): Is the answer fully supported by the context?\n"
         "2. Helpfulness (1-5): Does it give actionable guidance?\n"
         "3. Accuracy (1-5): Is the fraud assessment correct?\n"
         "Respond with JSON only: {\"faithfulness\": N, \"helpfulness\": N, \"accuracy\": N, \"reasoning\": \"...\"}"),
        ("human",
         "Query: {query}\nContext: {context}\nAnswer: {answer}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "query":   query,
        "context": "\n".join(context),
        "answer":  answer,
    })
    import json
    try:
        return json.loads(response.content)
    except Exception:
        return {"raw": response.content}


if __name__ == "__main__":
    print("=== DeepEval Results ===")
    print(run_deepeval())
    print("\n=== Ragas Results ===")
    print(run_ragas())
