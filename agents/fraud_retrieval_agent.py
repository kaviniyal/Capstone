"""
Fraud Retrieval Agent — finds historically similar fraudulent claims.
Implements Corrective RAG (CRAG): if retrieval confidence is low, it
refines the query and retries once before returning results.
"""

import os
import sys
from langchain.prompts import ChatPromptTemplate

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_factory import get_llm
from retrieval.retriever import get_retriever, RetrievedClaim

CONFIDENCE_THRESHOLD = 0.30  # cross-encoder score below this → refine & retry

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query optimizer for an insurance fraud investigation system. "
     "Given an original query that returned low-confidence results, rewrite it to be "
     "more specific and include relevant insurance/fraud terminology."),
    ("human", "Original query: {query}\nRewrite it to improve retrieval quality:"),
])


def run_fraud_retrieval_agent(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> dict:
    """
    Retrieve the most similar historical fraud claims for the given query.
    Returns a dict with 'claims', 'query_used', and 'crag_triggered'.
    """
    retriever = get_retriever()
    results: list[RetrievedClaim] = retriever.retrieve(query, top_k=top_k, filters=filters)

    crag_triggered = False
    query_used = query

    # CRAG: check average confidence; if low, refine and retry
    if results:
        avg_score = sum(r.score for r in results) / len(results)
        if avg_score < CONFIDENCE_THRESHOLD:
            crag_triggered = True
            llm = get_llm(temperature=0)
            chain = REFINE_PROMPT | llm
            refined = chain.invoke({"query": query})
            query_used = refined.content.strip()
            results = retriever.retrieve(query_used, top_k=top_k, filters=filters)

    return {
        "claims": [
            {
                "doc_id":    r.doc_id,
                "document":  r.document,
                "metadata":  r.metadata,
                "score":     round(r.score, 4),
            }
            for r in results
        ],
        "query_used":      query_used,
        "crag_triggered":  crag_triggered,
    }
