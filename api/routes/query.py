import os
import sys
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from api.schemas import QueryRequest, QueryResponse, ClaimResponse
from retrieval.retriever import get_retriever

router = APIRouter(prefix="/query", tags=["Retrieval"])


@router.post("", response_model=QueryResponse)
def semantic_query(req: QueryRequest):
    """Hybrid BM25 + semantic search over the claims vector store."""
    try:
        retriever = get_retriever()
        results = retriever.retrieve(
            query=req.query,
            top_k=req.top_k,
            filters=req.filters,
            rerank=req.rerank,
        )
        return QueryResponse(
            query_used=req.query,
            crag_triggered=False,
            results=[
                {"doc_id": r.doc_id, "document": r.document, "metadata": r.metadata, "score": round(r.score, 4)}
                for r in results
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claim/{claim_id}", response_model=ClaimResponse)
def get_claim(claim_id: str):
    """Retrieve a specific claim by its document ID."""
    try:
        retriever = get_retriever()
        result = retriever.collection.get(ids=[claim_id], include=["documents", "metadatas"])
        if not result["ids"]:
            raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
        return ClaimResponse(
            claim_id=claim_id,
            document=result["documents"][0],
            metadata=result["metadatas"][0],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
