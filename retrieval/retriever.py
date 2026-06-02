"""
Hybrid retriever: BM25 (keyword) + ChromaDB (semantic) fusion + cross-encoder reranker.
Implements Reciprocal Rank Fusion (RRF) to merge the two result lists.
"""

import os
import sys
from typing import Optional
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from ingestion.ingest import get_chroma_collection

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # RRF constant


@dataclass
class RetrievedClaim:
    doc_id: str
    document: str
    metadata: dict
    score: float
    source: str  # "semantic" | "bm25" | "hybrid"


class HybridRetriever:
    def __init__(self):
        self.collection = get_chroma_collection()
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metas: list[dict] = []
        self._reranker: Optional[CrossEncoder] = None
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Pull all documents from ChromaDB and build an in-memory BM25 index."""
        total = self.collection.count()
        if total == 0:
            print("Warning: ChromaDB collection is empty. Run ingestion first.")
            return

        result = self.collection.get(include=["documents", "metadatas"])
        self._bm25_docs  = result["documents"]
        self._bm25_ids   = result["ids"]
        self._bm25_metas = result["metadatas"]
        tokenized = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built over {total} claims.")

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def _semantic_search(self, query: str, top_k: int, filters: Optional[dict] = None) -> list[RetrievedClaim]:
        where = filters if filters else None
        result = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        claims = []
        for doc, meta, dist, cid in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ):
            score = 1.0 - float(dist)  # cosine similarity
            claims.append(RetrievedClaim(doc_id=cid, document=doc, metadata=meta, score=score, source="semantic"))
        return claims

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedClaim]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        claims = []
        for idx in top_indices:
            claims.append(RetrievedClaim(
                doc_id=self._bm25_ids[idx],
                document=self._bm25_docs[idx],
                metadata=self._bm25_metas[idx],
                score=float(scores[idx]),
                source="bm25",
            ))
        return claims

    def _rrf_fusion(self, sem_results: list[RetrievedClaim], bm25_results: list[RetrievedClaim]) -> list[RetrievedClaim]:
        """Reciprocal Rank Fusion of two ranked lists."""
        rrf_scores: dict[str, float] = {}
        doc_map:    dict[str, RetrievedClaim] = {}

        for rank, claim in enumerate(sem_results):
            rrf_scores[claim.doc_id] = rrf_scores.get(claim.doc_id, 0) + 1.0 / (RRF_K + rank + 1)
            doc_map[claim.doc_id] = claim

        for rank, claim in enumerate(bm25_results):
            rrf_scores[claim.doc_id] = rrf_scores.get(claim.doc_id, 0) + 1.0 / (RRF_K + rank + 1)
            if claim.doc_id not in doc_map:
                doc_map[claim.doc_id] = claim

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in fused:
            c = doc_map[doc_id]
            results.append(RetrievedClaim(
                doc_id=c.doc_id, document=c.document,
                metadata=c.metadata, score=score, source="hybrid"
            ))
        return results

    def _rerank(self, query: str, candidates: list[RetrievedClaim], top_k: int) -> list[RetrievedClaim]:
        """Cross-encoder reranking on the fused candidates."""
        if not candidates:
            return []
        reranker = self._get_reranker()
        pairs = [[query, c.document] for c in candidates]
        scores = reranker.predict(pairs)
        for claim, score in zip(candidates, scores):
            claim.score = float(score)
        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        rerank: bool = True,
    ) -> list[RetrievedClaim]:
        """
        Main entry point. Returns top_k claims ranked by hybrid + reranker score.

        Args:
            query:   Natural language query.
            top_k:   Number of final results to return.
            filters: ChromaDB metadata filter dict, e.g. {"fraud_label": "Y"}.
            rerank:  Whether to apply cross-encoder reranking.
        """
        fetch_k = max(top_k * 3, 20)

        sem_results  = self._semantic_search(query, fetch_k, filters)
        bm25_results = self._bm25_search(query, fetch_k)
        fused        = self._rrf_fusion(sem_results, bm25_results)

        if rerank:
            return self._rerank(query, fused[:fetch_k], top_k)
        return fused[:top_k]


# singleton
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
