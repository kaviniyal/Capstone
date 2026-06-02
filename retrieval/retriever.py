"""
Hybrid retriever: BM25 (keyword) + Pinecone (semantic) fusion + cross-encoder reranker.
Implements Reciprocal Rank Fusion (RRF) to merge the two result lists.

Semantic search  → Pinecone (cosine similarity over OpenAI embeddings)
Keyword search   → BM25Okapi built from docs_cache.json (written at ingest time)
Fusion           → Reciprocal Rank Fusion
Reranking        → cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations
import os
import sys
import json
from typing import Optional
from dataclasses import dataclass

import numpy as np
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from llm_factory import get_embeddings

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K           = 60
DOCS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "docs_cache.json")


@dataclass
class RetrievedClaim:
    doc_id:   str
    document: str
    metadata: dict
    score:    float
    source:   str  # "semantic" | "bm25" | "hybrid"


def _to_pinecone_filter(filters: dict) -> dict:
    """Convert flat dict {'fraud_label': 'Y'} → Pinecone filter {'fraud_label': {'$eq': 'Y'}}."""
    return {k: {"$eq": v} for k, v in filters.items()}


class HybridRetriever:
    def __init__(self):
        self._pc_index   = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
        self._embedder   = get_embeddings()
        self._reranker: Optional[CrossEncoder] = None

        # BM25 index — built from local docs_cache.json
        self._bm25:       Optional[BM25Okapi] = None
        self._bm25_docs:  list[str]  = []
        self._bm25_ids:   list[str]  = []
        self._bm25_metas: list[dict] = []
        self._build_bm25_index()

    # ── BM25 setup ─────────────────────────────────────────────────────────────

    def _build_bm25_index(self):
        if not os.path.exists(DOCS_CACHE_PATH):
            print("Warning: docs_cache.json not found. Run ingestion first.")
            return

        with open(DOCS_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)

        self._bm25_docs  = [c["document"] for c in cache]
        self._bm25_ids   = [c["id"]       for c in cache]
        self._bm25_metas = [c["metadata"] for c in cache]

        tokenized    = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25   = BM25Okapi(tokenized)
        print(f"BM25 index built over {len(cache)} claims.")

    # ── Semantic search via Pinecone ───────────────────────────────────────────

    def _semantic_search(
        self, query: str, top_k: int, filters: Optional[dict] = None
    ) -> list[RetrievedClaim]:
        query_vec  = self._embedder.embed_query(query)
        pc_filter  = _to_pinecone_filter(filters) if filters else None

        kwargs: dict = dict(vector=query_vec, top_k=top_k, include_metadata=True)
        if pc_filter:
            kwargs["filter"] = pc_filter

        results = self._pc_index.query(**kwargs)

        claims = []
        for match in results.matches:
            meta = dict(match.metadata)
            doc  = meta.pop("text", "")   # text was stored in metadata at ingest time
            claims.append(RetrievedClaim(
                doc_id=match.id,
                document=doc,
                metadata=meta,
                score=float(match.score),
                source="semantic",
            ))
        return claims

    # ── BM25 keyword search ────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedClaim]:
        if self._bm25 is None:
            return []
        scores      = self._bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedClaim(
                doc_id=self._bm25_ids[i],
                document=self._bm25_docs[i],
                metadata=self._bm25_metas[i],
                score=float(scores[i]),
                source="bm25",
            )
            for i in top_indices
        ]

    # ── RRF fusion ─────────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        sem_results:  list[RetrievedClaim],
        bm25_results: list[RetrievedClaim],
    ) -> list[RetrievedClaim]:
        rrf_scores: dict[str, float]        = {}
        doc_map:    dict[str, RetrievedClaim] = {}

        for rank, claim in enumerate(sem_results):
            rrf_scores[claim.doc_id] = rrf_scores.get(claim.doc_id, 0) + 1.0 / (RRF_K + rank + 1)
            doc_map[claim.doc_id] = claim

        for rank, claim in enumerate(bm25_results):
            rrf_scores[claim.doc_id] = rrf_scores.get(claim.doc_id, 0) + 1.0 / (RRF_K + rank + 1)
            if claim.doc_id not in doc_map:
                doc_map[claim.doc_id] = claim

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            RetrievedClaim(
                doc_id=doc_map[did].doc_id,
                document=doc_map[did].document,
                metadata=doc_map[did].metadata,
                score=score,
                source="hybrid",
            )
            for did, score in fused
        ]

    # ── Cross-encoder reranking ────────────────────────────────────────────────

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def _rerank(self, query: str, candidates: list[RetrievedClaim], top_k: int) -> list[RetrievedClaim]:
        if not candidates:
            return []
        reranker = self._get_reranker()
        scores   = reranker.predict([[query, c.document] for c in candidates])
        for claim, score in zip(candidates, scores):
            claim.score = float(score)
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]

    # ── Public entry point ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query:   str,
        top_k:   int = 5,
        filters: Optional[dict] = None,
        rerank:  bool = True,
    ) -> list[RetrievedClaim]:
        """
        Returns top_k claims ranked by hybrid BM25 + Pinecone score, optionally reranked.

        Args:
            query:   Natural language investigation query.
            top_k:   Number of final results to return.
            filters: Flat metadata filter dict e.g. {"fraud_label": "Y"}.
                     Converted to Pinecone filter syntax internally.
            rerank:  Apply cross-encoder reranking on fused candidates.
        """
        fetch_k      = max(top_k * 3, 20)
        sem_results  = self._semantic_search(query, fetch_k, filters)
        bm25_results = self._bm25_search(query, fetch_k)
        fused        = self._rrf_fusion(sem_results, bm25_results)

        if rerank:
            return self._rerank(query, fused[:fetch_k], top_k)
        return fused[:top_k]


# ── Singleton ──────────────────────────────────────────────────────────────────

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
