"""
Ingestion pipeline: loads fraud_oracle.csv, embeds each row with OpenAI,
and upserts vectors + metadata into Pinecone.

Also writes data/docs_cache.json — a local copy of all documents used to
build the BM25 keyword index at retrieval time (Pinecone is vector-only).
"""

from __future__ import annotations
import os
import sys
import json
import pandas as pd
from typing import Optional
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from llm_factory import get_embeddings

DATA_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "fraud_oracle.csv")
DOCS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "docs_cache.json")


# ── Document builders ─────────────────────────────────────────────────────────

def _row_to_document(row: dict) -> str:
    return (
        f"Claim ID: {row.get('policy_number', row.get('claim_id', 'N/A'))}. "
        f"Policy Type: {row.get('policy_type', 'N/A')}. "
        f"Accident Type: {row.get('incident_type', row.get('accident_type', 'N/A'))}. "
        f"Claim Amount: {row.get('total_claim_amount', row.get('claim_amount', 'N/A'))}. "
        f"Customer Region: {row.get('insured_zip', row.get('customer_region', 'N/A'))}. "
        f"Fraud Label: {row.get('fraud_reported', row.get('fraud_label', 'N/A'))}. "
        f"Incident Date: {row.get('incident_date', 'N/A')}. "
        f"Customer History: {row.get('insured_occupation', row.get('customer_history', 'N/A'))}. "
        f"Claim Status: {row.get('incident_severity', row.get('claim_status', 'N/A'))}. "
        f"Number of Vehicles Involved: {row.get('number_of_vehicles_involved', 'N/A')}. "
        f"Bodily Injuries: {row.get('bodily_injuries', 'N/A')}. "
        f"Property Damage: {row.get('property_damage', 'N/A')}. "
        f"Police Report: {row.get('police_report_available', 'N/A')}."
    )


def _build_metadata(row: dict, idx: int) -> dict:
    def safe(val):
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return "unknown"
        return str(val)

    return {
        "claim_id":        safe(row.get("policy_number", idx)),
        "policy_type":     safe(row.get("policy_type")),
        "accident_type":   safe(row.get("incident_type", row.get("accident_type"))),
        "claim_amount":    safe(row.get("total_claim_amount", row.get("claim_amount"))),
        "customer_region": safe(row.get("insured_zip", row.get("customer_region"))),
        "fraud_label":     safe(row.get("fraud_reported", row.get("fraud_label"))),
        "incident_date":   safe(row.get("incident_date")),
        "claim_status":    safe(row.get("incident_severity", row.get("claim_status"))),
    }


# ── Pinecone helpers ──────────────────────────────────────────────────────────

def get_pinecone_index(reset: bool = False):
    """Return a connected Pinecone index, creating it if necessary."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if reset and settings.pinecone_index_name in existing:
        pc.delete_index(settings.pinecone_index_name)
        existing = []

    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        print(f"Created Pinecone index '{settings.pinecone_index_name}'")

    return pc.Index(settings.pinecone_index_name)


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_csv(
    csv_path: Optional[str] = None,
    batch_size: int = 100,
    reset: bool = False,
) -> int:
    """
    Embed each claim row and upsert into Pinecone.
    Also writes docs_cache.json for the BM25 retriever.
    Returns the number of records ingested.
    """
    path = csv_path or DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download fraud_oracle.csv from Kaggle and place it in data/"
        )

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"Loaded {len(df)} rows from {path}")

    index = get_pinecone_index(reset=reset)

    # Skip if already populated and not resetting
    stats = index.describe_index_stats()
    existing_count = stats.get("total_vector_count", 0)
    if existing_count > 0 and not reset:
        print(f"Index already has {existing_count} vectors. Pass reset=True to re-ingest.")
        return existing_count

    embedder = get_embeddings()
    docs_cache: list[dict] = []
    total_upserted = 0

    rows = df.to_dict(orient="records")
    for batch_start in tqdm(range(0, len(rows), batch_size), desc="Embedding & upserting"):
        batch_rows = rows[batch_start: batch_start + batch_size]

        documents = [_row_to_document(r) for r in batch_rows]
        metadatas = [_build_metadata(r, batch_start + i) for i, r in enumerate(batch_rows)]
        ids       = [f"claim_{batch_start + i}" for i in range(len(batch_rows))]

        # Embed the batch
        embeddings = embedder.embed_documents(documents)

        # Build Pinecone upsert payload
        # Store document text inside metadata so retriever can reconstruct it
        vectors = [
            {
                "id":     doc_id,
                "values": emb,
                "metadata": {**meta, "text": doc},
            }
            for doc_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]
        index.upsert(vectors=vectors)
        total_upserted += len(vectors)

        # Accumulate local cache for BM25
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            docs_cache.append({"id": doc_id, "document": doc, "metadata": meta})

    # Persist local BM25 cache
    cache_path = DOCS_CACHE_PATH
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(docs_cache, f)
    print(f"docs_cache.json written ({len(docs_cache)} records)")

    print(f"Ingestion complete. Total vectors upserted: {total_upserted}")
    return total_upserted


if __name__ == "__main__":
    ingest_csv(reset=True)
