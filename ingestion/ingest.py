"""
Ingestion pipeline: loads fraud_oracle.csv, converts each row into a
structured text document, embeds it, and stores in ChromaDB.
"""

import os
import sys
import pandas as pd
from typing import Optional
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

COLLECTION_NAME = "insurance_claims"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fraud_oracle.csv")


def _row_to_document(row: dict) -> str:
    """Convert a CSV row into a human-readable text chunk for embedding."""
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
    """Extract flat metadata for ChromaDB filtering."""
    def safe(val):
        if val is None or (isinstance(val, float) and str(val) == 'nan'):
            return "unknown"
        return str(val)

    return {
        "claim_id":       safe(row.get("policy_number", idx)),
        "policy_type":    safe(row.get("policy_type")),
        "accident_type":  safe(row.get("incident_type", row.get("accident_type"))),
        "claim_amount":   safe(row.get("total_claim_amount", row.get("claim_amount"))),
        "customer_region":safe(row.get("insured_zip", row.get("customer_region"))),
        "fraud_label":    safe(row.get("fraud_reported", row.get("fraud_label"))),
        "incident_date":  safe(row.get("incident_date")),
        "claim_status":   safe(row.get("incident_severity", row.get("claim_status"))),
    }


def get_chroma_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    embed_fn = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
        api_base=settings.openai_base_url,
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest_csv(csv_path: Optional[str] = None, batch_size: int = 100, reset: bool = False) -> int:
    """
    Load the CSV, embed each row, and upsert into ChromaDB.
    Returns the number of records ingested.
    """
    path = csv_path or DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download fraud_oracle.csv from Kaggle and place it in capstone_project/data/"
        )

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"Loaded {len(df)} rows from {path}")

    collection = get_chroma_collection(reset=reset)
    existing = collection.count()
    if existing > 0 and not reset:
        print(f"Collection already has {existing} records. Skipping ingestion. Pass reset=True to re-ingest.")
        return existing

    documents, metadatas, ids = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding claims"):
        doc = _row_to_document(row.to_dict())
        meta = _build_metadata(row.to_dict(), idx)
        doc_id = f"claim_{idx}"

        documents.append(doc)
        metadatas.append(meta)
        ids.append(doc_id)

        if len(documents) >= batch_size:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            documents, metadatas, ids = [], [], []

    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    total = collection.count()
    print(f"Ingestion complete. Total records in ChromaDB: {total}")
    return total


if __name__ == "__main__":
    ingest_csv(reset=True)
