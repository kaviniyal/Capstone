import os
import sys
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from api.schemas import IngestRequest, IngestResponse
from ingestion.ingest import ingest_csv

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse)
def ingest_data(req: IngestRequest):
    """Load fraud_oracle.csv, embed every claim, and store in ChromaDB."""
    try:
        count = ingest_csv(csv_path=req.csv_path, reset=req.reset)
        return IngestResponse(
            status="success",
            records_ingested=count,
            message=f"Successfully ingested {count} claims into the vector store.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
