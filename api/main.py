import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from monitoring.tracer import setup_langsmith
from api.routes import ingest, query, analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_langsmith()
    yield


app = FastAPI(
    title="Insurance Claims Intelligence API",
    description=(
        "AI-powered fraud detection and investigation assistant. "
        "Uses hybrid RAG, multi-agent LangGraph pipeline, guardrails, and LangSmith monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(analyze.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Insurance Claims Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)
