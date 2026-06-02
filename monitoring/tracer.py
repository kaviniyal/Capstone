"""
LangSmith monitoring setup.
Enables automatic tracing of all LangChain / LangGraph calls when
LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set in .env.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings


def setup_langsmith():
    """Call once at app startup to enable LangSmith tracing."""
    if not settings.langchain_tracing_v2:
        print("LangSmith tracing disabled (LANGCHAIN_TRACING_V2=false).")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"]     = settings.langchain_project
    print(f"LangSmith tracing enabled → project: '{settings.langchain_project}'")


def get_run_url(run_id: str) -> str:
    """Return the LangSmith UI URL for a given run ID."""
    return f"https://smith.langchain.com/o/runs/{run_id}"
