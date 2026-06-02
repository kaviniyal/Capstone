"""
Central factory for LLM and embedding clients.
All agents import from here so the gateway URL is configured in one place.
"""

import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(__file__))
from config import settings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_llm_temp(temperature: float) -> ChatOpenAI:
    """Non-cached version for agents that need a specific temperature."""
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_openai_client() -> OpenAI:
    """Raw OpenAI client for direct API calls (e.g. ChromaDB embedding function)."""
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
