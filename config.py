from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = "learner030"
    openai_base_url: str = "https://keygateway.arshnivlabs.com/v1"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "insurance-claims-intelligence"

    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    hitl_low_threshold: float = 0.4
    hitl_high_threshold: float = 0.6

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
