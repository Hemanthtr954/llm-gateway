from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite+aiosqlite:///./gateway.db"

    # Auth
    master_key: str = "change-me-to-a-secure-random-key"

    # Rate limiting defaults
    default_rpm: int = 60
    default_tpd: int = 100_000

    # Caching
    cache_ttl_seconds: int = 3600

    # App
    app_title: str = "LLM Gateway"
    app_version: str = "1.0.0"
    debug: bool = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
