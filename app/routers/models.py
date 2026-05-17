import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

SUPPORTED_MODELS = [
    # OpenAI
    {"id": "gpt-4o", "provider": "openai"},
    {"id": "gpt-4o-mini", "provider": "openai"},
    {"id": "gpt-4-turbo", "provider": "openai"},
    {"id": "gpt-4", "provider": "openai"},
    {"id": "gpt-3.5-turbo", "provider": "openai"},
    # Anthropic
    {"id": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
    {"id": "claude-3-5-sonnet-20240620", "provider": "anthropic"},
    {"id": "claude-3-5-haiku-20241022", "provider": "anthropic"},
    {"id": "claude-3-opus-20240229", "provider": "anthropic"},
    {"id": "claude-3-haiku-20240307", "provider": "anthropic"},
    # Groq
    {"id": "llama-3.1-8b-instant", "provider": "groq"},
    {"id": "llama-3.1-70b-versatile", "provider": "groq"},
    {"id": "llama-3.3-70b-versatile", "provider": "groq"},
    {"id": "mixtral-8x7b-32768", "provider": "groq"},
    {"id": "gemma2-9b-it", "provider": "groq"},
]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    provider: str


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> Any:
    """Return the list of available models — OpenAI-compatible format."""
    created_ts = int(time.time())
    data = [
        ModelInfo(
            id=m["id"],
            created=created_ts,
            owned_by=m["provider"],
            provider=m["provider"],
        )
        for m in SUPPORTED_MODELS
    ]
    return ModelListResponse(data=data)
