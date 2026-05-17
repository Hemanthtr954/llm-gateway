import time
import logging
from typing import Any

import httpx

from app.config import get_settings
from app.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)

GROQ_API_BASE = "https://api.groq.com/openai/v1"


class GroqProvider(BaseProvider):
    """Groq uses an OpenAI-compatible endpoint."""

    name = "groq"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=GROQ_API_BASE,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in kwargs:
                payload[key] = kwargs[key]

        start = time.monotonic()
        response = await self._client.post(
            "/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        elapsed_ms = int((time.monotonic() - start) * 1000)

        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ProviderResponse(
            content=choice["message"]["content"] or "",
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
            provider=self.name,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/models", headers=self._headers())
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Groq health check failed: %s", exc)
            return False
