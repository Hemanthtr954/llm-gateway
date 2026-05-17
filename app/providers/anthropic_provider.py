import time
import logging
from typing import Any

import httpx

from app.config import get_settings
from app.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=ANTHROPIC_API_BASE,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        Split OpenAI-style messages into (system_prompt, user/assistant turns).
        Anthropic requires system as a top-level field, not a message role.
        """
        system_prompt: str | None = None
        converted: list[dict[str, str]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_prompt = content
            else:
                converted.append({"role": role, "content": content})

        return system_prompt, converted

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        system_prompt, converted_messages = self._convert_messages(messages)

        max_tokens = kwargs.get("max_tokens", 4096)
        payload: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]

        start = time.monotonic()
        response = await self._client.post(
            "/messages",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        elapsed_ms = int((time.monotonic() - start) * 1000)

        data = response.json()
        # Anthropic returns content as a list of blocks
        content_blocks = data.get("content", [])
        text_content = " ".join(
            block["text"] for block in content_blocks if block.get("type") == "text"
        )

        usage = data.get("usage", {})

        return ProviderResponse(
            content=text_content,
            model=data.get("model", model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=elapsed_ms,
            provider=self.name,
            finish_reason=data.get("stop_reason", "stop"),
        )

    async def health_check(self) -> bool:
        try:
            # Anthropic doesn't have a /models endpoint in the same way;
            # send a minimal messages call to verify connectivity.
            payload = {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
            response = await self._client.post(
                "/messages", json=payload, headers=self._headers()
            )
            return response.status_code in (200, 400)  # 400 = bad request but reachable
        except Exception as exc:
            logger.warning("Anthropic health check failed: %s", exc)
            return False
