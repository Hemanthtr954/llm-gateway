from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    provider: str
    finish_reason: str = "stop"


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Send a chat completion request to the provider.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            model: The model identifier.
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ProviderResponse with content and token counts.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
            Exception: On any other provider-side error.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
