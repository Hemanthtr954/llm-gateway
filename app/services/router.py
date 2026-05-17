import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from app.providers.base import BaseProvider, ProviderResponse
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

# Singleton provider instances
_openai = OpenAIProvider()
_anthropic = AnthropicProvider()
_groq = GroqProvider()


def _pick_primary_provider(model: str) -> BaseProvider:
    """Return the primary provider for the given model name."""
    m = model.lower()
    if m.startswith("claude"):
        return _anthropic
    if any(m.startswith(prefix) for prefix in ("llama", "mixtral", "gemma", "whisper-large")):
        return _groq
    return _openai


def _fallback_chain(primary: BaseProvider) -> list[BaseProvider]:
    """Return ordered list of providers to try, starting from primary."""
    all_providers = [_openai, _anthropic, _groq]
    chain = [primary] + [p for p in all_providers if p is not primary]
    return chain


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
async def _call_with_retry(
    provider: BaseProvider,
    messages: list[dict[str, str]],
    model: str,
    **kwargs: Any,
) -> ProviderResponse:
    return await provider.chat_complete(messages, model, **kwargs)


class ProviderRouter:
    """
    Routes requests to the appropriate provider based on model name.
    On failure, falls back to the next provider in the chain.
    """

    async def route(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> tuple[ProviderResponse, str]:
        """
        Returns (ProviderResponse, provider_name_used).
        Raises RuntimeError if all providers fail.
        """
        primary = _pick_primary_provider(model)
        chain = _fallback_chain(primary)

        last_exc: Exception | None = None
        for provider in chain:
            try:
                logger.info("Attempting provider=%s model=%s", provider.name, model)
                result = await _call_with_retry(provider, messages, model, **kwargs)
                if provider is not primary:
                    logger.warning(
                        "Used fallback provider=%s (primary=%s failed)",
                        provider.name,
                        primary.name,
                    )
                return result, provider.name
            except Exception as exc:
                logger.error(
                    "Provider %s failed for model=%s: %s",
                    provider.name,
                    model,
                    exc,
                )
                last_exc = exc
                continue

        raise RuntimeError(
            f"All providers exhausted for model={model}. Last error: {last_exc}"
        ) from last_exc


# Singleton router
provider_router = ProviderRouter()
