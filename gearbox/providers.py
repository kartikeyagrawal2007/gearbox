"""Model providers. The default talks to any backend LiteLLM supports
(Anthropic, OpenAI, Gemini, OpenRouter, Ollama, vLLM, ...)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from gearbox.config import Tier

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    latency_s: float = 0.0


class Provider(Protocol):
    async def complete(
        self, tier: Tier, messages: list[dict[str, str]], **kwargs: Any
    ) -> Completion: ...


class LiteLLMProvider:
    def __init__(self) -> None:
        # Imported here, not at module level or on first call: the import takes about a
        # second and is synchronous, so doing it inside the event loop would stall
        # every in-flight delegation and skew overlap measurements.
        import litellm

        litellm.suppress_debug_info = True  # otherwise it print()s help banners on errors
        self._litellm = litellm

    async def complete(
        self, tier: Tier, messages: list[dict[str, str]], **kwargs: Any
    ) -> Completion:
        params = {**tier.params, **kwargs}
        if tier.api_base:
            params["api_base"] = tier.api_base
        start = time.perf_counter()
        resp = await self._litellm.acompletion(model=tier.model, messages=messages, **params)
        latency = time.perf_counter() - start

        usage = resp.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        text = resp.choices[0].message.content or ""
        return Completion(
            text=_THINK_RE.sub("", text).strip(),
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            cached_tokens=cached,
            latency_s=latency,
        )
