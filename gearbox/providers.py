"""Model providers. The default talks to any backend LiteLLM supports
(Anthropic, OpenAI, Gemini, OpenRouter, Ollama, vLLM, ...)."""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from gearbox.config import Tier

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# LiteLLM downloads a model price list from raw.githubusercontent.com on import. Gearbox has
# its own pricing, and on networks that block or stall that host (seen on the lab VM) every
# start waits through retries. Use LiteLLM's bundled copy unless the user chose otherwise.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


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


class SimulatedProvider:
    """Offline stand-in for demos and dry runs: no model, fake text, latency that
    grows with the tier index. Never use its numbers as results."""

    def __init__(self, tier_names: list[str], base_s: float = 0.8, per_tier_s: float = 0.8) -> None:
        self._index = {name: i for i, name in enumerate(tier_names)}
        self.base_s = base_s
        self.per_tier_s = per_tier_s

    async def complete(
        self, tier: Tier, messages: list[dict[str, str]], **kwargs: Any
    ) -> Completion:
        latency = self.base_s + self.per_tier_s * self._index.get(tier.name, 0)
        await asyncio.sleep(latency)
        prompt = messages[-1]["content"]
        first_line = next((line for line in prompt.splitlines() if line and not line.startswith("#")), prompt)
        return Completion(
            text=f"[simulated output from {tier.name}] {first_line[:120]}",
            input_tokens=sum(len(m["content"]) for m in messages) // 4,
            output_tokens=150,
            latency_s=latency,
        )


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
