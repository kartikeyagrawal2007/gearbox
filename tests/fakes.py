"""Deterministic stand-ins for model backends."""

from __future__ import annotations

import asyncio

from gearbox.config import GearboxConfig, Pricing, Tier
from gearbox.providers import Completion


def make_config(n_tiers: int = 3, **overrides) -> GearboxConfig:
    tiers = tuple(
        Tier(name=f"t{i}", model=f"fake/t{i}", pricing=Pricing(input=float(i + 1), output=float(4 * (i + 1))))
        for i in range(n_tiers)
    )
    knobs = {"host": Pricing(input=10.0, output=50.0), "leverage": 0, "risk_scaled_leverage": False, **overrides}
    return GearboxConfig(tiers=tiers, **knobs)


class FakeProvider:
    """Replies per tier name. A reply may be a string, an Exception to raise,
    or a callable taking the messages. `delay` simulates model latency."""

    def __init__(self, replies: dict[str, object] | None = None, delay: float = 0.0) -> None:
        self.replies = replies or {}
        self.delay = delay
        self.calls: list[str] = []

    async def complete(self, tier: Tier, messages: list[dict[str, str]], **kwargs) -> Completion:
        self.calls.append(tier.name)
        await asyncio.sleep(self.delay)
        reply = self.replies.get(tier.name, f"result from {tier.name}")
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages)
        return Completion(text=reply, input_tokens=100, output_tokens=400, latency_s=self.delay)
