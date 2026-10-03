"""Configuration: an ordered ladder of model tiers plus routing/delegation knobs.

Tiers are ordered cheapest -> strongest. Index 0 is the lowest gear.
Prices are USD per million tokens. For local models, use API-equivalent
prices (what a hosted provider charges for the same open model) or leave 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATHS = ("gearbox.yaml", "~/.config/gearbox/gearbox.yaml")
RISK_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class Pricing:
    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None  # defaults to `input` when unset

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cache_rate = self.input if self.cache_read is None else self.cache_read
        uncached = max(input_tokens - cached_tokens, 0)
        return (uncached * self.input + cached_tokens * cache_rate + output_tokens * self.output) / 1e6


@dataclass(frozen=True)
class Tier:
    name: str
    model: str  # LiteLLM model string, e.g. "ollama_chat/qwen3:4b" or "anthropic/claude-haiku-4-5"
    pricing: Pricing = field(default_factory=Pricing)
    context_window: int = 32_768
    api_base: str | None = None
    params: dict[str, Any] = field(default_factory=dict)  # extra kwargs passed to the provider


@dataclass(frozen=True)
class GearboxConfig:
    tiers: tuple[Tier, ...]
    # The agent that calls Gearbox (e.g. the model inside Claude Code or Antigravity).
    # Gearbox never sees its tokens; its pricing is only used to estimate savings.
    host: Pricing = field(default_factory=Pricing)
    leverage: int = 1
    risk_scaled_leverage: bool = True
    difficulty: str = "heuristic"  # "heuristic" | "judge"
    judge_tier: int = 0
    max_concurrent: int = 4
    task_timeout_s: float = 600.0
    max_escalations: int = 1
    # Executable acceptance checks run model-written code (best-effort isolation, see
    # gearbox/verify/checks.py), so they are opt-in.
    code_checks: bool = False
    check_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("config needs at least one tier")
        names = [t.name for t in self.tiers]
        if len(set(names)) != len(names):
            raise ValueError(f"tier names must be unique, got {names}")
        if self.leverage < 0:
            raise ValueError("leverage must be >= 0")
        if self.difficulty not in ("heuristic", "judge"):
            raise ValueError(f"unknown difficulty estimator {self.difficulty!r}")
        if not 0 <= self.judge_tier < len(self.tiers):
            raise ValueError("judge_tier out of range")

    @property
    def top(self) -> int:
        return len(self.tiers) - 1

    def tier_index(self, name_or_index: str | int) -> int:
        if isinstance(name_or_index, int):
            if not 0 <= name_or_index <= self.top:
                raise KeyError(f"tier index {name_or_index} out of range")
            return name_or_index
        for i, t in enumerate(self.tiers):
            if t.name == name_or_index:
                return i
        raise KeyError(f"unknown tier {name_or_index!r}; known: {[t.name for t in self.tiers]}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GearboxConfig:
        tiers = tuple(_tier_from_dict(t) for t in data.get("tiers", []))
        host = _pricing_from_dict(data.get("host", {}))
        knobs = {k: v for k, v in data.items() if k not in ("tiers", "host")}
        return cls(tiers=tiers, host=host, **knobs)


def _pricing_from_dict(d: dict[str, Any]) -> Pricing:
    return Pricing(
        input=float(d.get("input_price", 0.0)),
        output=float(d.get("output_price", 0.0)),
        cache_read=None if d.get("cache_read_price") is None else float(d["cache_read_price"]),
    )


def _tier_from_dict(d: dict[str, Any]) -> Tier:
    return Tier(
        name=d["name"],
        model=d["model"],
        pricing=_pricing_from_dict(d),
        context_window=int(d.get("context_window", 32_768)),
        api_base=d.get("api_base"),
        params=dict(d.get("params", {})),
    )


def load_config(path: str | os.PathLike[str] | None = None) -> GearboxConfig:
    """Load config from `path`, $GEARBOX_CONFIG, ./gearbox.yaml or ~/.config/gearbox/gearbox.yaml."""
    candidates = [path] if path else [os.environ.get("GEARBOX_CONFIG"), *DEFAULT_CONFIG_PATHS]
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_file():
            with p.open() as f:
                return GearboxConfig.from_dict(yaml.safe_load(f) or {})
    raise FileNotFoundError(
        "no Gearbox config found; copy gearbox.example.yaml to gearbox.yaml or set GEARBOX_CONFIG"
    )
