"""Pick a tier: estimate difficulty, map it to a base tier, then shift up by leverage.

Leverage is the safety margin: route to a gear higher than the estimate says,
so an underestimated task does not land on a model that fails it. It is either
a constant (config.leverage) or risk-scaled: it grows with how costly a wrong
answer is (low/medium/high) and with how unsure the estimator is.
"""

from __future__ import annotations

from dataclasses import dataclass

from gearbox.config import RISK_LEVELS, GearboxConfig
from gearbox.cost.ledger import Ledger
from gearbox.difficulty import DifficultyEstimate, DifficultyEstimator, HeuristicEstimator, JudgeEstimator
from gearbox.providers import Provider

RISK_LEVERAGE = {"low": 0, "medium": 1, "high": 2}
LOW_CONFIDENCE = 0.5


@dataclass(frozen=True)
class RoutingDecision:
    score: float
    confidence: float
    base_tier: int
    leverage: int
    tier: int
    tier_name: str
    rationale: str

    def as_dict(self) -> dict:
        return {
            "tier": self.tier_name,
            "tier_index": self.tier,
            "base_tier_index": self.base_tier,
            "leverage": self.leverage,
            "difficulty": self.score,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def base_tier_for(score: float, n_tiers: int) -> int:
    """Split [0, 1] into n equal bands, one per tier."""
    return min(int(score * n_tiers), n_tiers - 1)


def leverage_for(
    config: GearboxConfig, risk: str | None, confidence: float, override: int | None = None
) -> int:
    if override is not None:
        if override < 0:
            raise ValueError("leverage must be >= 0")
        return override
    if config.risk_scaled_leverage and risk is not None:
        if risk not in RISK_LEVELS:
            raise ValueError(f"risk must be one of {RISK_LEVELS}, got {risk!r}")
        return RISK_LEVERAGE[risk] + (1 if confidence < LOW_CONFIDENCE else 0)
    return config.leverage


def decide(
    config: GearboxConfig,
    estimate: DifficultyEstimate,
    risk: str | None = None,
    leverage: int | None = None,
    min_tier: int = 0,
    max_tier: int | None = None,
) -> RoutingDecision:
    ceiling = config.top if max_tier is None else min(max_tier, config.top)
    base = base_tier_for(estimate.score, len(config.tiers))
    lev = leverage_for(config, risk, estimate.confidence, leverage)
    tier = min(max(base + lev, min_tier), ceiling)
    return RoutingDecision(
        score=estimate.score,
        confidence=estimate.confidence,
        base_tier=base,
        leverage=lev,
        tier=tier,
        tier_name=config.tiers[tier].name,
        rationale=estimate.rationale,
    )


class Router:
    def __init__(self, config: GearboxConfig, estimator: DifficultyEstimator) -> None:
        self.config = config
        self.estimator = estimator

    @classmethod
    def from_config(cls, config: GearboxConfig, provider: Provider, ledger: Ledger) -> Router:
        if config.difficulty == "judge":
            estimator: DifficultyEstimator = JudgeEstimator(provider, config.tiers[config.judge_tier], ledger)
        else:
            estimator = HeuristicEstimator()
        return cls(config, estimator)

    async def route(
        self,
        prompt: str,
        risk: str | None = None,
        leverage: int | None = None,
        min_tier: int = 0,
        max_tier: int | None = None,
    ) -> RoutingDecision:
        estimate = await self.estimator.estimate(prompt)
        return decide(self.config, estimate, risk, leverage, min_tier, max_tier)
