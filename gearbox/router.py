"""Pick a tier: estimate difficulty, map it to a base tier, then shift up by leverage.

Leverage is the safety margin: route to a gear higher than the estimate says,
so an underestimated task does not land on a model that fails it. It is either
a constant (config.leverage) or risk-scaled: it grows with how costly a wrong
answer is (low/medium/high) and with how unsure the estimator is.

A subtask with a check is different: a failed check escalates, so leverage only has to
cover the wrong answers the check lets through. It then comes from the check's strength
(config.check_leverage: weak +2, medium +1, strong +0 by default), plus the risk level.
"""

from __future__ import annotations

from dataclasses import dataclass

from gearbox.config import RISK_LEVELS, GearboxConfig
from gearbox.cost.ledger import Ledger
from gearbox.difficulty import (
    DifficultyEstimate, DifficultyEstimator, HeuristicEstimator, JudgeEstimator, LearnedEstimator, RouterModel,
)
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
    config: GearboxConfig, risk: str | None, confidence: float, override: int | None = None,
    check_strength: str | None = None,
) -> int:
    if risk is not None and risk not in RISK_LEVELS:
        raise ValueError(f"risk must be one of {RISK_LEVELS}, got {risk!r}")
    if override is not None:
        if override < 0:
            raise ValueError("leverage must be >= 0")
        return override
    if check_strength in config.check_leverage:
        return config.check_leverage[check_strength] + (RISK_LEVERAGE[risk] if risk else 0)
    if config.risk_scaled_leverage and risk is not None:
        return RISK_LEVERAGE[risk] + (1 if confidence < LOW_CONFIDENCE else 0)
    return config.leverage


def decide(
    config: GearboxConfig,
    estimate: DifficultyEstimate,
    risk: str | None = None,
    leverage: int | None = None,
    min_tier: int = 0,
    max_tier: int | None = None,
    check_strength: str | None = None,
) -> RoutingDecision:
    ceiling = config.top if max_tier is None else min(max_tier, config.top)
    min_tier = max(min_tier, config.floor)
    rationale = estimate.rationale
    if estimate.pass_prob is not None:
        # Learned: the cheapest tier (from the floor up) whose pass chance clears the bar for
        # this check. The bar is the safety margin, so only risk (or an explicit leverage) adds.
        tau = config.learned_tau.get(check_strength or "none", 0.9)
        base = next((i for i in range(min_tier, ceiling + 1)
                     if estimate.pass_prob[i] is not None and estimate.pass_prob[i] >= tau), ceiling)
        if risk is not None and risk not in RISK_LEVELS:
            raise ValueError(f"risk must be one of {RISK_LEVELS}, got {risk!r}")
        lev = leverage if leverage is not None else (RISK_LEVERAGE[risk] if risk else 0)
        p = estimate.pass_prob[base]
        rationale += f"; {config.tiers[base].name} passes with p={p:.2f}" if p is not None else ""
        rationale += f" (bar {tau:.2f}, {check_strength or 'no'} check)"
    else:
        base = base_tier_for(estimate.score, len(config.tiers))
        lev = leverage_for(config, risk, estimate.confidence, leverage, check_strength)
        if leverage is None and check_strength in config.check_leverage:
            rationale += f"; {check_strength} check: leverage +{config.check_leverage[check_strength]}"
    tier = min(max(base + lev, min_tier), ceiling)
    return RoutingDecision(
        score=estimate.score,
        confidence=estimate.confidence,
        base_tier=base,
        leverage=lev,
        tier=tier,
        tier_name=config.tiers[tier].name,
        rationale=rationale,
    )


class Router:
    def __init__(self, config: GearboxConfig, estimator: DifficultyEstimator) -> None:
        self.config = config
        self.estimator = estimator

    @classmethod
    def from_config(cls, config: GearboxConfig, provider: Provider, ledger: Ledger) -> Router:
        if config.difficulty == "judge":
            estimator: DifficultyEstimator = JudgeEstimator(provider, config.tiers[config.judge_tier], ledger)
        elif config.difficulty == "learned":
            estimator = LearnedEstimator(config, provider, RouterModel.load(config.router_model), ledger)
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
        check_strength: str | None = None,
    ) -> RoutingDecision:
        estimate = await self.estimator.estimate(prompt)
        return decide(self.config, estimate, risk, leverage, min_tier, max_tier, check_strength)
