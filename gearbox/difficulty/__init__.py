"""Difficulty estimators map a prompt to a score in [0, 1] plus a confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DifficultyEstimate:
    score: float  # 0 = trivial, 1 = hardest
    confidence: float  # how much to trust `score`; low confidence raises risk-scaled leverage
    rationale: str = ""
    # Pass chance per tier (config order), when the estimator can predict it (the learned
    # estimator); None for tiers it knows nothing about. The router then picks the cheapest
    # tier that clears a bar instead of mapping `score` to a tier.
    pass_prob: tuple[float | None, ...] | None = None


class DifficultyEstimator(Protocol):
    async def estimate(self, prompt: str) -> DifficultyEstimate: ...


from gearbox.difficulty.heuristic import HeuristicEstimator  # noqa: E402
from gearbox.difficulty.judge import JudgeEstimator  # noqa: E402
from gearbox.difficulty.learned import LearnedEstimator, RouterModel  # noqa: E402

__all__ = ["DifficultyEstimate", "DifficultyEstimator", "HeuristicEstimator", "JudgeEstimator",
           "LearnedEstimator", "RouterModel"]
