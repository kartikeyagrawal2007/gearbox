"""Difficulty estimators map a prompt to a score in [0, 1] plus a confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DifficultyEstimate:
    score: float  # 0 = trivial, 1 = hardest
    confidence: float  # how much to trust `score`; low confidence raises risk-scaled leverage
    rationale: str = ""


class DifficultyEstimator(Protocol):
    async def estimate(self, prompt: str) -> DifficultyEstimate: ...


from gearbox.difficulty.heuristic import HeuristicEstimator  # noqa: E402
from gearbox.difficulty.judge import JudgeEstimator  # noqa: E402

__all__ = ["DifficultyEstimate", "DifficultyEstimator", "HeuristicEstimator", "JudgeEstimator"]
