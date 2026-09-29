"""Zero-cost baseline estimator from surface features.

Deliberately simple and transparent: it is the baseline every learned or
LLM-judged estimator has to beat, and it never spends tokens.
"""

from __future__ import annotations

import math
import re

from gearbox.difficulty import DifficultyEstimate

_HARD = re.compile(
    r"\b(prove|proof|derive|architect|distributed|concurren|race condition|deadlock|optimi[sz]"
    r"|security|vulnerab|migrat|refactor|debug|root cause|trade-?off|algorithm|complexity"
    r"|formal|theorem|invariant|scalab)",
)
_EASY = re.compile(
    r"\b(summari[sz]e|rename|reformat|format|typo|translate|list|extract|classify|convert"
    r"|spell|lint|boilerplate|docstring|commit message)",
)
_CODE = re.compile(r"```|\bdef |\bclass |\bfunction\b|=>|#include|\bimport ")
_REQUIREMENT = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


class HeuristicEstimator:
    async def estimate(self, prompt: str) -> DifficultyEstimate:
        return self.estimate_sync(prompt)

    def estimate_sync(self, prompt: str) -> DifficultyEstimate:
        text = prompt.lower()
        words = len(text.split())
        hard = len(set(_HARD.findall(text)))
        easy = len(set(_EASY.findall(text)))
        requirements = len(_REQUIREMENT.findall(prompt))

        score = 0.15
        score += 0.25 * min(1.0, math.log10(1 + words) / 3.5)
        score += min(0.12 * hard, 0.36)
        score -= min(0.12 * easy, 0.24)
        score += 0.08 if _CODE.search(prompt) else 0.0
        score += min(0.05 * max(requirements - 2, 0), 0.2)
        score = min(max(score, 0.0), 1.0)

        # Surface features are weak evidence; only extreme scores earn moderate confidence.
        confidence = 0.35 + 0.3 * abs(score - 0.5) * 2
        rationale = f"words={words} hard={hard} easy={easy} requirements={requirements}"
        return DifficultyEstimate(score=round(score, 3), confidence=round(confidence, 3), rationale=rationale)
