"""LLM-as-judge estimator: a cheap tier rates task difficulty.

The judge's own tokens are routing overhead and are charged to the ledger,
so savings reports stay honest.
"""

from __future__ import annotations

import json
import re

from gearbox.config import Tier
from gearbox.cost.ledger import Ledger
from gearbox.difficulty import DifficultyEstimate
from gearbox.difficulty.heuristic import HeuristicEstimator
from gearbox.providers import Provider

MAX_PROMPT_CHARS = 6000
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

SYSTEM = (
    "You rate how difficult a task is for a language model. Reply with JSON only: "
    '{"difficulty": <integer 1-5>, "confidence": <0.0-1.0>, "reason": "<one short sentence>"}. '
    "1 = trivial lookup or formatting, 2 = simple single step, 3 = moderate multi-step, "
    "4 = hard reasoning or design, 5 = expert or research level."
)


class JudgeEstimator:
    def __init__(self, provider: Provider, tier: Tier, ledger: Ledger | None = None) -> None:
        self.provider = provider
        self.tier = tier
        self.ledger = ledger
        self._fallback = HeuristicEstimator()

    async def estimate(self, prompt: str) -> DifficultyEstimate:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt[:MAX_PROMPT_CHARS]},
        ]
        try:
            completion = await self.provider.complete(self.tier, messages, max_tokens=120, temperature=0)
        except Exception as e:  # judge down -> never block routing
            est = self._fallback.estimate_sync(prompt)
            return DifficultyEstimate(est.score, est.confidence, f"judge error ({e}); heuristic: {est.rationale}")

        if self.ledger is not None:
            self.ledger.record(self.tier, completion, role="judge")
        parsed = parse_judgement(completion.text)
        if parsed is None:
            est = self._fallback.estimate_sync(prompt)
            return DifficultyEstimate(est.score, est.confidence, f"unparseable judge reply; heuristic: {est.rationale}")
        return parsed


def parse_judgement(text: str) -> DifficultyEstimate | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        level = int(data["difficulty"])
        confidence = float(data.get("confidence", 0.5))
    except (ValueError, KeyError, TypeError):
        return None
    level = min(max(level, 1), 5)
    confidence = min(max(confidence, 0.0), 1.0)
    return DifficultyEstimate(score=(level - 1) / 4, confidence=confidence, rationale=str(data.get("reason", "")))
