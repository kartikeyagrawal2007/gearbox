"""Learned estimator: a judge rates the task, a trained model turns that into pass chances.

Trained by bench/router_train.py --export on HumanEval+ and MBPP+ outcomes (item response
theory): every model has an ability a, the task gets a difficulty b predicted from the
judge's 1-10 rating, and P(model solves it) = sigmoid(a - b). The router then picks the
cheapest tier whose pass chance clears a bar (see gearbox/router.py).

The judge must be asked exactly as during training (JUDGE_PROMPT), or its ratings drift
from what the model learned. Abilities are stored by model string (e.g.
"ollama_chat/qwen3.5:4b"), so a router file works with any config that uses those models;
tiers with unknown models get no pass chance and aren't chosen by it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from gearbox.config import GearboxConfig
from gearbox.cost.ledger import Ledger
from gearbox.difficulty import DifficultyEstimate
from gearbox.difficulty.heuristic import HeuristicEstimator
from gearbox.providers import Provider

JUDGE_PROMPT = (
    "Here is a programming task. Do not solve it. Rate how hard it is for a small language model "
    "to solve correctly on the first try, judged by hidden tests that include edge cases: 1 = trivial, "
    "10 = very hard. Reply with only the number.\n\n{task}"
)
MAX_PROMPT_CHARS = 6000
_NUMBER = re.compile(r"\b(10|[1-9])\b")


def parse_rating(text: str) -> int | None:
    m = _NUMBER.search(text)
    return int(m.group(1)) if m else None


class RouterModel:
    """The exported file: judge model, rating -> difficulty, and model abilities."""

    def __init__(self, data: dict) -> None:
        self.judge_model: str = data["judge_model"]
        self.mean, self.sd = data["rating_mean"], data["rating_sd"]
        self.intercept, self.slope = data["intercept"], data["slope"]
        self.abilities: dict[str, float] = data["abilities"]
        # How much harder (+) or easier (-) your tasks are than the training benchmarks; fit it on
        # ~30 labelled tasks (bench/router_train.py --export --fit-offset). Without it, the bar
        # can be off on a new kind of task even though the ranking of tasks is right.
        self.offset: float = data.get("offset", 0.0)
        self.trained_on: list[str] = data.get("trained_on", [])

    @classmethod
    def load(cls, path: str | Path) -> RouterModel:
        return cls(json.loads(Path(path).expanduser().read_text()))

    def difficulty(self, rating: float) -> float:
        return self.intercept + self.slope * (rating - self.mean) / self.sd + self.offset

    def pass_prob(self, model: str, rating: float) -> float | None:
        a = self.abilities.get(model)
        return None if a is None else 1 / (1 + math.exp(-(a - self.difficulty(rating))))


class LearnedEstimator:
    def __init__(self, config: GearboxConfig, provider: Provider, model: RouterModel,
                 ledger: Ledger | None = None) -> None:
        self.config, self.provider, self.model, self.ledger = config, provider, model, ledger
        judge = next((t for t in config.tiers if t.model == model.judge_model), None)
        if judge is None:
            raise ValueError(f"router model needs judge {model.judge_model!r}, which is not a tier in this config")
        self.judge = judge
        self._fallback = HeuristicEstimator()

    async def estimate(self, prompt: str) -> DifficultyEstimate:
        messages = [{"role": "user", "content": JUDGE_PROMPT.format(task=prompt[:MAX_PROMPT_CHARS])}]
        try:
            completion = await self.provider.complete(self.judge, messages, max_tokens=8, temperature=0)
        except Exception as e:  # judge down -> never block routing
            est = self._fallback.estimate_sync(prompt)
            return DifficultyEstimate(est.score, est.confidence, f"judge error ({e}); heuristic: {est.rationale}")
        if self.ledger is not None:
            self.ledger.record(self.judge, completion, role="judge")
        rating = parse_rating(completion.text)
        if rating is None:
            est = self._fallback.estimate_sync(prompt)
            return DifficultyEstimate(est.score, est.confidence, f"unparseable judge reply; heuristic: {est.rationale}")
        probs = tuple(self.model.pass_prob(t.model, rating) for t in self.config.tiers)
        return DifficultyEstimate(
            score=(rating - 1) / 9, confidence=1.0, rationale=f"judge {self.judge.name} rated {rating}/10",
            pass_prob=probs,
        )
