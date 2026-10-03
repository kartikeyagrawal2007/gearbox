"""Per-call accounting of every token Gearbox spends, plus a conservative
estimate of what the host saved by delegating (same cost model as breakeven.py)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from gearbox.config import Pricing, Tier
from gearbox.providers import Completion

SAVINGS_ASSUMPTIONS = (
    "Estimate. Direct = host writes the worker's output itself, with context already in its window. "
    "Delegation = host writes the brief (~4 chars/token) and reads the result, plus all worker "
    "attempts. Failed tasks count as pure overhead. Judge calls count as routing overhead."
)


@dataclass(frozen=True)
class CallRecord:
    timestamp: float
    tier: str
    role: str  # "delegate" | "judge" | "direct"
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_s: float
    cost_usd: float
    task_id: str | None = None
    ok: bool = True
    brief_tokens: int = 0  # host-written tokens that produced this call (delegate only)
    check_passed: bool | None = None  # None = no executable check ran on this call


class Ledger:
    def __init__(self, host: Pricing | None = None, path: str | Path | None = None) -> None:
        self.host = host or Pricing()
        self._records: list[CallRecord] = []
        self._path = Path(path) if path else None

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self._records)

    def record(
        self,
        tier: Tier,
        completion: Completion,
        role: str,
        task_id: str | None = None,
        ok: bool = True,
        brief_tokens: int = 0,
        check_passed: bool | None = None,
    ) -> CallRecord:
        rec = CallRecord(
            timestamp=time.time(),
            tier=tier.name,
            role=role,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cached_tokens=completion.cached_tokens,
            latency_s=round(completion.latency_s, 4),
            cost_usd=tier.pricing.cost(completion.input_tokens, completion.output_tokens, completion.cached_tokens),
            task_id=task_id,
            ok=ok,
            brief_tokens=brief_tokens,
            check_passed=check_passed,
        )
        self._records.append(rec)
        if self._path:
            with self._path.open("a") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def summary(self) -> dict:
        by_tier: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in self._records:
            t = by_tier[r.tier]
            t["calls"] += 1
            t["input_tokens"] += r.input_tokens
            t["output_tokens"] += r.output_tokens
            t["cached_tokens"] += r.cached_tokens
            t["cost_usd"] += r.cost_usd
            if r.check_passed is not None:
                t["checked"] += 1
                t["check_failures"] += not r.check_passed

        tasks: dict[str, list[CallRecord]] = defaultdict(list)
        for r in self._records:
            if r.role == "delegate" and r.task_id:
                tasks[r.task_id].append(r)

        savings = 0.0
        succeeded = 0
        for attempts in tasks.values():
            worker_cost = sum(a.cost_usd for a in attempts)
            final = attempts[-1]
            handoff = self.host.cost(input_tokens=final.output_tokens, output_tokens=attempts[0].brief_tokens)
            direct = self.host.cost(input_tokens=0, output_tokens=final.output_tokens) if final.ok else 0.0
            succeeded += final.ok
            savings += direct - handoff - worker_cost
        judge_cost = sum(r.cost_usd for r in self._records if r.role == "judge")
        savings -= judge_cost

        return {
            "calls": len(self._records),
            "total_cost_usd": round(sum(r.cost_usd for r in self._records), 6),
            "judge_overhead_usd": round(judge_cost, 6),
            "by_tier": {name: {k: round(v, 6) for k, v in t.items()} for name, t in by_tier.items()},
            "delegated_tasks": len(tasks),
            "delegated_tasks_succeeded": succeeded,
            "estimated_host_savings_usd": round(savings, 6),
            # Share of checked answers that claimed success but failed their check.
            "false_done_rate": {
                name: round(t["check_failures"] / t["checked"], 3) for name, t in by_tier.items() if t.get("checked")
            },
            "savings_assumptions": SAVINGS_ASSUMPTIONS,
        }
