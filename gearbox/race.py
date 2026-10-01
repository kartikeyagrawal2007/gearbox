"""Blocking vs async delegation on the same workload, with measured timelines.

The host tier does its own independent steps while subtasks go to the worker tier.
  blocking: delegate one subtask, wait for it, do one host step, repeat.
  async:    delegate every subtask up front, do the host steps, then collect results.
Both modes make the same model calls, so a wall-clock difference comes from overlap
(or, on a shared GPU, from contention eating it). Every model is loaded with an
untimed warm-up call first, so neither phase pays the cold start.
Remaining caveat: the async phase runs second, so backend prompt caches can favour it
slightly; the benchmark harness should alternate order across repeated rounds.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from gearbox.config import GearboxConfig
from gearbox.cost.ledger import Ledger
from gearbox.delegate.runtime import DelegationRuntime
from gearbox.providers import Provider

DEFAULT_SUBTASKS = [
    "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Code only.",
    "Write three pytest test functions for a function add(a, b) that adds two numbers. Code only.",
]
DEFAULT_HOST_STEPS = [
    "In two sentences: when should a cache use LRU instead of LFU eviction?",
    "In two sentences: what is the trade-off between optimistic and pessimistic locking?",
]
MODES = ("blocking", "async")
# Greedy decoding so both phases see the same answers as far as the backend allows;
# a sampled UNSURE in one phase but not the other changes the work done.
CALL_PARAMS = {"temperature": 0}


@dataclass
class HostSpan:
    kind: str  # "work" | "blocked"
    label: str
    start: float
    end: float | None = None


class RacePhase:
    def __init__(self, mode: str, runtime: DelegationRuntime) -> None:
        self.mode = mode
        self.runtime = runtime
        self.spans: list[HostSpan] = []
        self.task_ids: list[str] = []
        self.t0: float | None = None
        self.t1: float | None = None

    def worker_calls(self) -> list[tuple[int, str, str]]:
        """(subtask index, tier, outcome) for every finished worker attempt."""
        return [
            (i, a.tier, a.outcome)
            for i, tid in enumerate(self.task_ids)
            for a in self.runtime.task(tid).attempts
        ]

    def snapshot(self, now: float) -> dict:
        t0 = self.t0 or now

        def rel(t: float | None) -> float:
            return round((now if t is None else t) - t0, 3)

        host = [
            {"kind": s.kind, "label": s.label, "start": rel(s.start), "end": rel(s.end), "open": s.end is None}
            for s in self.spans
        ]
        workers = []
        for tid in self.task_ids:
            dt = self.runtime.task(tid)
            for a in dt.attempts:
                workers.append({
                    "task_id": tid, "label": dt.task, "tier": a.tier, "outcome": a.outcome,
                    "start": rel(a.started_at), "end": rel(a.started_at + a.latency_s), "open": False,
                })
            if dt.active:
                workers.append({
                    "task_id": tid, "label": dt.task, "tier": dt.active[0], "outcome": "running",
                    "start": rel(dt.active[1]), "end": rel(None), "open": True,
                })
        blocked = sum((s.end or now) - s.start for s in self.spans if s.kind == "blocked")
        return {
            "mode": self.mode,
            "started": self.t0 is not None,
            "done": self.t1 is not None,
            "wall_s": rel(self.t1) if self.t0 else 0.0,
            "host_blocked_s": round(blocked, 3),
            "worker_calls": len(self.worker_calls()),
            "host": host,
            "workers": workers,
        }


class Race:
    def __init__(
        self,
        config: GearboxConfig,
        provider: Provider,
        subtasks: list[str] | None = None,
        host_steps: list[str] | None = None,
        host_tier: str | int | None = None,
        worker_tier: str | int | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.config = config
        self.provider = provider
        self.subtasks = subtasks or DEFAULT_SUBTASKS
        self.host_steps = host_steps or DEFAULT_HOST_STEPS
        self.host_tier = config.tiers[config.tier_index(host_tier) if host_tier is not None else config.top]
        self.worker_tier = config.tiers[config.tier_index(worker_tier) if worker_tier is not None else 0].name
        # Separate runtimes and ledgers so the two modes never share state.
        self.phases = {
            mode: RacePhase(mode, DelegationRuntime(config, provider, Ledger(config.host), call_params=CALL_PARAMS))
            for mode in MODES
        }
        self.status = "pending"
        self.error: str | None = None

    async def run(self) -> None:
        try:
            await self._warm_up()
            self.status = "running"
            for mode in MODES:
                await self._run_phase(self.phases[mode])
            self.status = "done"
        except Exception as e:
            self.status = "failed"
            self.error = f"{type(e).__name__}: {e}"

    async def _warm_up(self) -> None:
        """Load each model before timing anything. On local backends the first call
        can take 10+ s to load weights, which would be charged to whichever phase ran first."""
        self.status = "warming up"
        worker = self.config.tiers[self.config.tier_index(self.worker_tier)]
        seen: set[str] = set()
        for tier in (self.host_tier, worker):
            if tier.model not in seen:
                seen.add(tier.model)
                await self.provider.complete(tier, [{"role": "user", "content": "Reply with OK."}], max_tokens=1)

    async def _run_phase(self, phase: RacePhase) -> None:
        phase.t0 = time.time()
        rt = phase.runtime
        if phase.mode == "blocking":
            for i in range(max(len(self.subtasks), len(self.host_steps))):
                if i < len(self.subtasks):
                    phase.task_ids.append(rt.delegate(self.subtasks[i], tier=self.worker_tier).id)
                    await self._wait(phase, phase.task_ids[-1])
                if i < len(self.host_steps):
                    await self._host_step(phase, self.host_steps[i])
        else:
            for subtask in self.subtasks:
                phase.task_ids.append(rt.delegate(subtask, tier=self.worker_tier).id)
            for step in self.host_steps:
                await self._host_step(phase, step)
            for tid in phase.task_ids:
                await self._wait(phase, tid)
        phase.t1 = time.time()

    async def _host_step(self, phase: RacePhase, prompt: str) -> None:
        span = HostSpan("work", prompt, time.time())
        phase.spans.append(span)
        completion = await self.provider.complete(self.host_tier, [{"role": "user", "content": prompt}], **CALL_PARAMS)
        phase.runtime.ledger.record(self.host_tier, completion, role="direct")
        span.end = time.time()

    async def _wait(self, phase: RacePhase, task_id: str) -> None:
        span = HostSpan("blocked", "waiting for worker", time.time())
        phase.spans.append(span)
        await phase.runtime.await_result(task_id, timeout_s=None)
        span.end = time.time()

    def snapshot(self) -> dict:
        now = time.time()
        phases = {mode: p.snapshot(now) for mode, p in self.phases.items()}
        b, a = phases["blocking"], phases["async"]
        finished = b["done"] and a["done"]
        speedup = round(b["wall_s"] / a["wall_s"], 2) if finished and a["wall_s"] > 0 else None
        # Only a fair comparison if both phases made the same worker calls with the same
        # outcomes; otherwise one phase did extra work (e.g. an escalation) and the ratio lies.
        comparable = (
            self.phases["blocking"].worker_calls() == self.phases["async"].worker_calls() if finished else None
        )
        # Async can at best remove the time the host spent blocked, so from the blocking
        # trace alone: speedup <= wall / (wall - blocked). Contention only lowers it.
        ceiling = (
            round(b["wall_s"] / (b["wall_s"] - b["host_blocked_s"]), 2)
            if b["done"] and b["wall_s"] - b["host_blocked_s"] > 0 else None
        )
        return {
            "race_id": self.id,
            "status": self.status,
            "error": self.error,
            "host_tier": self.host_tier.name,
            "worker_tier": self.worker_tier,
            "subtasks": self.subtasks,
            "host_steps": self.host_steps,
            "phases": phases,
            "speedup": speedup,
            "comparable": comparable,
            "speedup_ceiling": ceiling,
        }
