"""Asynchronous delegation: hand a subtask to a cheaper tier and keep working.

`delegate()` schedules the subtask and returns at once. The host calls
`await_result()` only when it needs the output. Time spent blocked there is
recorded per task: `blocked_s` versus the task's own run time measures how
much of the worker's time the host overlapped with useful work.

If a worker replies UNSURE, errors, or times out, the subtask moves up one
tier (at most `config.max_escalations` times).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

from gearbox.config import RISK_LEVELS, GearboxConfig, Tier
from gearbox.cost.ledger import Ledger
from gearbox.delegate.brief import build_messages, estimate_tokens, is_unsure
from gearbox.providers import Completion, LiteLLMProvider, Provider
from gearbox.router import Router, RoutingDecision


class TaskState(str, Enum):
    ROUTING = "routing"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = frozenset({TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED})


@dataclass
class Attempt:
    tier: str
    outcome: str  # "ok" | "unsure" | "error" | "timeout"
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float = 0.0  # epoch seconds, after a concurrency slot was acquired


@dataclass
class DelegatedTask:
    id: str
    task: str
    brief_tokens: int
    created_at: float
    state: TaskState = TaskState.ROUTING
    decision: RoutingDecision | None = None
    attempts: list[Attempt] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    routed_at: float | None = None
    finished_at: float | None = None
    blocked_s: float = 0.0  # host wall time spent inside await_result for this task
    awaited: bool = False  # the host asked for the result at least once
    active: tuple[str, float] | None = None  # (tier, started_at) of the attempt in flight
    job: asyncio.Task | None = field(default=None, repr=False)

    @property
    def run_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.created_at

    def view(self, include_result: bool = True) -> dict:
        d = {
            "task_id": self.id,
            "task": self.task,
            "state": self.state.value,
            "tier": self.attempts[-1].tier if self.attempts else (self.decision.tier_name if self.decision else None),
            "routing": self.decision.as_dict() if self.decision else None,
            "attempts": [asdict(a) for a in self.attempts],
            "run_s": round(self.run_s, 3),
            "host_blocked_s": round(self.blocked_s, 3),
            "created_at": self.created_at,
            "routed_at": self.routed_at,
            "finished_at": self.finished_at,
            "active": {"tier": self.active[0], "since": self.active[1]} if self.active else None,
        }
        if include_result and self.state in TERMINAL:
            d["result"] = self.result
            d["error"] = self.error
        return d


class DelegationRuntime:
    def __init__(
        self,
        config: GearboxConfig,
        provider: Provider | None = None,
        ledger: Ledger | None = None,
        router: Router | None = None,
        call_params: dict | None = None,
    ) -> None:
        self.config = config
        self.call_params = call_params or {}  # extra provider kwargs for every worker call, e.g. temperature
        self.provider = provider or LiteLLMProvider()
        self.ledger = ledger or Ledger(config.host)
        self.router = router or Router.from_config(config, self.provider, self.ledger)
        self._tasks: dict[str, DelegatedTask] = {}
        self._slots = asyncio.Semaphore(config.max_concurrent)

    def delegate(
        self,
        task: str,
        context: str = "",
        acceptance: str = "",
        risk: str | None = None,
        tier: str | int | None = None,
        leverage: int | None = None,
        timeout_s: float | None = None,
    ) -> DelegatedTask:
        """Schedule a subtask and return immediately. Must be called inside a running event loop."""
        if not task.strip():
            raise ValueError("task must not be empty")
        if risk is not None and risk not in RISK_LEVELS:
            raise ValueError(f"risk must be one of {RISK_LEVELS}, got {risk!r}")
        forced =self.config.tier_index(tier) if tier is not None else None
        messages = build_messages(task, context, acceptance)
        dt = DelegatedTask(
            id=uuid.uuid4().hex[:8],
            task=task,
            brief_tokens=estimate_tokens(task + context + acceptance),
            created_at=time.time(),
        )
        timeout = timeout_s or self.config.task_timeout_s
        dt.job = asyncio.get_running_loop().create_task(
            self._run(dt, messages, risk, forced, leverage, timeout), name=f"gearbox-{dt.id}"
        )
        self._tasks[dt.id] = dt
        return dt

    async def run(self, task: str, **kwargs) -> DelegatedTask:
        """Blocking delegation: the synchronous baseline."""
        dt = self.delegate(task, **kwargs)
        await self.await_result(dt.id, timeout_s=None)
        return dt

    def status(self, task_id: str) -> dict:
        return self._get(task_id).view(include_result=False)

    def task(self, task_id: str) -> DelegatedTask:
        return self._get(task_id)

    def list_tasks(self, include_result: bool = False) -> list[dict]:
        return [t.view(include_result=include_result) for t in self._tasks.values()]

    async def await_result(self, task_id: str, timeout_s: float | None = 60.0) -> dict:
        """Wait up to `timeout_s` (None = forever). Returns the task view, finished or not."""
        dt = self._get(task_id)
        dt.awaited = True
        if dt.job is not None and not dt.job.done():
            start = time.perf_counter()
            await asyncio.wait({dt.job}, timeout=timeout_s)
            dt.blocked_s += time.perf_counter() - start
        return dt.view()

    def cancel(self, task_id: str) -> bool:
        dt = self._get(task_id)
        if dt.job is None or dt.job.done():
            return False
        return dt.job.cancel()

    def stats(self) -> dict:
        # Overlap only means something for results the host consumed; fire-and-forget
        # tasks would otherwise count as perfect overlap.
        finished = [t for t in self._tasks.values() if t.state in TERMINAL and t.awaited]
        run = sum(t.run_s for t in finished)
        blocked = sum(t.blocked_s for t in finished)
        return {
            **self.ledger.summary(),
            "async": {
                "awaited_tasks": len(finished),
                "worker_run_s": round(run, 3),
                "host_blocked_s": round(blocked, 3),
                # Share of worker run time the host spent doing something else.
                "overlap_ratio": round(1 - blocked / run, 3) if run > 0 else None,
            },
        }

    def _get(self, task_id: str) -> DelegatedTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(f"unknown task_id {task_id!r}") from None

    async def _run(
        self,
        dt: DelegatedTask,
        messages: list[dict[str, str]],
        risk: str | None,
        forced: int | None,
        leverage: int | None,
        timeout: float,
    ) -> None:
        try:
            if forced is not None:
                dt.decision = RoutingDecision(
                    score=0.0, confidence=1.0, base_tier=forced, leverage=0, tier=forced,
                    tier_name=self.config.tiers[forced].name, rationale="tier chosen by caller",
                )
            else:
                dt.decision = await self.router.route(messages[-1]["content"], risk=risk, leverage=leverage)
            dt.routed_at = time.time()
            dt.state = TaskState.RUNNING

            tier_idx = dt.decision.tier
            escalations = self.config.max_escalations
            while True:
                outcome, completion = await self._attempt(dt, self.config.tiers[tier_idx], messages, timeout)
                if outcome == "ok":
                    dt.result, dt.error, dt.state = completion.text, None, TaskState.DONE
                    return
                if escalations > 0 and tier_idx < self.config.top:
                    escalations -= 1
                    tier_idx += 1
                    continue
                if outcome == "unsure":
                    dt.result = completion.text
                    dt.error = "worker reported UNSURE and no escalation was left"
                dt.state = TaskState.FAILED
                return
        except asyncio.CancelledError:
            dt.state = TaskState.CANCELLED
            raise
        except Exception as e:
            dt.state = TaskState.FAILED
            dt.error = f"{type(e).__name__}: {e}"
        finally:
            dt.finished_at = time.time()

    async def _attempt(
        self, dt: DelegatedTask, tier: Tier, messages: list[dict[str, str]], timeout: float
    ) -> tuple[str, Completion | None]:
        async with self._slots:
            began = time.time()
            start = time.perf_counter()
            dt.active = (tier.name, began)
            try:
                completion = await asyncio.wait_for(self.provider.complete(tier, messages, **self.call_params), timeout)
            except asyncio.TimeoutError:
                dt.attempts.append(Attempt(tier.name, "timeout", time.perf_counter() - start, started_at=began))
                dt.error = f"timed out after {timeout}s on {tier.name}"
                return "timeout", None
            except Exception as e:
                dt.attempts.append(Attempt(tier.name, "error", time.perf_counter() - start, started_at=began))
                dt.error = f"{type(e).__name__}: {e}"
                return "error", None
            finally:
                dt.active = None

        unsure = is_unsure(completion.text)
        outcome = "unsure" if unsure else "ok"
        self.ledger.record(
            tier, completion, role="delegate", task_id=dt.id, ok=not unsure, brief_tokens=dt.brief_tokens
        )
        dt.attempts.append(
            Attempt(
                tier.name, outcome, round(time.perf_counter() - start, 4),
                completion.input_tokens, completion.output_tokens, started_at=began,
            )
        )
        return outcome, completion
