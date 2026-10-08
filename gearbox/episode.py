"""One host episode, recorded for the dashboard's live view.

Three schedules, so the dashboard can compare them on the same batch:
  blocking  delegate one subtask, wait for it, do one host step; repeat
  async     delegate all, do the host steps while the workers run, then collect
  burst     delegate all, but the runtime holds them until the host waits; then they run together
Every model call is recorded in the shared ledger (workers as delegations, host steps as "direct"). Everything the screen shows comes from the
real runtime: each subtask's routing decision (pass chance per tier from the learned
router, the check's strength), every attempt with its tier, timing and check result,
escalations, and the host's own work and waiting, all on one clock.
"""

from __future__ import annotations

import dataclasses
import time
import uuid

from gearbox.config import GearboxConfig
from gearbox.cost.ledger import Ledger
from gearbox.delegate.runtime import TERMINAL, DelegationRuntime
from gearbox.providers import Provider
from gearbox.verify.strength import check_strength, count_cases

CALL_PARAMS = {"temperature": 0}


class Episode:
    def __init__(self, config: GearboxConfig, provider: Provider, subtasks: list[dict], host_steps: list[str],
                 host_tier: str | int | None = None, mode: str | None = None, ledger: Ledger | None = None) -> None:
        if not subtasks:
            raise ValueError("add at least one subtask")
        self.id = uuid.uuid4().hex[:8]
        if not config.code_checks:  # checks run model-written code; respect the config's opt-in
            subtasks = [{**s, "check": ""} for s in subtasks]
        if mode not in (None, "auto", "async", "burst", "blocking"):
            raise ValueError(f"mode must be auto, async, burst or blocking, got {mode!r}")
        self.blocking = mode == "blocking"
        runtime_mode = "async" if self.blocking else mode
        self.config = dataclasses.replace(config, delegation_mode=runtime_mode) if runtime_mode and runtime_mode != "auto" else config
        self.runtime = DelegationRuntime(self.config, provider, ledger=ledger, call_params=CALL_PARAMS)
        self.provider = provider
        self.subtasks = subtasks
        self.host_steps = [s for s in host_steps if s.strip()]
        self.host = self.config.tiers[self.config.tier_index(host_tier) if host_tier is not None else self.config.top]
        self.task_ids: list[str] = []
        self.spans: list[list] = []  # [kind, label, start, end or None]
        self.host_tokens = 0
        self.t0: float | None = None
        self.t1: float | None = None
        self.status, self.error = "pending", None

    async def run(self) -> None:
        self.status, self.t0 = "running", time.time()
        try:
            if self.blocking:  # one subtask at a time, with a host step after each
                for i, s in enumerate(self.subtasks):
                    self._delegate(s)
                    await self._wait(self.task_ids[-1])
                    if i < len(self.host_steps):
                        await self._host_step(self.host_steps[i])
                for step in self.host_steps[len(self.subtasks):]:
                    await self._host_step(step)
            else:
                for s in self.subtasks:
                    self._delegate(s)
                for step in self.host_steps:
                    await self._host_step(step)
                for tid in self.task_ids:
                    await self._wait(tid)
            self.status = "done"
        except Exception as e:
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            self.t1 = time.time()

    def _delegate(self, s: dict) -> None:
        self.task_ids.append(self.runtime.delegate(s["task"], check=s.get("check", ""),
                                                   acceptance=s.get("acceptance", "")).id)

    async def _host_step(self, step: str) -> None:
        span = ["work", step, time.time(), None]
        self.spans.append(span)
        c = await self.provider.complete(self.host, [{"role": "user", "content": step}],
                                         max_tokens=256, timeout=300, **CALL_PARAMS)
        self.runtime.ledger.record(self.host, c, role="direct")
        self.host_tokens += c.output_tokens
        span[3] = time.time()

    async def _wait(self, task_id: str) -> None:
        span = ["blocked", "waiting for workers", time.time(), None]
        self.spans.append(span)
        await self.runtime.await_result(task_id, timeout_s=None)
        span[3] = time.time()

    def snapshot(self) -> dict:
        now = time.time()
        t0 = self.t0 or now
        rel = lambda t: round((now if t is None else t) - t0, 2)  # noqa: E731
        tasks = []
        for tid, spec in zip(self.task_ids, self.subtasks):
            dt = self.runtime.task(tid)
            view = dt.view(include_result=True)
            attempts = [{"tier": a.tier, "outcome": a.outcome, "start": rel(a.started_at),
                         "end": rel(a.started_at + a.latency_s), "tokens": a.output_tokens,
                         "check": (a.check or {}).get("output", "")[-300:] if a.check else None}
                        for a in dt.attempts]
            if dt.active:
                attempts.append({"tier": dt.active[0], "outcome": "running", "start": rel(dt.active[1]),
                                 "end": rel(None), "tokens": 0, "check": None})
            accepted = dt.attempts[-1].output_tokens if dt.attempts and dt.state.value == "done" else 0
            tasks.append({
                "task_id": tid, "task": dt.task, "state": dt.state.value, "routing": view["routing"],
                "check_cases": None if not dt.check else (lambda n: None if n == float("inf") else int(n))(count_cases(dt.check)),
                "check_strength": check_strength(dt.check) if dt.check else None,
                "verified": view["verified"], "attempts": attempts, "result": view.get("result"),
                "error": view.get("error"),
                # Had the host written the answer itself, it would have produced these tokens;
                # instead it wrote only the brief.
                "host_tokens_saved": max(0, accepted - dt.brief_tokens),
            })
        blocked = sum((s[3] or now) - s[2] for s in self.spans if s[0] == "blocked")
        return {
            "episode_id": self.id, "status": self.status, "error": self.error,
            "mode": "blocking" if self.blocking else self.runtime.mode,
            "host_tier": self.host.name, "elapsed_s": rel(self.t1) if self.t0 else 0.0,
            "host": [{"kind": k, "label": label, "start": rel(s), "end": rel(e), "open": e is None}
                     for k, label, s, e in self.spans],
            "tasks": tasks,
            "summary": {
                "done": all(self.runtime.task(t).state in TERMINAL for t in self.task_ids) if self.task_ids else False,
                "host_blocked_s": round(blocked, 2), "host_tokens": self.host_tokens,
                "host_tokens_saved": sum(t["host_tokens_saved"] for t in tasks),
                "verified": sum(t["verified"] is True for t in tasks),
                "escalations": sum(max(0, len([a for a in t["attempts"] if a["outcome"] != "running"]) - 1) for t in tasks),
            },
        }
