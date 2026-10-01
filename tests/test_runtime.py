import asyncio
import time

import pytest
from fakes import FakeProvider, make_config

from gearbox.delegate.runtime import DelegationRuntime, TaskState


def make_runtime(replies=None, delay=0.0, **config_overrides) -> DelegationRuntime:
    return DelegationRuntime(make_config(**config_overrides), provider=FakeProvider(replies, delay))


def test_delegate_returns_immediately_and_host_overlaps():
    async def scenario():
        rt = make_runtime(delay=0.2)
        start = time.perf_counter()
        dt = rt.delegate("Write unit tests for parse_date")
        assert time.perf_counter() - start < 0.05
        assert dt.state in (TaskState.ROUTING, TaskState.RUNNING)
        await asyncio.sleep(0.2)  # the host does independent work meanwhile
        view = await rt.await_result(dt.id)
        return view, time.perf_counter() - start, rt.stats()

    view, total, stats = asyncio.run(scenario())
    assert view["state"] == "done" and view["result"].startswith("result from")
    assert view["host_blocked_s"] < 0.1
    assert total < 0.35  # overlapped, not 0.2 + 0.2
    assert stats["async"]["overlap_ratio"] > 0.5
    assert stats["async"]["awaited_tasks"] == 1


def test_sync_run_blocks_for_whole_task():
    async def scenario():
        rt = make_runtime(delay=0.1)
        return await rt.run("Summarise this changelog")

    dt = asyncio.run(scenario())
    assert dt.state is TaskState.DONE
    assert dt.blocked_s >= 0.09


def test_unsure_escalates_one_tier():
    async def scenario():
        rt = make_runtime({"t0": "UNSURE: need the schema"}, max_escalations=1)
        dt = await rt.run("Write the migration", tier="t0")
        return dt, rt

    dt, rt = asyncio.run(scenario())
    assert dt.state is TaskState.DONE
    assert dt.result == "result from t1"
    assert [a.outcome for a in dt.attempts] == ["unsure", "ok"]
    assert [r.ok for r in rt.ledger.records] == [False, True]


def test_unsure_without_escalation_fails_but_keeps_text():
    async def scenario():
        rt = make_runtime({"t0": "UNSURE: need the schema"}, max_escalations=0)
        return await rt.run("Write the migration", tier="t0")

    dt = asyncio.run(scenario())
    assert dt.state is TaskState.FAILED
    assert dt.result.startswith("UNSURE")
    assert "UNSURE" in dt.error


def test_timeouts_escalate_then_fail():
    async def scenario():
        rt = make_runtime(delay=0.3, max_escalations=1)
        return await rt.run("Slow task", tier="t0", timeout_s=0.05)

    dt = asyncio.run(scenario())
    assert dt.state is TaskState.FAILED
    assert [(a.tier, a.outcome) for a in dt.attempts] == [("t0", "timeout"), ("t1", "timeout")]


def test_provider_error_escalates():
    async def scenario():
        rt = make_runtime({"t0": ConnectionError("ollama down")}, max_escalations=1)
        return await rt.run("Anything", tier="t0")

    dt = asyncio.run(scenario())
    assert dt.state is TaskState.DONE
    assert [a.outcome for a in dt.attempts] == ["error", "ok"]


def test_await_timeout_returns_running_then_cancel():
    async def scenario():
        rt = make_runtime(delay=5)
        dt = rt.delegate("Very slow task")
        running = await rt.await_result(dt.id, timeout_s=0.05)
        assert rt.cancel(dt.id)
        final = await rt.await_result(dt.id)
        return running, final

    running, final = asyncio.run(scenario())
    assert running["state"] == "running" and "result" not in running
    assert final["state"] == "cancelled"


def test_routing_uses_difficulty_and_risk():
    async def scenario():
        rt = make_runtime(risk_scaled_leverage=True)
        dt = await rt.run("Fix the typo in this docstring", risk="high")
        return dt

    dt = asyncio.run(scenario())
    assert dt.decision.leverage >= 2
    assert dt.attempts[0].tier == "t2"


def test_bad_inputs_rejected_up_front():
    async def scenario():
        rt = make_runtime()
        with pytest.raises(ValueError):
            rt.delegate("   ")
        with pytest.raises(ValueError):
            rt.delegate("task", risk="extreme")
        with pytest.raises(KeyError):
            rt.delegate("task", tier="nope")
        with pytest.raises(KeyError):
            rt.status("missing")

    asyncio.run(scenario())


def test_overlap_ignores_tasks_nobody_waited_for():
    async def scenario():
        rt = make_runtime()
        dt = rt.delegate("Fire and forget")
        await asyncio.wait({dt.job})
        return rt.stats()["async"]

    stats = asyncio.run(scenario())
    assert stats["awaited_tasks"] == 0
    assert stats["overlap_ratio"] is None
