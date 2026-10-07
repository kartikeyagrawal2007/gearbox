"""Delegation modes: async (start at once) vs burst (hold until the host awaits)."""

import asyncio
import dataclasses
import time

import pytest
from fakes import FakeProvider, make_config

from gearbox.config import GearboxConfig, Tier
from gearbox.delegate.runtime import DelegationRuntime, TaskState


def ollama_config(**kw) -> GearboxConfig:
    tiers = (Tier("w", "ollama_chat/qwen3.5:4b"), Tier("x", "ollama_chat/qwen3.5:9b", api_base="http://127.0.0.1:11434"))
    return GearboxConfig(tiers=tiers, **kw)


def test_auto_mode_bursts_only_when_the_host_shares_the_workers_ollama():
    assert ollama_config().effective_delegation_mode == "async"  # host unknown: assume a cloud host
    assert ollama_config(host_model="anthropic/claude-sonnet").effective_delegation_mode == "async"
    assert ollama_config(host_model="ollama_chat/qwen3.5:27b").effective_delegation_mode == "burst"
    remote = ollama_config(host_model="ollama_chat/qwen3.5:27b", host_api_base="http://gpu-box:11434")
    assert remote.effective_delegation_mode == "async"
    assert ollama_config(delegation_mode="burst").effective_delegation_mode == "burst"
    with pytest.raises(ValueError):
        ollama_config(delegation_mode="sometimes")


def test_host_model_is_read_from_the_host_block():
    config = GearboxConfig.from_dict({"tiers": [{"name": "w", "model": "ollama_chat/qwen3.5:4b"}],
                                      "host": {"model": "ollama_chat/qwen3.5:27b", "input_price": 1.0}})
    assert config.host_model == "ollama_chat/qwen3.5:27b" and config.host.input == 1.0
    assert config.effective_delegation_mode == "burst"


def burst_runtime(delay=0.2, **kw):
    config = dataclasses.replace(make_config(1), delegation_mode="burst", **kw)
    provider = FakeProvider({"t0": "done"}, delay=delay)
    return DelegationRuntime(config, provider), provider


def test_burst_holds_subtasks_until_the_host_awaits_then_runs_them_together():
    async def scenario():
        rt, provider = burst_runtime()
        a, b = rt.delegate("first"), rt.delegate("second")
        await asyncio.sleep(0.3)  # the host works; nothing may run on the shared GPU meanwhile
        held = (a.state, b.state, list(provider.calls))
        start = time.perf_counter()
        await rt.await_result(a.id, timeout_s=None)
        await rt.await_result(b.id, timeout_s=None)
        return held, time.perf_counter() - start, a.state, b.state, rt.stats()["delegation_mode"]

    held, waited, a_state, b_state, mode = asyncio.run(scenario())
    assert held == (TaskState.QUEUED, TaskState.QUEUED, [])
    assert waited < 0.35  # both ran at once (0.2 s each), not one after the other
    assert a_state == b_state == TaskState.DONE and mode == "burst"


def test_burst_starts_held_subtasks_anyway_if_the_host_never_awaits():
    async def scenario():
        rt, _ = burst_runtime(delay=0.0, burst_max_wait_s=0.1)
        dt = rt.delegate("fire and forget")
        await asyncio.sleep(0.3)
        return dt.state

    assert asyncio.run(scenario()) == TaskState.DONE


def test_cancelling_a_held_subtask_never_calls_a_model():
    async def scenario():
        rt, provider = burst_runtime()
        dt = rt.delegate("not needed after all")
        cancelled = rt.cancel(dt.id)
        other = rt.delegate("needed")
        await rt.await_result(other.id, timeout_s=None)
        return cancelled, dt.state, provider.calls

    cancelled, state, calls = asyncio.run(scenario())
    assert cancelled and state == TaskState.CANCELLED and calls == ["t0"]  # only the second subtask ran


def test_async_mode_starts_at_once():
    async def scenario():
        config = dataclasses.replace(make_config(1), delegation_mode="async")
        provider = FakeProvider({"t0": "done"}, delay=0.0)
        rt = DelegationRuntime(config, provider)
        dt = rt.delegate("now")
        await asyncio.sleep(0.05)
        return dt.state, rt.mode

    assert asyncio.run(scenario()) == (TaskState.DONE, "async")
