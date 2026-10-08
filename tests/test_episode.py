"""gearbox/episode.py and the dashboard's /api/episode."""

import asyncio
import dataclasses

from fakes import FakeProvider, make_config
from starlette.testclient import TestClient

from gearbox.episode import Episode
from gearbox.ui.server import create_app


def test_episode_records_routing_attempts_checks_and_host_work():
    config = dataclasses.replace(make_config(3), code_checks=True, max_escalations=2, leverage=0,
                                 check_leverage={})
    replies = {"t0": "def f():\n    return 0", "t1": "def f():\n    return 1", "t2": "host notes"}
    ep = Episode(config, FakeProvider(replies, delay=0.05),
                 [{"task": "Write f() returning 1.", "check": "assert f() == 1"}], ["plan the rest"], mode="async")
    asyncio.run(ep.run())
    snap = ep.snapshot()
    task = snap["tasks"][0]
    assert snap["status"] == "done" and snap["mode"] == "async" and snap["host_tier"] == "t2"
    assert [a["tier"] for a in task["attempts"]] == ["t0", "t1"]           # failed check -> escalated
    assert [a["outcome"] for a in task["attempts"]] == ["check_failed", "ok"]
    assert task["verified"] is True and task["check_cases"] == 1 and task["check_strength"] == "weak"
    assert snap["summary"]["escalations"] == 1 and snap["summary"]["verified"] == 1
    assert [s["kind"] for s in snap["host"]] == ["work", "blocked"]


def test_episode_endpoint_runs_and_reports():
    config = dataclasses.replace(make_config(2), code_checks=True)
    app = create_app(config, provider=FakeProvider({"t0": "def g():\n    return 2"}), simulated=True)
    with TestClient(app) as client:
        r = client.post("/api/episode", json={"subtasks": [{"task": "Write g().", "check": "assert g() == 2"}]})
        assert r.status_code == 200
        snap = client.get(f"/api/episode/{r.json()['episode_id']}").json()
        assert snap["tasks"][0]["task"] == "Write g()."
        assert client.post("/api/episode", json={"subtasks": []}).status_code == 400


def test_blocking_waits_for_each_subtask_and_everything_lands_in_the_shared_ledger():
    from gearbox.cost.ledger import Ledger
    config = dataclasses.replace(make_config(3), code_checks=True)
    ledger = Ledger(config.host)
    replies = {"t0": "def f():\n    return 1", "t1": "def f():\n    return 1", "t2": "notes"}
    subtasks = [{"task": "Write f().", "check": "assert f() == 1"}] * 2
    ep = Episode(config, FakeProvider(replies, delay=0.05), subtasks, ["step a", "step b"],
                 mode="blocking", ledger=ledger)
    asyncio.run(ep.run())
    snap = ep.snapshot()
    assert snap["mode"] == "blocking"
    assert [s["kind"] for s in snap["host"]] == ["blocked", "work", "blocked", "work"]  # wait, work, wait, work
    roles = sorted(r.role for r in ledger.records)
    assert roles.count("direct") == 2 and len(roles) >= 4      # 2 host steps + the workers' calls


def test_unknown_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        Episode(make_config(2), FakeProvider(), [{"task": "x"}], [], mode="sometimes")
