import asyncio

from fakes import FakeProvider, make_config

from gearbox.race import Race


def test_async_overlaps_when_backend_is_concurrent():
    config = make_config(max_concurrent=4)
    provider = FakeProvider(delay=0.15)
    race = Race(config, provider, subtasks=["a", "b"], host_steps=["x", "y"])
    asyncio.run(race.run())
    snap = race.snapshot()

    # one untimed warm-up per model, before either phase
    assert provider.calls[:2] == ["t2", "t0"]

    assert snap["status"] == "done"
    blocking, async_ = snap["phases"]["blocking"], snap["phases"]["async"]
    # blocking: 4 sequential calls (~0.6s); async: host steps overlap the workers (~0.3s)
    assert blocking["wall_s"] >= 0.55
    assert async_["wall_s"] < 0.45
    assert snap["speedup"] > 1.3
    assert async_["host_blocked_s"] < blocking["host_blocked_s"]
    # every model call shows up on a timeline, and both modes made the same calls
    for phase in (blocking, async_):
        assert len(phase["workers"]) == 2
        assert [s["kind"] for s in phase["host"]].count("work") == 2
    assert snap["host_tier"] == "t2" and snap["worker_tier"] == "t0"


def test_race_reports_failure():
    race = Race(make_config(), FakeProvider({"t2": ConnectionError("host down")}), subtasks=["a"], host_steps=["x"])
    asyncio.run(race.run())
    snap = race.snapshot()
    assert snap["status"] == "failed"
    assert "host down" in snap["error"]
