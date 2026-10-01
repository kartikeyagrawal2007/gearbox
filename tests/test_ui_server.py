import time

from fakes import FakeProvider, make_config
from starlette.testclient import TestClient

from gearbox.ui.server import create_app


def wait_for(client, path, done, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(path).json()
        if done(data):
            return data
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting on {path}: {data}")


def make_client():
    return TestClient(create_app(make_config(), provider=FakeProvider(delay=0.05)))


def test_dashboard_page_and_config():
    with make_client() as client:
        page = client.get("/")
        assert "Gearbox" in page.text
        assert page.headers["cache-control"] == "no-cache"
        cfg = client.get("/api/config").json()
        assert [t["name"] for t in cfg["tiers"]] == ["t0", "t1", "t2"]
        assert cfg["simulated"] is False


def test_route_delegate_ledger_flow():
    with make_client() as client:
        decision = client.post("/api/route", json={"prompt": "Fix the typo", "risk": "low"}).json()
        assert decision["tier"] in {"t0", "t1", "t2"}

        task_id = client.post("/api/delegate", json={"task": "Write docs", "tier": "t0"}).json()["task_id"]
        tasks = wait_for(client, "/api/tasks", lambda d: d["tasks"][0]["state"] == "done")
        view = tasks["tasks"][0]
        assert view["task_id"] == task_id and view["result"] == "result from t0"
        assert view["attempts"][0]["started_at"] > 0

        ledger = client.get("/api/ledger").json()
        assert ledger["delegated_tasks_succeeded"] == 1


def test_bad_input_is_a_400_with_reason():
    with make_client() as client:
        res = client.post("/api/delegate", json={"task": "x", "tier": "nope"})
        assert res.status_code == 400 and "unknown tier" in res.json()["error"]
        res = client.post("/api/route", json={"prompt": "x", "risk": "extreme"})
        assert res.status_code == 400
        assert client.get("/api/race/missing").status_code == 400


def test_race_endpoint_runs_to_completion():
    with make_client() as client:
        race_id = client.post("/api/race", json={"subtasks": ["a"], "host_steps": ["x"]}).json()["race_id"]
        snap = wait_for(client, f"/api/race/{race_id}", lambda d: d["status"] in ("done", "failed"))
        assert snap["status"] == "done"
        assert snap["speedup"] is not None


def test_simulated_mode_needs_no_backend():
    with TestClient(create_app(make_config(), simulated=True)) as client:
        assert client.get("/api/config").json()["simulated"] is True
