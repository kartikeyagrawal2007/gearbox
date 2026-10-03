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


def test_delegate_with_check_over_http():
    good = "```python\ndef double(x):\n    return 2 * x\n```"
    app = create_app(make_config(code_checks=True), provider=FakeProvider({"t0": good}))
    with TestClient(app) as client:
        assert client.get("/api/config").json()["code_checks"] is True
        client.post("/api/delegate", json={"task": "Write double", "tier": "t0", "check": "assert double(3) == 6"})
        tasks = wait_for(client, "/api/tasks", lambda d: d["tasks"][0]["state"] in ("done", "failed"), timeout=15)
        assert tasks["tasks"][0]["verified"] is True
        assert client.get("/api/ledger").json()["false_done_rate"] == {"t0": 0.0}


def test_check_rejected_when_disabled():
    with make_client() as client:
        res = client.post("/api/delegate", json={"task": "x", "check": "assert True"})
        assert res.status_code == 400 and "code_checks" in res.json()["error"]


# --- download status and benchmark results (used on the VM while models download) ---

import json as _json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from gearbox.config import GearboxConfig, Tier


def fake_ollama(downloaded: list[str]) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = _json.dumps({"models": [{"name": n, "model": n} for n in downloaded]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def ollama_config(base: str) -> GearboxConfig:
    return GearboxConfig(tiers=(
        Tier("small", "ollama_chat/qwen3.5:0.8b", api_base=base),
        Tier("big", "ollama_chat/qwen3.5:27b", api_base=base),
        Tier("api", "anthropic/claude-haiku-4-5"),
    ), code_checks=True)


def test_config_reports_which_models_are_downloaded():
    server = fake_ollama(["qwen3.5:0.8b"])
    try:
        app = create_app(ollama_config(f"http://127.0.0.1:{server.server_port}"), provider=FakeProvider())
        with TestClient(app) as client:
            tiers = {t["name"]: t["available"] for t in client.get("/api/config").json()["tiers"]}
    finally:
        server.shutdown()
    assert tiers == {"small": True, "big": False, "api": None}


def test_unreachable_ollama_means_unknown_not_missing():
    app = create_app(ollama_config("http://127.0.0.1:9"), provider=FakeProvider())
    with TestClient(app) as client:
        assert {t["available"] for t in client.get("/api/config").json()["tiers"]} == {None}


def test_cannot_use_a_model_that_is_not_downloaded():
    server = fake_ollama(["qwen3.5:0.8b"])
    try:
        app = create_app(ollama_config(f"http://127.0.0.1:{server.server_port}"), provider=FakeProvider())
        with TestClient(app) as client:
            res = client.post("/api/delegate", json={"task": "x", "tier": "big"})
            assert res.status_code == 400 and "not downloaded yet" in res.json()["error"]
            assert "ollama pull qwen3.5:27b" in res.json()["error"]
            res = client.post("/api/race", json={"host_tier": "big", "worker_tier": "small"})
            assert res.status_code == 400
            assert client.post("/api/delegate", json={"task": "x", "tier": "small"}).status_code == 200
    finally:
        server.shutdown()


def test_runs_panel_merges_latest_result_per_model(tmp_path):
    older = [{"tier": "t1", "hatch": "on", "tasks": 8, "passed": 3, "unsure": 1, "false_done_rate": 0.5, "rows": []}]
    newer = [
        {"tier": "t1", "hatch": "on", "tasks": 8, "passed": 6, "unsure": 0, "false_done_rate": 0.25, "rows": []},
        {"tier": "t0", "hatch": "off", "tasks": 8, "passed": 1, "false_done_rate": None,
         "rows": [{"outcome": "unsure"}, {"outcome": "ok"}]},
    ]
    (tmp_path / "a.json").write_text(_json.dumps(older))
    (tmp_path / "b.json").write_text(_json.dumps(newer))
    import os
    os.utime(tmp_path / "a.json", (1, 1))  # make a.json the older file
    (tmp_path / "notes.json").write_text('{"not": "a benchmark"}')
    (tmp_path / "broken.json").write_text("{")

    app = create_app(make_config(), provider=FakeProvider(), runs_dir=tmp_path)
    with TestClient(app) as client:
        data = client.get("/api/runs").json()
    combined = [(r["tier"], r["hatch"], r["passed"], r["unsure"]) for r in data["combined"]]
    assert combined == [("t0", "off", 1, 1), ("t1", "on", 6, 0)]  # tier order; newer t1 wins
    # old result files have no logic-only rate: fall back to the plain rate
    assert data["combined"][1]["logic_false_done_rate"] == 0.25
    errors = {f["file"]: f.get("error") for f in data["files"]}
    assert errors["broken.json"] and errors["a.json"] is None and errors["notes.json"] is None


def test_runs_panel_without_folder(tmp_path):
    app = create_app(make_config(), provider=FakeProvider(), runs_dir=tmp_path / "missing")
    with TestClient(app) as client:
        assert client.get("/api/runs").json()["combined"] == []
