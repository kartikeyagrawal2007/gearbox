"""Local web dashboard: delegate subtasks, watch routing and runs live, race
blocking vs async delegation, and browse benchmark results. Serves on localhost
only; there is no auth."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from gearbox.config import RISK_LEVELS, GearboxConfig
from gearbox.delegate.runtime import DelegationRuntime
from gearbox.providers import Provider, SimulatedProvider
from gearbox.race import Race

STATIC = Path(__file__).parent / "static"
Handler = Callable[[Request], Awaitable[JSONResponse]]


def caller_errors(handler: Handler) -> Handler:
    """Bad input (unknown tier or task, invalid risk, missing field) -> 400 with a reason."""

    async def wrapped(request: Request) -> JSONResponse:
        try:
            return await handler(request)
        except (ValueError, KeyError, TypeError) as e:
            return JSONResponse({"error": str(e.args[0]) if e.args else type(e).__name__}, status_code=400)

    return wrapped


OLLAMA_PREFIXES = ("ollama_chat/", "ollama/")
DEFAULT_OLLAMA = "http://localhost:11434"


async def ollama_availability(config: GearboxConfig) -> dict[str, bool | None]:
    """Which Ollama tiers are downloaded. None = not an Ollama model, or Ollama unreachable."""
    by_base: dict[str, list] = {}
    for t in config.tiers:
        if t.model.startswith(OLLAMA_PREFIXES):
            by_base.setdefault(t.api_base or DEFAULT_OLLAMA, []).append(t)
    out: dict[str, bool | None] = {t.name: None for t in config.tiers}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for base, tiers in by_base.items():
            try:
                resp = await client.get(f"{base.rstrip('/')}/api/tags")
                names = {n for m in resp.json().get("models", []) for n in (m.get("name"), m.get("model")) if n}
            except (httpx.HTTPError, ValueError):
                continue
            for t in tiers:
                tag = t.model.split("/", 1)[1]
                out[t.name] = tag in names or f"{tag}:latest" in names
    return out


def load_runs(runs_dir: Path, tier_order: list[str]) -> dict:
    """Summaries of bench/false_done.py JSON files, plus the latest result per (model, hatch)."""
    files, latest = [], {}
    paths = sorted(runs_dir.glob("*.json"), key=lambda f: f.stat().st_mtime) if runs_dir.is_dir() else []
    for path in paths:  # oldest first, so newer files overwrite in `latest`
        entry = {"file": path.name, "modified": path.stat().st_mtime, "results": []}
        try:
            data = json.loads(path.read_text())
            for r in data if isinstance(data, list) else []:
                row = {
                    "tier": r["tier"], "hatch": r.get("hatch", "on"), "tasks": r["tasks"], "passed": r["passed"],
                    "unsure": r.get("unsure", sum(x.get("outcome") == "unsure" for x in r.get("rows", []))),
                    "false_done_rate": r.get("false_done_rate"),
                    # Newer results separate format slips from wrong code; prefer the logic-only rate.
                    "logic_false_done_rate": r.get("logic_false_done_rate", r.get("false_done_rate")),
                    "format_failures": r.get("format_failures"), "fences_repaired": r.get("fences_repaired"),
                    "file": path.name,
                }
                entry["results"].append(row)
                latest[(row["tier"], row["hatch"])] = row
        except (ValueError, KeyError, TypeError) as e:
            entry["error"] = f"not a benchmark result ({type(e).__name__})"
        files.append(entry)
    rank = {name: i for i, name in enumerate(tier_order)}
    combined = sorted(latest.values(), key=lambda r: (rank.get(r["tier"], len(rank)), r["tier"], r["hatch"]))
    return {"runs_dir": str(runs_dir), "files": files[::-1], "combined": combined}


def create_app(
    config: GearboxConfig,
    provider: Provider | None = None,
    simulated: bool = False,
    runtime: DelegationRuntime | None = None,
    runs_dir: str | Path = "runs",
) -> Starlette:
    """Pass `runtime` to watch an existing runtime, e.g. the MCP server's, live."""
    if runtime is None:
        if simulated and provider is None:
            provider = SimulatedProvider([t.name for t in config.tiers])
        runtime = DelegationRuntime(config, provider)
    races: dict[str, Race] = {}
    background: set[asyncio.Task] = set()  # keep references so race tasks aren't garbage-collected
    runs_path = Path(runs_dir)

    async def availability() -> dict[str, bool | None]:
        return {t.name: True for t in config.tiers} if simulated else await ollama_availability(config)

    async def require_downloaded(*tier_names: str | None) -> None:
        avail = await availability()
        for name in tier_names:
            if name and avail.get(name) is False:
                model = config.tiers[config.tier_index(name)].model.split("/", 1)[1]
                raise ValueError(f"{name} ({model}) is not downloaded yet: run `ollama pull {model}`")

    async def index(request: Request) -> FileResponse:
        # Revalidate every load: otherwise browsers keep running an old page after an upgrade.
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    async def get_config(request: Request) -> JSONResponse:
        avail = await availability()
        return JSONResponse({
            "simulated": simulated,
            "risks": list(RISK_LEVELS),
            "leverage": config.leverage,
            "risk_scaled_leverage": config.risk_scaled_leverage,
            "difficulty": config.difficulty,
            "code_checks": config.code_checks,
            "tiers": [
                {"name": t.name, "model": t.model, "input_price": t.pricing.input, "output_price": t.pricing.output,
                 "available": avail.get(t.name)}
                for t in config.tiers
            ],
        })

    @caller_errors
    async def route(request: Request) -> JSONResponse:
        body = await request.json()
        decision = await runtime.router.route(body["prompt"], risk=body.get("risk") or None)
        return JSONResponse({**decision.as_dict(), "model": config.tiers[decision.tier].model})

    @caller_errors
    async def delegate(request: Request) -> JSONResponse:
        body = await request.json()
        await require_downloaded(body.get("tier") or None)
        dt = runtime.delegate(
            body["task"],
            context=body.get("context", ""),
            acceptance=body.get("acceptance", ""),
            risk=body.get("risk") or None,
            tier=body.get("tier") or None,
            check=body.get("check", ""),
        )
        return JSONResponse({"task_id": dt.id, "state": dt.state.value})

    async def tasks(request: Request) -> JSONResponse:
        return JSONResponse({"tasks": list(reversed(runtime.list_tasks(include_result=True)))})

    @caller_errors
    async def cancel(request: Request) -> JSONResponse:
        task_id = request.path_params["task_id"]
        return JSONResponse({"task_id": task_id, "cancelled": runtime.cancel(task_id)})

    async def ledger(request: Request) -> JSONResponse:
        return JSONResponse(runtime.stats())

    @caller_errors
    async def start_race(request: Request) -> JSONResponse:
        body = await request.json() if await request.body() else {}
        await require_downloaded(body.get("host_tier") or None, body.get("worker_tier") or None)
        race = Race(
            config,
            runtime.provider,
            subtasks=body.get("subtasks") or None,
            host_steps=body.get("host_steps") or None,
            host_tier=body.get("host_tier") or None,
            worker_tier=body.get("worker_tier") or None,
        )
        races[race.id] = race
        job = asyncio.get_running_loop().create_task(race.run())
        background.add(job)
        job.add_done_callback(background.discard)
        return JSONResponse({"race_id": race.id})

    @caller_errors
    async def get_race(request: Request) -> JSONResponse:
        return JSONResponse(races[request.path_params["race_id"]].snapshot())

    async def runs(request: Request) -> JSONResponse:
        return JSONResponse(load_runs(runs_path, [t.name for t in config.tiers]))

    return Starlette(routes=[
        Route("/", index),
        Route("/api/config", get_config),
        Route("/api/route", route, methods=["POST"]),
        Route("/api/delegate", delegate, methods=["POST"]),
        Route("/api/tasks", tasks),
        Route("/api/tasks/{task_id}/cancel", cancel, methods=["POST"]),
        Route("/api/ledger", ledger),
        Route("/api/race", start_race, methods=["POST"]),
        Route("/api/race/{race_id}", get_race),
        Route("/api/runs", runs),
    ])


def serve(
    config: GearboxConfig, host: str = "127.0.0.1", port: int = 8790, simulated: bool = False, runs_dir: str = "runs"
) -> None:
    import uvicorn

    app = create_app(config, simulated=simulated, runs_dir=runs_dir)
    print(f"Gearbox dashboard: http://{host}:{port}" + ("  (SIMULATED models)" if simulated else ""))
    uvicorn.run(app, host=host, port=port, log_level="warning")
