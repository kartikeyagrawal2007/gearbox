"""Local web dashboard: delegate subtasks, watch routing and runs live, race
blocking vs async delegation. Serves on localhost only; there is no auth."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

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


def create_app(
    config: GearboxConfig,
    provider: Provider | None = None,
    simulated: bool = False,
    runtime: DelegationRuntime | None = None,
) -> Starlette:
    """Pass `runtime` to watch an existing runtime, e.g. the MCP server's, live."""
    if runtime is None:
        if simulated and provider is None:
            provider = SimulatedProvider([t.name for t in config.tiers])
        runtime = DelegationRuntime(config, provider)
    races: dict[str, Race] = {}
    background: set[asyncio.Task] = set()  # keep references so race tasks aren't garbage-collected

    async def index(request: Request) -> FileResponse:
        # Revalidate every load: otherwise browsers keep running an old page after an upgrade.
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    async def get_config(request: Request) -> JSONResponse:
        return JSONResponse({
            "simulated": simulated,
            "risks": list(RISK_LEVELS),
            "leverage": config.leverage,
            "risk_scaled_leverage": config.risk_scaled_leverage,
            "difficulty": config.difficulty,
            "code_checks": config.code_checks,
            "tiers": [
                {"name": t.name, "model": t.model, "input_price": t.pricing.input, "output_price": t.pricing.output}
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
    ])


def serve(config: GearboxConfig, host: str = "127.0.0.1", port: int = 8790, simulated: bool = False) -> None:
    import uvicorn

    app = create_app(config, simulated=simulated)
    print(f"Gearbox dashboard: http://{host}:{port}" + ("  (SIMULATED models)" if simulated else ""))
    uvicorn.run(app, host=host, port=port, log_level="warning")
