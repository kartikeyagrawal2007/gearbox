"""MCP server exposing Gearbox to any MCP host (Claude Code, Antigravity, Cursor, Cline, Codex).

The host agent is the "frontier" model. It calls `delegate` to hand off a
subtask, keeps working, and calls `await_result` when it needs the output.
Background subtasks live as long as this server process (one per host session).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from gearbox.config import load_config
from gearbox.cost.breakeven import break_even
from gearbox.delegate.runtime import DelegationRuntime

INSTRUCTIONS = """\
Gearbox lets you hand self-contained subtasks to cheaper models and keep working meanwhile.

Delegate: well-specified, low-risk, checkable subtasks whose output is much longer than the \
instructions you have to write, e.g. generating tests or boilerplate, drafting docs, \
transforming data.
Everything you write into `task` and `context` is billed as your own output tokens, so keep \
briefs short.
Flow: call `delegate`, which returns a task_id immediately. Continue with independent work, \
then call `await_result` only when you need the output. If a delegate reply says mode "burst" \
(your model shares the workers' GPU), subtasks wait until you call await_result, so delegate the \
whole batch first, then await: overlapping on one GPU slows both sides down.
Verify: for code, pass `check` (Python asserts or test_ functions). Gearbox runs it against the \
worker's answer, escalates to a stronger tier if it fails, and reports `verified`. A thorough \
check (10+ cases, or a loop over test data) lets Gearbox start on a cheaper model; a check with \
1-3 asserts starts two tiers higher, because it lets more wrong answers through. Without a check, \
verify the result yourself before relying on it.
Do not delegate: steps that need your full conversation history, open design decisions, or \
irreversible actions. Set `risk` honestly: higher risk routes to a stronger model.
"""

mcp = MCPServer("gearbox", instructions=INSTRUCTIONS)
_runtime: DelegationRuntime | None = None


def runtime() -> DelegationRuntime:
    global _runtime
    if _runtime is None:
        _runtime = DelegationRuntime(load_config())
    return _runtime


@contextmanager
def caller_errors() -> Iterator[None]:
    """Report mistakes the calling agent can fix (unknown tier or task_id, bad risk,
    missing config) as readable tool errors. MCP hides other exception messages."""
    try:
        yield
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise ToolError(str(e.args[0]) if e.args else type(e).__name__) from e


@mcp.tool()
async def route(prompt: str, risk: str | None = None) -> dict:
    """Recommend a model tier for a prompt from its estimated difficulty.

    risk: "low" | "medium" | "high". How costly a wrong answer would be; raises the tier (leverage).
    """
    with caller_errors():
        rt = runtime()
        decision = await rt.router.route(prompt, risk=risk)
        return {**decision.as_dict(), "model": rt.config.tiers[decision.tier].model}


@mcp.tool()
async def delegate(
    task: str,
    context: str = "",
    acceptance: str = "",
    risk: str = "low",
    tier: str | None = None,
    expected_output_tokens: int | None = None,
    check: str = "",
) -> dict:
    """Hand a self-contained subtask to a cheaper model. Returns immediately with a task_id.

    Keep working on anything that does not depend on this result, then call await_result.
    task: what to do, stated so it can be done without your conversation history.
    context: only the facts or snippets the worker needs (you pay output tokens for every word).
    acceptance: how to tell the result is correct.
    risk: "low" | "medium" | "high". Higher risk routes to a stronger tier.
    tier: force a specific tier by name (skips difficulty routing).
    expected_output_tokens: optional; if given, a break-even estimate is included in the reply.
    check: optional Python test code run against the answer. The worker's code is loaded first,
      then your asserts or test_* functions run; RESULT holds the raw reply text. Failing answers
      escalate to a stronger tier with the failure shown to it. The more cases it tests, the
      cheaper the starting tier (1-3 asserts: +2 tiers, 4-9: +1, 10 or more: +0).
    """
    with caller_errors():
        rt = runtime()
        dt = rt.delegate(task, context=context, acceptance=acceptance, risk=risk, tier=tier, check=check)
        reply = {"task_id": dt.id, "state": dt.state.value, "mode": rt.mode,
                 "next": "continue other work; call await_result when needed" if rt.mode == "async" else
                 "delegate the rest of the batch, then call await_result: subtasks start when you wait"}
        if expected_output_tokens:
            # Priced against the cheapest tier; routing may pick a higher one.
            estimate = break_even(rt.config.host, rt.config.tiers[0].pricing, dt.brief_tokens, expected_output_tokens)
            reply["break_even"] = estimate.as_dict()
        return reply


@mcp.tool()
async def await_result(task_id: str, timeout_s: float = 60.0) -> dict:
    """Wait up to timeout_s seconds for a delegated subtask. Returns its result once state is "done".

    If it is still running, you get state "running". Do other work or call again.
    """
    with caller_errors():
        return await runtime().await_result(task_id, timeout_s=timeout_s)


@mcp.tool()
async def status(task_id: str | None = None) -> dict:
    """Check one delegated subtask (by task_id) or list all of them, without waiting."""
    with caller_errors():
        rt = runtime()
        if task_id:
            return rt.status(task_id)
        return {"tasks": rt.list_tasks()}


@mcp.tool()
async def cancel(task_id: str) -> dict:
    """Cancel a running delegated subtask."""
    with caller_errors():
        return {"task_id": task_id, "cancelled": runtime().cancel(task_id)}


@mcp.tool()
async def ledger() -> dict:
    """Tokens and cost spent by Gearbox per tier, estimated host savings, and async overlap stats."""
    with caller_errors():
        return runtime().stats()


def protect_stdout() -> None:
    """stdout carries the JSON-RPC stream. The MCP SDK sets the root logger to INFO
    and LiteLLM routes records below WARNING to stdout, so a single completion can
    corrupt the stream. Only let ERROR records (routed to stderr) through.
    Must run before LiteLLM is first imported."""
    os.environ["LITELLM_LOG"] = "ERROR"


async def serve_with_dashboard(port: int) -> None:
    """Run the MCP stdio server and the web dashboard in one process over one runtime,
    so delegations made by the host agent show up live in the browser."""
    import sys

    import anyio
    import uvicorn

    from gearbox.ui.server import create_app

    rt = runtime()
    app = create_app(rt.config, runtime=rt)
    # access_log=False: uvicorn's access log writes to stdout, which carries JSON-RPC.
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    print(f"Gearbox dashboard: http://127.0.0.1:{port}", file=sys.stderr)
    async with anyio.create_task_group() as tg:
        tg.start_soon(server.serve)
        await mcp.run_stdio_async()
        server.should_exit = True


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="gearbox-mcp", description="Gearbox MCP server (stdio).")
    parser.add_argument("--dashboard", type=int, metavar="PORT", help="also serve the live dashboard on this port")
    args = parser.parse_args(argv)

    protect_stdout()
    # Build the runtime (and import LiteLLM) at startup rather than inside the first
    # tool call. A missing or invalid config then fails at launch, where the host shows it.
    runtime()
    if args.dashboard:
        import anyio

        anyio.run(serve_with_dashboard, args.dashboard)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
