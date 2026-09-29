"""Command line: `gearbox tiers | route | run | serve`."""

from __future__ import annotations

import argparse
import asyncio
import json

from gearbox.config import RISK_LEVELS, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gearbox", description=__doc__)
    parser.add_argument("--config", help="path to gearbox.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tiers", help="list configured tiers")

    p_route = sub.add_parser("route", help="show which tier a prompt would go to")
    p_route.add_argument("prompt")
    p_route.add_argument("--risk", choices=RISK_LEVELS)
    p_route.add_argument("--leverage", type=int)

    p_run = sub.add_parser("run", help="delegate one subtask and wait for it")
    p_run.add_argument("task")
    p_run.add_argument("--context", default="")
    p_run.add_argument("--acceptance", default="")
    p_run.add_argument("--risk", choices=RISK_LEVELS)
    p_run.add_argument("--tier")

    sub.add_parser("serve", help="run the MCP server on stdio")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        from gearbox.integrations.mcp_server import main as serve

        serve()
        return

    config = load_config(args.config)
    if args.cmd == "tiers":
        for i, t in enumerate(config.tiers):
            print(f"{i}  {t.name:<12} {t.model}  in=${t.pricing.input}/M out=${t.pricing.output}/M")
        return

    from gearbox.delegate.runtime import DelegationRuntime

    async def _go() -> dict:
        rt = DelegationRuntime(config)
        if args.cmd == "route":
            return (await rt.router.route(args.prompt, risk=args.risk, leverage=args.leverage)).as_dict()
        dt = await rt.run(args.task, context=args.context, acceptance=args.acceptance, risk=args.risk, tier=args.tier)
        return {"task": dt.view(), "ledger": rt.stats()}

    print(json.dumps(asyncio.run(_go()), indent=2))


if __name__ == "__main__":
    main()
