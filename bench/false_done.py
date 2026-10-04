"""False-done benchmark: how often does each tier claim success on a coding subtask
while failing its executable check?

Each task runs once per tier with no escalation, at temperature 0, so the numbers
are each model's own rate. Examples:

    python bench/false_done.py --config gearbox.yaml --tiers small large            # 8 smoke tasks
    python bench/false_done.py --config vm/models.vm.yaml --tasks humaneval+         # 164 HumanEval+ problems
    python bench/false_done.py --config gearbox.yaml --tasks humaneval+ --limit 20   # quick trial

Task sets are defined in bench/tasksets.py.

`--hatch off` removes the "reply UNSURE if information is missing" instruction from the
worker prompt. Comparing on/off shows whether a model's UNSURE answers are honest (it
fails without the hatch too) or timid (it passes once the hatch is gone).

Needs `code_checks: true` in the config (checks run model-written code).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import time

from tasksets import EVALPLUS_SETS, TASK_SETS, Task, load_tasks

from gearbox.config import load_config
from gearbox.delegate import brief
from gearbox.delegate.runtime import DelegationRuntime
from gearbox.verify import has_unbalanced_fences

NO_HATCH_SYSTEM = "You are a focused worker model. Do the subtask and reply with the result only, no preamble."

_UNDEFINED = re.compile(r"name '(\w+)' is not defined")


def failure_kind(check_output: str, targets: tuple[str, ...]) -> str:
    """'format' when a function the task asked for was never defined (the answer didn't
    compile, or held no such function); 'logic' when it was defined but wrong, even if the
    answer also crashed while loading (e.g. a demo call after the definition), or if a helper
    of its own is missing. Format failures say more about output style or our code extraction
    than about whether the model knows the answer, so they are reported apart from false dones."""
    missing = _UNDEFINED.search(check_output.strip().splitlines()[-1] if check_output.strip() else "")
    return "format" if missing and missing.group(1) in targets else "logic"


async def run_tier(config, tier: str, tasks: list[Task], concurrency: int) -> dict:
    rt = DelegationRuntime(dataclasses.replace(config, max_escalations=0, max_concurrent=concurrency),
                           call_params={"temperature": 0})
    try:  # load the model before timing, so the first task doesn't absorb the load time
        await rt.provider.complete(config.tiers[config.tier_index(tier)], [{"role": "user", "content": "Reply with OK."}],
                                   max_tokens=1, **rt.call_params)
    except Exception:
        pass
    async def one(task: Task) -> dict:
        dt = await rt.run(task.prompt, tier=tier, check=task.check, show_check=task.show_check)
        a = dt.attempts[-1] if dt.attempts else None
        check_output = a.check["output"] if a and a.check else ""
        return {
            "task": task.name,
            "outcome": a.outcome if a else "error",
            "kind": failure_kind(check_output, task.targets) if a and a.outcome == "check_failed" else None,
            # Model time only: tasks queue for a concurrency slot, and that wait isn't the model's.
            "seconds": round(sum(at.latency_s for at in dt.attempts), 2),
            "check_seconds": a.check["duration_s"] if a and a.check else None,
            "why": (
                check_output.strip().splitlines()[-1] if a and a.check and not a.check["passed"]
                else (dt.error or "") if a is None or a.outcome in ("error", "timeout") else ""
            ),
            "fences_repaired": has_unbalanced_fences(dt.result or ""),
            "answer": (dt.result or "")[:6000],
            "check_output": check_output,
        }

    # The runtime caps how many run at once (max_concurrent); results keep task order.
    rows = await asyncio.gather(*(one(t) for t in tasks))
    claimed = [r for r in rows if r["outcome"] in ("ok", "check_failed")]
    false_done = [r for r in claimed if r["outcome"] == "check_failed"]
    scorable = [r for r in claimed if r["kind"] != "format"]
    logic_false_done = [r for r in scorable if r["outcome"] == "check_failed"]
    return {
        "tier": tier,
        "tasks": len(rows),
        "passed": sum(r["outcome"] == "ok" for r in rows),
        "unsure": sum(r["outcome"] == "unsure" for r in rows),
        "format_failures": sum(r["kind"] == "format" for r in rows),
        "fences_repaired": sum(r["fences_repaired"] for r in rows),
        "false_done_rate": round(len(false_done) / len(claimed), 3) if claimed else None,
        "logic_false_done_rate": round(len(logic_false_done) / len(scorable), 3) if scorable else None,
        "unsure_or_error": sum(r["outcome"] not in ("ok", "check_failed") for r in rows),
        "rows": rows,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--tiers", nargs="+", help="tier names to measure (default: all)")
    parser.add_argument("--json", help="also write full results to this file")
    parser.add_argument("--hatch", choices=("on", "off"), default="on", help="offer the UNSURE escape hatch (default on)")
    parser.add_argument("--tasks", choices=TASK_SETS, default="smoke", help="task set (default: smoke)")
    parser.add_argument("--limit", type=int, help="only the first N tasks (quick trial)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="tasks in flight per model (default 4, matching OLLAMA_NUM_PARALLEL on the VM)")
    parser.add_argument("--check-timeout", type=float, help="seconds per check (default: 60 for HumanEval+/MBPP+, else config)")
    parser.add_argument("--verbose", action="store_true", help="print every task, not just failures")
    args = parser.parse_args()
    if args.hatch == "off":
        brief.WORKER_SYSTEM = NO_HATCH_SYSTEM  # build_messages reads it at call time

    config = load_config(args.config)
    if not config.code_checks:
        raise SystemExit("set `code_checks: true` in the config first (checks run model-written code)")
    tasks = load_tasks(args.tasks, args.limit)
    # EvalPlus inputs can be large (Mbpp/599's reference alone takes ~9 s), so allow a minute.
    check_timeout = args.check_timeout or (60.0 if args.tasks in EVALPLUS_SETS else config.check_timeout_s)
    config = dataclasses.replace(config, check_timeout_s=check_timeout)
    print(f"{len(tasks)} {args.tasks} tasks, concurrency {args.concurrency}, check timeout {check_timeout:g}s")
    results = []
    for tier in args.tiers or [t.name for t in config.tiers]:
        started = time.perf_counter()
        res = await run_tier(config, tier, tasks, args.concurrency)
        res.update(hatch=args.hatch, task_set=args.tasks, concurrency=args.concurrency,
                   pass_rate=round(res["passed"] / res["tasks"], 3), wall_s=round(time.perf_counter() - started, 1))
        results.append(res)
        print(f"\n{tier} (hatch {args.hatch}): {res['passed']}/{res['tasks']} passed ({res['pass_rate']:.1%}), "
              f"{res['unsure']} unsure, {res['format_failures']} format failures, {res['fences_repaired']} fences repaired, "
              f"false-done rate {res['false_done_rate']} (logic only: {res['logic_false_done_rate']}), {res['wall_s']}s")
        show_all = args.verbose or len(tasks) <= 20
        for r in res["rows"]:
            if show_all or r["outcome"] != "ok":
                kind = f"[{r['kind']}]" if r["kind"] else ""
                print(f"  {r['task']:<16} {r['outcome']:<13} {kind:<9} {r['seconds']:>6}s  {r['why'][:90]}")
        if args.json:  # write after every model, so a crash or Ctrl+C keeps finished results
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
