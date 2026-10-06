"""Check-and-escalate with realistic checks: how much does the cascade lose when the check is weak?

    python bench/weak_checks.py                  # re-check recorded answers (cached), then simulate
    python bench/weak_checks.py --ladder all
  python bench/weak_checks.py --json docs/paper/data/weak_checks.json

bench/routing_sim.py escalates on the hidden EvalPlus suite: a perfect check, so an upper bound.
A host's own check is weaker. Here we re-run every recorded answer against weaker checks:

  1 test     only the first base input (for MBPP+ this is the example shown in the prompt)
  3 tests    the first three base inputs
  base       the original HumanEval / MBPP tests (no EvalPlus extras)
  full       base + plus inputs: the hidden suite that decides "correct" everywhere

The cascade starts at the cheapest model and escalates when the check fails (or the model
answers UNSURE). It accepts the first answer that passes the check, so a weak check lets a
wrong answer through: a "false accept". Accuracy is always judged by the full suite.

Only answers that failed the full suite are re-run: an answer that passes every input
also passes any subset, and a check that fails on k inputs also fails on more. Results are
cached in runs/cache/weak_checks.json (needs the raw runs/ files, not just the summaries).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
from pathlib import Path

from routing_sim import SETS, make_ladder
from plots import params_b
from tasksets import EVALPLUS_SETS, evalplus_check

from gearbox.verify.checks import run_check

LEVELS = ["1 test", "3 tests", "base", "full"]
TRUNCATED_AT = 6000  # runs before Oct 2026 stored at most this many characters of each answer


def recorded_rows(runs_dir: Path, task_set: str) -> dict[str, dict[str, dict]]:
    """{tier: {problem: row}} for each model's hatch-on run (the newest file wins)."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(runs_dir.glob("*.json"), key=lambda f: f.stat().st_mtime):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        for r in data if isinstance(data, list) else []:
            if isinstance(r, dict) and r.get("task_set") == task_set and r.get("hatch", "on") == "on" \
                    and len(r.get("rows", [])) == SETS[task_set]:
                out[r["tier"]] = {x["task"]: x for x in r["rows"]}
    return out


def level_inputs(problem: dict) -> dict[str, list]:
    base = list(problem["base_input"])
    return {"1 test": base[:1], "3 tests": base[:3], "base": base}


async def recheck(runs_dir: Path, task_set: str, cache: dict, concurrency: int, timeout: float) -> None:
    """Fill cache[task_set][tier][problem] = {level: passed} for every recorded answer."""
    dataset, name, _ = EVALPLUS_SETS[task_set]
    with gzip.open(Path(__file__).parent / "data" / name, "rt") as f:
        problems = {p["task_id"]: p for p in map(json.loads, f)}
    rows = recorded_rows(runs_dir, task_set)
    store = cache.setdefault(task_set, {})
    sem = asyncio.Semaphore(concurrency)
    todo = []

    async def one(tier: str, pid: str, row: dict) -> None:
        result = {"full": row["outcome"] == "ok"}
        if row["outcome"] == "ok":
            result.update(dict.fromkeys(LEVELS[:-1], True))
        elif row["outcome"] != "check_failed":  # UNSURE, error, timeout: nothing to accept
            result.update(dict.fromkeys(LEVELS[:-1], False))
        elif len(row["answer"]) >= TRUNCATED_AT:
            # The stored answer was cut off, so re-running it proves nothing; cascade() bounds it.
            result.update(dict.fromkeys(LEVELS[:-1], None))
        else:
            failed = False
            for level, inputs in level_inputs(problems[pid]).items():
                if not failed:  # failing on k inputs means failing on any superset
                    async with sem:
                        check = await run_check(row["answer"], evalplus_check(problems[pid], dataset, inputs), timeout)
                    failed = not check.passed
                result[level] = not failed
        store.setdefault(tier, {})[pid] = result

    for tier, by_problem in rows.items():
        for pid, row in by_problem.items():
            if pid not in store.get(tier, {}):
                todo.append(one(tier, pid, row))
    if todo:
        print(f"{task_set}: re-checking {len(todo)} answers ...", flush=True)
        await asyncio.gather(*todo)


def cascade(ladder: list[str], results: dict, level: str, unknown_passes: bool = False) -> dict:
    """Start cheapest; escalate while the check fails; accept the first pass (or the top's answer).
    unknown_passes: what a truncated wrong answer does on a weak check (run both for bounds)."""
    problems = list(results[ladder[0]])
    costs = [params_b(t) for t in ladder]
    correct = cost = false_accepts = 0
    for p in problems:
        for i, tier in enumerate(ladder):
            cost += costs[i]
            r = results[tier][p]
            passes = unknown_passes if r[level] is None else r[level]
            if passes or i == len(ladder) - 1:
                correct += r["full"]
                false_accepts += passes and not r["full"]
                break
    n = len(problems)
    return {"level": level, "accuracy": round(correct / n, 4), "cost_b_params": round(cost / n, 2),
            "false_accepts": false_accepts, "problems": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--ladder", choices=["qwen", "all"], default="qwen")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--json", help="also write the cascade rows to this file")
    args = ap.parse_args()
    runs = Path(args.runs)
    cache_path = runs / "cache" / "weak_checks.json"  # a subfolder: runs/*.json are benchmark results
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for ts in SETS:
        asyncio.run(recheck(runs, ts, cache, args.concurrency, args.timeout))
        cache_path.write_text(json.dumps(cache))

    out = []
    for ts in SETS:
        results = cache[ts]
        ladder = make_ladder(args.ladder, results)
        top = ladder[-1]
        top_acc = sum(r["full"] for r in results[top].values()) / len(results[top])
        print(f"\n== {ts}  (ladder: {', '.join(ladder)})")
        unknown = sum(v is None for t in ladder for r in results[t].values() for v in r.values())
        print(f"  {'check':<22} {'accuracy':>15} {'cost (B params)':>16} {'false accepts':>14}")
        print(f"  {'always ' + top:<22} {top_acc:>15.1%} {params_b(top):>16.1f} {'-':>14}")
        for level in LEVELS:
            best, worst = cascade(ladder, results, level, False), cascade(ladder, results, level, True)
            row = {"task_set": ts, "ladder": args.ladder, "start": 0, **best,
                   "accuracy_low": worst["accuracy"], "cost_low": worst["cost_b_params"],
                   "false_accepts_high": worst["false_accepts"]}
            out.append(row)
            acc = f"{best['accuracy']:.1%}" if best == worst else f"{worst['accuracy']:.1%}-{best['accuracy']:.1%}"
            fa = str(best["false_accepts"]) if best == worst else f"{best['false_accepts']}-{worst['false_accepts']}"
            print(f"  cascade, {level:<13} {acc:>15} {best['cost_b_params']:>16.1f} {fa:>14}")
        if unknown:
            print(f"  ranges: {unknown} weak-check results unknown (stored answer truncated at {TRUNCATED_AT} chars)")
        # Leverage as a start tier: skip the cheapest models when the check can't be trusted.
        print(f"\n  leverage (start tier) vs check strength: accuracy @ cost")
        print(f"  {'check':<9}" + "".join(f"{'start +' + str(k) + ' (' + ladder[k] + ')':>30}" for k in range(len(ladder) - 1)))
        for level in LEVELS:
            cells = []
            for k in range(len(ladder) - 1):
                worst, best = cascade(ladder[k:], results, level, True), cascade(ladder[k:], results, level, False)
                out.append({"task_set": ts, "ladder": args.ladder, "start": k, **best,
                            "accuracy_low": worst["accuracy"], "cost_low": worst["cost_b_params"]})
                acc = f"{best['accuracy']:.0%}" if best == worst else f"{worst['accuracy']:.0%}-{best['accuracy']:.0%}"
                cells.append(f"{acc} @ {best['cost_b_params']:.1f}")
            print(f"  {level:<9}" + "".join(f"{c:>30}" for c in cells))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
