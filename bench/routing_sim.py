"""Routing simulation: replay recorded benchmark answers under different routing strategies.

    python bench/routing_sim.py                      # HumanEval+ and MBPP+, Qwen3.5 ladder
    python bench/routing_sim.py --ladder all         # all 12 models, sorted by size
    python bench/routing_sim.py --json out.json

Every model has already answered every problem (bench/false_done.py, temperature 0, so one
answer per model per problem). That lets us ask "what if the router had sent problem X to
model Y?" without running anything new: look up Y's recorded outcome for X.

Cost is a proxy: the model's parameter count in billions, once per attempt. A failed attempt
that is escalated still costs. Strategies:

  fixed         always the same model (the baselines)
  oracle        the cheapest model that solves the problem (the best any router could do)
  heuristic     Gearbox's free difficulty heuristic + leverage, one attempt, no check
  shuffled      same tier counts as the heuristic but assigned at random (does it have any signal?)
  +escalate     start at that tier; on a failed check, retry one tier up, until the top

Escalation assumes a perfect check. Here the check is the hidden EvalPlus test suite, so the
"+escalate" rows are an upper bound; a real check (the host's own tests) will be weaker.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from plots import params_b
from results import tier_order

from tasksets import load_tasks
from gearbox.difficulty import HeuristicEstimator
from gearbox.router import base_tier_for
from gearbox.runs import load_runs

QWEN = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]
SETS = {"humaneval+": 164, "mbpp+": 378}


def outcomes(runs_dir: Path, task_set: str) -> dict[str, dict[str, bool]]:
    """{tier: {problem: passed}} for the hatch-on run of each model (the newest file wins)."""
    n = SETS[task_set]
    out: dict[str, dict[str, bool]] = {}
    for path in sorted(runs_dir.glob("*.json"), key=lambda f: f.stat().st_mtime):
        for r in json.loads(path.read_text()):
            if r.get("task_set", "") == task_set and r.get("hatch", "on") == "on" and len(r["rows"]) == n:
                out[r["tier"]] = {x["task"]: x["outcome"] == "ok" for x in r["rows"]}
    return out


def evaluate(ladder, solved, costs, problems, start_tier, escalate):
    """Accuracy and mean cost when each problem starts at start_tier[problem]."""
    passed, cost = 0, 0.0
    for p in problems:
        i = start_tier[p]
        while True:
            cost += costs[i]
            if solved[ladder[i]][p]:
                passed += 1
                break
            if not escalate or i == len(ladder) - 1:
                break
            i += 1
    return passed / len(problems), cost / len(problems)


def simulate(runs_dir: Path, task_set: str, ladder_name: str, shuffles: int, seed: int) -> list[dict]:
    solved = outcomes(runs_dir, task_set)
    if ladder_name == "qwen":
        ladder = QWEN
    else:
        ladder = sorted(solved, key=lambda t: (params_b(t), tier_order().index(t) if t in tier_order() else 0))
    costs = [params_b(t) for t in ladder]
    tasks = load_tasks(task_set)
    problems = [t.name for t in tasks]
    est = HeuristicEstimator()
    scores = {t.name: est.estimate_sync(t.prompt) for t in tasks}
    top = len(ladder) - 1
    rows: list[dict] = []

    def add(name, start, escalate=False, note=""):
        acc, cost = evaluate(ladder, solved, costs, problems, start, escalate)
        rows.append({"task_set": task_set, "strategy": name, "accuracy": round(acc, 4),
                     "cost_b_params": round(cost, 2), "note": note})

    for i, t in enumerate(ladder):
        add(f"always {t}", dict.fromkeys(problems, i))
    add("always smallest + escalate", dict.fromkeys(problems, 0), True, "cascade")
    # Oracle: start at the cheapest tier that solves it; one attempt, never wasted.
    oracle = {p: next((i for i, t in enumerate(ladder) if solved[t][p]), top) for p in problems}
    add("oracle (cheapest that passes)", oracle)

    for lev in (0, 1, 2):
        start = {p: min(base_tier_for(scores[p].score, len(ladder)) + lev, top) for p in problems}
        tiers_used = sorted(start.values())
        dist = {ladder[i]: tiers_used.count(i) for i in range(len(ladder)) if i in tiers_used}
        add(f"heuristic, leverage +{lev}", start, False, f"tiers used: {dist}")
        add(f"heuristic, leverage +{lev}, +escalate", start, True)
        rng = random.Random(seed)
        accs, cs = [], []
        for _ in range(shuffles):
            shuf = rng.sample(tiers_used, len(tiers_used))
            a, c = evaluate(ladder, solved, costs, problems, dict(zip(problems, shuf)), False)
            accs.append(a)
            cs.append(c)
        rows.append({"task_set": task_set, "strategy": f"shuffled control, +{lev}",
                     "accuracy": round(sum(accs) / shuffles, 4), "cost_b_params": round(sum(cs) / shuffles, 2),
                     "note": f"mean of {shuffles} shuffles"})
    return rows


def render(rows: list[dict]) -> str:
    lines, current = [], None
    for r in rows:
        if r["task_set"] != current:
            current = r["task_set"]
            lines += ["", f"== {current}", f"  {'strategy':<34} {'accuracy':>9} {'cost (B params)':>16}  note"]
        lines.append(f"  {r['strategy']:<34} {r['accuracy']:>9.1%} {r['cost_b_params']:>16.1f}  {r['note']}")
    return "\n".join(lines).lstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--ladder", choices=["qwen", "all"], default="qwen")
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the rows to this file")
    args = ap.parse_args()
    rows = []
    for ts in SETS:
        rows += simulate(Path(args.runs), ts, args.ladder, args.shuffles, args.seed)
    print(render(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
