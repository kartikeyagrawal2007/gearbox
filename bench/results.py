"""Print the benchmark results table: the dashboard's table, for terminals without a browser.

    python bench/results.py              # results from runs/
    python bench/results.py other/dir    # another folder
    python bench/results.py --export     # compact JSON summary, to copy off a machine
                                         # you can only reach through a terminal

Shows the latest result per (task set, model, UNSURE hatch). The benchmark writes its JSON
after every model, so this also shows a run's progress while it's going.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gearbox.config import load_config
from gearbox.runs import load_runs

MODELS_CONFIG = Path(__file__).resolve().parent.parent / "vm" / "models.vm.yaml"


def tier_order() -> list[str]:
    """Model order from the VM config (size ladder first), so rows don't sort alphabetically."""
    try:
        return [t.name for t in load_config(MODELS_CONFIG).tiers]
    except (OSError, ValueError, KeyError):
        return []


def pct(x: float | None) -> str:
    return "  -" if x is None else f"{x * 100:3.0f}%"


def render(runs_dir: Path) -> str:
    data = load_runs(runs_dir, tier_order())
    if not data["combined"]:
        return f"No results in {runs_dir}/ yet."
    lines = []
    task_set = None
    for r in data["combined"]:
        if r["task_set"] != task_set:
            task_set = r["task_set"]
            lines += ["", f"== {task_set}",
                      f"  {'model':<20} {'hatch':<5} {'passed':>14} {'unsure':>7} {'false-done':>10} {'format':>7} {'fences':>7}"]
        passed = f"{r['passed']}/{r['tasks']} ({r['passed'] / r['tasks']:.0%})"
        fmt = "-" if r.get("format_failures") is None else str(r["format_failures"])
        fences = "-" if r.get("fences_repaired") is None else str(r["fences_repaired"])
        lines.append(f"  {r['tier']:<20} {r['hatch']:<5} {passed:>14} {r['unsure']:>7} "
                     f"{pct(r['logic_false_done_rate']):>10} {fmt:>7} {fences:>7}")
    broken = [f["file"] for f in data["files"] if f.get("error")]
    if broken:
        lines += ["", "Not benchmark results: " + ", ".join(broken)]
    return "\n".join(lines).lstrip("\n")


SUMMARY_FIELDS = ("task_set", "tier", "hatch", "tasks", "passed", "unsure", "false_done_rate",
                  "logic_false_done_rate", "format_failures", "fences_repaired")


def export(runs_dir: Path) -> str:
    """One compact JSON line: the per-model numbers without answers or check output.
    Save it as a .json file in another runs/ folder and every tool here can read it."""
    rows = [{k: r.get(k) for k in SUMMARY_FIELDS} for r in load_runs(runs_dir, tier_order())["combined"]]
    return json.dumps(rows, separators=(",", ":"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--export"]
    folder = Path(args[0] if args else "runs")
    print(export(folder) if "--export" in sys.argv else render(folder))
