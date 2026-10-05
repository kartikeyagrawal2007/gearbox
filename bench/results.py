"""Print the benchmark results table: the dashboard's table, for terminals without a browser.

    python bench/results.py              # results from runs/
    python bench/results.py other/dir    # another folder

Shows the latest result per (task set, model, UNSURE hatch). The benchmark writes its JSON
after every model, so this also shows a run's progress while it's going.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gearbox.config import load_config
from gearbox.ui.server import load_runs

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


if __name__ == "__main__":
    print(render(Path(sys.argv[1] if len(sys.argv) > 1 else "runs")))
