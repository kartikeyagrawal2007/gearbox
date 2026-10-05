"""Reading benchmark results: the JSON files bench/false_done.py writes into runs/.

Each file holds one list entry per model ("tier"). Several files can cover the same
model (e.g. a run that was stopped and resumed), so `load_runs` keeps the newest
result per (model, UNSURE hatch, task set). The dashboard, bench/results.py and
bench/plots.py all read results through this one function.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_runs(runs_dir: Path, tier_order: list[str]) -> dict:
    """Summaries of bench/false_done.py JSON files, plus the latest result per (model, hatch, task set)."""
    files, latest = [], {}
    paths = sorted(runs_dir.glob("*.json"), key=lambda f: f.stat().st_mtime) if runs_dir.is_dir() else []
    for path in paths:  # oldest first, so newer files overwrite in `latest`
        entry = {"file": path.name, "modified": path.stat().st_mtime, "results": []}
        try:
            data = json.loads(path.read_text())
            for r in data if isinstance(data, list) else []:
                row = {
                    "tier": r["tier"], "hatch": r.get("hatch", "on"), "task_set": r.get("task_set", "smoke"),
                    "tasks": r["tasks"], "passed": r["passed"],
                    "unsure": r.get("unsure", sum(x.get("outcome") == "unsure" for x in r.get("rows", []))),
                    "false_done_rate": r.get("false_done_rate"),
                    # Newer results separate format slips from wrong code; prefer the logic-only rate.
                    "logic_false_done_rate": r.get("logic_false_done_rate", r.get("false_done_rate")),
                    "format_failures": r.get("format_failures"), "fences_repaired": r.get("fences_repaired"),
                    "file": path.name,
                }
                entry["results"].append(row)
                latest[(row["tier"], row["hatch"], row["task_set"])] = row
        except (ValueError, KeyError, TypeError) as e:
            entry["error"] = f"not a benchmark result ({type(e).__name__})"
        files.append(entry)
    rank = {name: i for i, name in enumerate(tier_order)}
    combined = sorted(latest.values(), key=lambda r: (r["task_set"], rank.get(r["tier"], len(rank)), r["tier"], r["hatch"]))
    return {"runs_dir": str(runs_dir), "files": files[::-1], "combined": combined}
