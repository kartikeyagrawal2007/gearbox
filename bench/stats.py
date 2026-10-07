"""Uncertainty for the paper's numbers: Wilson intervals and paired McNemar tests.

    python bench/stats.py              # every pass rate with its 95% interval, and the paired tests

Every model answered the same problems, so two models (or one model with the UNSURE hatch on
and off) are compared problem by problem: McNemar's exact test on the problems where exactly
one of the two passed. That is far more sensitive than comparing two pass rates as if they
came from different problems.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from results import tier_order
from routing_sim import SETS


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def outcomes(runs_dir: Path) -> dict[tuple[str, str, str], dict[str, bool]]:
    """{(task set, model, hatch): {problem: passed}}, newest file per key."""
    out = {}
    for path in sorted(runs_dir.glob("*.json"), key=lambda f: f.stat().st_mtime):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        for r in data if isinstance(data, list) else []:
            if isinstance(r, dict) and r.get("task_set") in SETS and len(r.get("rows", [])) == SETS[r["task_set"]]:
                out[(r["task_set"], r["tier"], r.get("hatch", "on"))] = {x["task"]: x["outcome"] == "ok" for x in r["rows"]}
    return out


def compare(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    only_a = sum(a[p] and not b[p] for p in a)
    only_b = sum(b[p] and not a[p] for p in a)
    return only_a, only_b, mcnemar(only_a, only_b)


PAIRS = [  # (task set, model A, hatch A, model B, hatch B, why)
    *[("humaneval+", m, "on", m, "off", "UNSURE hatch on vs off") for m in tier_order()],
    ("humaneval+", "granite4.2-8b", "on", "qwen3.5-9b", "on", "vendor at ~8B"),
    ("mbpp+", "granite4.2-8b", "on", "qwen3.5-9b", "on", "vendor at ~8B"),
    ("humaneval+", "qwen3.5-4b", "on", "qwen3.5-9b", "on", "size plateau 4B -> 9B"),
    ("mbpp+", "qwen3.5-4b", "on", "qwen3.5-9b", "on", "size plateau 4B -> 9B"),
    ("humaneval+", "qwen2.5-coder-0.5b", "on", "qwen3.5-0.8b", "on", "code-tuned 0.5B vs 0.8B"),
    ("mbpp+", "qwen2.5-coder-0.5b", "on", "qwen3.5-0.8b", "on", "code-tuned 0.5B vs 0.8B"),
    ("humaneval+", "gemma3-4b", "on", "qwen3.5-4b", "on", "vendor at ~4B"),
    ("mbpp+", "gemma3-4b", "on", "qwen3.5-4b", "on", "vendor at ~4B"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default="runs")
    args = ap.parse_args()
    res = outcomes(Path(args.runs))
    order = {t: i for i, t in enumerate(tier_order())}
    print("Pass rate with 95% Wilson interval")
    for key in sorted(res, key=lambda k: (k[0], order.get(k[1], 99), k[2])):
        v = res[key]
        k, n = sum(v.values()), len(v)
        lo, hi = wilson(k, n)
        print(f"  {key[0]:<11} {key[1]:<20} hatch {key[2]:<3} {k:>3}/{n} {k / n:6.1%}  [{lo:5.1%}, {hi:5.1%}]")
    print("\nPaired comparisons (McNemar exact): problems only A solved / only B solved, p-value")
    for ts, ma, ha, mb, hb, why in PAIRS:
        if (ts, ma, ha) in res and (ts, mb, hb) in res:
            a_only, b_only, p = compare(res[(ts, ma, ha)], res[(ts, mb, hb)])
            star = " *" if p < 0.05 else ""
            print(f"  {ts:<11} {ma} ({ha}) vs {mb} ({hb}): {a_only:>3} / {b_only:<3} p={p:.3f}{star}   [{why}]")


if __name__ == "__main__":
    main()
