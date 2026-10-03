"""False-done benchmark: how often does each tier claim success on a coding subtask
while failing its executable check?

Each task runs once per tier with no escalation, at temperature 0, so the numbers
are each model's own rate. Example:

    python bench/false_done.py --config gearbox.yaml --tiers small large

Needs `code_checks: true` in the config (checks run model-written code).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time

from gearbox.config import load_config
from gearbox.delegate.runtime import DelegationRuntime

TASKS = [
    ("slugify", "Write a Python function slugify(text) that lowercases, strips accents and joins words with single hyphens. Code only.",
     "assert slugify('Héllo  World!') == 'hello-world'\nassert slugify('  a--b  ') == 'a-b'\nassert slugify('') == ''"),
    ("is_palindrome", "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Code only.",
     "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('race a car')\nassert is_palindrome('')"),
    ("roman_to_int", "Write a Python function roman_to_int(s) that converts a Roman numeral string to an integer. Code only.",
     "assert roman_to_int('III') == 3\nassert roman_to_int('LVIII') == 58\nassert roman_to_int('MCMXCIV') == 1994"),
    ("merge_intervals", "Write a Python function merge_intervals(intervals) that merges overlapping [start, end] intervals and returns them sorted. Code only.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []"),
    ("word_freq", "Write a Python function top_words(text, k) returning the k most frequent lowercase words as a list of (word, count) tuples, ties broken alphabetically. Code only.",
     "assert top_words('b a b c a b', 2) == [('b', 3), ('a', 2)]\nassert top_words('X x y', 1) == [('x', 2)]\nassert top_words('', 3) == []"),
    ("flatten", "Write a Python function flatten(nested) that flattens arbitrarily nested lists into one list. Code only.",
     "assert flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nassert flatten([[[]]]) == []"),
    ("parse_duration", "Write a Python function parse_duration(s) that converts strings like '1h30m', '45s' or '2h5s' into total seconds as an int. Code only.",
     "assert parse_duration('1h30m') == 5400\nassert parse_duration('45s') == 45\nassert parse_duration('2h5s') == 7205"),
    ("valid_brackets", "Write a Python function balanced(s) that returns True if every (), [] and {} in s is properly balanced and nested; other characters are ignored. Code only.",
     "assert balanced('a(b[c]{d})')\nassert not balanced('(]')\nassert not balanced('((')\nassert balanced('')"),
]


async def run_tier(config, tier: str) -> dict:
    rt = DelegationRuntime(dataclasses.replace(config, max_escalations=0), call_params={"temperature": 0})
    rows = []
    for name, task, check in TASKS:
        start = time.perf_counter()
        dt = await rt.run(task, tier=tier, check=check)
        a = dt.attempts[-1] if dt.attempts else None
        rows.append({
            "task": name,
            "outcome": a.outcome if a else "error",
            "seconds": round(time.perf_counter() - start, 2),
            "why": (a.check["output"].strip().splitlines()[-1] if a and a.check and not a.check["passed"] else ""),
        })
    claimed = [r for r in rows if r["outcome"] in ("ok", "check_failed")]
    false_done = [r for r in claimed if r["outcome"] == "check_failed"]
    return {
        "tier": tier,
        "tasks": len(rows),
        "passed": sum(r["outcome"] == "ok" for r in rows),
        "false_done_rate": round(len(false_done) / len(claimed), 3) if claimed else None,
        "unsure_or_error": sum(r["outcome"] not in ("ok", "check_failed") for r in rows),
        "rows": rows,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--tiers", nargs="+", help="tier names to measure (default: all)")
    parser.add_argument("--json", help="also write full results to this file")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.code_checks:
        raise SystemExit("set `code_checks: true` in the config first (checks run model-written code)")
    results = []
    for tier in args.tiers or [t.name for t in config.tiers]:
        res = await run_tier(config, tier)
        results.append(res)
        print(f"\n{tier}: {res['passed']}/{res['tasks']} passed, false-done rate {res['false_done_rate']}")
        for r in res["rows"]:
            print(f"  {r['task']:<16} {r['outcome']:<13} {r['seconds']:>6}s  {r['why'][:90]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
