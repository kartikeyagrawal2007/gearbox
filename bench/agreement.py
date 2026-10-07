"""The agreement check: escalate when a model's own answers disagree, no test needed.

    python bench/agreement.py                 # needs runs/ results and runs/cache/samples.json

Routing can't tell in advance which problems a model will fail (docs/paper/results.md,
finding 12), and a weak check lets wrong answers through (finding 9). This asks the model
instead: take its answer plus extra sampled answers (bench/sample_answers.py), run them all
on a few sample *inputs* (no expected outputs, which a host can't always supply) and compare
what they return. If they all behave the same, accept the answer; if not, escalate.

Sample inputs are the first --probes EvalPlus base inputs, used only as inputs. Outputs and
exceptions are compared exactly, in a canonical form (sets and dicts sorted, memory addresses
removed), so identical code can't look different from one run to the next. The cascade is 4B -> 9B -> 27B. Each tier
costs (1 + samples) answers when it is checked, and the top model is accepted as is. It is
compared with fixed models, the weak-check cascade, and random mixing at equal quality.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
from evalplus_compat import mbpp_deserialize_inputs
from plots import params_b
from router_fair import mix_cost
from router_train import load_set
from tasksets import EVALPLUS_SETS

from gearbox.verify.checks import _command, extract_code

LADDER = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]

HARNESS = r'''
import io, json, pickle, re, resource, sys, copy, contextlib
ADDR = re.compile(r" at 0x[0-9a-f]+")  # object reprs carry memory addresses, which differ run to run

def canon(x):
    """Sets and dicts in a fixed order, so equal values print the same in any process."""
    if isinstance(x, (set, frozenset)):
        return "{" + ", ".join(sorted(canon(i) for i in x)) + "}"
    if isinstance(x, dict):
        return "{" + ", ".join(sorted(canon(k) + ": " + canon(v) for k, v in x.items())) + "}"
    if isinstance(x, (list, tuple)):
        inner = ", ".join(canon(i) for i in x)
        return ("[" + inner + "]") if isinstance(x, list) else ("(" + inner + ("," if len(x) == 1 else "") + ")")
    return ADDR.sub("", repr(x))
cpu = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
entry, inputs = pickle.load(open("inputs.pkl", "rb"))
ns = {}
out = []
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(open("solution.py").read(), ns)
    fn = ns[entry]
except BaseException as e:
    print(json.dumps(["LOAD:" + type(e).__name__] * len(inputs))); sys.exit(0)
for args in inputs:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = fn(*copy.deepcopy(args))
        out.append(canon(r)[:300])
    except BaseException as e:
        out.append("EXC:" + type(e).__name__)
print(json.dumps(out))
'''


async def behaviour(answer: str, entry: str, inputs: list, timeout: float = 10.0) -> tuple[str, ...]:
    """What the answer's function returns (or raises) on each input: its observable behaviour."""
    with tempfile.TemporaryDirectory(prefix="gearbox-agree-") as tmp:
        work = Path(tmp)
        (work / "runner.py").write_text(HARNESS)
        (work / "solution.py").write_text(extract_code(answer))
        (work / "inputs.pkl").write_bytes(pickle.dumps((entry, inputs)))
        proc = await asyncio.create_subprocess_exec(*_command(work, int(timeout) + 2), cwd=work,
                                                    env={"PATH": "/usr/bin:/bin", "HOME": tmp},
                                                    stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
            return tuple(json.loads(out.decode().strip().splitlines()[-1]))
        except (asyncio.TimeoutError, ValueError, IndexError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ("TIMEOUT",) * len(inputs)


def agree(behaviours: list[tuple]) -> bool:
    return len(set(behaviours)) == 1


async def compute(runs: Path, samples: dict, probes: int, concurrency: int, cache: dict) -> None:
    sem = asyncio.Semaphore(concurrency)
    for ts in ("humaneval+", "mbpp+"):
        dataset, gz, _ = EVALPLUS_SETS[ts]
        with gzip.open(Path(__file__).parent / "data" / gz, "rt") as f:
            problems = {p["task_id"]: p for p in map(json.loads, f)}
        recorded = {}
        for path in sorted(runs.glob("*.json"), key=lambda p: p.stat().st_mtime):
            data = json.loads(path.read_text())
            for r in data if isinstance(data, list) else []:
                if isinstance(r, dict) and r.get("task_set") == ts and r.get("hatch", "on") == "on" and r["tier"] in LADDER:
                    recorded[r["tier"]] = {x["task"]: x["answer"] for x in r["rows"]}
        store = cache.setdefault(ts, {})

        async def one(tier, pid):
            p = problems[pid]
            inputs = list(p["base_input"])[:probes]
            if dataset == "mbpp":
                inputs = mbpp_deserialize_inputs(pid, inputs)
            answers = [recorded[tier][pid], *samples.get(ts, {}).get(tier, {}).get(pid, [])]
            async with sem:
                store.setdefault(tier, {})[pid] = [list(await behaviour(a, p["entry_point"], inputs)) for a in answers]

        todo = [(t, pid) for t in LADDER[:-1] for pid in problems if pid not in store.get(t, {})
                and samples.get(ts, {}).get(t, {}).get(pid)]
        if todo:
            print(f"{ts}: running {len(todo)} problem x model answer sets on sample inputs ...", flush=True)
            await asyncio.gather(*(one(t, pid) for t, pid in todo))


def evaluate(runs: Path, cache: dict, n_samples: int) -> list[str]:
    lines = ["Agreement cascade 4B -> 9B -> 27B: accept a model's answer when its answers behave alike on the",
             "sample inputs, else escalate. Cost counts every answer generated; quality is vs the 27B."]
    costs = {t: params_b(t) for t in LADDER}
    for ts in ("humaneval+", "mbpp+"):
        d = load_set(runs, ts)
        ids = [i for i in d["ids"] if all(i in cache.get(ts, {}).get(t, {}) for t in LADDER[:-1])]
        if not ids:
            lines.append(f"\n{ts}: no sampled answers yet")
            continue
        idx = {pid: k for k, pid in enumerate(d["ids"])}
        passed = {t: np.array([d["passed"][t][idx[i]] for i in ids]) for t in LADDER}
        top = passed[LADDER[-1]].mean()
        fixed = [(passed[t].mean() / top, costs[t]) for t in LADDER]
        correct, cost, accepted_at = [], [], {t: 0 for t in LADDER}
        agree_rate, agree_correct = {}, {}
        for t in LADDER[:-1]:
            a = np.array([agree([tuple(b) for b in cache[ts][t][i]]) for i in ids])
            agree_rate[t] = a.mean()
            agree_correct[t] = passed[t][a].mean() if a.any() else float("nan")
        for k, pid in enumerate(ids):
            c = 0.0
            for t in LADDER:
                if t == LADDER[-1]:
                    c += costs[t]
                    correct.append(passed[t][k]); accepted_at[t] += 1
                    break
                c += costs[t] * (1 + n_samples)
                if agree([tuple(b) for b in cache[ts][t][pid]]):
                    correct.append(passed[t][k]); accepted_at[t] += 1
                    break
            cost.append(c)
        q, c = np.mean(correct) / top, float(np.mean(cost))
        lines += ["", f"== {ts} ({len(ids)} problems)",
                  *[f"  {t}: answers agree on {agree_rate[t]:.0%} of problems; when they agree, {agree_correct[t]:.0%} are correct"
                    f" (vs {passed[t].mean():.0%} overall)" for t in LADDER[:-1]],
                  f"  accepted at: " + ", ".join(f"{t} {n}" for t, n in accepted_at.items()),
                  f"  agreement cascade: quality {q:.1%} of the 27B at cost {c:.1f}"
                  f" | random mix of fixed models at that quality: {mix_cost(fixed, q):.1f}"
                  f" -> {mix_cost(fixed, q) / c:.2f}x  (always 27B: cost 27.0)"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--samples-file", default="runs/cache/samples.json")
    ap.add_argument("--probes", type=int, default=5, help="sample inputs per problem")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    runs = Path(args.runs)
    samples = json.loads(Path(args.samples_file).read_text())
    cache_path = runs / "cache" / "behaviours.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    asyncio.run(compute(runs, samples, args.probes, args.concurrency, cache))
    cache_path.write_text(json.dumps(cache))
    n = max((len(v) for ts in samples.values() for t in ts.values() for v in t.values()), default=0)
    print("\n".join(evaluate(runs, cache, n)))


if __name__ == "__main__":
    sys.exit(main())
