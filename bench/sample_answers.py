"""Extra sampled answers per model and problem, for the agreement check (bench/agreement.py).

    python bench/sample_answers.py --config vm/models.vm.yaml --tiers qwen3.5-4b,qwen3.5-9b --samples 2

The benchmark (bench/false_done.py) recorded one answer per model at temperature 0. Here each
model answers every HumanEval+ and MBPP+ problem `--samples` more times at --temperature,
with the same worker brief, and no check. Answers are cached in
runs/cache/samples.json ({task set: {tier: {problem: [answers]}}}), so a rerun only fills gaps.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path

from tasksets import load_tasks

from gearbox.config import load_config
from gearbox.delegate.runtime import DelegationRuntime


async def sample(config, tier: str, tasks, samples: int, temperature: float, concurrency: int,
                 store: dict, save) -> None:
    cfg = dataclasses.replace(config, max_escalations=0, max_concurrent=concurrency, code_checks=False,
                              delegation_mode="async")
    rt = DelegationRuntime(cfg, call_params={"temperature": temperature})
    todo = [(t, i) for t in tasks for i in range(len(store.get(t.name, [])), samples)]
    started = time.time()

    async def one(task, _):
        dt = await rt.run(task.prompt, tier=tier, show_check=False)
        store.setdefault(task.name, []).append(dt.result or "")

    for start in range(0, len(todo), 50):  # save every 50 answers: the lab's connection drops
        await asyncio.gather(*(one(t, i) for t, i in todo[start:start + 50]))
        save()
        print(f"  {tier}: {min(start + 50, len(todo))}/{len(todo)} answers, {time.time() - started:.0f}s", flush=True)


async def main_async(args) -> None:
    config = load_config(Path(args.config))
    out = Path(args.out)
    cache = json.loads(out.read_text()) if out.exists() else {}

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cache))

    for ts in ("humaneval+", "mbpp+"):
        tasks = load_tasks(ts)
        for tier in args.tiers.split(","):
            print(f"{ts} / {tier}: {args.samples} samples per problem at temperature {args.temperature}", flush=True)
            await sample(config, tier, tasks, args.samples, args.temperature, args.concurrency,
                         cache.setdefault(ts, {}).setdefault(tier, {}), save)
    save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="vm/models.vm.yaml")
    ap.add_argument("--tiers", default="qwen3.5-4b,qwen3.5-9b")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="runs/cache/samples.json")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
