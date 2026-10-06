"""Async delegation experiment (RQ1): does a host that keeps working finish sooner?

    python bench/async_bench.py --config vm/async.vm.yaml --host qwen3.5-27b \\
        --workers qwen3.5-4b,qwen3.5-4b-cpu --out runs/async.jsonl      # run (resumable)
    python bench/async_bench.py --summary runs/async.jsonl              # tables only
    python bench/async_bench.py --host remote:1.0:80,remote:2.0:40 --workers qwen3.5-4b \
        --reps 3 --out runs/async_remote.jsonl                          # cloud host, emulated
    python bench/async_bench.py --simulate --k 2 --reps 1               # dry run, no models

An episode is one host agent's job: k coding subtasks (HumanEval+ problems, verified by
their hidden tests) and k host steps of its own (listing the edge cases each function's
tests must cover, capped at --host-tokens). The same episode runs in four modes:

  host_only  the host solves the subtasks itself (no delegation)
  blocking   delegate a subtask, wait for it, do a host step; repeat
  parallel   delegate all subtasks, wait for all, then do the host steps
  async      delegate all subtasks, do the host steps meanwhile, then collect

blocking -> parallel isolates the gain from workers running side by side;
parallel -> async isolates the gain from the host not waiting (the paper's claim).
The blocking trace bounds async: speedup <= wall / (wall - host blocked).

Fairness: temperature 0; both models warmed up (untimed) before each condition; the
mode order rotates across repetitions (a Latin square when --reps is a multiple of the
number of modes) so
no mode always runs first or benefits from the backend's prompt cache; and Ollama's
/api/ps is recorded per condition, so a model that spilled off the GPU is visible.
Placement is chosen by the worker tier: a tier with api_base on a CPU-only Ollama
(vm/ollama_cpu.sh) puts the worker on the CPU and leaves the GPU to the host.

Remote host (--host remote:TTFT:RATE): Gearbox's intended setting is a cloud host (e.g. the
model inside Claude Code) with workers on local hardware. We can't call a paid API, so the
host is emulated by time: each host step takes TTFT + host_tokens / RATE seconds and uses no
local hardware, while the workers stay real (GPU, real answers, real checks). host_only is
skipped for an emulated host: it would need the cloud model's own answers.

Each episode is appended to --out as one JSON line, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gzip
import json
import random
import statistics
import time
from pathlib import Path

from tasksets import EVALPLUS_SETS, Task, load_tasks

from gearbox.config import GearboxConfig, Tier, load_config
from gearbox.cost.energy import EnergyMeter
from gearbox.delegate.runtime import DelegationRuntime
from gearbox.providers import LiteLLMProvider, Provider, SimulatedProvider

MODES = ("host_only", "blocking", "parallel", "async")
CALL_PARAMS = {"temperature": 0}
HOST_STEP = (
    "Another engineer is implementing the function below. List the edge cases a thorough test "
    "suite for it must cover, one per line, each with a one-line reason. Be thorough.\n\n{spec}"
)


@dataclasses.dataclass(frozen=True)
class RemoteHost:
    """A cloud host emulated by time: it uses no local hardware, only the clock."""

    ttft_s: float  # time to first token
    tokens_per_s: float

    @property
    def name(self) -> str:
        return f"remote-{self.ttft_s:g}s-{self.tokens_per_s:g}tps"

    def seconds(self, tokens: int) -> float:
        return self.ttft_s + tokens / self.tokens_per_s


def parse_host(spec: str, config: GearboxConfig) -> Tier | RemoteHost:
    """A tier name, or remote:TTFT:RATE (e.g. remote:1.0:80)."""
    if spec.startswith("remote:"):
        try:
            _, ttft, rate = spec.split(":")
            return RemoteHost(float(ttft), float(rate))
        except ValueError:
            raise SystemExit(f"bad host {spec!r}: use remote:TTFT_SECONDS:TOKENS_PER_SECOND, e.g. remote:1.0:80")
    return config.tiers[config.tier_index(spec)]


def modes_for(host: Tier | RemoteHost) -> tuple[str, ...]:
    return MODES[1:] if isinstance(host, RemoteHost) else MODES


@dataclasses.dataclass(frozen=True)
class Episode:
    subtasks: tuple[Task, ...]
    host_steps: tuple[str, ...]


def make_episode(tasks: list[Task], specs: dict[str, str], k: int, rep: int, seed: int) -> Episode:
    """The same k problems for every mode and placement within one (k, rep)."""
    chosen = random.Random(f"{seed}/{k}/{rep}").sample(tasks, k)
    return Episode(tuple(chosen), tuple(HOST_STEP.format(spec=specs[t.name]) for t in chosen))


class Timeline:
    """Host spans ("work" or "blocked") on the wall clock, relative to the episode start."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.spans: list[tuple[str, float, float]] = []

    async def timed(self, kind: str, coro):
        start = time.time()
        try:
            return await coro
        finally:
            self.spans.append((kind, start - self.t0, time.time() - self.t0))

    def total(self, kind: str) -> float:
        return sum(end - start for k, start, end in self.spans if k == kind)


async def run_episode(mode: str, episode: Episode, config: GearboxConfig, provider: Provider,
                      host: Tier | RemoteHost, worker: str, host_tokens: int) -> dict:
    rt = DelegationRuntime(config, provider, call_params=CALL_PARAMS)
    tl = Timeline()
    host_out = 0

    async def host_step(prompt: str) -> None:
        nonlocal host_out
        if isinstance(host, RemoteHost):
            await asyncio.sleep(host.seconds(host_tokens))
            host_out += host_tokens
            return
        c = await provider.complete(host, [{"role": "user", "content": prompt}], max_tokens=host_tokens, **CALL_PARAMS)
        host_out += c.output_tokens

    def delegate(t: Task, tier: str):
        return rt.delegate(t.prompt, tier=tier, check=t.check, show_check=False)

    tasks = []
    if mode == "host_only":
        if isinstance(host, RemoteHost):
            raise ValueError("host_only needs a real host model")
        for t, step in zip(episode.subtasks, episode.host_steps):
            tasks.append(await tl.timed("work", rt.run(t.prompt, tier=host.name, check=t.check, show_check=False)))
            await tl.timed("work", host_step(step))
    elif mode == "blocking":
        for t, step in zip(episode.subtasks, episode.host_steps):
            tasks.append(delegate(t, worker))
            await tl.timed("blocked", rt.await_result(tasks[-1].id, timeout_s=None))
            await tl.timed("work", host_step(step))
    else:
        tasks = [delegate(t, worker) for t in episode.subtasks]
        if mode == "parallel":
            for dt in tasks:
                await tl.timed("blocked", rt.await_result(dt.id, timeout_s=None))
        for step in episode.host_steps:
            await tl.timed("work", host_step(step))
        if mode == "async":
            for dt in tasks:
                await tl.timed("blocked", rt.await_result(dt.id, timeout_s=None))
    wall = time.time() - tl.t0

    attempts = [a for dt in tasks for a in dt.attempts]
    starts = [a.started_at - tl.t0 for a in attempts]
    ends = [a.started_at - tl.t0 + a.latency_s for a in attempts]
    return {
        "mode": mode,
        "wall_s": round(wall, 3),
        "host_work_s": round(tl.total("work"), 3),
        "host_blocked_s": round(tl.total("blocked"), 3),
        "worker_busy_s": round(sum(a.latency_s for a in attempts), 3) if mode != "host_only" else 0.0,
        "worker_span_s": round(max(ends) - min(starts), 3) if attempts and mode != "host_only" else 0.0,
        "check_s": round(sum(a.check["duration_s"] for a in attempts if a.check), 3),
        "host_out_tokens": host_out + (sum(a.output_tokens for a in attempts) if mode == "host_only" else 0),
        "worker_out_tokens": sum(a.output_tokens for a in attempts) if mode != "host_only" else 0,
        "outcomes": [dt.attempts[-1].outcome if dt.attempts else (dt.error or "error") for dt in tasks],
        "spans": [[k, round(s, 3), round(e, 3)] for k, s, e in tl.spans],
    }


async def ollama_placement(tiers: list[Tier]) -> dict:
    """Share of each loaded model in GPU memory, from Ollama's /api/ps (None if not Ollama)."""
    import httpx

    out: dict[str, float | None] = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for tier in tiers:
            if not tier.model.startswith("ollama"):
                out[tier.name] = None
                continue
            base = (tier.api_base or "http://localhost:11434").rstrip("/")
            name = tier.model.split("/", 1)[1]
            try:
                models = (await client.get(f"{base}/api/ps")).json().get("models", [])
                m = next((m for m in models if m.get("name") == name or m.get("model") == name), None)
                out[tier.name] = round(m["size_vram"] / m["size"], 3) if m and m.get("size") else None
            except (httpx.HTTPError, ValueError):
                out[tier.name] = None
    return out


async def warm_up(provider: Provider, tiers: list[Tier]) -> None:
    for tier in tiers:
        await provider.complete(tier, [{"role": "user", "content": "Reply with OK."}], max_tokens=1)


def condition_key(r: dict) -> tuple:
    return (r["host"], r["worker"], r["k"], r["host_tokens"], r["rep"], r["mode"])


async def run(args) -> None:
    workers = args.workers.split(",")
    if args.simulate:
        from gearbox.config import Pricing

        names = [h for h in args.host.split(",") if not h.startswith("remote:")] + workers
        config = GearboxConfig(tiers=tuple(Tier(n, f"simulated/{n}") for n in names), host=Pricing())
        # Simulated latency grows with tier index, so list the host last: the slowest model.
        provider: Provider = SimulatedProvider(list(reversed(names)), base_s=0.2, per_tier_s=0.2)
    else:
        config = load_config(Path(args.config))
        provider = LiteLLMProvider()
    config = dataclasses.replace(config, max_escalations=0, max_concurrent=args.concurrency,
                                 code_checks=True, check_timeout_s=60, task_timeout_s=1800)
    hosts = [parse_host(h, config) for h in args.host.split(",")]
    for w in workers:
        config.tier_index(w)  # fail early on a typo

    _, gz_name, _ = EVALPLUS_SETS[args.tasks]
    tasks = load_tasks(args.tasks)
    with gzip.open(Path(__file__).parent / "data" / gz_name, "rt") as f:
        specs = {p["task_id"]: p["prompt"].strip() for p in map(json.loads, f)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {condition_key(json.loads(line)) for line in out.read_text().splitlines() if line.strip()}
    meter = EnergyMeter()
    ks = [int(x) for x in args.k.split(",")]
    host_tokens_list = [int(x) for x in args.host_tokens.split(",")]
    conditions = [(host, w, k, h, rep) for host in hosts for w in workers for k in ks
                  for h in host_tokens_list for rep in range(args.reps)]
    started, finished = time.time(), 0
    print(f"{len(conditions)} conditions; hosts {', '.join(h.name for h in hosts)}; "
          f"GPU energy {'on' if meter.available else 'unavailable'}; writing {out}", flush=True)

    for host, worker, k, host_tokens, rep in conditions:
        all_modes = modes_for(host)
        modes = all_modes[rep % len(all_modes):] + all_modes[:rep % len(all_modes)]  # rotate: Latin square
        todo = [m for m in modes if (host.name, worker, k, host_tokens, rep, m) not in done]
        if not todo:
            finished += 1
            continue
        worker_tier = config.tiers[config.tier_index(worker)]
        local = [t for t in (host, worker_tier) if isinstance(t, Tier)]  # an emulated host has nothing to load
        await warm_up(provider, local)
        placement = await ollama_placement(local) if not args.simulate else {}
        episode = make_episode(tasks, specs, k, rep, args.seed)
        for position, mode in enumerate(modes):
            if mode not in todo:
                continue
            with meter.span() as energy:
                record = await run_episode(mode, episode, config, provider, host, worker, host_tokens)
            record.update({
                "host": host.name, "worker": worker, "k": k, "host_tokens": host_tokens, "rep": rep,
                "order": position, "problems": [t.name for t in episode.subtasks],
                "gpu_joules": round(energy.joules, 1) if energy.joules is not None else None,
                "placement": placement, "finished_at": time.time(),
            })
            with out.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"  {host.name:<20} {worker:<16} k={k:<2} host_tokens={host_tokens:<5} rep={rep} {mode:<9} "
                  f"wall {record['wall_s']:7.1f}s  blocked {record['host_blocked_s']:6.1f}s  "
                  f"{','.join(record['outcomes'])}", flush=True)
            if args.cooldown:
                await asyncio.sleep(args.cooldown)
        finished += 1
        elapsed = time.time() - started
        print(f"  [{finished}/{len(conditions)} conditions, {elapsed / 60:.0f} min so far, "
              f"about {elapsed / finished * (len(conditions) - finished) / 60:.0f} min left]", flush=True)
    print(summarize(load_records(out)))


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(records: list[dict]) -> str:
    """Per condition: median speedups over repetitions, paired within each repetition."""
    groups: dict[tuple, dict[int, dict[str, dict]]] = {}
    for r in records:
        groups.setdefault((r["host"], r["worker"], r["k"], r["host_tokens"]), {}).setdefault(r["rep"], {})[r["mode"]] = r
    lines = ["", "Speedups are medians over repetitions (min-max). ceiling = blocking wall / (wall - host blocked).",
             f"  {'host':<20} {'worker':<18} {'k':>2} {'h.tok':>5} {'reps':>4} {'host_only s':>11} {'async s':>8}"
             f" {'blk/par':>12} {'par/async':>12} {'blk/async':>12} {'ceiling':>8} {'host_only/async':>16}"
             f" {'same answers':>12} {'host GPU':>8} {'worker GPU':>10}"]

    def ratio(reps, a, b):
        xs = [m[a]["wall_s"] / m[b]["wall_s"] for m in reps if a in m and b in m and m[b]["wall_s"] > 0]
        return f"{statistics.median(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})" if xs else "-"

    def median_of(reps, mode, key):
        xs = [m[mode][key] for m in reps if mode in m]
        return statistics.median(xs) if xs else None

    for (host, worker, k, ht), by_rep in sorted(groups.items()):
        reps = list(by_rep.values())
        ceilings = [m["blocking"]["wall_s"] / (m["blocking"]["wall_s"] - m["blocking"]["host_blocked_s"])
                    for m in reps if "blocking" in m and m["blocking"]["wall_s"] > m["blocking"]["host_blocked_s"]]
        # Delegating modes ran the same prompts at temperature 0; different outcomes mean different work.
        same = sum(len({tuple(m[x]["outcomes"]) for x in ("blocking", "parallel", "async") if x in m}) == 1 for m in reps)
        placements = [m["blocking"]["placement"] for m in reps if m.get("blocking", {}).get("placement")]

        def on_gpu(tier):
            share = placements[0].get(tier) if placements else None
            return "-" if share is None else f"{share:.0%}"

        ho, asy = median_of(reps, "host_only", "wall_s"), median_of(reps, "async", "wall_s")
        lines.append(
            f"  {host:<20} {worker:<18} {k:>2} {ht:>5} {len(reps):>4} {'-' if ho is None else f'{ho:.1f}':>11} {'-' if asy is None else f'{asy:.1f}':>8}"
            f" {ratio(reps, 'blocking', 'parallel'):>12} {ratio(reps, 'parallel', 'async'):>12}"
            f" {ratio(reps, 'blocking', 'async'):>12} {statistics.median(ceilings) if ceilings else 0:>8.2f}"
            f" {ratio(reps, 'host_only', 'async'):>16} {f'{same}/{len(reps)}':>12}"
            f" {on_gpu(host):>8} {on_gpu(worker):>10}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="vm/async.vm.yaml")
    ap.add_argument("--host", default="qwen3.5-27b",
                    help="host tier, or remote:TTFT:RATE for an emulated cloud host; comma-separate several")
    ap.add_argument("--workers", default="qwen3.5-4b,qwen3.5-4b-cpu", help="comma-separated worker tiers")
    ap.add_argument("--k", default="2,4,8", help="subtasks per episode (comma-separated)")
    ap.add_argument("--host-tokens", default="256,768", help="max tokens per host step (comma-separated)")
    ap.add_argument("--reps", type=int, default=4,
                    help="repetitions; a multiple of the mode count (4, or 3 for a remote host) balances mode order")
    ap.add_argument("--tasks", default="humaneval+", choices=sorted(EVALPLUS_SETS))
    ap.add_argument("--concurrency", type=int, default=4, help="worker calls at once (match OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to pause between episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/async.jsonl")
    ap.add_argument("--simulate", action="store_true", help="fake models with fixed latencies (tests the harness)")
    ap.add_argument("--summary", metavar="JSONL", help="only print the summary of a results file")
    args = ap.parse_args()
    if args.summary:
        print(summarize(load_records(Path(args.summary))))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
