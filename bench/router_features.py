"""Collect model-based features for the learned router: judge ratings and embeddings.

    python bench/router_features.py --config vm/models.vm.yaml --judges qwen3.5-4b,qwen3.5-9b \\
        --embed-model nomic-embed-text --out runs/cache/router_features.json

Surface text features (bench/router_train.py) can't tell how hard a coding problem is. These
can, if anything can:

  judge      a model reads the problem (it does not solve it) and answers one question about it,
             a few output tokens per problem. --variant picks the question:
               rate   difficulty 1-10 (what gearbox/difficulty/learned.py asks)
               pass   chance 0-100 that a 4B model passes on the first try
               rate3  difficulty 1-10 asked 3 times at temperature 0.7, averaged
             Every call's time and tokens are kept too (key "latency:..."), to cost the judge.
  embedding  a vector for the problem's meaning, from an Ollama embedding model
             (`ollama pull nomic-embed-text` first).

Results are cached per (feature, problem) in --out, so a rerun only fills in what's missing.
To time judges without other calls running at once: --concurrency 1 --limit 100 --out <new file>.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import re
from pathlib import Path

import httpx
from tasksets import EVALPLUS_SETS

from gearbox.config import load_config
from gearbox.difficulty.learned import JUDGE_PROMPT, parse_rating  # one prompt for training and routing
from gearbox.providers import LiteLLMProvider
PASS_PROMPT = (
    "Here is a programming task. Do not solve it. Estimate the probability, from 0 to 100, that a small "
    "4-billion-parameter language model writes a solution that passes hidden tests, including edge cases, "
    "on its first try. Reply with only the number.\n\n{task}"
)
_PERCENT = re.compile(r"\b(100|\d{1,2})\b")


def parse_percent(text: str) -> int | None:
    m = _PERCENT.search(text)
    return int(m.group(1)) if m else None


# variant -> (prompt, parser, samples, temperature, cache key prefix)
VARIANTS = {
    "rate": (JUDGE_PROMPT, parse_rating, 1, 0.0, "judge"),
    "pass": (PASS_PROMPT, parse_percent, 1, 0.0, "judge-pass"),
    "rate3": (JUDGE_PROMPT, parse_rating, 3, 0.7, "judge-rate3"),
}


def problems() -> dict[str, dict[str, str]]:
    """{task_set: {problem id: prompt}} for HumanEval+ and MBPP+."""
    out = {}
    for ts in ("humaneval+", "mbpp+"):
        _, gz_name, _ = EVALPLUS_SETS[ts]
        with gzip.open(Path(__file__).parent / "data" / gz_name, "rt") as f:
            out[ts] = {p["task_id"]: p["prompt"].strip() for p in map(json.loads, f)}
    return out


async def judge_all(provider, tier, variant: str, todo: list[tuple[str, str]], store: dict, latency: dict,
                    concurrency: int) -> None:
    prompt, parse, samples, temperature, _ = VARIANTS[variant]
    sem = asyncio.Semaphore(concurrency)

    async def one(pid: str, text: str) -> None:
        values, seconds, tokens_in, tokens_out = [], 0.0, 0, 0
        for _ in range(samples):
            async with sem:
                c = await provider.complete(tier, [{"role": "user", "content": prompt.format(task=text)}],
                                            max_tokens=8, temperature=temperature)
            seconds += c.latency_s
            tokens_in += c.input_tokens
            tokens_out += c.output_tokens
            if (v := parse(c.text)) is not None:
                values.append(v)
        store[pid] = round(sum(values) / len(values), 3) if values else None
        latency[pid] = [round(seconds, 3), tokens_in, tokens_out]

    await asyncio.gather(*(one(pid, text) for pid, text in todo))


async def embed_all(base: str, model: str, todo: list[tuple[str, str]], store: dict, batch: int = 16) -> None:
    async with httpx.AsyncClient(timeout=300) as client:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            r = await client.post(f"{base}/api/embed", json={"model": model, "input": [t for _, t in chunk]})
            r.raise_for_status()
            for (pid, _), vec in zip(chunk, r.json()["embeddings"]):
                store[pid] = [round(x, 5) for x in vec]


async def main_async(args) -> None:
    out = Path(args.out)
    cache = json.loads(out.read_text()) if out.exists() else {}
    probs = problems()
    if args.limit:  # a fixed random sample, the same for every judge
        pool = sorted((ts, pid) for ts, ps in probs.items() for pid in ps)
        keep = set(random.Random(0).sample(pool, min(args.limit, len(pool))))
        probs = {ts: {pid: t for pid, t in ps.items() if (ts, pid) in keep} for ts, ps in probs.items()}

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cache))

    if args.judges:
        config = load_config(Path(args.config))
        provider = LiteLLMProvider()
        for name in args.judges.split(","):
            tier = config.tiers[config.tier_index(name)]
            key = f"{VARIANTS[args.variant][4]}:{name}"
            store = cache.setdefault(key, {})
            latency = cache.setdefault(f"latency:{key}", {})
            todo = [(pid, t) for ps in probs.values() for pid, t in ps.items() if pid not in store]
            if todo:
                print(f"{key}: judging {len(todo)} problems ...", flush=True)
                await judge_all(provider, tier, args.variant, todo, store, latency, args.concurrency)
                save()
            missing = sum(v is None for v in store.values())
            secs = [v[0] for v in latency.values()]
            timing = f", median {sorted(secs)[len(secs) // 2]:.2f}s per problem" if secs else ""
            print(f"{key}: done ({missing} unparseable replies{timing})", flush=True)
    if args.embed_model:
        store = cache.setdefault(f"embed:{args.embed_model}", {})
        todo = [(pid, t) for ps in probs.values() for pid, t in ps.items() if pid not in store]
        if todo:
            print(f"embedding {len(todo)} problems with {args.embed_model} ...", flush=True)
            await embed_all(args.ollama, args.embed_model, todo, store)
            save()
        print(f"embeddings: done ({len(next(iter(store.values())))} dimensions)", flush=True)
    save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="vm/models.vm.yaml")
    ap.add_argument("--judges", default="", help="comma-separated judge tiers")
    ap.add_argument("--embed-model", default="", help="Ollama embedding model, e.g. nomic-embed-text")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="rate", help="the question judges answer")
    ap.add_argument("--limit", type=int, default=0, help="only a fixed random sample of this many problems")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="runs/cache/router_features.json")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
