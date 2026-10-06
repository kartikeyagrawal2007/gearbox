"""Collect model-based features for the learned router: judge ratings and embeddings.

    python bench/router_features.py --config vm/models.vm.yaml --judges qwen3.5-4b,qwen3.5-9b \\
        --embed-model nomic-embed-text --out runs/cache/router_features.json

Surface text features (bench/router_train.py) can't tell how hard a coding problem is. These
can, if anything can:

  judge      a model reads the problem (it does not solve it) and rates its difficulty 1-10.
             Costs one prompt read and a few output tokens per problem.
  embedding  a vector for the problem's meaning, from an Ollama embedding model
             (`ollama pull nomic-embed-text` first).

Results are cached per (feature, problem) in --out, so a rerun only fills in what's missing.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import re
from pathlib import Path

import httpx
from tasksets import EVALPLUS_SETS

from gearbox.config import load_config
from gearbox.providers import LiteLLMProvider

JUDGE_PROMPT = (
    "Here is a programming task. Do not solve it. Rate how hard it is for a small language model "
    "to solve correctly on the first try, judged by hidden tests that include edge cases: 1 = trivial, "
    "10 = very hard. Reply with only the number.\n\n{task}"
)
_NUMBER = re.compile(r"\b(10|[1-9])\b")


def problems() -> dict[str, dict[str, str]]:
    """{task_set: {problem id: prompt}} for HumanEval+ and MBPP+."""
    out = {}
    for ts in ("humaneval+", "mbpp+"):
        _, gz_name, _ = EVALPLUS_SETS[ts]
        with gzip.open(Path(__file__).parent / "data" / gz_name, "rt") as f:
            out[ts] = {p["task_id"]: p["prompt"].strip() for p in map(json.loads, f)}
    return out


async def judge_all(provider, tier, todo: list[tuple[str, str]], store: dict, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(pid: str, text: str) -> None:
        async with sem:
            c = await provider.complete(tier, [{"role": "user", "content": JUDGE_PROMPT.format(task=text)}],
                                        max_tokens=8, temperature=0)
        m = _NUMBER.search(c.text)
        store[pid] = int(m.group(1)) if m else None

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

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cache))

    if args.judges:
        config = load_config(Path(args.config))
        provider = LiteLLMProvider()
        for name in args.judges.split(","):
            tier = config.tiers[config.tier_index(name)]
            store = cache.setdefault(f"judge:{name}", {})
            todo = [(pid, t) for ps in probs.values() for pid, t in ps.items() if pid not in store]
            if todo:
                print(f"judge {name}: rating {len(todo)} problems ...", flush=True)
                await judge_all(provider, tier, todo, store, args.concurrency)
                save()
            missing = sum(v is None for v in store.values())
            print(f"judge {name}: done ({missing} unparseable replies)", flush=True)
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
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="runs/cache/router_features.json")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
