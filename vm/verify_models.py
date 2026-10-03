"""Check every downloaded model in a config actually answers, and that thinking is off.

    .venv/bin/python vm/verify_models.py --config vm/models.vm.yaml

Models not downloaded yet are listed as "not pulled" and skipped. For each pulled model it
reports latency, output tokens, whether the reply contained hidden reasoning ("thinking"),
and the reply itself. Catches wrong tags, unsupported parameters, and models that think
when they should not before a benchmark wastes hours on them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request

import litellm

from gearbox.config import load_config

PROMPT = "Write a Python function add(a, b) that returns their sum. Code only."


def pulled_models() -> set[str]:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
        return {m["name"] for m in json.load(resp)["models"]}


async def probe(tier) -> dict:
    start = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=tier.model, messages=[{"role": "user", "content": PROMPT}], temperature=0, **tier.params
        )
    except Exception as e:
        return {"tier": tier.name, "status": "ERROR", "detail": f"{type(e).__name__}: {str(e)[:160]}"}
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    content = msg.content or ""
    return {
        "tier": tier.name,
        "status": "OK" if content.strip() and not reasoning and "<think>" not in content else "CHECK",
        "seconds": round(time.perf_counter() - start, 2),
        "out_tokens": resp.usage.completion_tokens,
        "thinking": bool(reasoning) or "<think>" in content,
        "reply": " ".join(content.split())[:70],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="vm/models.vm.yaml")
    args = parser.parse_args()
    litellm.suppress_debug_info = True
    config = load_config(args.config)
    have = pulled_models()
    print("=================== PASTE EVERYTHING BELOW THIS LINE ===================")
    for tier in config.tiers:
        tag = tier.model.split("/", 1)[1]
        if tag not in have and f"{tag}:latest" not in have:
            print(f"{tier.name:<20} not pulled")
            continue
        r = await probe(tier)  # one at a time: each model loads into the GPU in turn
        if r["status"] == "ERROR":
            print(f"{r['tier']:<20} ERROR  {r['detail']}")
        else:
            print(f"{r['tier']:<20} {r['status']:<5}  {r['seconds']:>6}s  {r['out_tokens']:>4} tok  "
                  f"thinking={'YES' if r['thinking'] else 'no'}  | {r['reply']}")
    print("========================================================================")


if __name__ == "__main__":
    asyncio.run(main())
