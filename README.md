# Gearbox

**An automatic transmission for LLM agents.**

Gearbox does two things:

1. **Picks the gear.** It estimates how hard a prompt is and routes it to the cheapest model tier that should handle it. **Leverage** keeps it one or more tiers above that estimate as a safety margin. Leverage can be constant, or scaled by how costly a wrong answer would be (`risk: low | medium | high`) and by how unsure the estimator is.
2. **Downshifts subtasks, asynchronously.** An expensive "host" model (the agent in Claude Code, Antigravity, Cursor, ...) hands a self-contained subtask to a cheaper model and **keeps working**. It blocks only when it actually needs the result. If a worker replies `UNSURE`, errors, or times out, the subtask escalates one tier.

Every token Gearbox spends is recorded in a ledger. The ledger also gives a conservative estimate of what the host saved, and measures **how much of the worker's run time the host overlapped** with other work.

> Status: early alpha (v0.1). This is the tool half of a research project on *asynchronous heterogeneous delegation*. See [docs/related-work.md](docs/related-work.md) for how it relates to existing routers and delegation work.

## Why async?

Most delegation today is blocking: the big model waits for the small one. Waiting on a stateless API costs no tokens, so the token savings come from delegating at all. Async adds **wall-clock** savings on top.

Async has costs of its own:
- work done while waiting may need redoing once the result arrives
- prompt caches may expire during the wait
- integrating the result costs tokens

Gearbox is built to measure when async delegation wins.

## The cost model, in one paragraph

When a host delegates through a tool call, everything it writes into the brief is billed as **its own output tokens**, usually its most expensive rate. Delegation therefore pays off when the worker's output is much longer than the brief, or when the worker reads context by reference instead of having it pasted in. `gearbox.cost.break_even` makes that decision explicit. Pass `expected_output_tokens` to `delegate` to see it.

## Install

```bash
pip install -e ".[dev]"      # add ,gpu for NVML energy measurement on NVIDIA GPUs
cp gearbox.example.yaml gearbox.yaml
```

Edit `gearbox.yaml`. Tiers are any [LiteLLM](https://docs.litellm.ai/) model strings: Ollama, vLLM, Anthropic, OpenAI, Gemini, OpenRouter, and others. Prices in the example are placeholders.

## See it running

**The dashboard** (`gearbox ui`) lets you delegate subtasks, preview which gear a prompt lands in and why, watch attempts and escalations live, and read the ledger. Its **race** panel runs the same workload blocking and then async, and draws both measured timelines.

```bash
gearbox ui --simulate          # instant: fake models, no backend needed
gearbox ui                     # real models from gearbox.yaml (e.g. Ollama)
```

Then open http://127.0.0.1:8765. The dashboard binds to localhost only and has no auth.

**With a real agent as the host**, let the MCP server serve the dashboard too. Delegations Claude Code makes then appear live:

```bash
claude mcp add gearbox -e GEARBOX_CONFIG=/abs/path/gearbox.yaml -- /abs/path/.venv/bin/gearbox-mcp --dashboard 8777
```

Open http://127.0.0.1:8777, then ask Claude Code to "use gearbox to delegate writing the tests for X".

Notes on the race numbers:
- Each model is loaded with an untimed warm-up call first. Without it, the first phase pays the cold start, which inflated one early run to 1.31× when the real figure was about 1.1×.
- When host and worker share one GPU, the host stops waiting but the overlapped calls slow each other down, so async gains little wall-clock time. On an M4 with one 14B model we measured 1.08× and 1.16× over two warm runs. Separate hardware for host and worker is where async should pay.

## CLI

```bash
gearbox tiers
gearbox route "Design a lock-free queue and prove it is linearizable" --risk high
gearbox run "Write pytest cases for a slugify(text) function" --acceptance "covers unicode and empty input"
```

## Use it from an agent (MCP)

Gearbox exposes these MCP tools: `route`, `delegate`, `await_result`, `status`, `cancel` and `ledger`.

**Claude Code** (add `--dashboard PORT` after `gearbox-mcp` to watch it live)
```bash
claude mcp add gearbox -e GEARBOX_CONFIG=/abs/path/gearbox.yaml -- /abs/path/.venv/bin/gearbox-mcp
```

**Antigravity**: add this to `~/.gemini/config/mcp_config.json`. **Cursor**: add the same block to `.cursor/mcp.json`.
```json
{
  "mcpServers": {
    "gearbox": {
      "command": "/abs/path/.venv/bin/gearbox-mcp",
      "env": { "GEARBOX_CONFIG": "/abs/path/gearbox.yaml" }
    }
  }
}
```

## Library

```python
import asyncio
from gearbox import DelegationRuntime, load_config

async def main():
    rt = DelegationRuntime(load_config())
    job = rt.delegate("Write docstrings for utils.py", context=open("utils.py").read(), risk="low")
    ...                                  # keep working
    print(await rt.await_result(job.id))
    print(rt.stats())

asyncio.run(main())
```

## Development

```bash
pytest
```

## License

Apache-2.0
