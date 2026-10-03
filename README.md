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

Then open http://127.0.0.1:8790 (change it with `--port`). The dashboard binds to localhost only and has no auth.

**With a real agent as the host**, let the MCP server serve the dashboard too. Delegations Claude Code makes then appear live:

```bash
claude mcp add gearbox -e GEARBOX_CONFIG=/abs/path/gearbox.yaml -- /abs/path/.venv/bin/gearbox-mcp --dashboard 8777
```

Open http://127.0.0.1:8777, then ask Claude Code to "use gearbox to delegate writing the tests for X".

How the race keeps its numbers honest:
- **Warm-up.** Each model is loaded with an untimed call first. Without it, the first phase pays the cold start; one early run showed 1.31× when the real figure was about 1.1×.
- **Same work.** All calls use temperature 0. If the phases still made different worker calls (say an `UNSURE` escalation in only one phase), the race reports "not comparable" instead of a speedup.
- **Ceiling.** Async can only remove time the host spent blocked, so the race reports the best async could have done: `wall / (wall − host blocked)`, computed from the blocking phase.

What it showed on an M4 Mac (host qwen2.5-coder:14b, worker qwen2.5-coder:1.5b, one shared GPU):
- Delegation itself paid: the 1.5B finished the default subtasks in about 1s, and the workload dropped from about 25s (when everything escalated to the 14B) to about 15s.
- Async did not: three fair runs gave 0.98–1.04×, under a ceiling of about 1.09×. The host was only blocked about 1.3s of 15s, and overlapped calls contend for the same GPU. Async should pay where workers run long and on separate hardware from the host.

## Verify, don't trust: executable checks

Models often claim success when they're wrong. Pass a `check`, Python asserts or `test_*` functions, and Gearbox runs it against the worker's answer. An answer is accepted only if it passes. A failing answer (a "false done") escalates one tier, and the next model sees the failed answer and exactly what went wrong, e.g. `slugify('Héllo  World!') returned 'héllo-world', expected 'hello-world'`.

```bash
gearbox run "Write add(a, b). Code only." --tier small --check "assert add(2, 3) == 5"
```

The same `check` parameter works on the MCP `delegate` tool and in the dashboard. Results report `verified: true/false`, and the ledger tracks each tier's false-done rate.

Checks **execute model-written code**, so they are off unless `code_checks: true`. Isolation is best effort, not a security boundary: a temp dir, an empty environment, Python isolated mode, a timeout, CPU and file limits, and no network (`sandbox-exec` on macOS, `unshare -rn` on Linux when permitted).

To measure each model's false-done rate on 8 coding tasks:

```bash
python bench/false_done.py --config gearbox.yaml --tiers small large
```

First result: qwen2.5-coder:1.5b passed 5/8 and claimed success on all 8. That's a 37.5% false-done rate, and not one of its wrong answers said "unsure".

For real statistics, use **HumanEval+**: the 164 HumanEval problems with EvalPlus's extended tests (~760 inputs each), downloaded on first use. Its checks are hidden from the worker, as in the official evaluation:

```bash
python bench/false_done.py --config gearbox.yaml --tasks humaneval+ --json runs/humaneval.json
```

Use `--tasks humaneval+mini` for a faster run with EvalPlus's reduced tests, or `--limit 20` for a trial.

**Harness validation:** qwen2.5-coder:1.5b (Ollama 4-bit, temperature 0) scored 65.2% (107/164). The Qwen2.5-Coder report gives 66.5% for this model. All 164 reference solutions pass their own generated checks (a regression test). The 57 failed answers were all claimed as done, and none said "unsure".

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
