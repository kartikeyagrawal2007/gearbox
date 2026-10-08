# Gearbox

**An automatic transmission for LLM agents.** An expensive "boss" model (the agent in Claude Code, Antigravity, Cursor, …) hands self-contained subtasks to cheaper models, keeps working while they run, and accepts an answer only once it passes an executable check.

- **Picks the gear.** Each subtask goes to the cheapest model that should handle it. It never starts on the tiniest ones, and it starts higher when the check is weak.
- **Checks, doesn't trust.** Cheap models say "done" when they're wrong and almost never say "unsure". Gearbox runs tests on every answer and escalates a failure to a stronger model, showing it what failed.
- **Doesn't wait, when that helps.** With a cloud boss, delegated work runs while the boss keeps going. When the boss shares the workers' GPU, Gearbox holds the work until the boss waits, so the two never fight over it.

## What we measured

12 open models × 542 coding problems with hidden tests (HumanEval+ and MBPP+), on an RTX A5000:

| | Result |
|---|---|
| Cheap model first, check, escalate on failure | **95.7%** vs 92.7% for always using Qwen3.5 27B, at **about 1/3 of the compute** |
| Starting at a 4B instead of the tiniest model | same accuracy, **2.8× faster** |
| Boss's own writing when coding is delegated | **about half** |
| Boss keeps working (cloud boss, local workers) | **1.1–1.6× faster**, as our formula predicts |
| Picking a model per task up front (our router, and RouteLLM) | predicts, but **saves nothing** on code when measured fairly |

Details and caveats: [REPORT.md](REPORT.md).

## Start here

| If you want to… | Read |
|---|---|
| Know what we found, and whether it worked | [REPORT.md](REPORT.md) |
| See what's built, what's missing, and what's next | [ROADMAP.md](ROADMAP.md) |
| Understand how the code works | [EXPLANATION.md](EXPLANATION.md) |
| Contribute (setup, lab PC, rules for experiments) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Run or extend an experiment | [bench/README.md](bench/README.md) |
| Read the paper draft | [docs/paper/results.md](docs/paper/results.md), [docs/paper/outline.md](docs/paper/outline.md) |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # extras: plots (figures), gpu (energy), baselines (RouteLLM)
cp gearbox.example.yaml gearbox.yaml
.venv/bin/gearbox ui --simulate      # dashboard with fake models: http://127.0.0.1:8790
```

In the dashboard, **Watch a delegation → Run the batch** shows a batch being routed, handed off, worked on in parallel, checked and returned.

For real models, set the tiers in `gearbox.yaml`. A tier is any [LiteLLM](https://docs.litellm.ai/) model string (Ollama, vLLM, Anthropic, OpenAI, Gemini, OpenRouter, …). Then:

```bash
.venv/bin/gearbox ui                                                        # dashboard, real models
.venv/bin/gearbox route "Design a lock-free queue and prove it" --risk high # which model, and why
.venv/bin/gearbox run "Write add(a, b). Code only." --check "assert add(2, 3) == 5"
```

Checks run model-written code, so they're off unless `code_checks: true`. Isolation is best effort (a temp folder, an empty environment, time and CPU limits, no network where supported). It is **not** a security boundary; see ROADMAP T1.

## Use it from an agent (MCP)

Tools: `delegate`, `await_result`, `status`, `cancel`, `ledger`, `route`. Add `--dashboard 8777` to watch delegations live.

```bash
claude mcp add gearbox -e GEARBOX_CONFIG=/abs/path/gearbox.yaml -- /abs/path/.venv/bin/gearbox-mcp
```

For **Antigravity** (`~/.gemini/config/mcp_config.json`) or **Cursor** (`.cursor/mcp.json`):

```json
{ "mcpServers": { "gearbox": { "command": "/abs/path/.venv/bin/gearbox-mcp",
                               "env": { "GEARBOX_CONFIG": "/abs/path/gearbox.yaml" } } } }
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## License

[Apache-2.0](LICENSE)
