# Gearbox

**An automatic transmission for LLM agents.** An expensive "host" model (the agent in Claude Code, Antigravity, Cursor, …) hands self-contained subtasks to cheaper models, keeps working while they run, and accepts an answer only once it passes an executable check.

- **Picks the gear.** Routes each subtask to the cheapest model tier that should handle it, plus **leverage**: a safety margin that grows with risk and with how unsure the difficulty estimate is.
- **Doesn't wait.** `delegate()` returns at once; the host blocks only when it needs the result. Gearbox measures how much waiting that actually saved.
- **Verifies, doesn't trust.** Cheap models say "done" when they're wrong and almost never say "unsure". Gearbox runs a check against each answer and escalates a failed one to a stronger tier, along with exactly what failed.

> **New here? Read [EXPLANATION.md](EXPLANATION.md).** It explains every part in plain language: the idea, the vocabulary, how a delegation flows through the code, the benchmarks, and what we've found.
>
> Status: research prototype (v0.1), the tool half of a paper on asynchronous, verified delegation to cheaper models. See [docs/paper/outline.md](docs/paper/outline.md) and [docs/lit-review.md](docs/lit-review.md).

## Quick start

```bash
pip install -e ".[dev]"           # extras: gpu (energy measurement), plots (figures)
cp gearbox.example.yaml gearbox.yaml
gearbox ui --simulate             # dashboard with fake models, no setup needed → http://127.0.0.1:8790
```

Edit `gearbox.yaml` to use real models. A tier is any [LiteLLM](https://docs.litellm.ai/) model string (Ollama, vLLM, Anthropic, OpenAI, Gemini, OpenRouter, …). Then:

```bash
gearbox ui                                                         # dashboard with real models
gearbox route "Design a lock-free queue and prove it" --risk high  # which tier, and why
gearbox run "Write add(a, b). Code only." --check "assert add(2, 3) == 5"
```

Checks run model-written code, so they're off unless `code_checks: true`. Isolation is best effort (temp dir, empty environment, limits, no network where supported), not a security boundary.

## Use it from an agent (MCP)

Tools: `route`, `delegate`, `await_result`, `status`, `cancel`, `ledger`. Add `--dashboard 8777` to watch delegations live.

```bash
claude mcp add gearbox -e GEARBOX_CONFIG=/abs/path/gearbox.yaml -- /abs/path/.venv/bin/gearbox-mcp
```

For **Antigravity** (`~/.gemini/config/mcp_config.json`) or **Cursor** (`.cursor/mcp.json`):

```json
{ "mcpServers": { "gearbox": { "command": "/abs/path/.venv/bin/gearbox-mcp",
                               "env": { "GEARBOX_CONFIG": "/abs/path/gearbox.yaml" } } } }
```

## Benchmarks

```bash
python bench/false_done.py --config gearbox.yaml --tasks humaneval+ --json runs/he.json   # or mbpp+, smoke
python bench/results.py                      # results table
python bench/plots.py                        # paper figures → docs/paper/figures/
```

HumanEval+ (164 problems) and MBPP+ (378) use hidden tests graded exactly like the official EvalPlus evaluator. Validation with qwen2.5-coder:1.5b: **65.2% / 58.7%** here vs **66.5% / 59.4%** published; all 542 reference solutions pass their own checks. Running on a lab GPU over WSL: [vm/README.md](vm/README.md).

## Development

```bash
.venv/bin/python -m pytest -q     # 96 tests
```

## License

Apache-2.0
