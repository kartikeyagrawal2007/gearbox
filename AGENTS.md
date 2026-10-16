# AGENTS.md: working on Gearbox as an AI coding agent

Full context for an AI agent (Claude Code, Codex, Cursor, …) contributing to this repository. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## What this project is
Gearbox lets an expensive "boss" model hand subtasks to cheaper worker models over MCP. It checks the workers' answers with executable tests, escalates failures to stronger models, and schedules the work by where the boss runs (async, or burst on a shared GPU). It is both a tool (`gearbox/`) and a research project (`bench/`, `docs/paper/`).

Read first: [IDEA.md](IDEA.md) (the idea), [REPORT.md](REPORT.md) (results with numbers), [ROADMAP.md](ROADMAP.md) (what's missing). How the code works: [EXPLANATION.md](EXPLANATION.md).

## Repository map
```
gearbox/                the installable package (keep it light: no PyTorch or other heavy deps)
  config.py             YAML config: tiers, start_floor, leverage, check_leverage, delegation_mode, ...
  router.py             picks a tier: difficulty estimate + leverage (check-aware), or the learned router's pass chances
  difficulty/           heuristic.py, judge.py, learned.py (IRT router; file in routers/)
  delegate/runtime.py   DelegationRuntime: delegate / await_result / cancel, escalation, async vs burst
  delegate/brief.py     the worker's instructions
  verify/checks.py      the sandboxed check runner;  verify/strength.py counts a check's test cases
  episode.py            one boss episode (5 schedules, local or simulated-cloud boss) for the dashboard
  providers.py          LiteLLMProvider (real models), SimulatedProvider (fake)
  cost/                 ledger, break-even, GPU energy
  integrations/mcp_server.py   the MCP tools
  ui/server.py, ui/static/index.html   the dashboard (one HTML file, no build step)
  cli.py                `gearbox` command: tiers, route, run, serve, ui
bench/                  experiments; index in bench/README.md; results go to runs/ (not in git)
tests/                  pytest, using fake models (tests/fakes.py); no GPU or network needed
vm/                     lab PC scripts and configs (RTX A5000, WSL)
docs/paper/             paper draft (results.md is the source of truth for numbers), figures, data
routers/                trained router files
```

## Commands
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup (see SETUP.md for extras)
.venv/bin/python -m pytest -q                                 # all tests, about 50 s; must pass before any commit
.venv/bin/gearbox ui --simulate                               # dashboard with fake models on :8790
.venv/bin/python bench/results.py                             # results table (needs runs/)
```

## Rules you must not break
1. **Tests never need a model, a GPU or the network.** Use `FakeProvider` and `make_config` from `tests/fakes.py`. Add a test for every behaviour change. Run the full suite before committing.
2. **Don't edit `bench/evalplus_compat.py`.** It's EvalPlus's grading code, copied verbatim. It's the reason our scores match the published ones.
3. **Keep `gearbox/` light.** Heavy dependencies go in an optional extra used only by `bench/` (see `pyproject.toml`).
4. **Honest experiments.** Each of these rules came from a mistake we made:
   - temperature 0 and a warm-up before timing
   - choose thresholds on training folds only, never on the test set
   - compare a router with **random mixing of fixed models at equal quality**, with its own cost included (`bench/router_fair.py`), never with "always the biggest model"
   - report 95% intervals or paired tests (`bench/stats.py`)
   - look for the boring explanation first (the "async is 2× slower" result turned out to be GPU memory)
   - write down negative results
5. **One source for each number.** A number in the docs must come from a script you can run. When results change, update `docs/paper/results.md` first, then `REPORT.md`, the table in `README.md`, and `ROADMAP.md`.
6. **Security.**
   - Checks execute model-written code: never weaken the sandbox in `verify/checks.py`, and keep `code_checks` off by default.
   - Never put API keys in configs, code or git; they go in environment variables.
   - The dashboard must stay bound to `127.0.0.1`.
7. **Never commit** `runs/`, `bench/data/`, `gearbox.yaml` (personal config) or `.venv/`. They're in `.gitignore`. Keep it that way.
8. **Commits and PRs.** Small commits whose message says what changed and why. Don't add AI tools as co-authors or contributors, and don't mention them in commit messages, PR descriptions, READMEs or changelogs: credit goes to the project's human authors.

## The lab PC
Big runs happen on a lab PC that you **cannot reach**. A human pastes commands into its WSL terminal. So:
- Give lab commands in ` ```text ` blocks, one per block, ready to paste.
- Long runs go in `tmux`. Scripts must save progress as they go and **resume** when rerun.
- One GPU job at a time.
- Estimate run times **from measured speeds** (e.g. the `seconds` field in `runs/*.json`), never by guessing.
- Results come back as a `tar czf` archive on `/mnt/c/Users/PRO-LAB-2/Desktop/`, copied over by hand.

## Pitfalls we already hit
| Symptom | Cause |
|---|---|
| `No module named 'gearbox'` on macOS | the project was in an iCloud folder, which hides `.venv`'s `.pth` files |
| MCP client sees garbage | LiteLLM printed to stdout; `protect_stdout()` in the MCP server prevents it |
| `FastMCP` import fails | MCP SDK 2.x renamed it `MCPServer` (`mcp.server.mcpserver`); only `ToolError` messages reach clients |
| Qwen3.5 or Granite answers slowly, or "think" | set `params: {reasoning_effort: none}` on the tier |
| A model's code "is not defined" though it wrote it | unbalanced code fences; `extract_code` repairs them, and the benchmark counts repairs |
| Identical code "disagrees" with itself | set and dict order changes between processes; compare outputs in a canonical form. Note `python -I` ignores `PYTHONHASHSEED` |
| Re-checking old answers gives different verdicts | runs before Oct 2026 stored answers truncated at 6,000 characters |
| Async 2× slower, or "CUDA error: illegal memory access" | GPU memory nearly full; the lab uses 8k contexts |
| `ProcessLookupError` killing a check | kill race on Linux; already handled in `run_check` |

## How to verify a change
- **Code:** `pytest -q` passes, plus a new test for the new behaviour.
- **Dashboard:** run `gearbox ui --simulate`, click through the changed part, and check both a desktop width and a phone width (375 px). There's no build step; edit `gearbox/ui/static/index.html` directly.
- **Experiments:** try them on `--limit 10` or `--tasks smoke` before asking for a lab run.
- **Docs:** if you changed behaviour or numbers, update the docs listed in rule 5, and [EXPLANATION.md](EXPLANATION.md) if the code structure changed.
