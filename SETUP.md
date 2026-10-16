# Setup

Every way to get Gearbox running, from a laptop demo to the lab GPU. If you're contributing, also read [CONTRIBUTING.md](CONTRIBUTING.md). If you're an AI agent, read [AGENTS.md](AGENTS.md).

## 1. Requirements
- **Python 3.10+** and git
- *Optional:* [Ollama](https://ollama.com) for local models, or an API key for a cloud model
- *Optional:* an NVIDIA GPU, to run the benchmarks at full scale

## 2. Laptop: try it in 2 minutes (fake models, no GPU)

```bash
git clone https://github.com/kartikeyagrawal2007/gearbox.git && cd gearbox
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp gearbox.example.yaml gearbox.yaml
.venv/bin/python -m pytest -q          # all tests pass, no models needed
.venv/bin/gearbox ui --simulate        # http://127.0.0.1:8790
```

In the dashboard, **Watch a delegation → Run the batch**.

> **macOS:** don't keep the project in an iCloud-synced folder (Desktop or Documents). iCloud hides files inside `.venv`, and Python then can't find the installed package (`No module named 'gearbox'`). Use a folder like `~/code/gearbox`. A shortcut on the Desktop is fine.

Optional extras: `.[plots]` (paper figures), `.[gpu]` (GPU energy readings), `.[baselines]` (the RouteLLM comparison; installs PyTorch).

## 3. Real models

### Local, with Ollama
```bash
ollama pull qwen2.5-coder:1.5b       # a small worker; add bigger ones if your machine has the memory
```
In `gearbox.yaml`, list tiers from cheapest to strongest:
```yaml
tiers:
  - {name: small, model: ollama_chat/qwen2.5-coder:1.5b}
  - {name: large, model: ollama_chat/qwen2.5-coder:7b}
code_checks: true        # run tests on answers (this executes model-written code; see section 7)
```
Then run `.venv/bin/gearbox ui` (real models), or `.venv/bin/gearbox run "Write add(a, b). Code only." --check "assert add(2, 3) == 5"`.

### Cloud models
A tier can be any [LiteLLM](https://docs.litellm.ai/) model string, e.g. `anthropic/claude-haiku-4-5` or `gemini/gemini-2.5-flash`. Put API keys in **environment variables** (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …), **never** in `gearbox.yaml` or in git.

### The settings you'll touch most
| Setting | What it does |
|---|---|
| `start_floor` | the cheapest tier work may start on (e.g. your 4B) |
| `code_checks` | allow executable checks (off by default) |
| `max_escalations` | how many tiers a failed answer may climb; use the number of tiers − 1 |
| `delegation_mode` | `auto`, `async` or `burst` (see EXPLANATION.md, "The four schedules") |
| `host: {model: ...}` | the boss's model, if it runs locally; lets `auto` pick burst |
| `difficulty` | `heuristic` (default), `judge`, or `learned` (experimental) |

Every setting is explained in `gearbox.example.yaml`.

## 4. Use it from an agent (MCP)

**Claude Code:**
```bash
claude mcp add gearbox -e GEARBOX_CONFIG=/abs/path/gearbox.yaml -- /abs/path/.venv/bin/gearbox-mcp
```
**Antigravity** (`~/.gemini/config/mcp_config.json`) or **Cursor** (`.cursor/mcp.json`):
```json
{ "mcpServers": { "gearbox": { "command": "/abs/path/.venv/bin/gearbox-mcp",
                               "env": { "GEARBOX_CONFIG": "/abs/path/gearbox.yaml" } } } }
```
The agent gets the tools `delegate`, `await_result`, `status`, `cancel`, `ledger` and `route`. Add `--dashboard 8777` to the command to watch its delegations live.

## 5. The lab PC (RTX A5000, Windows + WSL Ubuntu)

The full guide is [vm/README.md](vm/README.md). The essentials:
- **One-time setup** (inside WSL): `git clone … ~/gearbox && cd ~/gearbox && bash vm/setup_wsl.sh`, then `bash vm/pull_models.sh` (the 12 benchmark models) and `.venv/bin/python vm/verify_models.py`.
- **Ollama settings that matter:**
  - **8k context:** `/etc/systemd/system/ollama.service.d/context.conf`, so two models fit in GPU memory without crashing
  - **models unload after 5 idle minutes:** `zz-keepalive.conf`, so the GPU is freed and remote desktop stays responsive
- **Long runs go in `tmux`.** Every experiment script resumes where it stopped.
- **One GPU job at a time.**
- **Dashboard demo with real models and the learned router:**
  ```text
  pkill -f "gearbox.*ui"; cd ~/gearbox && git pull -q && nohup .venv/bin/gearbox --config vm/demo.vm.yaml ui > ~/gearbox-ui.log 2>&1 & sleep 5; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8790
  ```
  It should print `200`. Then open http://localhost:8790 on the lab PC.
- **Moving results to a laptop:** `tar czf` them onto `/mnt/c/Users/PRO-LAB-2/Desktop/`, then copy with AnyDesk.

## 6. Results data

Raw results (`runs/`, about 30 MB of model answers) are not in git. Get the latest archive from the project owner and unpack it in the repo root. Then:
```bash
.venv/bin/python bench/results.py      # the results table
.venv/bin/python bench/stats.py        # intervals and paired tests
```
Every other script is listed in [bench/README.md](bench/README.md).

## 7. Safety
Checks **run code that a model wrote**. Isolation is best effort: a temporary folder, an empty environment, time and CPU limits, and no network on Linux (`unshare`) and macOS (`sandbox-exec`). **The code can still read your files.** Turn on `code_checks` only on a machine where that's acceptable, or run Gearbox in a VM or container. The dashboard has no login; it only listens on `127.0.0.1`.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'gearbox'` on macOS | the project is in an iCloud folder: move it (section 2) |
| `address already in use` on port 8790 | an old dashboard is still running: `pkill -f "gearbox.*ui"` |
| Dashboard says a model is "not downloaded" | `ollama pull <model>` (the message gives the exact name) |
| Remote desktop is laggy, or GPU memory is full | unload models: `for m in $(ollama ps \| awk 'NR>1{print $1}'); do ollama stop "$m"; done` |
| A model call hangs or crashes with "CUDA error: illegal memory access" | GPU memory is full: use the 8k context setting and don't run two GPU jobs |
| Tools misbehave inside WSL | conda's `(base)` environment is active: `conda deactivate` |
| LiteLLM warns it can't fetch a cost map | harmless; it falls back to a local copy |
