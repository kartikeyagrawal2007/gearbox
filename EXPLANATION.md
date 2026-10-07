# Gearbox, explained

This is the guide to the whole project in plain language: what it is, why each piece exists, how the pieces fit, and what we've found so far. Read it top to bottom once. After that, use it as a map.

---

## 1. The idea in one minute

AI coding agents (Claude Code, Antigravity, Cursor) run on **expensive** models. Much of what they do is routine: writing a test, a small helper function, a docstring. A **cheaper** model could do that work.

Gearbox lets the expensive model **hand such a subtask to a cheaper model** (a *delegation*) and **keep working instead of waiting**. That's *async* delegation, and it's the part nobody has measured.

Two things make it hard:

1. **Does async actually save time?** Only if the expensive model has useful work to do while it waits, and only if the two models aren't fighting over the same hardware.
2. **Can we trust the cheap model's answer?** We found that cheap models say "done" even when they're wrong, and almost never say "I'm unsure". So Gearbox **runs the answer against tests** before accepting it, and **escalates** to a stronger model if the tests fail.

The research paper measures both. The code is the tool that does it, and the benchmark that proves it.

---

## 2. Words you'll see everywhere

| Word | Meaning |
|---|---|
| **Host** | The expensive model doing the main job (e.g. Claude in Claude Code). It delegates. |
| **Worker** | The cheaper model that receives a delegated subtask. |
| **Tier** | One model in Gearbox's ladder, ordered cheapest → strongest (e.g. `q0.8b`, `q2b`, …, `q27b`). "Gear" in the gearbox metaphor. |
| **Routing** | Choosing which tier gets a subtask, from an estimate of how hard it is. |
| **Leverage** | A safety margin: route one or more tiers *higher* than the estimate says. Can scale with risk. |
| **Delegate** | Hand a subtask to a worker. Gearbox returns a *task id* at once. |
| **Blocking vs async** | Blocking: the host waits for the worker. Async: the host keeps working and collects the result later. |
| **Check** | Test code (Python asserts) run against the worker's answer. Passing means *verified*. |
| **Hidden check** | A check the worker never sees (benchmarks use these, like real exams). |
| **False done** | The worker claimed success, but the answer fails its check. Our key reliability measure. |
| **UNSURE / escape hatch** | We tell the worker it may reply `UNSURE: …` if it can't do the task. "Hatch on/off" = whether that instruction is included. |
| **Escalation** | When an answer fails, retry on the next stronger tier, showing it what went wrong. |
| **Format failure** | The answer didn't even contain the requested function (an output-formatting problem, not wrong logic). |
| **Fence repair** | Fixing broken ```` ``` ```` code markers so the code can be extracted. |
| **Race** | Our experiment: the same workload run blocking, then async, with measured timelines. |
| **Speedup ceiling** | The best async can possibly do: `total time ÷ (total time − time the host spent waiting)`. |
| **Ledger** | The record of every model call: tokens, cost, time. |
| **MCP** | Model Context Protocol, the standard way agents (Claude Code etc.) call external tools. Gearbox is an MCP server. |
| **Ollama** | The program that runs open models locally on a GPU. |
| **LiteLLM** | A Python library that talks to any model provider (Ollama, Anthropic, OpenAI, …) through one interface. |
| **Quantization (4-bit)** | Compressing a model so it fits in memory. It's why a 27-billion-parameter model is only 17 GB. |
| **HumanEval+ / MBPP+** | Standard coding exams for AI: 164 and 378 small Python problems, each with many hidden test inputs. "+" means EvalPlus's much stricter tests. |
| **EvalPlus** | The project that made HumanEval+/MBPP+ and their official grader. We copy its grading rules exactly. |

---

## 3. What happens when a subtask is delegated

Follow one request through the code. Say Claude Code calls Gearbox's `delegate` tool with *"Write tests for parse_date"*, risk `low`, and a check.

1. **The request arrives**: `gearbox/integrations/mcp_server.py`, function `delegate`. It passes everything to the runtime and replies with a `task_id` **immediately**, in a few milliseconds. Claude keeps working.
2. **A background job starts**: `gearbox/delegate/runtime.py`, `DelegationRuntime.delegate` → `_run`. Everything below happens without the host waiting.
3. **Routing**: `gearbox/router.py`.
   - A **difficulty estimator** (`gearbox/difficulty/heuristic.py` is free and keyword-based; `judge.py` asks a cheap model) scores the task from 0 to 1.
   - The score maps to a base tier.
   - **Leverage** shifts it up: +0, +1 or +2 for low, medium or high risk, plus 1 more if the estimate was unsure.
   - **A floor** (`start_floor`) stops it starting on a tiny model: they're slow and usually wrong, so on HumanEval+ starting at the 4B gave the same accuracy as starting at the 0.8B in 2.8× less time.
   - **The learned router** (`difficulty: learned`, experimental: measured fairly it saves nothing over the floor plus checks on code) replaces the guess: a 4B judge rates the task 1–10 (a few tokens), a model trained on our 542 problems × 12 models (`routers/judge-qwen3.5-4b.json`) turns that into each tier's pass chance, and the cheapest tier over a bar is picked. The bar depends on the check: 0.9 with no or a weak check, 0.8 with a medium one, and with a strong check it starts at the floor because escalation catches the misses.
   - **With a check, leverage comes from the check's strength instead** (`gearbox/verify/strength.py` counts its test cases): weak (1–3) +2, medium (4–9) +1, strong (10+, or a loop over test data) +0, plus the risk level. A failed check escalates anyway, so leverage only has to cover the wrong answers the check lets through. This is the weak-check finding, built into the tool.
4. **The brief**: `gearbox/delegate/brief.py`. The worker receives a short instruction (system prompt + subtask + "done when" + the check, if it's meant to be visible). It doesn't get the host's whole conversation, which keeps it cheap.
5. **The model call**: `gearbox/providers.py`, `LiteLLMProvider.complete`. It talks to the model through LiteLLM, with a timeout. A concurrency limit stops too many calls at once.
6. **Judging the answer**, in `runtime.py` `_attempt`:
   - If the answer starts with `UNSURE:` → outcome **unsure**.
   - Else, if there's a check → `gearbox/verify/checks.py` `run_check` runs it in a **sandbox** (next section). Pass → **ok**; fail → **check_failed** (a false done).
   - Model error or timeout → **error** / **timeout**.
7. **Escalation**: if the outcome isn't ok and a stronger tier is allowed, retry there. After a failed check, the next model sees its predecessor's answer plus the exact failure (e.g. *"slugify('Héllo') returned 'héllo', expected 'hello'"*), so it can repair rather than start over.
8. **Bookkeeping**: `gearbox/cost/ledger.py` records tokens, cost, and whether a check passed. That's where the per-tier false-done rate comes from.
9. **Collecting the result**: when the host calls `await_result(task_id)`, it gets the answer, `verified: true/false`, and every attempt. The runtime also records **how long the host was actually blocked**. That's how we measure what async saved.

---

## 4. The check sandbox, and why it matters

A check runs **code written by a model** on your machine. `gearbox/verify/checks.py` limits the damage:

- a throwaway temporary folder
- an **empty environment**, so no API keys or tokens are visible
- Python's isolated mode
- input closed
- CPU and file-size limits
- a wall-clock timeout
- **no network** where the OS supports it (`sandbox-exec` on macOS, `unshare` on Linux/WSL)

This isn't a perfect security boundary, so checks are **off by default** (`code_checks: false`). Turn them on in the config only where that's acceptable, like the lab VM.

How a check runs: the worker's code is extracted from its answer (code fences, with broken fences repaired). It's loaded first, then the check's asserts or `test_*` functions run in the same space. For a failed plain `assert f(x) == y`, the runner reports the **actual value**, which is what makes the repair feedback in step 7 useful.

---

## 5. The code, folder by folder

```
gearbox/                  the installable Python package
  config.py               reads the YAML config: the tier ladder, prices, switches
  router.py               difficulty + leverage → which tier
  difficulty/             heuristic.py (free), judge.py (asks a cheap model), learned.py (judge + trained model)
  delegate/               runtime.py (background jobs, escalation) and brief.py (worker instructions)
  verify/checks.py        the sandboxed check runner
  verify/strength.py      how strong a check is (counts its test cases), which sets its leverage
  providers.py            LiteLLM (real models) and a SimulatedProvider (fake, for demos)
  cost/                   ledger.py, breakeven.py (is delegating worth it?), energy.py (GPU joules)
  race.py                 blocking vs async experiment with timelines
  runs.py                 reads benchmark result files (used by dashboard, results, plots)
  integrations/mcp_server.py   the MCP tools: route, delegate, await_result, status, cancel, ledger
  ui/                     the dashboard: server.py (API) and static/index.html (the page)
  cli.py                  the `gearbox` command (tiers, route, run, serve, ui)
bench/                    benchmarks (not part of the installed package)
  false_done.py           the benchmark runner: every task × every model → JSON in runs/
  tasksets.py             the task sets: smoke (8), humaneval+ (164), humaneval+mini, mbpp+ (378)
  evalplus_compat.py      EvalPlus's grading code, copied verbatim (do not edit)
  results.py              prints the results table in a terminal (and --export for copying)
  plots.py                makes the paper figures
  routing_sim.py          replays recorded answers under routing strategies (no GPU needed)
  weak_checks.py          the cascade with weaker checks, and leverage as a start tier (no GPU needed)
  async_bench.py          the async delegation experiment: four modes, repeated, two placements
  router_train.py         the learned router: predicts which models solve a problem (IRT), assigns the cheapest
  router_features.py      collects judge ratings and embeddings for the learned router (lab GPU)
  router_fair.py          the fair test: does a router beat randomly mixing fixed models at equal quality?
  routellm_baseline.py    RouteLLM's BERT router as a baseline (needs the `baselines` extra)
  stats.py                95% intervals and paired tests for the paper's numbers
routers/                  trained learned-router files (judge-qwen3.5-4b.json)
vm/                       setting up and using the A5000 lab machine
  setup_wsl.sh            one-shot setup inside WSL Ubuntu (Ollama, Python, tests)
  pull_models.sh          downloads the 12-model benchmark set
  verify_models.py        checks every model answers, with "thinking" off
  models.vm.yaml          all 12 models as tiers (for benchmarks)
  gearbox.vm.yaml         the Qwen3.5 ladder (for routing experiments)
  async.vm.yaml           host and worker tiers for the async experiment, on GPU and CPU
  ollama_cpu.sh           a second, CPU-only Ollama (port 11435) for the worker-on-CPU placement
docs/                     lit-review.md (the go/no-go check), related-work.md, paper/ (outline, results, figures)
tests/                    121 automated tests: run them with `.venv/bin/python -m pytest -q`
```

**Two configs, two jobs.**
- A *routing* config is a **ladder**: cheapest to strongest, one family (`vm/gearbox.vm.yaml`, or `gearbox.yaml` on your Mac).
- A *benchmark* config is just a **list of models to test** (`vm/models.vm.yaml`). Its order means nothing to the router.

---

## 6. How the benchmark works

`bench/false_done.py` answers one question: **for each model, how often is its "done" actually right?**

1. **Load a task set** (`bench/tasksets.py`):
   - *smoke*: 8 hand-written tasks. The worker sees the checks. Only for testing the pipeline.
   - *humaneval+* and *mbpp+*: the real exams. They download on first use. The worker sees only the problem statement; the tests are **hidden**.
2. **Warm up each model** with a tiny untimed call, so loading time isn't counted.
3. **Send every task to the model** as a delegation: no escalation, temperature 0 (same input, same output), up to 4 at once.
4. **Grade each answer**, for HumanEval+/MBPP+, by running EvalPlus's **reference solution** and the worker's function on **every test input** and comparing outputs. It uses the official rules: float tolerance, special cases, input conversion (all copied from EvalPlus).
5. **Label each failure**:
   - **format**: the requested function was never defined
   - **logic**: it ran and gave a wrong answer
   - **unsure**: the model declined
6. **Save** one JSON entry per model to `runs/`. The file is rewritten after **every** model, so a crash keeps everything finished so far.

**We checked that the grader is right:**
1. All 542 official reference solutions pass their own checks.
2. Qwen2.5-Coder 1.5B scores 65.2% / 58.7% here, against the published 66.5% / 59.4%.

**Reading the results:**
- `bench/results.py` prints the table.
- The dashboard shows the same table in a browser.
- `bench/plots.py` draws the figures.
- They all read results through `gearbox/runs.py`, which keeps the newest result per (task set, model, hatch).

---

## 7. The race (async vs blocking)

`gearbox/race.py` runs one workload twice: the host's own steps plus some delegated subtasks.

- **Blocking**: delegate → wait → do a host step → delegate → wait…
- **Async**: delegate everything → do all host steps → collect the results.

Both make **the same model calls**, so any time difference comes from overlapping work. Three safeguards keep it fair, each added after it bit us:

1. **Warm-up first.** Without it, the first phase pays the model-loading time.
2. **Temperature 0, plus a comparability check.** If the phases still made different calls (e.g. one escalated), the result is reported as "not comparable" rather than as a speedup.
3. **The speedup ceiling.** Async can only remove time the host spent waiting.

On your Mac (one GPU, both models on it) async gave about 1.0×, against a ceiling of about 1.09×. There was little waiting to remove, and the models slowed each other down.

The race is the dashboard's demo. **The paper's experiment is `bench/async_bench.py`**, which runs one episode (k HumanEval+ subtasks plus k host steps) in four modes:

| Mode | What the host does |
|---|---|
| host_only | solves the subtasks itself: no delegation |
| blocking | delegate one, wait, do a host step; repeat |
| parallel | delegate all, wait for all, then do the host steps |
| async | delegate all, do the host steps meanwhile, then collect |

Comparing them splits the saving by cause: **blocking → parallel** is the gain from workers running side by side, and **parallel → async** is the gain from the host not waiting (the paper's claim). It repeats each condition with the mode order rotated, so no mode always goes first. It runs with the worker on the same GPU as the host, or on the CPU (`vm/ollama_cpu.sh`). It also records whether each model really sat on the GPU. Steps are in `vm/README.md`, section 6.

**Remote host.** Gearbox's intended setting is a cloud host (the model in Claude Code) with workers on your own hardware. We can't pay for an API, so `--host remote:1.0:80` *emulates* one: each host step takes 1.0 s + tokens ÷ 80 per second and uses no local hardware, while the workers stay real. On the Mac this gave async **1.45–1.51×** over blocking, the full ceiling. That's because host and workers no longer compete for the same hardware.

---

## 8. Measurement traps we hit (all fixed)

Each of these changed a headline number before it was caught. They make a good "lessons for evaluators" section in the paper.

| Trap | What happened | Fix |
|---|---|---|
| Cold start | The first phase also paid for loading the model; it showed 1.31× when the truth was about 1.1× | untimed warm-up |
| Sampled outputs | A random UNSURE in one phase gave a fake 2.34× speedup | temperature 0 + comparability check |
| Escape-hatch wording | The 1.5B replied UNSURE to everything until the instruction was reworded | reworded worker prompt |
| Broken code fences | Ministral 8B scored 0/8 under strict parsing, 7/8 with repair | repair stray fences, count repairs |
| Format vs logic | Crashes after a correct definition were labelled "format" | "format" only if the function is missing |
| `test_` names | Mbpp/19 asks for `test_duplicate`; our runner called it as a test | only run tests defined by the check |
| Not-None problems | EvalPlus compares these as yes/no; we compared raw objects | match EvalPlus |
| Cancelled checks | A cancelled check kept running; a whole run froze | kill the check process on any interruption |
| Queue time | Per-task time included waiting for a free slot | time only the model call |

---

## 9. What we've found so far

From HumanEval+ and MBPP+ on the A5000 (details in `docs/paper/results.md`):

- **Bigger is more reliable, with a plateau.** Qwen3.5 0.8B → 27B: 23% → 93% solved, false-done 70% → 7%. The 4B and 9B are about equal.
- **Models almost never say "unsure".** Across 11 current models only the 0.8B ever did (8 times), while being wrong on 70% of what it claimed. "Escalate when the worker asks for help" can't work, so **checks are necessary**.
- **The escape hatch didn't help anyone**, and it hurt the smallest model.
- **Vendor rankings don't transfer between benchmarks.** Granite 8B led HumanEval+ (87%), but on MBPP+ it tied with Gemma 12B and Qwen 9B (71–72%).
- **Code specialization beats size at the low end.** An older 0.5B coder (55%) beat Qwen3.5 0.8B and 2B.
- **Formatting can decide the score.** 38 of Ministral 8B's answers needed fence repair.
- **MBPP+ repeats the main findings:** reliability rises with size, only the 0.8B ever says unsure, and even the 27B claims "done" on 21% wrong answers.
- **Start cheap, check, escalate** beats always using the 27B on accuracy and cost, *if the check is perfect*: 95.7% at about a third of the cost on HumanEval+ (`bench/routing_sim.py`).
- **Our difficulty heuristic has no signal.** It does no better than assigning the same tiers at random.
- **The cascade is only as good as its check** (`bench/weak_checks.py`). With a 1-test check, wrong answers slip through and HumanEval+ accuracy drops to 62%.
- **Leverage compensates for a weak check.** Starting the cascade at the 4B instead of the 0.8B restores 82% at cost 5.2 with a 1-test check. So leverage should scale with **how weak the check is**, not how hard the prompt looks.
- **Async works when host and workers don't share hardware.** An emulated cloud host with workers on the A5000: 1.1–1.6× faster than blocking, reaching the predicted ceiling. Host and workers on the same GPU: up to 2× *slower* (they slow each other down far more than taking turns). Workers on the CPU: async helps (1.2–1.4×), but the CPU is too slow to be worth it.
- **Don't start on tiny models.** Starting the cascade at the 4B instead of the 0.8B: same accuracy, 2.8× less time on HumanEval+ (1.6× on MBPP+).
- **The guesser predicts, but doesn't save.** Text features can't predict which model solves a problem (AUC ~0.55), and neither can RouteLLM's off-the-shelf chat router (0.43–0.63). A 4B–27B judge *reading* the problem can (0.67–0.74). But measured fairly, against randomly mixing two fixed models at the same quality with the bar chosen without peeking, no router saves anything on code (0.5–1.3×, `bench/router_fair.py`). An earlier version of this file overstated it. The real savings come from the start floor and from check-and-escalate.

---

## 10. Cheat sheet

On your Mac (from `~/code/gearbox`):

```bash
.venv/bin/python -m pytest -q                       # all tests
.venv/bin/gearbox ui --simulate                      # dashboard with fake models
.venv/bin/python bench/plots.py docs/paper/data      # regenerate figures
```

On the VM, inside WSL (`cd ~/gearbox`):

```text
.venv/bin/python bench/results.py                    # results table
.venv/bin/python bench/results.py --export           # one-line summary to copy to the Mac
tmux ls                                              # is a background run still going?
tmux attach -t bench                                 # watch it (leave with Ctrl+B then D)
```

Running a benchmark (VM):

```text
.venv/bin/python bench/false_done.py --config vm/models.vm.yaml --tasks humaneval+ --json runs/NAME.json
```

Useful options: `--tasks mbpp+`, `--hatch off`, `--tiers qwen3.5-4b gemma3-4b`, `--limit 20` (quick trial).

---

## 11. What's next

1. **Run the async experiment on the lab PC** (`vm/README.md`, section 6). It decides how the paper is framed.
2. Statistics (confidence intervals, paired tests) and the `docs/paper/results.md` update.
3. Write the paper (`docs/paper/outline.md`), post it to arXiv, then submit to TMLR.
