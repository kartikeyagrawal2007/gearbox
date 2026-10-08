# Roadmap: what's built, what's missing, what's next

*Updated 8 October 2026.* The results behind each line are in [REPORT.md](REPORT.md). How the code works is in [EXPLANATION.md](EXPLANATION.md).

## Built and working

| Area | What exists | Evidence |
|---|---|---|
| **Delegation over MCP** | `delegate` / `await_result` / `status` / `cancel` / `ledger` / `route` tools (`gearbox/integrations/mcp_server.py`) | end-to-end stdio test (`tests/test_stdio_e2e.py`) |
| **Checks and escalation** | sandboxed Python checks; a failed answer moves up the ladder with the failure shown to the next model | 95.7% vs 92.7% for the 27B on HumanEval+, at about 1/3 the compute |
| **Start floor** | `start_floor`: never start on the tiniest models | same accuracy, 2.8× faster |
| **Leverage from check strength** | `gearbox/verify/strength.py` counts a check's cases: weak +2 tiers, medium +1, strong +0 | 1-test check: 62% → 82% by starting at the 4B |
| **Async and burst modes** | `delegation_mode: auto/async/burst`; burst holds work until the boss waits | async 1.1–1.6× with a cloud boss; burst 1.00× on a shared GPU (no slowdown, no crash) |
| **Agreement check** *(analysis only)* | `bench/sample_answers.py` + `bench/agreement.py`: escalate when a model's own answers disagree on sample inputs | real signal (87% correct when the 4B agrees vs 77%), but costs more than it saves (0.72–0.97× of random mixing). **One test case does far better: 2.4–2.7×** |
| **Learned router** *(experimental)* | judge + IRT model (`routers/judge-qwen3.5-4b.json`), with a calibration offset | predicts (AUC 0.67–0.74) but **saves nothing** vs random mixing on code |
| **Dashboard** | `gearbox ui`: watch a batch get routed, handed off, run in parallel, checked and returned | `tests/test_episode.py`, `tests/test_ui_server.py` |
| **Benchmarks** | 12 models × 542 problems (HumanEval+, MBPP+), graded exactly like EvalPlus | our scores within about 1 point of the published ones |
| **Lab PC tooling** | `vm/`: setup, model pulls, CPU-only Ollama, demo config | used for every run so far |

## In progress
- Nothing running. The agreement check finished (Finding 18 in `docs/paper/results.md`).

## Missing: research (what a reviewer will attack)

Ranked by how much each limits what we can claim.

| # | Gap | Why it matters | How to close it | Size |
|---|---|---|---|---|
| R1 | **No real agent in the loop.** The cloud boss is simulated by timing | The headline time and token claims rest on an emulation | Run real sessions with and without Gearbox: a free Gemini key as the boss, or Claude Code on a real repo task | medium |
| R2 | **Only single Python functions** (HumanEval+, MBPP+) | Real subtasks span files and need context | Add a multi-file or test-writing benchmark | large |
| R3 | **The cost of writing briefs is unmeasured** | The "about half the boss's tokens" saving ignores it | Comes free with R1 | small, after R1 |
| R4 | **Weak checks were simulated** by subsetting the hidden tests | Real boss-written tests may catch more or less | Have a model write the tests; measure how many wrong answers they catch | medium |
| R5 | **Cost is a stand-in** (model size × answers) | Weak support for "saves money" | Report the GPU energy (joules) already recorded in `runs/async*.jsonl` | small |
| R6 | One machine, 4-bit models, one answer per problem | Limits generality | Repeat the key results on another GPU or a cloud API | medium |

## Missing: tool (what a user will hit)

| # | Gap | How to close it | Size |
|---|---|---|---|
| T1 | **Security:** check code can **read your files** on Linux (only the network is blocked); macOS's `sandbox-exec` is deprecated; the dashboard has no login | Run checks under bubblewrap or a container with a read-only filesystem; add a dashboard token | medium |
| T2 | **Context costs the boss tokens:** everything a worker needs is pasted into the brief | Context by reference: file paths the worker may read, or a memory tool such as Waggle | medium |
| T3 | **Workers have no tools** (can't open files, can't rerun their code) | Let workers read listed files and retry against the check | medium |
| T4 | **Python-only checks** | A runner per language (JS, Go, shell) | medium |
| T5 | **Not published:** no CI, not on PyPI, not in the MCP registry | GitHub Actions for `pytest`; publish so `uvx gearbox-mcp` works | small |
| T6 | Tasks are lost on restart; placement detection only knows Ollama | Persist tasks; detect vLLM and llama.cpp servers | small |

## Next, in order
1. **R1: a real agent in the loop.** The most valuable experiment left; it also closes R3.
2. **T1: the security pass.**
3. **Make "write one test" easy for the boss:** the strongest lever we found. For example, Gearbox could suggest a test case when a subtask arrives without one.
4. **Figures**, then **write the paper** (`docs/paper/outline.md`): arXiv (needs an endorser), then TMLR or a workshop.
5. R5 (energy) and T5 (CI, PyPI) are good first tasks for a new contributor.
