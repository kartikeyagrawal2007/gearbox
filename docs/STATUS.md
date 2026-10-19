# Project status

*20 October 2026.* One page on where Gearbox stands: what works, what the numbers say, what's broken, whether it's novel, the open questions, and the plan. Details live elsewhere: [REPORT.md](../REPORT.md) (results), [ROADMAP.md](../ROADMAP.md) (tasks), [EXPLANATION.md](../EXPLANATION.md) (code).

## 1. In one paragraph
Gearbox is a working tool and a mostly finished set of experiments. An expensive boss model (an agent in Claude Code, Cursor, …) hands coding subtasks to cheaper models over MCP. Gearbox checks every answer with tests, escalates failures to stronger models, and schedules the work by where the boss runs. On 542 coding problems × 12 open models, **checking cheap models' work beats always using a big model** on both accuracy and cost. **Picking a model per task up front doesn't pay.** **Overlapping work only helps when boss and workers are on different hardware.** What's missing is a **real agent in the loop** (the cloud boss is simulated) and **tasks beyond single Python functions**.

## 2. How it works
```
boss agent ──delegate(task, check)──▶ Gearbox ──▶ pick a worker tier ──▶ worker model
     │                                   │       (start floor; leverage from      │
     │  keeps working (async),           │        check strength)                 ▼
     │  or waits once (burst)            │                              answer ──▶ run the check in a sandbox
     │                                   │                                 pass ✓ │ fail ✕ → next tier, shown the failure
     └────────await_result ◀─────────────┴──── verified answer + ledger (tokens, cost, time)
```
- **Tool:** MCP server (`delegate`, `await_result`, `status`, `cancel`, `ledger`, `route`), a CLI, and a dashboard that runs a batch under five schedules and compares them.
- **Research:** the `bench/` scripts reproduce every number. Raw data lives in `runs/` (not in git).

## 3. The results so far

| Question | Answer | Strength of evidence |
|---|---|---|
| Can cheap models do the subtasks? | **Only with checking.** The 4B alone solves 77% / 65% and almost never says "unsure" | strong: 12 models × 542 problems, with intervals |
| Does check + escalate pay? | **Yes:** 95.7% vs 92.7% for the 27B, at about 1/3 of the compute | strong, but assumes a perfect check |
| With a weak check? | One test case, starting at the 4B: **89–98% of the 27B's quality, 2.4–2.7× cheaper** than the best random mix | strong (simulated weak checks) |
| Does it save time? | **About 1.5×** on one GPU (delegating vs the boss doing it all); **2.8×** from not starting on tiny models | measured, one machine |
| Does it save the boss's tokens? | **About half** in our workload (e.g. 1,031 → 315 tokens on the dashboard batch) | measured, but brief-writing cost not counted |
| Does async help? | **1.1–1.6×** with a (simulated) cloud boss; **no gain** on a shared GPU; up to 2× slower when GPU memory is full | measured; the cloud boss is simulated |
| Can a router pick the model up front? | **No.** Judges predict (AUC 0.67–0.74), but no router, including RouteLLM, beats random mixing on code | strong negative result |
| Can a model check itself (agreement)? | **A real signal** (87% right when its answers agree, vs 77%), **but too costly** (0.72–0.97× of mixing) | measured |

## 4. Known issues
- **Fixed 20 Oct:** the dashboard's "tokens saved" was an *estimate* (worker tokens minus the brief), identical in every run and below the real saving. It now shows **measured** numbers: boss tokens, worker tokens, and the saving against your own "Do it all yourself" run. A failed answer could also escalate in "Do it all yourself", which made no sense for a baseline; that's fixed too.
- **The cloud boss is simulated** by timing. Every async claim for cloud bosses rests on that (ROADMAP R1).
- **The check sandbox doesn't stop file reads** on Linux (ROADMAP T1).
- **Only Python function-level tasks** were tested (R2).
- The routing ladder is one family (Qwen3.5, 5 sizes). The 12-model set is for benchmarks only.

## 5. Is it really novel?

**Honest answer: partly.** The parts exist separately; what's new is the combination, the measurements, and a few specific findings.

| Already known (prior work) | New here |
|---|---|
| Cascades that try cheap models first (FrugalGPT, and others) | **Leverage set by how strong the test is:** how far up to start depends on how much the check can catch, measured on 542 problems |
| Escalating on failed tests (e.g. arXiv 2405.15842) | **A fair, negative router result on code:** judges and RouteLLM don't beat random mixing once thresholds are chosen without peeking and the router's cost is counted |
| Routers that pick a model per query (RouteLLM, IRT-Router) | **Async delegation measured by hardware placement,** with a ceiling formula that predicts the gain, and the shared-GPU memory failure |
| Agents running subagents in the background (Claude Code, Codex) | **A practical rule:** one test case + a capable small starting model is 2.4–2.7× cheaper than the best random mix of models |
| Models overclaiming "done" (2026 papers, e.g. arXiv 2606.09863) | A size ladder and vendor comparison for **small open** models, and the finding that they essentially never abstain |

**What it isn't:** a new algorithm or model. It's a **systems + empirical** contribution with an open-source tool. That fits a **workshop paper or TMLR**. A top-venue main track would need real agent sessions and more task types.

## 6. Open questions
These are genuinely open, as far as we know, and each could be a follow-up project:
1. **Where does the one test come from?** Our strongest lever needs a test. Can a cheap model write a test that's good enough, and how do you know when it isn't?
2. **Does "one test beats prediction" hold beyond single functions:** multi-file changes, refactors, data transformations, non-code tasks?
3. **A better measure of check strength than counting asserts:** coverage, mutation score, or diversity of the inputs?
4. **When does delegation pay, once the brief is counted?** The boss must describe the subtask; real agent traces would show the break-even point.
5. **Scheduling across many devices:** with several GPUs and cloud models, what's the best placement and batching? This is a queueing problem.
6. **Context by reference:** can workers read the files they need (or use a memory tool like Waggle) without losing the cost advantage?
7. **Untestable tasks:** for work with no executable check, is there any signal cheaper than the work itself?
8. **Security at scale:** running model-written code from many delegations safely.

## 7. Feasibility

| Goal | Feasible? | What it needs |
|---|---|---|
| The tool, for personal use | **Yes, today** | Ollama + a GPU or cheap API tier; harden the sandbox before wider use |
| A workshop or TMLR paper | **Yes** | real-agent sessions (R1, a free API key), figures, writing |
| A top-venue main-track paper | **Hard** | a second task family (R2), real-agent data at scale, possibly a second GPU type |
| Staying at $0 | **Yes** | the lab GPU for local models; free API tiers for the real-agent check |

## 8. Approach from here
1. **R1, real agents.** Run real sessions (Gemini free tier, or Claude Code) with and without Gearbox, on a few real repository tasks. Measure time, the boss's tokens (including briefs) and correctness. This turns the simulated claims into measured ones.
2. **T1, sandbox hardening** (bubblewrap or a container), before anyone else runs it.
3. **"Help the boss write one test":** have Gearbox suggest a test when none is given, then measure how good those tests are (open question 1).
4. **Figures and writing** (`docs/paper/outline.md`), then arXiv (needs an endorser) and TMLR.
5. *If there's time:* a second task family (R2).

## 9. Where things are (the docs map)
| File | What it's for |
|---|---|
| `README.md` | entry point and headline numbers |
| `IDEA.md` | the idea, and what's new |
| `docs/STATUS.md` | **this page:** the state of everything |
| `REPORT.md` | results in plain language |
| `ROADMAP.md` | built, missing, next |
| `EXPLANATION.md` | how the code works |
| `SETUP.md` / `CONTRIBUTING.md` / `AGENTS.md` | setup / contributing / AI-agent context |
| `bench/README.md` | which experiment script does what |
| `docs/paper/` | paper outline, results with statistics, figures |
| `docs/lit-review.md`, `docs/related-work.md` | the literature check |
