# Paper outline (draft 1, 2026-10-04)

Working title: **"Don't Wait Up, But Do Check: Measuring Asynchronous and Verified Delegation to Cheaper LLMs"**

Alternative: *"When Does Async Delegation Pay? Cost, Latency and Reliability of Delegating Subtasks to Small Language Models"*

**Target:** arXiv first, then TMLR (no length limit, no fees). A shortened version (4–8 pages) can go to an ICLR or ICML 2027 workshop. Write the long version first; a short one is easy to cut from it.

Status markers: ✅ result in hand · 🔄 runnable now, needs the VM · ⬜ not built yet

---

## Abstract (fill the numbers last)

AI agents increasingly hand subtasks to cheaper models, and products now run these delegations **asynchronously**: the expensive model keeps working instead of waiting. Whether that actually saves time or cost has not been measured. We study delegation to small open models from three angles:

1. When async delegation beats blocking delegation, and what bounds the gain. We give a simple ceiling derived from a blocking trace, and show how a shared accelerator's contention erases the gain.
2. How reliable cheap workers' completion claims are across a 0.8B–27B size ladder and four vendors, on HumanEval+ and MBPP+ with hidden tests.
3. What executable verification with escalation costs and recovers.

Headline numbers to report: [async speedup vs ceiling by placement]; [false-done rate vs size]; [abstention rate, about 0 for current models]; [accuracy recovered by escalation and its cost].

---

## 1. Introduction

- Motivation: delegation to cheaper models is now standard (orchestrator/worker patterns). Async subagents ship by default in Claude Code (v2.1.198+), Hermes Agent and LiveKit, but no study compares them with blocking delegation on time *and* cost.
- A second problem: a cheap worker's "done" is unreliable. False success is documented for frontier agents (2606.09863, OverclaimBench), but not for **small** models in **delegation**, which is the setting where it matters.
- **Research questions:**
  - **RQ1:** When does async delegation reduce wall-clock time compared with blocking, and what limits the gain?
  - **RQ2:** How often do cheap workers claim success while wrong, by model size and vendor, and can their own abstention ("UNSURE") trigger escalation?
  - **RQ3:** What do executable checks with escalation recover, and at what cost?
- **Contributions:**
  1. The first head-to-head measurement of async vs blocking delegation, plus the speedup ceiling.
  2. A reliability study of 12 small models on hidden-test benchmarks.
  3. An open-source tool (Gearbox: MCP server, runtime, dashboard, benchmarks).
  4. A list of measurement traps we hit and fixed, as lessons for evaluators.

## 2. Related work

Use `docs/lit-review.md`.

- Routing and cascades: RouteLLM, RouterArena, LLMRouterBench, FrugalGPT, "Is Escalation Worth It?"
- Delegation inside agents: DecisionBench, RSI-Router, SWE-Router, TRACE-Router, "Can Small Agents Collaborate…", AgentInfer
- Async execution: ClawArena-Team (async but homogeneous workers), AsyncFC (async tools), CAID (sequential integration)
- False success: 2606.09863, OverclaimBench, Confident and Wrong
- Test-based escalation: Model Cascading for Code (2024), selective code generation
- Abstention: "LLM Abstention Can Be a Prompt Artifact"

Positioning: we are the only work that measures *async vs blocking* delegation to *cheaper* models with time and cost.

## 3. System: Gearbox (about 1 page)

- Runtime: `delegate()` returns a handle at once and `await_result()` blocks only when needed. Each task records host-blocked time.
- Routing: a difficulty estimate plus leverage (constant, or scaled by risk and confidence).
- Executable checks: the host's asserts or hidden tests, run in a sandbox (temp dir, empty environment, limits, no network). A failed check escalates one tier with repair feedback.
- Ledger (tokens, cost, false-done rate per tier) and NVML energy meter.
- MCP server (Claude Code, Antigravity, Cursor) and the dashboard.
- **Figure 1:** architecture diagram (reuse the one from the README discussion).

## 4. Method

### 4.1 Models ✅
- Size ladder: Qwen3.5 0.8B / 2B / 4B / 9B / 27B.
- Matched sizes: Granite 4.2 3B/8B, Ministral 3 3B/8B, Gemma 3 4B/12B.
- Reference: Qwen2.5-Coder 0.5B (older generation).
- All 4-bit quantized via Ollama 0.35.1, temperature 0. Thinking switched off where supported (Qwen3.5, Granite; verified per model).

### 4.2 Tasks
- HumanEval+ (164) ✅ and MBPP+ (378) ✅ with hidden checks, graded exactly like EvalPlus. The rules and conversions are copied verbatim from upstream.
- The smoke set (8) is used only for pipeline checks.
- **Harness validation** ✅: qwen2.5-coder:1.5b scores 65.2% on HumanEval+ (report: 66.5%) and 58.7% on MBPP+ (report: 59.4%). All 542 reference solutions pass their own checks.
- Grader fidelity: EvalPlus's input conversion and special oracles are copied verbatim (commit 26d6d00). Building it surfaced data quirks we handle and document: Mbpp/793's empty `{}` inputs, and Mbpp/19's function named `test_duplicate`, which a naive test runner would call as a test.

### 4.3 Metrics
- Pass rate.
- False-done rate (logic-only): wrong answers presented as done ÷ answers presented as done.
- Format failures and fence repairs, reported separately.
- UNSURE rate with the hatch on, and pass rate with it off.
- Wall-clock time, host-blocked time, overlap ratio, and the **speedup ceiling** = wall ÷ (wall − host blocked) from the blocking trace.
- Energy in joules (NVML).

### 4.4 Async protocol (the race)
- An untimed warm-up of every model; temperature 0.
- A comparability check: both phases must make identical worker calls.
- ⬜ Alternate phase order across rounds, and use ≥5 repetitions with confidence intervals.
- Placements:
  - (a) host and worker on one GPU ✅ (Mac)
  - (b) host on GPU, worker on CPU 🔄
  - (c) host and worker on GPU with spare capacity 🔄
- ⬜ Workloads that vary worker-task length and the share of independent work.

### 4.5 Hardware ✅
- RTX A5000 (24 GB) under WSL2: 36 CPU threads, 98 GB RAM.
- Apple M4 (24 GB) for development.

## 5. Results

### 5.1 Harness validation ✅
| Benchmark | Ours (qwen2.5-coder:1.5b, 4-bit, T=0) | Published |
|---|---|---|
| HumanEval+ | 65.2% | 66.5% |
| MBPP+ | 58.7% | 59.4% |

Plus the reference-solution self-check: 542/542 pass.

### 5.2 RQ2: reliability of cheap workers
- 🔄 **Table 1 (main):** 12 models × {HumanEval+, MBPP+}, with pass rate, false-done (logic), UNSURE rate, format failures and fence repairs.
- 🔄 **Figure 2:** false-done rate vs parameter count (the Qwen3.5 ladder as a line, other vendors as points).
- 🔄 **Figure 3:** hatch on vs off, i.e. honest vs timid abstention.
- Early signal ✅ (8 smoke tasks; report as a pilot only): 0 UNSURE in 88 attempts by current models, including 21 wrong answers; Ministral 8B scored 0/8 under a strict parser and 7/8 with fence repair.

### 5.3 RQ1: when async pays
- ✅ The ceiling explains the Mac results: 0.98–1.04× measured vs a ~1.09× ceiling, because the host was blocked only ~9% of the time and both models shared one GPU.
- 🔄 **Figure 4:** measured speedup vs ceiling for each placement (a/b/c).
- ⬜ **Figure 5:** speedup vs worker-task length (where async starts to matter).
- 🔄 Energy per completed workload, async vs blocking.

### 5.4 RQ3: verify and escalate ⬜
- Accuracy recovered when a failed check escalates (small → large, with repair feedback), against the extra tokens, time and energy.
- Compare with "always use the large model" and "trust the small model".
- Pilot ✅: a same-size repair did not fix slugify even with the exact failure, so escalation needs a stronger tier.

## 6. Discussion

- Practical guidance:
  - Delegate async only when the ceiling says there is blocked time to remove, and only when the host and worker don't share one accelerator.
  - Never accept a cheap worker's "done" without an executable check.
  - The brief is billed as host **output** tokens, so pass context by reference.
- Lessons for evaluators (appendix-worthy). Each of these traps changed a headline number:
  - cold-start bias (1.31× became ~1.1×)
  - sampled outputs making phases do different work (a fake 2.34×)
  - unbalanced code fences (0/8 became 7/8)
  - format-vs-logic mislabelling

## 7. Threats to validity
- 4-bit quantization and a single serving stack (Ollama).
- One GPU machine.
- Function-level coding tasks only; no SWE-bench-style agentic tasks yet.
- Benchmark contamination: HumanEval and MBPP are old and likely seen in training. That matters less here, because we compare models *relative to each other* and claims vs truth.
- Temperature 0 only.

## 8. Limitations and future work
Agentic benchmarks, API-hosted frontier hosts, speculative continuation, cache-aware scheduling, calibrated (conformal) leverage, context by reference.

## 9. Conclusion
⬜

## Appendix
- Prompts: worker system prompt, hatch on/off.
- Grading rules (EvalPlus parity) and the check sandbox.
- Model tags and download sizes.
- How to reproduce: `vm/README.md` and commit hashes.

---

## Work plan to finish

1. 🔄 **VM, Monday:** HumanEval+ and MBPP+ for all 12 models, hatch on and off (Table 1, Figures 2–3).
2. ⬜ **Race upgrade:** repetitions, alternating order, CPU-worker placement, variable worker length. Then the runs (Figures 4–5).
3. ⬜ **Verify-and-escalate experiment** (§5.4).
4. ⬜ Plots from `runs/*.json` (a small script).
5. ⬜ Write-up: §1, §3, §4 can be drafted now; §5–6 once numbers land.
6. ⬜ arXiv (needs an endorser), then TMLR.
