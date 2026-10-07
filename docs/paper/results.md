# Draft: Results (§5)

*Draft 2.1, 2026-10-07: §5.4 corrected to a fair router evaluation.* Every number below comes from the A5000 runs in `runs/`. Each subsection names the script that reproduces it. Intervals are 95% Wilson. Paired comparisons use McNemar's exact test on per-problem outcomes (`bench/stats.py`). We ran 20 paired tests, so treat p-values between 0.0025 and 0.05 as exploratory: only p < 0.0025 survives a Bonferroni correction.

Setup common to all sections:
- Open models served by Ollama 0.35.1 at 4-bit quantization, on one RTX A5000 (24 GB).
- Temperature 0, so there is one answer per model per problem.
- Hidden EvalPlus tests decide correctness: HumanEval+ has 164 problems, MBPP+ has 378.
- The routing ladder is Qwen3.5 0.8B / 2B / 4B / 9B / 27B.
- "Cost" is the model's parameter count in billions, once per attempt. That's a proxy for compute, not dollars.

---

## 5.1 Harness validation

Our harness reproduces published scores to within about one point. With Qwen2.5-Coder-1.5B-Instruct we measure 65.2% on HumanEval+ and 58.7% on MBPP+, against the published 66.5% and 59.4%.

Grading copies EvalPlus's input deserialization and special oracles verbatim. All 542 reference solutions pass their own generated checks.

## 5.2 How reliable is a cheap worker's "done"? (RQ2)

Twelve models answered every problem as delegated workers (`bench/false_done.py`, `bench/stats.py`). An answer counts as *claimed done* unless the model replies UNSURE. The *false-done rate* is the share of claimed-done answers that are wrong.

**Table 1: pass rate (95% interval), UNSURE hatch on.**

| Model | HumanEval+ (164) | MBPP+ (378) |
|---|---|---|
| Qwen3.5 0.8B | 23.2% [17.4, 30.2] | 29.4% [25.0, 34.1] |
| Qwen3.5 2B | 50.6% [43.0, 58.2] | 47.6% [42.6, 52.7] |
| Qwen3.5 4B | 76.8% [69.8, 82.6] | 64.8% [59.9, 69.5] |
| Qwen3.5 9B | 80.5% [73.8, 85.8] | 70.6% [65.9, 75.0] |
| Qwen3.5 27B | 92.7% [87.6, 95.8] | 78.0% [73.6, 81.9] |
| Granite 4.2 3B | 73.8% [66.6, 79.9] | 66.7% [61.8, 71.2] |
| Granite 4.2 8B | 87.2% [81.2, 91.5] | 72.5% [67.8, 76.7] |
| Ministral 3 3B | 64.6% [57.1, 71.5] | 59.3% [54.2, 64.1] |
| Ministral 3 8B | 79.3% [72.4, 84.8] | 66.9% [62.0, 71.5] |
| Gemma 3 4B | 63.4% [55.8, 70.4] | 67.7% [62.9, 72.2] |
| Gemma 3 12B | 77.4% [70.5, 83.2] | 72.0% [67.2, 76.2] |
| Qwen2.5-Coder 0.5B (older, code-tuned) | 54.9% [47.2, 62.3] | 44.2% [39.3, 49.2] |

**Finding 1: reliability rises with size, but even the largest model claims done on wrong answers.**
- Across the Qwen3.5 ladder, the pass rate rises from 23% to 93% on HumanEval+ and from 29% to 78% on MBPP+.
- The 27B still presents 7% (HumanEval+) and 21% (MBPP+) of its claimed answers as done when they're wrong.
- The 4B → 9B step is not significant on HumanEval+ (p = 0.38) but is on MBPP+ (p = 0.005). So there's a plateau on one benchmark only.

**Finding 2: current models almost never abstain.** Across 11 current models and 5,962 delegated problems (both benchmarks), only Qwen3.5 0.8B ever answered UNSURE: 8 times on HumanEval+ and 9 on MBPP+. It was wrong on about 70% of the answers it did claim. A delegation scheme that waits for the worker to ask for help would almost never trigger, so **an executable check is necessary**.

**Finding 3: offering the hatch changes nothing detectable.** With the hatch off, no model's pass rate differs significantly (all 12 McNemar p > 0.14). Its one visible effect is format: Qwen3.5 0.8B's missing-function failures rose from 9 to 29 when the hatch was offered.

**Finding 4: vendor rankings don't transfer between benchmarks.**
- Granite 8B beats Qwen3.5 9B on HumanEval+ (16 vs 5 discordant problems, p = 0.027), but not on MBPP+ (p = 0.42).
- Qwen3.5 4B beats Gemma 3 4B on HumanEval+ (p = 0.003), but not on MBPP+ (p = 0.22), where Gemma is nominally ahead.

**Finding 5: specialization beats size at the low end.** The year-older, code-tuned Qwen2.5-Coder 0.5B beats Qwen3.5 0.8B on both benchmarks (53 vs 1 and 76 vs 20 discordant problems, both p < 0.001).

**Finding 6: output format can decide measured accuracy.** Ministral 3 8B left code fences unbalanced in 38–41 of 164 HumanEval+ answers and in 0 of 378 MBPP+ answers, so the prompt format triggers it. Strict parsing would have scored all of those as failures. We repair fences and report repairs separately.

## 5.3 Check and escalate (RQ3)

We replay the recorded answers: what each model answered for each problem is known, so any routing policy can be evaluated without new model calls (`bench/routing_sim.py`, `bench/weak_checks.py`).

**Finding 7: start cheap, check, escalate beats the largest model on both accuracy and cost.** This is with the full hidden suite as the check:

| | HumanEval+ | MBPP+ |
|---|---|---|
| Always Qwen3.5 27B | 92.7%, cost 27.0 | 78.0%, cost 27.0 |
| Cascade from 0.8B, escalate on failure | 95.7%, cost 8.0 | 82.0%, cost 13.0 |
| Oracle (cheapest model that passes) | 95.7%, cost 5.1 | 82.0%, cost 8.2 |

The cascade beats the 27B on accuracy because smaller models solve some problems the 27B fails.

**Finding 8: tiny models waste time; start at the 4B.** Tiny models write long, wrong answers. This uses the recorded per-problem model time:

| | Accuracy | Cost | Time per problem |
|---|---|---|---|
| HumanEval+: cascade from 0.8B | 95.7% | 8.0 | 57.0 s |
| HumanEval+: **cascade from 4B** | **95.7%** | 9.2 | **20.7 s** |
| MBPP+: cascade from 0.8B | 82.0% | 13.0 | 45.0 s |
| MBPP+: **cascade from 4B** | **81.5%** | 13.8 | **27.9 s** |

That's the same accuracy in 2.8× (HumanEval+) and 1.6× (MBPP+) less time. Gearbox's `start_floor` setting applies this.

**Finding 9: the cascade is only as good as its check.** We re-ran every recorded wrong answer against weaker checks. Wrong answers that pass a weak check are accepted (*false accepts*). The ranges cover 51 answers whose stored text was truncated:

| Check | HumanEval+ | MBPP+ |
|---|---|---|
| 1 test | 62–63%, cost 3.2 (59–62 false accepts) | 62–64%, cost 6.5 |
| 3 tests | 76–79%, cost 4.8 | 67–70%, cost 7.8 |
| Original base tests | 88–91%, cost 6.9 | 67–70%, cost 7.8 |
| Full hidden suite | 95.7%, cost 8.0 | 82.0%, cost 13.0 |

**Finding 10: leverage should follow check strength, not prompt difficulty.**
- **With a weak check, starting higher recovers most of the loss.**
  - HumanEval+ with a 1-test check: starting at the 4B (+2 tiers) gives 82–83% at cost 5.2, against 62–63% from the 0.8B.
  - MBPP+ with 3 tests: starting at the 4B gives 77–78% at cost 9.5, against 78% for the 27B at cost 27.
- **With the full suite, leverage only adds cost.**
- **A prompt-difficulty heuristic carries no signal.** It does no better than the same tier assignment shuffled at random (HumanEval+ at +0: 50.6% vs 51.0%).

Gearbox counts a check's test cases and sets leverage from them: weak (1–3) +2, medium (4–9) +1, strong (10+, or a loop over data) +0.

## 5.4 A learned router (RQ3b)

The router uses item response theory, as in IRT-Router (`bench/router_train.py`):
- Each model has an ability `a`, each problem a difficulty `b`, and `P(solve) = sigmoid(a − b)`.
- The router predicts `b` for a new problem from features, then assigns the cheapest model whose `P` clears a bar.
- It's evaluated by 5-fold cross-validation within each benchmark, and by training on one benchmark and testing on the other.

**Finding 11: only a model that reads the problem predicts its difficulty.** Per-model AUC (how well `P` separates the problems a model solves from those it fails; 0.5 is chance), as a range over the four evaluations:

| Features | AUC |
|---|---|
| Surface text: length, keywords, TF-IDF | 0.45–0.65 |
| Embeddings: Qwen3-Embedding 0.6B / 4B | 0.48–0.63 |
| Nearest similar problems (embedding kNN) | 0.45–0.56 |
| **Judge: a model rates difficulty 1–10** | |
| Qwen3.5 4B judge | **0.66–0.68** |
| Qwen3.5 27B judge | 0.70–0.73 |
| 27B judge, three samples averaged | 0.73–0.75 |
| All three judges | 0.72–0.74 |
| Judge asked for a pass probability (0–100) | 0.43–0.72 (worse than rating) |
| RouteLLM BERT router (off the shelf; see Finding 13) | 0.43–0.63 |

With true difficulties, the same model reaches 0.93–0.96. **The bottleneck is reading the problem, not the routing model.**

**Finding 12: per-problem routing saves nothing on code once it's measured fairly.** Two things inflate router savings, and an earlier draft of this section fell for both: choosing the bar after seeing the test answers, and comparing against "always use the strongest model".
- **The fair baseline is random mixing.** Any quality between two fixed models is reachable with no router at all, by sending a random share of tasks to each. A router must be cheaper than that mix *at the same quality*.
- **The fair protocol** (`bench/router_fair.py`) picks the bar on training folds only, averages over 10 splits, and includes the router's own compute (its parameters × the ~146 tokens it reads).

| Router (one shot, no check) | HumanEval+ at 85 / 90 / 95% of the 27B's quality | MBPP+ at 85 / 90 / 95% |
|---|---|---|
| Qwen3.5 4B judge | 0.51 / 0.82 / 0.99× | 0.72 / 0.77 / 1.03× |
| RouteLLM BERT (off the shelf) | 0.70 / 1.01 / 1.27× | 0.93 / 0.90 / 0.92× |

The table shows cost relative to random mixing; above 1 means the router is cheaper. Inside weak-check cascades the result is the same: on HumanEval+, both routers land at 0.52–1.08× of a fixed starting model plus mixing (`--check "1 test"`).

The judge does predict which problems a model will solve (Finding 11), so why doesn't it save anything?
- **Its own cost.** A 4B judge reading every task costs 1.7 of our cost units, against 4–27 per answer.
- **Its mistakes fall where they matter.** On code, the gap between adjacent models is a few hard problems. AUC 0.67 can't find them reliably enough to beat a coin flip weighted by price.

This matches the routing literature's own caveat that routers barely beat random on hard benchmarks without in-domain data. RouteLLM's routers were near random on GSM8K and MMLU until augmented.

**Finding 13: an off-the-shelf chat router doesn't transfer to code correctness.** RouteLLM's BERT router (trained on Chatbot Arena preferences between GPT-4 and Mixtral) scores 0.43–0.63 AUC on our problems, below 0.5 on MBPP+ cross-validation. That's no better than surface text features. "Would a person prefer the strong model's chat answer?" and "will a small model's code pass hidden tests?" are different questions.

**Finding 14: the judge is cheap in time, and its bar must be recalibrated on new kinds of task.**
- Measured one call at a time, the judge reads about 144 tokens and writes 2: 0.12 s (4B), 0.14 s (9B), 0.36 s (27B). That's about 1–3% of a worker's answer time.
- Trained on one benchmark, its ranking transfers to the other (AUC holds), but its bar drifts in opposite directions: too cheap HumanEval+ → MBPP+, too cautious MBPP+ → HumanEval+.
- Fitting one offset on 20–30 labelled problems of the new kind restores the target quality (94–97%) over 50 random draws (`bench/router_train.py --calibration-study`).

**What we ship.** Gearbox's practical gains come from the start floor and from check-and-escalate, not from per-problem routing. The learned mode (`difficulty: learned`, `routers/judge-qwen3.5-4b.json`) stays in the tool as an experimental option, with the evidence above.

## 5.5 Asynchronous delegation (RQ1)

Each episode is k HumanEval+ subtasks plus k host steps. It runs in four modes (`bench/async_bench.py`):
- **host-only:** the host does everything itself
- **blocking:** delegate, wait, continue
- **parallel:** delegate all, wait for all, then do the host work
- **async:** delegate all, do the host work meanwhile, then collect

Comparing them splits the gain into two parts:
- blocking → parallel: workers running side by side
- parallel → async: the host not waiting (the claim)

The mode order rotates across repetitions. Models are warmed up, temperature is 0, and the delegating modes produced identical answers in every repetition. The worker is Qwen3.5 4B. The table gives the median over 4 repetitions (3 for the remote host), with the range in brackets.

**Table 5: async vs blocking.**

| Host | Worker | k | Async vs blocking | Ceiling | Parallel → async |
|---|---|---|---|---|---|
| Emulated cloud host, 1 s + 80 tok/s, 256-token steps | 4B on GPU | 2 / 4 / 8 | **1.59 / 1.49 / 1.50×** | 1.59 / 1.73 / 1.50 | 1.57 / 1.43 / 1.42× |
| Same host, 768-token steps | 4B on GPU | 2 / 4 / 8 | 1.25 / 1.29 / 1.20× | 1.25 / 1.29 / 1.20 | 1.23 / 1.27 / 1.18× |
| Emulated cloud host, 2 s + 40 tok/s, 256-token steps | 4B on GPU | 2 / 4 / 8 | 1.31 / 1.36 / 1.26× | 1.31 / 1.36 / 1.26 | 1.28 / 1.33 / 1.22× |
| Same host, 768-token steps | 4B on GPU | 2 / 4 / 8 | 1.12 / 1.15 / 1.10× | 1.13 / 1.15 / 1.10 | 1.11 / 1.13 / 1.09× |
| Qwen3.5 27B on the same GPU | 4B on GPU | 2 / 4 / 8 | 0.99 / **0.77 / 0.47×** | 1.31 / 1.30 / 1.25 | 0.96 / 0.77 / 0.46× |
| Qwen3.5 27B on GPU | 4B on CPU | 2 / 4 / 8 | 1.23 / 1.35 / 1.36× | 4.87 / 3.76 / 3.36 | 1.18 / 1.27 / 1.32× |

**Finding 15: async reaches its predicted ceiling when host and workers don't share hardware.**
- With a cloud host, async is 1.10–1.59× faster than blocking and comes within 0.01 of the ceiling `wall / (wall − host blocked)` in 11 of 12 conditions. The exception is the fast host at k = 4: 1.49× against 1.73×. The ceiling is computed from the blocking run alone.
- Nearly all of the gain is the host not waiting (parallel → async). Workers running side by side add only 1.01–1.05×.
- The gain is largest when the host's own steps are short relative to the work it delegates.

**Finding 16: on a shared accelerator, async is harmful.**
- With the 27B host and 4B workers on one GPU, async is 0.99× (k = 2), 0.77× (k = 4) and 0.47× (k = 8) of blocking speed. Both models stayed fully in GPU memory.
- The two model processes slow each other far more than taking turns would. At k = 8, the host's own work grew from 70 s to 204 s and the workers' from 17 s to 141 s. Run in turn, they would need about 87 s in total; overlapped, they took 204 s.
- Delegating at all still helps on one GPU. Delegating to the 4B was 1.42× faster than the 27B doing everything at k = 2, and burst delegation (parallel) is never slower than blocking.

**Finding 17: a CPU worker frees the GPU but is too slow.** Async recovers 1.23–1.36× over blocking with the 4B on 36 CPU threads. But every delegating mode is slower than the 27B alone (0.52–0.89×), because the CPU runs the 4B at about one subtask per 26 s, and four parallel slots barely help.

Gearbox's design follows from these findings: async when the host is remote, burst delegation when host and workers share a device.

---

## Threats to validity

- **The cloud host is emulated** (TTFT + tokens/rate) and has no latency variance. A small real-API check remains to be done.
- **One machine** (A5000, WSL2, WDDM). The shared-GPU slowdown may depend on the GPU's scheduling between processes; MPS or MIG could change it.
- **Quantized local models at temperature 0:** one sample per problem. The intervals reflect problem sampling, not decoding noise.
- **Cost is parameter count,** not dollars or measured energy. The judge's cost uses the same proxy (tokens read × parameters).
- **Two benchmarks, both Python functions,** both possibly seen in pretraining. The router's cross-benchmark test is the only out-of-domain evidence.
- **Weak checks were simulated** by subsetting EvalPlus inputs; real host-written tests may differ in what they catch.

## Still to add
- Figures for §5.3–5.5 (the cascade cost–accuracy curve, leverage × check strength, async speedup vs ceiling).
- A small real cloud-host run to validate the emulation.
