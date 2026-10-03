# Related work and the go/no-go gate

Snapshot from a search on 2026-09-29. **Phase 0 gate:** read the "Read in full" rows. If any of them already measures *asynchronous* delegation from an expensive host to cheaper workers, with cost accounting, pivot the paper's headline from async delegation to cache-aware scheduling or risk-scaled leverage (see the plan).

## Difficulty / query routing (mature: not our contribution)

| Work | What it is | Notes |
|---|---|---|
| [RouteLLM](https://github.com/lm-sys/routellm) (LMSYS, 2024) | Trained strong/weak routers, OpenAI-compatible server | Classic baseline |
| [RouterArena](https://arxiv.org/abs/2510.00202) (2025) | Open router leaderboard; difficulty levels from Bloom's taxonomy | Difficulty-aware evaluation already exists |
| [LLMRouterBench](https://arxiv.org/abs/2601.07206) (2026) | 21 datasets, many routers | Many routers fail to beat the best single model |
| [RouterEval](https://aclanthology.org/2025.findings-emnlp.208/) (EMNLP'25 Findings) | Results for 8,500 LLMs on 12 benchmarks | |
| [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | Production mixture-of-models router | ~5k stars |
| [claude-code-router](https://github.com/musistudio/claude-code-router) | Per-request routing for Claude Code (default / background / think / longContext) | Closest in UX to a proxy mode |

## Risk-aware routing (prior art for "leverage")

| Work | Notes |
|---|---|
| [RACER](https://arxiv.org/abs/2603.06616) | Conformal, distribution-free control of misrouting risk |
| [CR2](https://arxiv.org/abs/2605.12001) | Conservative local acceptance under asymmetric error costs |
| [Cost-Aware Protocol Routing](https://arxiv.org/abs/2608.14927) | Measures under- vs over-escalation directly |
| [Is Escalation Worth It?](https://arxiv.org/abs/2605.06350) | Decision theory of cascades; pre-generation routers often beat cascades |

## Delegation within agents (closest; read in full)

| Work | Sync or async? | Cost-heterogeneous? | Why it matters |
|---|---|---|---|
| [DecisionBench](https://arxiv.org/abs/2605.19099) | Sync (blocking `call_model`) | Yes, 11 peers | The delegation benchmark to compare against |
| [RSI-Router](https://arxiv.org/abs/2609.34712) | Sync (sequential subtasks) | Yes, 1:30 price ratio assumed | Subtask-level routing on SWE-bench Verified and Terminal-Bench |
| [ClawArena-Team](https://arxiv.org/abs/2606.31174) | **Async** (background subagents) | No: fixed, free local pool | Async exists, but no cost-heterogeneity or wall-clock metrics |
| [Minions](https://arxiv.org/abs/2502.15964) (ICML'25) | Parallel local jobs per round | Yes, local plus cloud | Cost-efficient local/cloud collaboration |
| [TRACE-Router](https://arxiv.org/abs/2607.22465) | Per-task pinning | Yes | Argues the routing unit should match the feedback unit |
| [SWE-Router](https://arxiv.org/abs/2607.00053) | Per-instance | Yes | Coding-agent routing |
| [Uno-Orchestra](https://arxiv.org/abs/2605.05007) | Learned decomposition plus routing | Yes | About 10x lower cost per query |
| [Towards Efficient Agents](https://arxiv.org/abs/2512.18337) | Small model runs, large one rescues | Yes | Inverse direction (escalation) |
| [Anthropic: optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) | Sync (advisor tool; orchestrator/workers) | Yes | Published numbers; "must beat the single model's whole curve" |

## Async execution (mechanism prior art)

| Work | Notes |
|---|---|
| [AsyncFC](https://arxiv.org/abs/2605.15077) | Future-based async function calling; overlaps decoding with tool latency |
| [Second Thought](https://arxiv.org/abs/2608.13667) | Reasoning continues while the agent acts and observes |
| [LAMaS](https://arxiv.org/abs/2601.10560) | Latency-aware orchestration objective |

## Working hypothesis for the gap

No work found measures an expensive host **continuing to work** while a **cheaper** worker runs a delegated subtask, with joint accounting of quality, cost or energy, host-token share, and wall-clock time. In particular, none separates async's latency win from its hidden costs:
- speculative work that gets wasted
- cache expiry during waits
- the cost of reading the result back into the host

Early observations from building the tool, worth testing properly:
- In MCP-style delegation the brief is billed as **host output tokens**. Passing context inline can erase the savings, and passing it by reference (paths) preserves them.
- Workers can report success while failing the stated acceptance check (seen in the first local smoke test). Executed acceptance checks may matter more than routing accuracy.
- **Small models over-use an escape hatch.** With "if you cannot complete it reliably, reply UNSURE", qwen2.5-coder:1.5b answered UNSURE (just restating the task) on trivial subtasks it solves fine without the hatch. Every subtask then escalated, and delegation saved nothing. Rewording the hatch ("almost every subtask can be done: just do it; only if information is missing…") fixed it. Self-reported uncertainty from small models is not a reliable escalation signal, which strengthens the case for executed checks.
- **Self-reported success is unreliable, so measure the false-done rate.** On 8 checked coding tasks (bench/false_done.py, temperature 0), qwen2.5-coder:1.5b passed 5. It claimed success on all 3 failures, a 37.5% false-done rate, and never replied UNSURE. Same-model repair with exact failure feedback (actual vs expected value) did not fix the slugify failure. Escalating to a stronger model is the remedy to test next. Candidate paper metric: false-done rate per tier, and how it changes with model size.
- **Async speedup has an Amdahl-style ceiling**: `wall_blocking / (wall_blocking − host_blocked)`. With fast cheap workers there is little blocked time to remove; on one shared GPU we measured 0.98–1.04× against a ceiling of about 1.09×.
- **Measurement traps found while building the race**: cold-start model loading charged to whichever phase runs first (it inflated a result to 1.31×), and sampled outputs making the two phases do different work (an escalation in one phase only produced a fake 2.34×).
