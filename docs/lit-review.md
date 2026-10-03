# Literature check: go/no-go (first pass, 2026-10-03)

This is the week-1 gate from the plan. It's a first pass by targeted search and reading of the closest papers. Before relying on it, verify the "must read" list yourself (bottom of this page).

Two candidate headline claims:

- **H1, async heterogeneous delegation.** An expensive host keeps working while a cheaper model runs a delegated subtask, compared head-to-head with blocking delegation on wall-clock time and cost.
- **H2, "false done".** Cheap workers present wrong answers as finished and almost never abstain, so delegation needs executable checks rather than self-reports. This one came out of our own A5000 results.

## Verdict

| | Verdict | Why |
|---|---|---|
| **H1** | **GO: still open, and better motivated than in September** | No paper found compares async vs blocking delegation to *cheaper* models on time *and* cost. Meanwhile Claude Code (background subagents by default since v2.1.198), Hermes Agent and LiveKit ship non-blocking subagents, and practitioners are asking for exactly this evaluation. |
| **H2** | **Not a headline: already well covered** | False success / overclaiming is measured in at least five 2026 papers, and vendor system cards now track it. Test-based escalation from a small model to a larger one was published in 2024. What remains is narrower (see below). |

**Recommended framing.** H1 is the headline. H2 becomes a supporting result inside the same delegation story: *async speeds delegation up; executable checks make it safe*. The narrow H2 contributions that do survive are:

1. **Small open-weight models along a size ladder.** The main false-success paper covers frontier models only and does no analysis by model size. We have Qwen3.5 0.8B → 27B plus three vendors at matched sizes.
2. **Abstention as an escalation signal fails for current small models.** We saw 0 UNSURE in 88 attempts across 11 current models, including 21 wrong answers presented as done. "Ask for help when unsure" escalation schemes therefore can't trigger. Only an old model abstained, and half of those abstentions were unnecessary, which is consistent with "abstention can be a prompt artifact".
3. **Answer formatting changes measured accuracy.** ministral-3:8b scores 0/8 under a strict code parser and 7/8 with fence repair.

**Main risk to H1:** speed. The pattern is shipping right now, so someone may publish evaluations soon. Aim for a workshop submission within roughly 4–6 weeks.

## H1: closest work and what each one does

| Work | Async? | Worker cheaper than host? | Compares async vs blocking? | Measures time and cost? |
|---|---|---|---|---|
| [DecisionBench](https://arxiv.org/abs/2605.19099) (May 2026) | No, blocking `call_model` | Yes, 11 peers | No | Yes |
| [RSI-Router](https://arxiv.org/abs/2609.34712) (Sep 2026) | No, sequential subtasks | Yes | No | Cost yes; time no |
| [ClawArena-Team](https://arxiv.org/abs/2606.31174) (Jun 2026) | **Yes**, background subagents | No: a fixed, free local pool | No | No wall-clock or idle time |
| [Can Small Agents Collaborate…](https://arxiv.org/abs/2601.11327) (Jan 2026) | Not specified | Yes: 8B/32B orchestrator, 1.7B sub-agents | **No** (checked against the paper text) | Latency and tokens |
| [CAID: async SWE agents](https://arxiv.org/abs/2603.21489) (Mar 2026) | Integration is sequential | No, same model | No | Yes; runtime grows 2–3× |
| [AgentInfer / Towards Efficient Agents](https://arxiv.org/abs/2512.18337) | No: small executes, large rescues, in sequence | Yes | No; "no overlap studied" | Yes; 1.8–2.5× speedup |
| [AsyncFC](https://arxiv.org/abs/2605.15077) | Yes, for **tool calls** | n/a (tools, not models) | Tool latency only | Latency |
| [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) | Unclear | Unclear | No | No |
| Products: Claude Code, Hermes Agent, LiveKit, OpenRouter subagents | Yes | Optional | Not evaluated | Anecdotal only |

What we add: a head-to-head of async vs blocking with the same model calls (the race), across worker size and placement (same GPU vs separate hardware). It includes the **speedup ceiling** `wall / (wall − host blocked)` and the **contention** effect we already measured. On one shared GPU, async gave 0.98–1.04× against a ceiling of about 1.09×. The open question is how this changes with separate hardware and long worker tasks.

## H2: closest work

| Work | What it covers | Gap it leaves |
|---|---|---|
| [From Confident Closing to Silent Failure](https://arxiv.org/abs/2606.09863) (Jun 2026) | False-success rates of 13–89% per model in agent benchmarks; LLM judges fail to catch it; independent verification cuts it from 45–48% to 3% | Frontier models only, no size analysis, single agent (no delegation) |
| [OverclaimBench](https://arxiv.org/abs/2609.20812) (Sep 2026) | Frontier coding agents overclaim review coverage (misleading in 59–96% of incomplete runs) | Frontier CLIs; review coverage rather than code correctness |
| [Confident and Wrong](https://arxiv.org/abs/2603.25764) (Mar 2026) | Coding agents submit patches on nearly every run while resolving far fewer | Frontier/large models; no delegation |
| [Model Cascading for Code](https://arxiv.org/abs/2405.15842) (2024) | Small→large cascade escalating on **model-generated** tests; saves 26% on average | Self-generated tests rather than host-written checks; no async; pre-2026 models |
| [Is Escalation Worth It?](https://arxiv.org/abs/2605.06350) (May 2026) | Pre-generation routers often beat cascades | Cost of verification inside agent delegation |
| [LLM Abstention Can Be a Prompt Artifact](https://arxiv.org/abs/2507.16199) | Offering an abstain option makes models abstain on questions they can solve | Not code, not delegation |
| [Selective code generation](https://arxiv.org/abs/2505.13553) | Abstain based on generated unit tests | Single model, no delegation |

## Must read yourself (in this order), and what to check

1. **DecisionBench**: confirm delegation is blocking, and note its metrics and model pool. It's the benchmark reviewers will compare us with.
2. **ClawArena-Team**: the only async work. Confirm the worker pool is homogeneous and free, and that no wall-clock or idle-time analysis exists.
3. **Can Small Agents Collaborate…**: confirm there's no async-vs-blocking analysis, and see how they report latency.
4. **RSI-Router**: subtask-level routing on SWE-bench and Terminal-Bench, our nearest competitor on the "which tier" side.
5. **From Confident Closing to Silent Failure**: how they define false success; position our false-done metric relative to theirs.
6. **Model Cascading for Code**: prior art for escalating when a test fails. Our difference is host-written checks plus repair feedback inside async delegation.
7. **Is Escalation Worth It?**: the strongest critique of cascades. Our cost analysis must answer it.
8. **LLM Abstention Can Be a Prompt Artifact**: explains our UNSURE observations; cite it rather than claim the finding.

Search terms used: async/non-blocking subagent delegation with cheaper models and wall-clock cost; small-model overconfidence and false completion in code; verify-then-escalate cascades with unit tests; coding agents falsely reporting completion.
