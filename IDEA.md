# The idea behind Gearbox

## The problem

AI coding agents (Claude Code, Antigravity, Cursor) run one expensive model for everything: the hard design decision and the boring helper function alike. Two things are wasted:
- **Money.** Every token of routine code is written at frontier prices.
- **Time.** When the agent hands work to a subagent, it usually waits for the answer before it continues.

Meanwhile, cheap models (small open models on your own GPU, or cheap API tiers) can write a lot of that routine code correctly, but **not reliably**, and they rarely say when they're unsure.

## The approach: an automatic transmission

Like a car's gearbox, which picks the gear so the driver doesn't have to, Gearbox sits between the expensive **boss** model and a **ladder** of cheaper **worker** models (cheapest to strongest):

1. **Hand off.** The boss calls `delegate("write function X", check="assert X(2) == 4")` over MCP and gets a ticket back at once.
2. **Pick the gear.** Gearbox picks a worker tier:
   - never start on the tiniest models (`start_floor`)
   - start higher when the check is weak (**leverage** from check strength)
3. **Work without waiting.** The worker runs while the boss keeps going (**async**). If the boss shares the workers' GPU, the work is held until the boss pauses (**burst**), so the two never compete.
4. **Check, don't trust.** The worker's answer runs against the check in a sandbox. If it fails, the subtask **escalates** to a stronger tier, along with what failed.
5. **Collect.** The boss calls `await_result` when it needs the answer, which is **verified**. A ledger records the tokens, cost and time spent.

## What's new

| Prior work | What it does | What Gearbox adds |
|---|---|---|
| Agent subagents (Claude Code, Codex) | delegate to the **same** class of model, usually without verifying | delegation to **cheap or local** models, with executable checks and escalation |
| Routers (RouteLLM, IRT-Router, Not Diamond) | pick one model per query **up front** | measure fairly whether that pays on code (it doesn't), and use the check instead |
| Cascades (FrugalGPT and similar) | try cheap, escalate on a score | escalate on **executable tests**; set where to start from **how strong the test is** |
| Async agents | run work in the background | measure **when** overlap helps (separate hardware) and when it hurts (a shared GPU), with a formula that predicts the gain |

## Research questions
- **RQ1, async:** does a boss that keeps working finish sooner, and when?
- **RQ2, reliability:** how often is a cheap worker's "done" wrong, and do workers ever admit they're unsure?
- **RQ3, routing and verification:** can we pick the right worker up front, or is checking and escalating better? How much does a weak check cost us?

## How the results changed the idea

We started with "pick the right model up front, then run async". The data reshaped both halves. The details are in [REPORT.md](REPORT.md).

- **Checking beats predicting.** Starting cheap and escalating on a failed test beat the 27B (95.7% vs 92.7%) at about a third of the compute. Even **a single test case**, starting at the 4B, is 2.4–2.7× cheaper than the best random mix of models at the same quality. No predictor we built or borrowed (text features, model judges, RouteLLM, self-agreement) beat simple mixing on code.
- **Leverage is about the test, not the task.** How much safety margin to add depends on how much the check can catch, not on how hard the prompt looks.
- **Async is about hardware.** It's 1.1–1.6× faster when the boss runs elsewhere (a cloud boss with local workers). On a shared GPU it gains nothing, and with GPU memory nearly full it was 2× slower. So Gearbox chooses the schedule from where the boss runs.
- **Workers never ask for help.** Across nearly 6,000 answers, only the smallest model ever said "unsure". Escalation has to be triggered by tests, not by the worker.

## The one-line version

> **Hand routine subtasks to cheap models, verify them with even one test, start at a capable small model, escalate on failure, and overlap work only when the boss and workers don't share hardware.**

## Where it could go next
- **Real agents in the loop:** measure real Claude Code or Gemini sessions with and without Gearbox ([ROADMAP.md](ROADMAP.md) R1).
- **Help the boss write that one test:** suggest a test case when a subtask arrives without one.
- **Context by reference:** let workers read the files they need instead of the boss pasting them into the brief (for example via a memory tool such as Waggle).
- **Beyond single functions:** multi-file tasks, test writing, other languages.
