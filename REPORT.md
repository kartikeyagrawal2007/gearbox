# What we did, what happened, and whether it worked

*A plain-language report on the Gearbox project, as of 7 October 2026.*

This file answers the questions you'd ask about the project:
- Did handing subtasks to cheaper models work?
- Did it save time and tokens?
- Were the results good?

[EXPLANATION.md](EXPLANATION.md) explains how the code works. The paper-style version with every number and test is [docs/paper/results.md](docs/paper/results.md).

---

## 1. The question

An expensive AI model (the "boss", like Claude inside Claude Code) does a big task. Many parts of that task are small and routine: write this helper function, write these tests. We asked:

1. **Can a cheap model do those parts instead,** and still get them right?
2. **Does that save money (tokens) and time?**
3. **Can the boss keep working** while the cheap model works, instead of waiting?
4. **Can we pick the right cheap model automatically** for each part?

## 2. What we built

**Gearbox** is a tool any AI agent can plug into, through MCP (the standard way agents use tools). The boss calls `delegate("write function X")`, gets a ticket back instantly, keeps working, and collects the answer later with `await_result`. Behind that, Gearbox:
- **picks a worker model** from a ladder of cheap to strong models
- **checks the worker's answer** by running tests in a sandbox, because cheap models often say "done" when they're wrong
- **escalates to a stronger model** if the check fails, showing it what went wrong
- **tracks cost and time**, and has a dashboard to watch it all

## 3. How we tested it

- **A real GPU at $0:** the college lab PC (RTX A5000, 24 GB) running free open models through Ollama.
- **12 models:** Qwen3.5 at five sizes (0.8B, 2B, 4B, 9B, 27B), plus models from IBM (Granite), Mistral (Ministral) and Google (Gemma) at matched sizes, and an older code-tuned Qwen 0.5B.
- **542 real coding problems** with hidden tests, from **HumanEval+** (164) and **MBPP+** (378), the standard benchmarks for this.
- **We checked the grader first:** our setup reproduces the published scores to within about 1 point, and all 542 official solutions pass their own tests.
- **Fair-testing rules:**
  - a fixed random seed and temperature 0, so the same input gives the same answer
  - models warmed up before timing
  - the order of experiments rotated across repeats
  - 95% confidence intervals and paired statistical tests
  - the router (Section 4.5) judged only on problems it never saw

## 4. The answers

### 4.1 Can a cheap model do the subtasks? **Yes, but only with checking.**

On their own, cheap models are unreliable:

| Model | HumanEval+ | MBPP+ |
|---|---|---|
| Qwen3.5 0.8B | 23% | 29% |
| Qwen3.5 4B | 77% | 65% |
| Qwen3.5 27B (the "expensive" one) | 93% | 78% |

Worse, **they almost never admit they're unsure.** Across 11 current models and nearly 6,000 answers, only the smallest one ever said "unsure", and only 17 times. Even the 27B claimed "done" on 7–21% of answers that were wrong. So you can't trust a worker's "done". You must **check**.

With checking, it works:

| Strategy (HumanEval+) | Accuracy | Compute cost |
|---|---|---|
| Always use the 27B | 92.7% | 27 |
| **Start with a cheap model, check, move up if it fails** | **95.7%** | **8–9** |

**The cheap-first strategy is more accurate than always using the big model, at about a third of the cost.** It's more accurate because small models sometimes solve problems the big one gets wrong. On MBPP+: 82% vs 78%, at about half the cost.

### 4.2 Did it save time? **Yes, about 1.5× on one GPU, and up to 2.8× from one simple rule.**

On the lab GPU, with the 27B as the boss:

| Subtasks | 27B does everything itself | Hands subtasks to the 4B | Faster by |
|---|---|---|---|
| 2 | 37.9 s | 24.3 s | 1.56× |
| 4 | 71.0 s | 46.1 s | 1.54× |
| 8 | 134.6 s | 89.1 s | 1.51× |

**One simple rule saves even more: don't start on the tiniest models.** Tiny models (0.8B, 2B) write long, wrong answers slowly. Starting the check-and-move-up ladder at the 4B instead of the 0.8B gives **the same accuracy, 2.8× faster** on HumanEval+ (57 s → 21 s per problem) and 1.6× faster on MBPP+. Gearbox does this with the `start_floor` setting.

### 4.3 Did it save tokens? **Yes: the expensive model wrote about half as much.**

Words the 27B had to write (tokens), with the 4B doing the coding:

| Subtasks | 27B writes everything | 27B delegates the coding |
|---|---|---|
| 2 | 1,046 | 510 |
| 4 | 1,953 | 985 |
| 8 | 3,792 | 1,964 |

All the code writing moved to the cheap model; what's left is the boss's own work. If the boss were a paid cloud model, that's roughly **half its output bill** in this workload.

There's an important catch:
- In real use, the boss has to **write a short description of each subtask**, and that costs tokens too. Delegation pays off when the answer is much longer than its description, like code or tests. It isn't worth it for tiny jobs.
- In this test the 4B passed a few fewer subtasks (6 of 8 vs 7 of 8) because checking was off. With checking on (Section 4.1), those get caught and redone.

### 4.4 Can the boss keep working instead of waiting (async)? **Yes, when the boss and workers are on separate hardware.**

This was the paper's main idea. The answer depends on *where* the models run:

| Setup | Async vs waiting for each subtask |
|---|---|
| **Boss in the cloud, workers on your GPU** (Gearbox's intended setup; cloud speed simulated) | **1.1–1.6× faster**, matching our formula's prediction |
| Boss and workers on the **same GPU**, with free memory | 1.00×, no gain (they share one GPU's power) |
| Same GPU, **memory nearly full** | up to **2× slower**, and it crashed once |
| Workers on the CPU | async helps (1.2–1.4×), but the CPU is too slow to be worth it |

We also wrote a formula that **predicts the speedup in advance** from a normal run. It matched within 0.01 in 11 of 12 cases.

**What we built from this:** Gearbox now has two modes, chosen automatically:
- **async** when the boss is a cloud model
- **burst** when the boss shares the GPU: Gearbox holds the subtasks until the boss pauses to wait, then runs them all at once, so the two never fight over the GPU

Burst mode measured 1.00×: never slower, and it prevents the memory crash.

### 4.5 Can we pick the right model automatically for each subtask? **It predicts, but it doesn't save, on code.**

We tried hard to build a "guesser" that looks at a task and assigns the right model directly:
- **Simple text clues** (length, keywords): about as good as a coin flip.
- **A small model reading the problem and rating its difficulty:** genuinely predictive, with AUC 0.67–0.74 (0.5 is a coin flip, 1.0 is perfect).
- **RouteLLM,** the best-known open-source router (from the Chatbot Arena team): a coin flip on code (0.43–0.63). It was trained on chat preferences, which is a different question from "will this code pass tests?".

**But measured fairly, no guesser saved anything** compared with simply sending a random share of tasks to each of two fixed models. The guesser's own cost and its mistakes cancel out what it predicts correctly. Starting at a sensible model (the 4B) and checking does better than any guesser we built.

This is a negative result, but a real and useful one. We kept the guesser in Gearbox as an experimental option and report the result honestly.

### 4.6 Other things we learned
- **Leverage (starting a step higher for safety) should depend on how good your test is,** not on how hard the task looks.
  - A weak test (1–3 checks) lets wrong answers through, so start two steps higher.
  - A thorough test catches mistakes, so start cheap.
  - Gearbox now counts a test's checks and sets this automatically. With a 1-check test, starting at the 4B instead of the 0.8B raised accuracy from 62% to 82%.
- **Model rankings don't transfer between benchmarks.** IBM's Granite 8B was clearly best mid-size on HumanEval+ but only tied on MBPP+. Test on your own kind of task.
- **Code specialization beats size at the small end.** An older 0.5B code model beat the newer, larger 0.8B on both benchmarks, by a wide margin.
- **Formatting can fake a failure.** One model left its code blocks unclosed in about a quarter of its answers. A strict reader would have marked them all wrong, so Gearbox repairs this.

## 5. What went wrong along the way (and how we fixed it)

Good research shows its mistakes. Ours, each caught and corrected:
- **Unfair timing** (cold starts, random answers, measurement bugs): fixed, and listed in EXPLANATION.md.
- **Overstated router savings:**
  - I first reported the guesser saving 1.3–1.9×. That came from two errors: picking its settings after seeing the test answers, and comparing against "always use the biggest model" instead of the fair baseline.
  - Re-measured fairly, the savings disappeared, and the write-up says so.
- **A misread cause:** "async is 2× slower on a shared GPU" turned out to be **memory pressure**, not the models competing for compute. A control run with smaller memory settings proved it, and the claim is now more precise.
- **Statistics corrected two claims:** "offering an 'unsure' option hurts small models" and "the 4B and 9B are equal" weren't statistically supported on both benchmarks. Both were softened.

## 6. So, were the results good?

**Yes. Not every idea worked, but the project produced solid, honest, publishable findings:**

| Claim | Status |
|---|---|
| Cheap models + checking can replace the expensive model on routine coding subtasks | ✅ **Strong:** better accuracy at about 1/3 of the compute |
| Delegation saves time | ✅ about 1.5× on one GPU; 2.8× from the start-floor rule |
| Delegation saves the expensive model's tokens | ✅ about half its output in our workload (minus the cost of writing the descriptions) |
| Async (boss keeps working) saves time | ✅ 1.1–1.6×, **but only with separate hardware** (like a cloud boss) |
| A guesser can assign the right model directly | ❌ predicts, but doesn't beat simple strategies on code; RouteLLM doesn't either |
| Leverage should follow test strength | ✅ new and useful; now automatic in Gearbox |

**What it means for the paper:** the story is *"cheap models are safe workers when you check their work; async helps only when boss and workers don't share hardware; per-task routing doesn't pay on code."* That fits a workshop paper or TMLR. Top conferences would want more benchmarks and a real cloud boss.

## 7. What's left
1. **A real cloud boss:** about 30 runs with a real API model as the boss, to confirm the simulated results. Needs a free Gemini API key.
2. **Figures** for the paper.
3. **Writing the paper's remaining sections,** then arXiv (needs an endorser) and TMLR.
4. *(Optional)* A third benchmark, to show the findings hold beyond Python functions.
