# bench/: the experiments

These scripts produce every number in [REPORT.md](../REPORT.md) and [docs/paper/results.md](../docs/paper/results.md). None of them are part of the installed `gearbox` package.

Run them from the repository root, e.g. `.venv/bin/python bench/results.py`. Results go to `runs/`, which isn't in git (see [CONTRIBUTING.md](../CONTRIBUTING.md#data)).

## By question

| Question | Script | Needs a GPU? | Reads | Writes |
|---|---|---|---|---|
| How often is a model's "done" wrong? | `false_done.py` | yes (lab PC) | models | `runs/*.json` |
| What are the scores, with uncertainty? | `results.py`, `stats.py` | no | `runs/*.json` | terminal |
| Paper figures | `plots.py` | no | `runs/*.json` | `docs/paper/figures/` |
| Does start cheap, check, escalate pay? | `routing_sim.py` | no | `runs/*.json` | terminal |
| How much does a weak check hurt, and does leverage fix it? | `weak_checks.py` | no (runs checks on CPU) | `runs/*.json` | `runs/cache/weak_checks.json` |
| Can a router pick the model up front? | `router_features.py` (judges and embeddings) → `router_train.py` | features: yes; training: no | `runs/`, features | `runs/cache/router_features.json`, `routers/*.json` |
| Is that router better than random mixing? (the fair test) | `router_fair.py` | no | the above | terminal |
| Does RouteLLM's router work on code? | `routellm_baseline.py` | no (CPU is fine) | problems | `runs/cache/router_features.json` |
| Does the boss finish sooner if it doesn't wait? | `async_bench.py` | yes | models | `runs/async*.jsonl` |
| Can a model check itself by agreement? | `sample_answers.py` → `agreement.py` | sampling: yes; analysis: no | `runs/`, samples | `runs/cache/samples.json`, `runs/cache/behaviours.json` |

## Shared pieces
- **`tasksets.py`**: the task sets (smoke 8, HumanEval+ 164, MBPP+ 378) and the hidden-test check builder.
- **`evalplus_compat.py`**: EvalPlus's grading code, **copied verbatim**. Don't edit it: it's why our scores match the published ones.
- **`data/`**: the EvalPlus problem files, downloaded on first use (not in git).

## Rules every script follows
These came from mistakes we made and fixed. Keep them when you add a script.
- Temperature 0 and a warm-up call before any timing.
- Write results after every model or episode, so a crash loses nothing, and **resume** instead of starting over.
- Judge a router only on problems it never saw, with its threshold chosen on training data, against **random mixing at equal quality**, with the router's own cost included.
- Report 95% intervals or paired tests (`stats.py`) before calling a difference real.
