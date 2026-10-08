# Contributing to Gearbox

Welcome! This page gets you from zero to your first change.

## 1. Read these first, in this order (about 1.5 hours)

| # | File | What you'll get from it | Time |
|---|---|---|---|
| 1 | [README.md](README.md) | What Gearbox is, in one screen | 5 min |
| 2 | [REPORT.md](REPORT.md) | **What we found, with all the numbers.** Example: cheap models + checks beat the 27B (95.7% vs 92.7%) at about 1/3 of the compute | 20 min |
| 3 | [ROADMAP.md](ROADMAP.md) | What's built, what's missing (ranked), what's next | 10 min |
| 4 | [EXPLANATION.md](EXPLANATION.md) | How the code works: the vocabulary, one delegation traced through the files, the sandbox, the benchmark | 30 min |
| 5 | [bench/README.md](bench/README.md) | Which experiment script answers which question | 5 min |
| 6 | [docs/paper/results.md](docs/paper/results.md) | The paper version: 17 findings with intervals and tests | 20 min |

Optional background: [docs/lit-review.md](docs/lit-review.md) (why the idea is new) and [docs/paper/outline.md](docs/paper/outline.md) (the paper plan).

## 2. Set up (Mac or Linux)

```bash
git clone https://github.com/kartikeyagrawal2007/gearbox.git && cd gearbox
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,plots]"
cp gearbox.example.yaml gearbox.yaml
.venv/bin/python -m pytest -q          # all tests should pass
.venv/bin/gearbox ui --simulate        # dashboard with fake models: http://127.0.0.1:8790
```

In the dashboard, click **Run the batch** under "Watch a delegation" to see the whole pipeline. With real models (Ollama), set your tiers in `gearbox.yaml`.

## 3. Where things live

- `gearbox/`: the tool itself (installable, kept light: no PyTorch or other heavy dependencies here)
- `bench/`: experiments; heavier optional extras are fine here (`.[plots]`, `.[baselines]`)
- `vm/`: the lab PC (RTX A5000) scripts and configs
- `docs/paper/`: the paper draft, data and figures
- `routers/`: trained router files

## 4. Data

Raw results live in `runs/`, which is **not in git** (about 30 MB of model answers). Ask Kartikey for the latest `runs` archive and unpack it in the repo root. The summaries the paper cites are in git (`docs/paper/data/`).

## 5. The lab PC

Big runs happen on the lab PC (Windows + WSL Ubuntu, RTX A5000 24 GB, Ollama with 12 models). Its README is [vm/README.md](vm/README.md).
- **Run anything long inside `tmux`,** so a dropped remote connection doesn't kill it. Every script resumes where it stopped.
- **One GPU job at a time.** Two jobs on the GPU distort timings and can run it out of memory. Ollama is set to an 8k context (`/etc/systemd/system/ollama.service.d/context.conf`) for the same reason.
- **Moving results to a laptop:** `tar czf` them onto the Windows Desktop (`/mnt/c/Users/PRO-LAB-2/Desktop/`), then copy with AnyDesk.

## 6. Rules for experiments

Each of these was a real mistake we made and fixed. A number that breaks one of them doesn't go in the paper.
1. **Same input, same output:** temperature 0, and a warm-up call before timing anything.
2. **Never tune on the test set.** Thresholds and settings are chosen on training data only (cross-validation).
3. **Compare against the fair baseline.** For a router, that's *random mixing of fixed models at equal quality*, with the router's own cost included, not "always the biggest model".
4. **Report uncertainty:** 95% intervals or paired tests (`bench/stats.py`) before calling a difference real.
5. **Look for the boring explanation first.** The 2× async slowdown turned out to be GPU memory, not compute; a control run found it.
6. **Write down negative results.** The router that doesn't save anything is a finding, not a failure.

## 7. Code

- **Readable over clever.** Comments explain *why*, not *what*. Every module starts with a docstring that says what it's for.
- **Tests for every behaviour change:** `tests/` uses pytest with fake models (`tests/fakes.py`), so tests never need a GPU.
- **Don't edit `bench/evalplus_compat.py`.** It's EvalPlus's grading code, copied verbatim, and it's why our scores match the published ones.
- **Configs explain themselves:** every new setting gets a comment in `gearbox.example.yaml`.

## 8. Workflow

- Work on a branch and open a pull request into `main`. Run `.venv/bin/python -m pytest -q` before pushing.
- Small commits. The message says what changed and **why**.
- Good first tasks, from [ROADMAP.md](ROADMAP.md):
  - **R5:** report the GPU energy already recorded in `runs/async*.jsonl`
  - **T5:** add GitHub Actions to run the tests on every push
  - **T6:** keep delegated tasks across a server restart
