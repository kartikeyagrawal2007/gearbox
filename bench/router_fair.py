"""The fair test for a router: does it beat randomly mixing fixed models at the same quality?

    python bench/router_fair.py                           # 4B judge and RouteLLM BERT, single shot
    python bench/router_fair.py --check "1 test"          # inside a weak-check cascade instead

Two things make router savings look better than they are, and this script avoids both:

  1. Choosing the bar after seeing the test answers. Here the bar is picked on the training
     folds only, then applied to unseen problems (5-fold, averaged over 10 random splits).
  2. Comparing against "always the strongest model". Any quality between two fixed models is
     reachable with no router at all, by sending a random share of tasks to each. A router is
     only useful if it is cheaper than that mix at the same quality. (With a check, the "fixed
     models" are cascades with a fixed starting model.)

The router's own cost is included (its parameters x the ~146 tokens it reads, against ~350
for an answer). Result on HumanEval+ and MBPP+: neither the judge nor RouteLLM's BERT router
reliably beats the mix (see docs/paper/results.md, finding 12).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plots import params_b
from router_train import QWEN, TextModel, cheapest_tau, fit_irt, load_set, pick, sigmoid
from routing_sim import SETS

ROUTERS = {"4B judge": (["judge:qwen3.5-4b"], 4.0), "RouteLLM BERT": (["routellm:bert"], 0.11)}
READ, ANSWER = 146, 350  # tokens the router reads vs tokens in a typical answer


def mix_cost(points: list[tuple[float, float]], quality: float) -> float:
    """Cheapest cost to reach `quality` by randomly mixing two of the (quality, cost) strategies."""
    best = np.inf
    for q1, c1 in points:
        if q1 >= quality:
            best = min(best, c1)
        for q2, c2 in points:
            if q2 > q1 and q1 <= quality <= q2:
                best = min(best, c1 + (quality - q1) / (q2 - q1) * (c2 - c1))
    return best


def outcome(ts, ids, start, Y, costs, weak, level):
    """(correct, cost) per problem when problem i starts at model start[i]: one shot, or a cascade
    that escalates while the weak check fails (a truncated answer counts as caught)."""
    if level is None:
        return Y[np.arange(len(ids)), start], costs[start]
    corr, cost = np.zeros(len(ids)), np.zeros(len(ids))
    for i, pid in enumerate(ids):
        for m in range(int(start[i]), len(QWEN)):
            r = weak[ts][QWEN[m]][pid]
            cost[i] += costs[m]
            if r[level] or m == len(QWEN) - 1:
                corr[i] = r["full"]
                break
    return corr, cost


def evaluate(data, extra, weak, level, targets, seeds) -> list[str]:
    costs = np.array([params_b(t) for t in QWEN])
    lines = []
    for name, (kinds, size) in ROUTERS.items():
        if any(k not in extra for k in kinds):
            lines.append(f"  {name}: features missing, skipped")
            continue
        for ts in SETS:
            d = data[ts]
            ids = d["ids"]
            Y = np.stack([d["passed"][t] for t in QWEN], 1)
            top = Y[:, -1].mean()
            fixed = []
            for s in range(len(QWEN)):
                c, k = outcome(ts, ids, np.full(len(ids), s), Y, costs, weak, level)
                fixed.append((c.mean() / top, k.mean()))
            for target in targets:
                res = []
                for seed in range(seeds):
                    order = np.random.default_rng(seed).permutation(len(ids))
                    start = np.zeros(len(ids), int)
                    for f in range(5):
                        te = order[f::5]
                        tr = np.setdiff1d(order, te)
                        a, b = fit_irt(Y[tr])
                        view = lambda idx: ([ids[i] for i in idx], [""] * len(idx), [""] * len(idx))
                        tm = TextModel(kinds, extra).fit(*view(tr), b)
                        P_tr = sigmoid(a[None, :] - tm.predict(*view(tr))[:, None])
                        if level is None:
                            tau = cheapest_tau(P_tr, Y[tr], costs, target)
                        else:  # the cheapest bar whose cascade reaches the target on the training problems
                            tr_ids, tau, best = [ids[i] for i in tr], 0.99, np.inf
                            for t in np.round(np.arange(0.3, 0.991, 0.02), 2):
                                c, k = outcome(ts, tr_ids, pick(P_tr, t), Y[tr], costs, weak, level)
                                if c.mean() >= target * top and k.mean() < best:
                                    tau, best = float(t), k.mean()
                        start[te] = pick(sigmoid(a[None, :] - tm.predict(*view(te))[:, None]), tau)
                    c, k = outcome(ts, ids, start, Y, costs, weak, level)
                    res.append((c.mean() / top, k.mean() + size * READ / ANSWER))
                quality, cost = np.array(res).mean(0)
                mix = mix_cost(fixed, quality)
                lines.append(f"  {name:<14} {ts:<11} target {target:.0%}: quality {quality:6.1%}  cost {cost:5.1f}"
                             f"  | random mix {mix:5.1f}  -> router {mix / cost:4.2f}x")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--features-file", default="runs/cache/router_features.json")
    ap.add_argument("--check", choices=["1 test", "3 tests", "base"], help="weak-check cascade instead of one shot")
    ap.add_argument("--targets", default="0.85,0.9,0.95", help="quality targets, as a share of the top model")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    runs = Path(args.runs)
    extra = json.loads(Path(args.features_file).read_text())
    weak = json.loads((runs / "cache" / "weak_checks.json").read_text()) if args.check else None
    data = {ts: load_set(runs, ts) for ts in SETS}
    targets = [float(x) for x in args.targets.split(",")]
    print(f"Router vs random mixing of fixed {'cascades' if args.check else 'models'} at the same quality "
          f"({args.check or 'no'} check; bar from training folds; router cost included). Above 1.00x = router wins.")
    print("\n".join(evaluate(data, extra, weak, args.check, targets, args.seeds)))


if __name__ == "__main__":
    main()
