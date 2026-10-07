"""Learned router: predict which models will solve a problem, and assign the cheapest one directly.

    python bench/router_train.py                     # evaluate on HumanEval+ and MBPP+ (Qwen ladder)
    python bench/router_train.py --ladder all        # all 12 models
    python bench/router_train.py --json out.json

Method (item response theory, as in IRT-Router): every model m has an ability a_m and every
problem p a difficulty b_p, fitted on recorded outcomes so that

    P(m solves p) = sigmoid(a_m - b_p).

For a new problem only b_p is unknown, so the router learns to predict it from the problem's
text (ridge regression on text features), gets a pass probability for every model, and
assigns the cheapest model whose probability reaches a bar tau. The bar is the leverage knob:
a higher bar buys accuracy with cost.

Evaluation never sees the test problems' outcomes:
  cv     5-fold cross-validation within one benchmark (abilities and the text model are
         fitted on the other folds)
  cross  train on one benchmark, test on the other (different authors and prompt format)

Reported per strategy: single-shot accuracy, cost (billions of parameters per problem) and
worker time (the recorded model seconds per problem), plus the same with check-and-escalate
from the router's pick. Baselines: fixed models, the cascade from the cheapest model, the
heuristic router, and the oracle.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from plots import params_b
from routing_sim import QWEN, SETS, make_ladder
from tasksets import EVALPLUS_SETS

from gearbox.difficulty import HeuristicEstimator
from gearbox.router import base_tier_for

TAUS = [round(0.3 + 0.05 * i, 2) for i in range(14)]  # 0.30 ... 0.95


# --- Data -------------------------------------------------------------------------------

def load_set(runs_dir: Path, task_set: str) -> dict:
    """Problems, their text, and per model: passed (0/1) and model seconds, from hatch-on runs."""
    rows: dict[str, dict[str, dict]] = {}
    for path in sorted(runs_dir.glob("*.json"), key=lambda f: f.stat().st_mtime):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        for r in data if isinstance(data, list) else []:
            if isinstance(r, dict) and r.get("task_set") == task_set and r.get("hatch", "on") == "on" \
                    and len(r.get("rows", [])) == SETS[task_set]:
                rows[r["tier"]] = {x["task"]: x for x in r["rows"]}
    _, gz_name, _ = EVALPLUS_SETS[task_set]
    with gzip.open(Path(__file__).parent / "data" / gz_name, "rt") as f:
        problems = {p["task_id"]: p for p in map(json.loads, f)}
    ids = list(problems)
    return {
        "ids": ids,
        "text": {i: problems[i]["prompt"] for i in ids},
        "entry": {i: problems[i]["entry_point"] for i in ids},
        "passed": {t: np.array([rows[t][i]["outcome"] == "ok" for i in ids], dtype=float) for t in rows},
        "seconds": {t: np.array([rows[t][i]["seconds"] for i in ids], dtype=float) for t in rows},
    }


# --- Item response theory ---------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_irt(Y: np.ndarray, l2: float = 0.01, steps: int = 3000, lr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Rasch model on a (problems x models) 0/1 matrix: returns (ability per model, difficulty per problem)."""
    n_p, n_m = Y.shape
    a, b = np.zeros(n_m), np.zeros(n_p)
    for _ in range(steps):
        p = sigmoid(a[None, :] - b[:, None])
        err = p - Y
        a -= lr * (err.mean(0) + l2 * a)
        b -= lr * (-err.mean(1) + l2 * b)
    return a, b


# --- Text features ----------------------------------------------------------------------

_WORD = re.compile(r"[a-z_]+")
_STOP = set("a an the of to and or in is be for that with as by on it this from are at if not into its "
            "given return returns function write python def".split())


def describe(text: str, entry: str) -> str:
    """The task description without code examples, signature or doctest lines."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith((">>>", "assert", "def ", "import ", "from ", '"""', "'''")) or s == '"""':
            continue
        keep.append(s.strip('"\''))
    return " ".join(keep).lower()


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text) if w not in _STOP and len(w) > 2]


def handcrafted(text: str, entry: str) -> list[float]:
    """Format-independent signals of how much the task asks for."""
    desc = describe(text, entry)
    w = words(desc)
    n_examples = text.count(">>>") + text.count("assert ")
    sig = re.search(rf"def {re.escape(entry)}\((.*?)\)", text)
    n_args = len([x for x in (sig.group(1).split(",") if sig and sig.group(1).strip() else [])])
    lower = text.lower()
    def has(*keys):
        return float(any(k in lower for k in keys))
    return [
        math.log1p(len(desc)), math.log1p(len(w)), len(set(w)) / (len(w) or 1),
        math.log1p(n_examples), float(n_args),
        desc.count(",") / (len(w) or 1), float(desc.count(". ") + desc.count("; ")),
        has("if ", "otherwise", "else", "unless", "except"),          # conditions
        has("all ", "every", "each"), has("only", "must", "should"),
        has("prime", "factor", "divisor", "gcd"), has("fibonacci", "recurs", "sequence"),
        has("sort", "order", "largest", "smallest", "max", "min"),
        has("string", "character", "letter", "word", "vowel"), has("list", "array", "tuple"),
        has("dict", "key", "map"), has("digit", "binary", "decimal", "base "),
        has("matrix", "grid", "row", "column"), has("float", "round", "decimal places"),
        has("count", "number of", "how many"), has("regex", "pattern", "match"),
        has("date", "time", "year", "month"), has("shortest", "longest", "minimum", "maximum"),
    ]


class Featurizer:
    """Feature blocks for a problem, each standardized on the training problems.

    kinds: "hand" (handcrafted text signals), "tfidf" (word and word-pair TF-IDF of the
    description), or a key of `extra` such as "judge:qwen3.5-9b" or "embed:nomic-embed-text"
    (from bench/router_features.py). Blocks are scaled by 1/sqrt(width) so a 768-wide
    embedding doesn't drown out a one-number judge rating."""

    def __init__(self, kinds: list[str], extra: dict[str, dict] | None = None, min_df: int = 2) -> None:
        self.kinds, self.extra, self.min_df = kinds, extra or {}, min_df

    def _block(self, kind, ids, texts, entries, fit):
        if kind == "hand":
            return np.array([handcrafted(t, e) for t, e in zip(texts, entries)])
        if kind == "tfidf":
            docs = []
            for t, e in zip(texts, entries):
                w = words(describe(t, e))
                docs.append(w + [f"{a}_{b}" for a, b in zip(w, w[1:])])
            if fit:
                df = Counter(w for d in docs for w in set(d))
                terms = sorted(w for w, c in df.items() if c >= self.min_df)
                self.vocab = {w: i for i, w in enumerate(terms)}
                self.idf = np.array([math.log((1 + len(docs)) / (1 + df[w])) + 1 for w in terms])
            X = np.zeros((len(docs), len(self.vocab)))
            for i, d in enumerate(docs):
                for w, c in Counter(d).items():
                    j = self.vocab.get(w)
                    if j is not None:
                        X[i, j] = (1 + math.log(c)) * self.idf[j]
            return X
        values = self.extra[kind]
        rows = [values.get(i) for i in ids]
        width = len(next(v for v in rows if v is not None)) if isinstance(rows[0], list) else 1
        fill = [0.0] * width if width > 1 else float("nan")
        X = np.array([v if v is not None else fill for v in rows], dtype=float).reshape(len(ids), width)
        if width == 1:  # a missing judge rating gets the training mean
            if fit:
                self.fill = np.nanmean(X)
            X = np.where(np.isnan(X), self.fill, X)
        return X

    def transform(self, ids, texts, entries, fit=False) -> np.ndarray:
        blocks = []
        if fit:
            self.stats = {}
        for kind in self.kinds:
            X = self._block(kind, ids, texts, entries, fit)
            if fit:
                self.stats[kind] = (X.mean(0), X.std(0) + 1e-9)
            mu, sd = self.stats[kind]
            blocks.append((X - mu) / sd / math.sqrt(X.shape[1]))
        return np.hstack(blocks)


ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def ridge(X, y, alpha):
    y0 = y.mean()
    w = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ (y - y0))
    return w, y0


class TextModel:
    """Ridge regression from problem features to IRT difficulty, with the ridge strength chosen
    by 3-fold cross-validation on the training problems only."""

    def __init__(self, kinds: list[str], extra: dict | None = None) -> None:
        self.feat = Featurizer(kinds, extra)

    def fit(self, ids, texts, entries, y):
        X = self.feat.transform(ids, texts, entries, fit=True)
        folds = np.arange(len(y)) % 3
        def cv_error(alpha):
            err = 0.0
            for f in range(3):
                tr, te = folds != f, folds == f
                w, y0 = ridge(X[tr], y[tr], alpha)
                err += ((X[te] @ w + y0 - y[te]) ** 2).sum()
            return err
        self.alpha = min(ALPHAS, key=cv_error)
        self.w, self.y0 = ridge(X, y, self.alpha)
        return self

    def predict(self, ids, texts, entries) -> np.ndarray:
        return self.feat.transform(ids, texts, entries) @ self.w + self.y0


# --- Routing ----------------------------------------------------------------------------

def route_eval(start: np.ndarray, Y: np.ndarray, costs: np.ndarray, secs: np.ndarray, escalate: bool) -> dict:
    """Accuracy, cost and worker time when problem i starts at model start[i]."""
    n, top = len(start), Y.shape[1] - 1
    correct = cost = time = attempts = 0.0
    for i in range(n):
        m = int(start[i])
        while True:
            cost += costs[m]
            time += secs[i, m]
            attempts += 1
            if Y[i, m] or not escalate or m == top:
                correct += Y[i, m]
                break
            m += 1
    return {"accuracy": correct / n, "cost": cost / n, "time_s": time / n, "attempts": attempts / n}


def pick(P: np.ndarray, tau: float) -> np.ndarray:
    """Cheapest model (columns are cheapest first) with pass probability >= tau, else the top."""
    ok = P >= tau
    return np.where(ok.any(1), ok.argmax(1), P.shape[1] - 1)


def predictions_cv(data, ladder, kinds, extra, folds, seed) -> np.ndarray:
    """Out-of-fold pass probabilities (problems x ladder)."""
    ids = data["ids"]
    Y = np.stack([data["passed"][t] for t in ladder], 1)
    order = np.random.default_rng(seed).permutation(len(ids))
    P = np.zeros_like(Y)

    def view(idx):
        sel = [ids[i] for i in idx]
        return sel, [data["text"][i] for i in sel], [data["entry"][i] for i in sel]

    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        a, b = fit_irt(Y[train])
        tm = TextModel(kinds, extra).fit(*view(train), b)
        P[test] = sigmoid(a[None, :] - tm.predict(*view(test))[:, None])
    return P


def predictions_cross(train, test, ladder, kinds, extra) -> np.ndarray:
    """Train on one benchmark (abilities and the text model), predict the other."""
    Y = np.stack([train["passed"][t] for t in ladder], 1)
    a, b = fit_irt(Y)
    def view(data):
        return data["ids"], [data["text"][i] for i in data["ids"]], [data["entry"][i] for i in data["ids"]]
    tm = TextModel(kinds, extra).fit(*view(train), b)
    return sigmoid(a[None, :] - tm.predict(*view(test))[:, None])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def evaluate(name: str, data, ladder, P: np.ndarray) -> list[dict]:
    Y = np.stack([data["passed"][t] for t in ladder], 1)
    S = np.stack([data["seconds"][t] for t in ladder], 1)
    costs = np.array([params_b(t) for t in ladder])
    rows = []

    def add(strategy, start, note=""):
        for esc in (False, True):
            r = route_eval(start, Y, costs, S, esc)
            rows.append({"eval": name, "strategy": strategy + (" + escalate" if esc else ""), **r, "note": note})

    for i, t in enumerate(ladder):
        add(f"always {t}", np.full(len(Y), i))
    oracle = np.where(Y.any(1), Y.argmax(1), len(ladder) - 1)
    add("oracle", oracle)
    est = HeuristicEstimator()
    heur = np.array([base_tier_for(est.estimate_sync(data["text"][i]).score, len(ladder)) for i in data["ids"]])
    add("heuristic", heur)
    if P is not None:
        pooled = auc(P.ravel(), Y.ravel())
        within = np.nanmean([auc(P[:, j], Y[:, j]) for j in range(len(ladder))])
        for tau in TAUS:
            add(f"learned, tau {tau:.2f}", pick(P, tau), f"AUC pooled {pooled:.3f}, per model {within:.3f}")
    return rows


def render(rows: list[dict]) -> str:
    out, current = [], None
    for r in rows:
        if r["eval"] != current:
            current = r["eval"]
            out += ["", f"== {current}",
                    f"  {'strategy':<40} {'accuracy':>8} {'cost':>6} {'time s':>7} {'attempts':>8}  note"]
        out.append(f"  {r['strategy']:<40} {r['accuracy']:>8.1%} {r['cost']:>6.1f} {r['time_s']:>7.2f}"
                   f" {r['attempts']:>8.2f}  {r['note']}")
    return "\n".join(out)


def export(runs: Path, extra: dict, judge: str, path: Path) -> dict:
    """Train the deployable router on every problem of both benchmarks and all recorded models,
    with one judge's rating as the only feature, and write it for gearbox/difficulty/learned.py."""
    from results import MODELS_CONFIG

    from gearbox.config import load_config

    data = [load_set(runs, ts) for ts in SETS]
    models = sorted(set.intersection(*(set(d["passed"]) for d in data)))
    ids = [i for d in data for i in d["ids"]]
    Y = np.vstack([np.stack([d["passed"][t] for t in models], 1) for d in data])
    a, b = fit_irt(Y)
    key = f"judge:{judge}"
    tm = TextModel([key], extra).fit(ids, [""] * len(ids), [""] * len(ids), b)
    mu, sd = tm.feat.stats[key]
    by_name = {t.name: t.model for t in load_config(MODELS_CONFIG).tiers}
    out = {
        "judge_model": by_name[judge],
        "rating_mean": float(mu[0]), "rating_sd": float(sd[0]),
        "intercept": float(tm.y0), "slope": float(tm.w[0]),
        "abilities": {by_name[m]: round(float(x), 4) for m, x in zip(models, a)},
        "trained_on": [f"{ts} ({SETS[ts]} problems)" for ts in SETS],
        "note": "P(model solves task) = sigmoid(ability - (intercept + slope * (rating - mean) / sd))",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def load_extra(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--ladder", choices=["qwen", "all"], default="qwen")
    ap.add_argument("--features", default="hand",
                    help="comma-separated: hand, tfidf, and any key in the features file "
                         "(e.g. judge:qwen3.5-9b, embed:nomic-embed-text)")
    ap.add_argument("--features-file", default="runs/cache/router_features.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--export", metavar="PATH", help="train the deployable router (one judge) and write it here")
    ap.add_argument("--judge", default="qwen3.5-4b", help="judge tier for --export")
    args = ap.parse_args()
    runs = Path(args.runs)
    extra = load_extra(Path(args.features_file))
    if args.export:
        out = export(runs, extra, args.judge, Path(args.export))
        print(f"wrote {args.export}: judge {out['judge_model']}, {len(out['abilities'])} models")
        return
    kinds = args.features.split(",")
    for k in kinds:
        if k not in ("hand", "tfidf") and k not in extra:
            raise SystemExit(f"feature {k!r} not found; available: hand, tfidf, {', '.join(extra) or '(none)'}")
    data = {ts: load_set(runs, ts) for ts in SETS}
    ladder = make_ladder(args.ladder, data["humaneval+"]["passed"])
    rows = []
    for ts in SETS:
        P = predictions_cv(data[ts], ladder, kinds, extra, args.folds, args.seed)
        rows += evaluate(f"{ts}: {args.folds}-fold CV, features {args.features}", data[ts], ladder, P)
    for tr, te in (("humaneval+", "mbpp+"), ("mbpp+", "humaneval+")):
        P = predictions_cross(data[tr], data[te], ladder, kinds, extra)
        rows += evaluate(f"train {tr} -> test {te}, features {args.features}", data[te], ladder, P)
    print(render(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
