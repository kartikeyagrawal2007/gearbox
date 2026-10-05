"""Paper figures from benchmark results.

    python bench/plots.py                              # runs/ -> docs/paper/figures/
    python bench/plots.py docs/paper/data --tasks mbpp+

Figures (PNG to look at, PDF for LaTeX):
  1. size_ladder:  pass rate and false-done rate vs parameter count for the Qwen3.5 ladder,
                   with the older Qwen2.5-Coder 0.5B as a grey reference point.
  2. vendors:      pass rate and false-done rate for four vendors at matched sizes.
  3. hatch:        pass rate with the UNSURE escape hatch on vs off, per Qwen3.5 size.

Model size and vendor are read from the model's name in the config (e.g. "granite4.2-8b").
Needs the optional `plots` extra: pip install -e ".[plots]".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; works without a display (e.g. over a terminal)
import matplotlib.pyplot as plt  # noqa: E402

from results import tier_order  # noqa: E402

from gearbox.runs import load_runs  # noqa: E402

# Reference palette, categorical slots 1-4 in fixed order (validated: CVD-safe on white;
# slots 3-4 are low-contrast, so every bar carries a printed value label).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REFERENCE = "#8b8a84"  # neutral grey for the non-series reference point
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

VENDORS = [  # (name prefix, label) in plotting order
    ("qwen3.5", "Qwen3.5"),
    ("granite", "Granite 4.2 (IBM)"),
    ("ministral", "Ministral 3 (Mistral)"),
    ("gemma", "Gemma 3 (Google)"),
]
_SIZE = re.compile(r"(\d+(?:\.\d+)?)b$")


def params_b(tier: str) -> float | None:
    """Parameters in billions, from a name like 'qwen3.5-0.8b'."""
    m = _SIZE.search(tier)
    return float(m.group(1)) if m else None


def vendor(tier: str) -> str | None:
    return next((label for prefix, label in VENDORS if tier.startswith(prefix)), None)


def style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)


def save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def pct(row: dict, key: str) -> float | None:
    if key == "pass":
        return 100 * row["passed"] / row["tasks"]
    value = row.get("logic_false_done_rate")
    return None if value is None else 100 * value


def size_ladder(rows: list[dict], task_set: str, out: Path) -> list[Path]:
    ladder = sorted((r for r in rows if r["tier"].startswith("qwen3.5") and r["hatch"] == "on"),
                    key=lambda r: params_b(r["tier"]))
    reference = next((r for r in rows if r["tier"].startswith("qwen2.5-coder") and r["hatch"] == "on"), None)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in ((axes[0], "pass", "Problems solved"), (axes[1], "fd", "False-done rate")):
        xs = [params_b(r["tier"]) for r in ladder]
        ys = [pct(r, key) for r in ladder]
        ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label="Qwen3.5", zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=INK)
        if reference:
            x, y = params_b(reference["tier"]), pct(reference, key)
            ax.plot([x], [y], marker="D", markersize=6, color=REFERENCE, linestyle="none", zorder=3)
            # Put the label where the ladder's line isn't: above the point when the line rises
            # from below it (pass rate), below the point when the line falls from above (false-done).
            above = key == "pass"
            ax.annotate(f"Qwen2.5-Coder 0.5B\n(older, code-tuned): {y:.0f}%", (x, y), textcoords="offset points",
                        xytext=(4, 10 if above else -10), fontsize=7.5, color=INK_MUTED, va="bottom" if above else "top")
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:g}B" for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("Parameters (log scale)", color=INK_MUTED)
        style(ax, "% of problems" if key == "pass" else "% of answers claimed done that were wrong")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.suptitle(f"Qwen3.5 size ladder on {task_set} (UNSURE hatch on)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return save(fig, out, f"{task_set.replace('+', 'plus')}_size_ladder")


def vendors(rows: list[dict], task_set: str, out: Path) -> list[Path]:
    on = [r for r in rows if r["hatch"] == "on" and vendor(r["tier"])]
    bands = [("Small (3–4B)", lambda b: 2.5 <= b <= 4.5), ("Mid (8–12B)", lambda b: 7.5 <= b <= 12.5)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    width = 0.19
    for ax, key, title in ((axes[0], "pass", "Problems solved"), (axes[1], "fd", "False-done rate")):
        for i, (_, label) in enumerate(VENDORS):
            for j, (_, in_band) in enumerate(bands):
                match = [r for r in on if vendor(r["tier"]) == label and in_band(params_b(r["tier"]))]
                if not match:
                    continue
                r = match[0]
                x = j + (i - 1.5) * (width + 0.02)  # 0.02 = surface gap between adjacent bars
                y = pct(r, key)
                ax.bar(x, y, width, color=SERIES[i], label=label if j == 0 else None, zorder=3)
                ax.annotate(f"{y:.0f}%\n{params_b(r['tier']):g}B", (x, y), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(bands)), [b for b, _ in bands])
        style(ax, "% of problems" if key == "pass" else "% of answers claimed done that were wrong")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.suptitle(f"Vendors at matched sizes on {task_set} (UNSURE hatch on)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    # One shared legend below both panels, clear of the bars and their labels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK)
    return save(fig, out, f"{task_set.replace('+', 'plus')}_vendors")


def hatch(rows: list[dict], task_set: str, out: Path) -> list[Path] | None:
    by = {(r["tier"], r["hatch"]): r for r in rows if r["tier"].startswith("qwen3.5")}
    tiers = sorted({t for t, h in by if (t, "on") in by and (t, "off") in by}, key=params_b)
    if not tiers:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.36
    for i, (h, label) in enumerate((("on", "Hatch on (may answer UNSURE)"), ("off", "Hatch off"))):
        for j, t in enumerate(tiers):
            r = by[(t, h)]
            x = j + (i - 0.5) * (width + 0.02)
            y = pct(r, "pass")
            ax.bar(x, y, width, color=SERIES[i], label=label if j == 0 else None, zorder=3)
            note = f"{y:.0f}%" + (f"\n{r['unsure']} unsure" if r["unsure"] else "")
            ax.annotate(note, (x, y), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(tiers)), [f"{params_b(t):g}B" for t in tiers])
    style(ax, "% of problems solved")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.set_title(f"Qwen3.5 on {task_set}: offering an UNSURE option", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    return save(fig, out, f"{task_set.replace('+', 'plus')}_hatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="?", default="runs", help="folder of result JSON files (default: runs)")
    parser.add_argument("--tasks", default="humaneval+", help="task set to plot (default: humaneval+)")
    parser.add_argument("--out", default="docs/paper/figures", help="output folder")
    args = parser.parse_args()
    rows = [r for r in load_runs(Path(args.runs), tier_order())["combined"] if r["task_set"] == args.tasks]
    if not rows:
        raise SystemExit(f"no {args.tasks} results in {args.runs}/")
    out = Path(args.out)
    made = size_ladder(rows, args.tasks, out) + vendors(rows, args.tasks, out) + (hatch(rows, args.tasks, out) or [])
    print("\n".join(str(p) for p in made))


if __name__ == "__main__":
    main()
