"""Figures and tables for the final report.

Inputs:  one or more training run dirs (train.csv + eval.csv from train.py) and one evaluation dir
         (games.csv + summary.json from evaluate.py).
Outputs: PNGs + results_table.md in --out.

    learning_curves.png   periodic greedy win rate (mean over seeds, min-max band) and training score
    win_rate.png          win rate per policy with Wilson 95% CI bars
    score_hist.png        final-score distribution per policy (small multiples, shared axes)
    speed_vs_safety.png   steps per apple vs win rate: the speed/safety trade-off
    death_heatmap.png     where the head is when the game is lost, per policy
    results_table.md      the ablation / comparison table (with CIs and tests vs the reference)

Usage (from the repo root):
    python -m dqn.v2.report --runs dqn/v2/runs/seed0 dqn/v2/runs/seed1 --eval dqn/v2/eval/main --out dqn/v2/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical palette (fixed order; a policy keeps its color in every figure).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#1f1f1e", "#5c5b55", "#e4e3de"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
})


def colors_for(names):
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


# --------------------------------------------------------------------------------------------- learning
def learning_curves(run_dirs, out, smooth=200):
    evals = [pd.read_csv(Path(d) / "eval.csv") for d in run_dirs]
    trains = [pd.read_csv(Path(d) / "train.csv") for d in run_dirs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    c = SERIES[0]

    # Win rate: align seeds on the evaluation episodes they share.
    ep = evals[0]["episode"].to_numpy()
    wins = np.array([e["win_rate"].to_numpy()[:len(ep)] for e in evals]) * 100
    if len(evals) > 1:
        ax1.fill_between(ep, wins.min(0), wins.max(0), color=c, alpha=0.18, linewidth=0, label="min-max over seeds")
    else:  # single seed: show the Wilson interval of that evaluation instead
        e = evals[0]
        ax1.fill_between(ep, e["win_lo"] * 100, e["win_hi"] * 100, color=c, alpha=0.18, linewidth=0,
                         label="Wilson 95% CI")
    ax1.plot(ep, wins.mean(0), color=c, label=f"mean of {len(evals)} seed(s)" if len(evals) > 1 else "greedy eval")
    ax1.set(xlabel="training episode", ylabel="win rate (%)", title="Greedy evaluation win rate", ylim=(0, 100))
    ax1.legend(loc="upper left")

    # Training score, rolling mean per seed then mean across seeds.
    n = min(len(t) for t in trains)
    rolled = np.array([t["score"].rolling(smooth, min_periods=1).mean().to_numpy()[:n] for t in trains])
    x = np.arange(1, n + 1)
    if len(trains) > 1:
        ax2.fill_between(x, rolled.min(0), rolled.max(0), color=c, alpha=0.18, linewidth=0)
    ax2.plot(x, rolled.mean(0), color=c)
    ax2.axhline(61, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax2.text(x[-1], 61, "board full (61)", ha="right", va="bottom", color=INK_2, fontsize=8)
    ax2.set(xlabel="training episode", ylabel=f"score (rolling mean, {smooth} ep)",
            title="Training score (with exploration)", ylim=(0, 64))
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- evaluation
def win_rate_bars(summary, names, colors, out):
    p = summary["policies"]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(names) + 1))
    y = np.arange(len(names))[::-1]
    rates = np.array([p[n]["win_rate"] for n in names]) * 100
    # clip: at 0% / 100% the CI bound equals the rate up to float rounding
    lo = np.clip(rates - np.array([p[n]["win_rate_ci"][0] for n in names]) * 100, 0, None)
    hi = np.clip(np.array([p[n]["win_rate_ci"][1] for n in names]) * 100 - rates, 0, None)
    ax.barh(y, rates, height=0.6, color=[colors[n] for n in names])
    ax.errorbar(rates, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for yi, r, h in zip(y, rates, hi):
        ax.text(r + h + 1.5, yi, f"{r:.1f}%", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, names)
    ax.set(xlim=(0, 112), xlabel="win rate (%), Wilson 95% CI",
           title=f"Win rate on {p[names[0]]['games']} fixed-seed 8x8 games")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "win_rate.png")
    plt.close(fig)


def score_hist(games, names, colors, out):
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.1 * rows), sharex=True, sharey=True, squeeze=False)
    bins = np.arange(0, 63, 3)
    for ax, n in zip(axes.flat, names):
        s = games.loc[games["policy"] == n, "score"]
        ax.hist(s, bins=bins, color=colors[n], edgecolor="white", linewidth=1)
        ax.set_title(n, fontsize=9, color=INK)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.supxlabel("final score (61 = win)", fontsize=9)
    fig.supylabel("games", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "score_hist.png")
    plt.close(fig)


def speed_vs_safety(summary, names, colors, out):
    p = summary["policies"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in names:
        s = p[n]
        if not s["steps_per_apple"] == s["steps_per_apple"]:  # NaN: never ate
            continue
        x, y = s["steps_per_apple"], s["win_rate"] * 100
        pos = lambda v: [max(0.0, v)]
        ax.errorbar(x, y, yerr=[pos(y - s["win_rate_ci"][0] * 100), pos(s["win_rate_ci"][1] * 100 - y)],
                    xerr=[pos(x - s["steps_per_apple_ci"][0]), pos(s["steps_per_apple_ci"][1] - x)],
                    fmt="o", ms=8, color=colors[n], ecolor=colors[n], elinewidth=1, mec="white", mew=1.5, label=n)
        if s["wins"]:  # direct-label only the policies that win; the 0% cluster relies on the legend
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8, color=INK)
    ax.set(xlabel="steps per apple (lower = faster), bootstrap 95% CI", ylabel="win rate (%), Wilson 95% CI",
           title="Speed vs. safety", ylim=(-5, 108))
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(out / "speed_vs_safety.png")
    plt.close(fig)


def death_heatmaps(games, names, out, size=8):
    lost = games[games["win"] == 0]
    shown = [n for n in names if (lost["policy"] == n).any()]
    if not shown:
        return
    cols = min(4, len(shown))
    rows = int(np.ceil(len(shown) / cols))
    grids = {}
    for n in shown:
        g = lost[lost["policy"] == n]
        grid = np.zeros((size, size))
        np.add.at(grid, (g["head_r"].clip(0, size - 1), g["head_c"].clip(0, size - 1)), 1)
        grids[n] = (grid / len(g), len(g))
    vmax = max(gr.max() for gr, _ in grids.values())  # one shared scale so panels are comparable
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.4 * rows), squeeze=False,
                             layout="constrained")
    for ax, n in zip(axes.flat, shown):
        grid, count = grids[n]
        im = ax.imshow(grid, cmap=SEQ_CMAP, vmin=0, vmax=vmax)
        ax.set_title(f"{n} ({count} losses)", fontsize=8, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.flat[len(shown):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.7, label="share of losses").ax.tick_params(labelsize=7)
    fig.suptitle("Head position at the end of lost games", fontsize=9, color=INK)
    fig.savefig(out / "death_heatmap.png")
    plt.close(fig)


def results_table(summary, names, out):
    p, cmp_ = summary["policies"], summary["comparisons"]
    pct = lambda v: f"{100 * v:.1f}"
    lines = ["| policy | win % [95% CI] | mean score [CI] | steps/apple [CI] | steps to win [CI] | "
             "p (win rate vs ref, Fisher) |", "|---|---|---|---|---|---|"]
    for n in names:
        s = p[n]
        test = next((c for k, c in cmp_.items() if k.startswith(f"{n} vs ")), None)
        stw = (f"{s['steps_to_win']:.0f} [{s['steps_to_win_ci'][0]:.0f}, {s['steps_to_win_ci'][1]:.0f}]"
               if s["wins"] else "-")
        lines.append(
            f"| {n} | {pct(s['win_rate'])} [{pct(s['win_rate_ci'][0])}, {pct(s['win_rate_ci'][1])}] | "
            f"{s['mean_score']:.1f} [{s['mean_score_ci'][0]:.1f}, {s['mean_score_ci'][1]:.1f}] | "
            f"{s['steps_per_apple']:.1f} [{s['steps_per_apple_ci'][0]:.1f}, {s['steps_per_apple_ci'][1]:.1f}] | "
            f"{stw} | {'reference' if test is None else f'{test['p_fisher']:.2g}'} |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="training run dirs (several seeds -> bands)")
    ap.add_argument("--eval", default=None, help="evaluate.py output dir")
    ap.add_argument("--out", default="dqn/v2/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.runs:
        learning_curves(args.runs, out)
    if args.eval:
        summary = json.loads((Path(args.eval) / "summary.json").read_text())
        games = pd.read_csv(Path(args.eval) / "games.csv")
        names = list(summary["policies"])
        colors = colors_for(names)
        win_rate_bars(summary, names, colors, out)
        score_hist(games, names, colors, out)
        speed_vs_safety(summary, names, colors, out)
        death_heatmaps(games, names, out, summary["config"]["size"])
        results_table(summary, names, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
