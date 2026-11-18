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


def fmt_p(p):
    """p-value for tables; 0.0 means it underflowed double precision, not that it is exactly 0."""
    return "< 1e-300" if p == 0 else f"{p:.2g}"


def colors_for(names):
    """Fixed categorical order, never cycled: entries past the 8th slot get neutral grays (they are the
    reference baselines at the end of the list, and are also identified by labels)."""
    grays = ["#8a8983", "#55544f"]
    return {n: SERIES[i] if i < len(SERIES) else grays[(i - len(SERIES)) % len(grays)]
            for i, n in enumerate(names)}


# --------------------------------------------------------------------------------------------- learning
def learning_curves(groups, out, smooth=200):
    """groups: {label: [run dirs (one per seed)]}. Line = mean over seeds, band = min-max over seeds
    (or the Wilson CI of the single run when a group has one seed)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for (label, run_dirs), c in zip(groups.items(), SERIES):
        evals = [pd.read_csv(Path(d) / "eval.csv") for d in run_dirs]
        trains = [pd.read_csv(Path(d) / "train.csv") for d in run_dirs]
        k = len(run_dirs)

        # Win rate: seeds are aligned by evaluation index (all runs use the same eval interval).
        m = min(len(e) for e in evals)
        ep = evals[0]["episode"].to_numpy()[:m]
        wins = np.array([e["win_rate"].to_numpy()[:m] for e in evals]) * 100
        if k > 1:
            ax1.fill_between(ep, wins.min(0), wins.max(0), color=c, alpha=0.18, linewidth=0)
        else:
            ax1.fill_between(ep, evals[0]["win_lo"][:m] * 100, evals[0]["win_hi"][:m] * 100, color=c,
                             alpha=0.18, linewidth=0)
        ax1.plot(ep, wins.mean(0), color=c, label=f"{label} ({k} seed{'s' if k > 1 else ''})")

        # Training score: rolling mean per seed, then mean across seeds.
        n = min(len(t) for t in trains)
        rolled = np.array([t["score"].rolling(smooth, min_periods=1).mean().to_numpy()[:n] for t in trains])
        x = np.arange(1, n + 1)
        if k > 1:
            ax2.fill_between(x, rolled.min(0), rolled.max(0), color=c, alpha=0.18, linewidth=0)
        ax2.plot(x, rolled.mean(0), color=c, label=label)

    band = "band = min-max over seeds" if any(len(d) > 1 for d in groups.values()) else "band = Wilson 95% CI"
    ax1.set(xlabel="training episode", ylabel="win rate (%)",
            title=f"Greedy validation win rate ({band})", ylim=(0, 100))
    ax1.legend(loc="upper left", fontsize=8)
    ax2.axhline(61, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax2.text(ax2.get_xlim()[1], 61, "board full (61)", ha="right", va="bottom", color=INK_2, fontsize=8)
    ax2.set(xlabel="training episode", ylabel=f"score (rolling mean, {smooth} ep)",
            title="Training score (with exploration)", ylim=(0, 64))
    ax2.legend(loc="lower right", fontsize=8)
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- seeds
def pool_seeds(summary, games):
    """Merge policies named 'variant|sK' into one 'variant' entry per variant.

    Returns (pooled summary, pooled games, per-seed win rates {variant: [rate per seed]}).
    The pooled CI treats all games as independent, so it reflects game-to-game noise only; the per-seed
    spread (seed-to-seed variation of training) is reported next to it."""
    from .evaluate import compare, summarize
    games = games.copy()
    games["variant"] = games["policy"].str.split("|").str[0]
    order = list(dict.fromkeys(n.split("|")[0] for n in summary["policies"]))
    per_seed = {v: [summary["policies"][n]["win_rate"] for n in summary["policies"] if n.split("|")[0] == v]
                for v in order}
    records = {v: games[games["variant"] == v].assign(policy=v).to_dict("records") for v in order}
    pooled = {"config": summary["config"], "policies": {v: summarize(records[v]) for v in order},
              "comparisons": {}, "_records": records}
    ref = order[0]
    for v in order[1:]:
        pooled["comparisons"][f"{v} vs {ref}"] = compare(records[v], records[ref])
    return pooled, games.assign(policy=games["variant"]).drop(columns="variant"), per_seed


def ablation_steps(pooled, steps, out):
    """Tests between consecutive steps of the ablation ladder (each step adds one change)."""
    from .evaluate import compare
    lines = ["| change | win % before -> after | p (Fisher) | steps to win before -> after | p (Mann-Whitney) |",
             "|---|---|---|---|---|"]
    rec = pooled["_records"]
    for a, b in zip(steps, steps[1:]):
        c = compare(rec[b], rec[a])
        pa, pb = pooled["policies"][a], pooled["policies"][b]
        lines.append(f"| {a} -> {b} | {100 * pa['win_rate']:.1f} -> {100 * pb['win_rate']:.1f} | "
                     f"{fmt_p(c['p_fisher'])} | {pa['steps_to_win']:.0f} -> {pb['steps_to_win']:.0f} | "
                     f"{fmt_p(c['p_mann_whitney'])} |")
    (out / "ablation_tests.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------- evaluation
def win_rate_bars(summary, names, colors, out, per_seed=None):
    p = summary["policies"]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(names) + 1))
    y = np.arange(len(names))[::-1]
    rates = np.array([p[n]["win_rate"] for n in names]) * 100
    # clip: at 0% / 100% the CI bound equals the rate up to float rounding
    lo = np.clip(rates - np.array([p[n]["win_rate_ci"][0] for n in names]) * 100, 0, None)
    hi = np.clip(np.array([p[n]["win_rate_ci"][1] for n in names]) * 100 - rates, 0, None)
    ax.barh(y, rates, height=0.6, color=[colors[n] for n in names])
    ax.errorbar(rates, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for yi, r, h, n in zip(y, rates, hi, names):
        right = max([r + h] + [100 * s for s in (per_seed or {}).get(n, [])])  # clear the CI and seed dots
        ax.text(right + 2, yi, f"{r:.1f}%", va="center", color=INK, fontsize=8)
    if per_seed:  # one white-ringed dot per training seed on top of the pooled bar
        labeled = False
        for yi, n in zip(y, names):
            seeds = per_seed.get(n, [])
            if len(seeds) > 1:
                ax.scatter(np.array(seeds) * 100, [yi] * len(seeds), s=22, color=INK, edgecolor="white",
                           linewidth=1.2, zorder=3, label=None if labeled else "one training seed")
                labeled = True
        if labeled:
            ax.legend(loc="lower right", fontsize=7)
    ax.set_yticks(y, names)
    ax.set(xlim=(0, 112), xlabel="win rate (%), Wilson 95% CI on pooled games",
           title="Win rate on 1000 fixed test-seed 8x8 games per model")
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
    labeled = []
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
            labeled.append((y, x, n))
    # Highest point first: when two points are close, the upper one is labeled above, the lower one below.
    placed = []
    for y, x, n in sorted(labeled, reverse=True):
        crowded = any(abs(x - px) < 1.0 and abs(y - py) < 8 for py, px in placed)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(7, -13 if crowded else 5), fontsize=8, color=INK)
        placed.append((y, x))
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


def results_table(summary, names, out, per_seed=None):
    p, cmp_ = summary["policies"], summary["comparisons"]
    per_seed = per_seed or {}
    pct = lambda v: f"{100 * v:.1f}"
    lines = ["| policy | games | win % [95% CI] | win % per seed | mean score [CI] | steps/apple [CI] | "
             "steps to win [CI] | p (win rate vs ref, Fisher) |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = p[n]
        test = next((c for k, c in cmp_.items() if k.startswith(f"{n} vs ")), None)
        stw = (f"{s['steps_to_win']:.0f} [{s['steps_to_win_ci'][0]:.0f}, {s['steps_to_win_ci'][1]:.0f}]"
               if s["wins"] else "-")
        seeds = " / ".join(pct(r) for r in per_seed.get(n, [])) if len(per_seed.get(n, [])) > 1 else "-"
        lines.append(
            f"| {n} | {s['games']} | {pct(s['win_rate'])} [{pct(s['win_rate_ci'][0])}, {pct(s['win_rate_ci'][1])}] | "
            f"{seeds} | {s['mean_score']:.1f} [{s['mean_score_ci'][0]:.1f}, {s['mean_score_ci'][1]:.1f}] | "
            f"{s['steps_per_apple']:.1f} [{s['steps_per_apple_ci'][0]:.1f}, {s['steps_per_apple_ci'][1]:.1f}] | "
            f"{stw} | {'reference' if test is None else fmt_p(test['p_fisher'])} |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[],
                    help="training runs, either plain dirs or 'label=dir1,dir2,...' groups (seeds -> bands)")
    ap.add_argument("--eval", default=None, help="evaluate.py output dir (policies named 'variant|sK' are "
                                                 "pooled over seeds)")
    ap.add_argument("--ablation", nargs="*", default=[],
                    help="ordered variant names; consecutive pairs are tested in ablation_tests.md")
    ap.add_argument("--out", default="dqn/v2/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.runs:
        groups = {}
        for spec in args.runs:
            label, _, dirs = spec.rpartition("=")
            groups[label or Path(dirs).name] = dirs.split(",")
        learning_curves(groups, out)
    if args.eval:
        summary = json.loads((Path(args.eval) / "summary.json").read_text())
        games = pd.read_csv(Path(args.eval) / "games.csv")
        per_seed = None
        if any("|" in n for n in summary["policies"]):
            summary, games, per_seed = pool_seeds(summary, games)
            if args.ablation:
                ablation_steps(summary, args.ablation, out)
        names = list(summary["policies"])
        colors = colors_for(names)
        win_rate_bars(summary, names, colors, out, per_seed)
        score_hist(games, names, colors, out)
        speed_vs_safety(summary, names, colors, out)
        death_heatmaps(games, names, out, summary["config"]["size"])
        results_table(summary, names, out, per_seed)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
