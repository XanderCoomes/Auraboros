"""Plot compare_tail.py results in the style of test_result/space_compare_*/compare.png.

Usage: python plot_tail_compare.py [test_result/tail_compare_<timestamp>]   (default: the latest one)
Dots = seeds, marker = mean, bars = std. x axis = training episodes, colour = feature set.
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

d = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("test_result/tail_compare_*"))[-1]
s = json.load(open(os.path.join(d, "summary.json")))
sets = list(s["results"])
episodes = list(next(iter(s["results"].values())))
games = s["config"]["eval_games"]

fig, ax = plt.subplots(figsize=(10, 5))
rng = np.random.default_rng(0)
offsets = (np.arange(len(sets)) - (len(sets) - 1) / 2) * 0.25
for i, name in enumerate(sets):
    for x, e in enumerate(episodes):
        r = s["results"][name][e]
        v = np.array(r["per_seed"])
        pos = x + offsets[i]
        ax.scatter(pos + rng.uniform(-0.05, 0.05, len(v)), v, s=14, alpha=0.45, color=COLORS[i])
        ax.errorbar(pos, v.mean(), yerr=v.std(), fmt="o", color=COLORS[i], capsize=4, ms=7,
                    label=f"{name} ({r['n_states']})" if x == 0 else None)
ax.axhline(61, color="gray", ls="--", lw=0.8)
ax.text(len(episodes) - 0.5, 61.5, "full board (61)", ha="right", fontsize=8, color="gray")
ax.set_xticks(range(len(episodes)))
ax.set_xticklabels([f"{int(e) // 1000}k episodes" for e in episodes])
ax.set_ylabel(f"mean apples over {games} greedy games")
ax.set_title(f"{s['seeds']} seeds: dots = seeds, marker = mean, bars = std")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = os.path.join(d, "compare.png")
fig.savefig(out, dpi=130)
print("saved", out)
