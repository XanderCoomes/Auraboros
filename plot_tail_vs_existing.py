"""Overlay the tail result (compare_tail.py) on the existing default / length_bucket comparison.

Usage: python plot_tail_vs_existing.py
Reads  test_result/seeds10_*/summary.json (default, length_bucket) and the latest test_result/tail_compare_*/summary.json.
Writes compare_all.png into the tail_compare folder. Dots = seeds, marker = mean, bars = std.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4c78a8", "#f58518", "#54a24b"]

old_dir = sorted(glob.glob("test_result/seeds10_*"))[-1]
new_dir = sorted(glob.glob("test_result/tail_compare_*"))[-1]
old = json.load(open(os.path.join(old_dir, "summary.json")))
new = json.load(open(os.path.join(new_dir, "summary.json")))
episodes = ["20000", "40000", "60000"]

# set name -> {episodes -> per-seed list}
data = {name: {e: old["by_episodes"][e][name]["per_seed"] for e in episodes} for name in old["sets"]}
data["tail"] = {e: new["results"]["danger,apple_dir,heading,tail"][e]["per_seed"] for e in episodes}
labels = {"default": "default (288)", "length_bucket": "length_bucket (1152)", "tail": "tail (7776)"}
for k in data:
    assert all(len(v) == 10 for v in data[k].values()), k

fig, ax = plt.subplots(figsize=(10, 5.5))
rng = np.random.default_rng(0)
offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * 0.25
for i, (name, per_e) in enumerate(data.items()):
    for x, e in enumerate(episodes):
        v = np.array(per_e[e])
        pos = x + offsets[i]
        ax.scatter(pos + rng.uniform(-0.05, 0.05, len(v)), v, s=14, alpha=0.45, color=COLORS[i])
        ax.errorbar(pos, v.mean(), yerr=v.std(), fmt="o", color=COLORS[i], capsize=4, ms=7,
                    label=labels[name] if x == 0 else None)
ax.axhline(61, color="gray", ls="--", lw=0.8)
ax.text(2.45, 61.7, "full board (61)", ha="right", fontsize=8, color="gray")
ax.set_xticks(range(3))
ax.set_xticklabels([f"{int(e) // 1000}k" for e in episodes])
ax.set_xlabel("training episodes")
ax.set_ylabel("mean score over 100 greedy games")
ax.set_title("10 seeds: dots = seeds, marker = mean, bars = std")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = os.path.join(new_dir, "compare_all.png")
fig.savefig(out, dpi=130)
print("saved", out)
for name, per_e in data.items():
    print(f"{name:14s}", "  ".join(f"{e}: {np.mean(per_e[e]):5.2f}+-{np.std(per_e[e]):4.2f}" for e in episodes))
