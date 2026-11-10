"""Compare feature sets (default vs space) under selected reward settings. Same metrics as sweep_rewards.py.

Usage: python compare_space.py --episodes 20000 --seeds 10
"""
import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from sweep_rewards import BASE, run_one

FEATURE_SETS = {"default (288)": "danger,apple_dir,heading", "space (972)": "space,apple_dir,heading"}
REWARDS = {"baseline": dict(BASE), "r_death=-10": {**BASE, "r_death": -10.0}, "r_death=-100": {**BASE, "r_death": -100.0}}
COLORS = {"default (288)": "#4c78a8", "space (972)": "#f58518"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    jobs, keys = [], []
    for rn, rw in REWARDS.items():
        for fname, feats in FEATURE_SETS.items():
            for s in range(a.seeds):
                jobs.append((f"{rn}|{fname}", rw, s, a.size, a.episodes, feats, a.max_steps, 100))
    with Pool(a.workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)

    d = f"test_result/space_compare_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(d)
    get = lambda key, i: np.array([r[i] for r in res if r[0] == key])
    summary = {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    rng = np.random.default_rng(0)
    for ax, i, yl in [(axes[0], 2, "apples / 100 steps"), (axes[1], 3, "mean apples over 100 greedy games")]:
        for x, rn in enumerate(REWARDS):
            for j, fname in enumerate(FEATURE_SETS):
                v = get(f"{rn}|{fname}", i)
                pos = x + (j - 0.5) * 0.3
                ax.scatter(pos + rng.uniform(-0.05, 0.05, len(v)), v, s=14, alpha=0.4, color=COLORS[fname])
                ax.errorbar(pos, v.mean(), yerr=v.std(), fmt="o", color=COLORS[fname], capsize=4, ms=7,
                            label=fname if x == 0 else None)
                summary.setdefault(f"{rn}|{fname}", {})[["", "", "apples_per_100_steps", "mean_apples"][i]] = \
                    {"mean": float(v.mean()), "std": float(v.std()), "per_seed": v.tolist()}
        ax.set_xticks(range(len(REWARDS)))
        ax.set_xticklabels(list(REWARDS))
        ax.set_ylabel(yl)
        ax.set_title(f"{a.seeds} seeds: dots = seeds, marker = mean, bars = std")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{d}/compare.png", dpi=130)
    json.dump({"args": vars(a), "results": summary}, open(f"{d}/summary.json", "w"), indent=2)
    print("saved", d)
    for k, v in summary.items():
        print(f"{k:28s} apples/100steps {v['apples_per_100_steps']['mean']:6.2f} +- {v['apples_per_100_steps']['std']:5.2f}"
              f" | mean apples {v['mean_apples']['mean']:5.2f} +- {v['mean_apples']['std']:5.2f}")


if __name__ == "__main__":
    main()
