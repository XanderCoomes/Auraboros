"""Sweep reward values (apple / step / closer / death), train Q-learning for each, and evaluate over 100 greedy trials.

Metric: apples per 100 steps = 100 * (total apples) / (total steps) over the 100 eval games (max_steps cap applies).
Usage: python sweep_rewards.py --size 8 --episodes 20000 --seeds 10
Outputs in test_result/reward_sweep_<timestamp>/: results.csv, summary.json, sweep.png
"""
import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from feature_library import make_feature_set
from q_tabular import QAgent
from snake_env import SnakeEnv
from train import run_episode

BASE = dict(r_apple=100.0, r_step=-0.01, r_closer=0.1, r_death=-6350.0)
# One-at-a-time variations around the baseline (the baseline itself is included once)
VARIANTS = {
    "r_apple": [10.0, 100.0, 1000.0],
    "r_step": [0.0, -0.01, -0.1, -1.0],
    "r_closer": [0.0, 0.1, 1.0, 5.0],
    "r_death": [-10.0, -100.0, -1000.0, -6350.0],
}


def configs():
    out = {"baseline": dict(BASE)}
    for k, vals in VARIANTS.items():
        for v in vals:
            if v != BASE[k]:
                out[f"{k}={v:g}"] = {**BASE, k: v}
    return out


def run_one(job):
    name, rw, seed, size, episodes, features, max_steps, n_eval = job
    fs = make_feature_set(features)
    env = SnakeEnv(size=size, seed=seed, features=fs, max_steps=max_steps, **rw)
    agent = QAgent(fs.shape, alpha=0.1, gamma=0.95, seed=seed)
    decay = max(1, int(episodes * 0.6))
    for ep in range(episodes):
        eps = 1.0 + min(1.0, ep / decay) * (0.01 - 1.0)
        run_episode(env, agent, eps)
    apples = steps = 0
    per_game = []
    for t in range(n_eval):  # fixed eval seeds, shared by all configs
        e = SnakeEnv(size=size, seed=100000 + t, features=fs, max_steps=max_steps, **rw)
        sc, st, _, _ = run_episode(e, agent, 0.0, learn=False)
        apples += sc
        steps += st
        per_game.append(sc)
    return name, seed, 100.0 * apples / max(steps, 1), float(np.mean(per_game))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--features", default="danger,apple_dir,heading")
    p.add_argument("--eval-trials", type=int, default=100)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    cfgs = configs()
    jobs = [(n, rw, s, a.size, a.episodes, a.features, a.max_steps, a.eval_trials)
            for n, rw in cfgs.items() for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)

    d = f"test_result/reward_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(d)
    names = list(cfgs)
    rate = {n: np.array([r[2] for r in res if r[0] == n]) for n in names}
    mean = {n: np.array([r[3] for r in res if r[0] == n]) for n in names}
    with open(f"{d}/results.csv", "w") as f:
        f.write("config,seed,apples_per_100_steps,mean_apples\n")
        for r in res:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f}\n")
    summ = {n: {"rewards": cfgs[n], "apples_per_100_steps_mean": float(rate[n].mean()),
                "apples_per_100_steps_std": float(rate[n].std()),
                "mean_apples_mean": float(mean[n].mean()), "mean_apples_std": float(mean[n].std())}
            for n in names}
    json.dump({"args": vars(a), "results": summ}, open(f"{d}/summary.json", "w"), indent=2)

    group_color = {"baseline": "#555555", "r_apple": "#4c78a8", "r_step": "#f58518",
                   "r_closer": "#54a24b", "r_death": "#e45756"}
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(names) * 0.8), 9))
    rng = np.random.default_rng(0)
    for ax, data, title, yl in [(axes[0], rate, "apples per 100 steps (higher = better)", "apples / 100 steps"),
                                (axes[1], mean, "mean apples per game", "mean apples over 100 greedy games")]:
        for i, n in enumerate(names):
            c = group_color[n.split("=")[0] if n != "baseline" else "baseline"]
            v = data[n]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, alpha=0.4, color=c)
            ax.errorbar(i, v.mean(), yerr=v.std(), fmt="o", color=c, capsize=4, ms=7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(yl)
        ax.set_title(f"{a.seeds} seeds: dots = seeds, marker = mean, bars = std - {title}")
    plt.tight_layout()
    plt.savefig(f"{d}/sweep.png", dpi=130)
    print("saved", d)
    for n in sorted(names, key=lambda n: -rate[n].mean()):
        print(f"{n:18s} {rate[n].mean():6.2f} +- {rate[n].std():5.2f} apples/100 steps | mean apples {mean[n].mean():5.2f}")


if __name__ == "__main__":
    main()
