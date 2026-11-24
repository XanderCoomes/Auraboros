"""Plot learning curves from a training log.

Usage:  python plot.py --log runs/<run>/log.csv [--window 200] [--outdir test_result]
Outputs in <outdir>/train_<trained>_test_<tested>/: curves.png and a copy of the log.
The plot title comes from the run's model.json when it sits beside the log.
"""
import argparse
import csv
import json
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def load_train_cfg(log_path):
    """model.json written by train_dqn.py next to the given log, or None."""
    path = os.path.join(os.path.dirname(log_path), "model.json")
    return json.load(open(path)) if os.path.exists(path) else None


def train_header(cfg):
    """One-line description of the training run for plot titles."""
    if not cfg:
        return "trained: unknown (no model.json beside the log)"
    when = cfg["trained_at"].replace("T", " ")[:19]
    greedy = cfg.get("final_greedy_eval_score_100_games")
    return (f"trained {when}  |  {cfg['completed_episodes']} eps, {cfg['board_size']}x{cfg['board_size']}, "
            f"{cfg['model']['name']} + {cfg['encoder']['name']}  |  greedy "
            + ("unknown" if greedy is None else f"{greedy:.2f}"))


def moving_avg(x, k):
    k = min(k, len(x))
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_curves(path, window, out, cfg=None):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    score = np.array([float(r["score"]) for r in rows])
    steps = np.array([float(r["steps"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(moving_avg(score, window)); ax[0].set_title(f"Score (moving avg {window})")
    ax[1].plot(moving_avg(steps, window)); ax[1].set_title(f"Episode steps (moving avg {window})")
    for a in ax:
        a.set_xlabel("episode")
    channels = ", ".join(cfg["state"]["features"]) if cfg and "state" in cfg else "unknown"
    fig.suptitle(f"{train_header(cfg)}\nboard channels: {channels}", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--outdir", default="test_result")
    a = p.parse_args()
    cfg = load_train_cfg(a.log)
    # One folder per plot, named by when the model was TRAINED (then when it was plotted), e.g.
    # test_result/train_20261102_202627_test_20261102_211333/ (sorts chronologically by training)
    trained = (datetime.fromisoformat(cfg["trained_at"]).strftime("%Y%m%d_%H%M%S") if cfg else "unknown")
    run_dir = os.path.join(a.outdir, f"train_{trained}_test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    plot_curves(a.log, a.window, os.path.join(run_dir, "curves.png"), cfg)
    shutil.copy2(a.log, run_dir)  # keep the log with the curves drawn from it
    print("copied training log to", run_dir)
