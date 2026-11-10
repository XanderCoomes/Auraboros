"""Evaluate a trained Q-table: greedy play, fixed seeds, mean apples over N trials (max_steps cap applies).

Usage: python evaluate_100.py --q q_table.pkl [--size 8] [--trials 100] [--max-steps 2000]
Features are read from the matching .json written by train.py.
"""
import argparse
import json

import numpy as np

from feature_library import make_feature_set
from q_tabular import QAgent
from snake_env import SnakeEnv
from train import run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", default="q_table.pkl")
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()
    cfg = json.load(open(a.q.rsplit(".", 1)[0] + ".json"))
    fs = make_feature_set(",".join(cfg["features"]))
    agent = QAgent(fs.shape)
    agent.load(a.q)
    scores, reasons = [], {}
    for t in range(a.trials):
        env = SnakeEnv(size=a.size, seed=a.seed + t, features=fs, max_steps=a.max_steps)  # fixed seed per trial
        score, _, _, info = run_episode(env, agent, 0.0, learn=False)
        scores.append(score)
        r = info.get("reason", "?")
        reasons[r] = reasons.get(r, 0) + 1
    print(f"{a.trials} trials: mean apples {np.mean(scores):.2f} (std {np.std(scores):.2f}, "
          f"min {min(scores)}, max {max(scores)}); end reasons {reasons}")


if __name__ == "__main__":
    main()
