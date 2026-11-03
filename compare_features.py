"""Compare several feature sets (e.g. with / without length_bucket) with the same hyperparameters and seeds.

Usage:
  python compare_features.py --size 8 --episodes 20000 --seeds 5 \
      --sets "danger,apple_dir,heading" "danger,apple_dir,heading,length_bucket"

Outputs in test_result/compare_<timestamp>/:
  curves.png    training score curves (mean over seeds, band = min..max)
  buckets.png   per length-bucket statistics from greedy evaluation games
  summary.json  all numbers behind the plots

Bucket statistics come from greedy (epsilon = 0, no learning) games played after training, and are computed
for every feature set (even those without length_bucket), because buckets are derived from the snake length.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import feature_library as feat
from feature_library import make_feature_set
from feature_library.length_bucket import BUCKET_THRESHOLDS, LengthBucket, length_bucket
from plot import moving_avg
from q_tabular import QAgent
from snake_env import SnakeEnv
from train import run_episode

BUCKETS = [b.name for b in LengthBucket]
REASONS = ["collision", "timeout", "win"]
COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2"]


def train_one(size, fs, seed, args):
    env = SnakeEnv(size=size, seed=seed, features=fs)
    agent = QAgent(fs.shape, alpha=args.alpha, gamma=args.gamma, seed=seed)
    decay_eps = max(1, int(args.episodes * args.eps_decay_frac))
    scores = []
    for ep in range(args.episodes):
        eps = args.eps_start + min(1.0, ep / decay_eps) * (args.eps_end - args.eps_start)
        scores.append(run_episode(env, agent, eps)[0])
    return agent, np.array(scores)


def eval_buckets(agent, size, fs, n_games, seed=1000):
    """Greedy games; returns overall stats and per-bucket counts (steps spent, games reached, deaths)."""
    env = SnakeEnv(size=size, seed=seed, features=fs)
    steps_in = np.zeros(len(BUCKETS))
    reached = np.zeros(len(BUCKETS))
    deaths = np.zeros(len(BUCKETS))
    reasons = {r: 0 for r in REASONS}
    reasons_by_bucket = {b: {r: 0 for r in REASONS} for b in BUCKETS}
    scores = []
    for _ in range(n_games):
        s, done, seen = env.reset(), False, set()
        while not done:
            b = int(length_bucket(len(env.snake), size))  # bucket in which this step is taken
            steps_in[b] += 1
            seen.add(b)
            s, _, done, info = env.step(agent.act(s, 0.0))
        for b in seen:
            reached[b] += 1
        reason = info["reason"]
        reasons[reason] += 1
        if reason != "win":  # collision / timeout count as a death in the bucket where the game ended
            deaths[info["bucket"]] += 1
        reasons_by_bucket[BUCKETS[info["bucket"]]][reason] += 1
        scores.append(info["score"])
    return {"mean_score": float(np.mean(scores)), "max_score": int(max(scores)),
            "steps_in_bucket": steps_in.tolist(), "reached": reached.tolist(),
            "deaths": deaths.tolist(), "end_reasons": reasons,
            "end_reasons_by_bucket": reasons_by_bucket, "games": n_games}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..seeds-1)")
    p.add_argument("--sets", nargs="+", default=[",".join(feat.DEFAULT_FEATURES),
                                                  ",".join(feat.DEFAULT_FEATURES + ("length_bucket",))],
                   help="feature sets to compare, each a comma-separated string")
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-frac", type=float, default=0.6)
    p.add_argument("--outdir", default="test_result")
    args = p.parse_args()

    sets = [make_feature_set(t) for t in args.sets]
    run_dir = os.path.join(args.outdir, "compare_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    results = {}  # set name -> {"curves": [..per seed..], "evals": [..per seed..]}
    for fs in sets:
        curves, evals = [], []
        for seed in range(args.seeds):
            agent, scores = train_one(args.size, fs, seed, args)
            ev = eval_buckets(agent, args.size, fs, args.eval_games)
            curves.append(scores)
            evals.append(ev)
            print(f"[{fs}] seed {seed}: greedy mean score {ev['mean_score']:.2f}")
        results[str(fs)] = {"fs": fs, "curves": np.array(curves), "evals": evals}

    # ---------------------------------------------------------------- curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, r) in enumerate(results.items()):
        sm = np.array([moving_avg(c, args.window) for c in r["curves"]])
        x = np.arange(sm.shape[1])
        ax.plot(x, sm.mean(0), color=COLORS[i % len(COLORS)], label=f"{name} ({r['fs'].n_states} states)")
        ax.fill_between(x, sm.min(0), sm.max(0), color=COLORS[i % len(COLORS)], alpha=0.15)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"training score (moving avg {args.window})")
    ax.set_title(f"{args.size}x{args.size}, {args.seeds} seeds (line = mean, band = min..max)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "curves.png"), dpi=120)

    # ---------------------------------------------------------- bucket stats
    summary = {"created_at": datetime.now().isoformat(timespec="seconds"),
               "board_size": args.size, "episodes": args.episodes, "seeds": args.seeds,
               "eval_games_per_seed": args.eval_games,
               "hyperparameters": {"alpha": args.alpha, "gamma": args.gamma, "eps_start": args.eps_start,
                                   "eps_end": args.eps_end, "eps_decay_frac": args.eps_decay_frac},
               "bucket_thresholds_fraction_of_board": BUCKET_THRESHOLDS, "sets": {}}
    agg = {}
    for name, r in results.items():
        ev = r["evals"]
        steps = np.sum([e["steps_in_bucket"] for e in ev], axis=0)
        reached = np.sum([e["reached"] for e in ev], axis=0)
        deaths = np.sum([e["deaths"] for e in ev], axis=0)
        games = sum(e["games"] for e in ev)
        reasons = {k: sum(e["end_reasons"][k] for e in ev) for k in REASONS}
        agg[name] = {"steps": steps / games, "reached": reached / games,
                     "death_rate": np.divide(deaths, reached, out=np.zeros_like(deaths), where=reached > 0),
                     "reasons": {k: v / games for k, v in reasons.items()}}
        means = [e["mean_score"] for e in ev]
        summary["sets"][name] = {
            "n_states": r["fs"].n_states,
            "final_greedy_score_mean": float(np.mean(means)), "final_greedy_score_std": float(np.std(means)),
            "per_seed_greedy_score": means,
            "last_window_train_score_mean": float(np.mean([c[-args.window:].mean() for c in r["curves"]])),
            "per_bucket": {b: {"mean_steps_per_game": float(agg[name]["steps"][j]),
                               "fraction_of_games_reaching": float(agg[name]["reached"][j]),
                               "death_rate_given_reached": float(agg[name]["death_rate"][j]),
                               "end_reasons": {k: sum(e["end_reasons_by_bucket"][b][k] for e in ev) for k in REASONS}}
                           for j, b in enumerate(BUCKETS)},
            "end_reason_fraction": agg[name]["reasons"],
        }

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    w = 0.8 / len(agg)
    x = np.arange(len(BUCKETS))
    panels = [("reached", "fraction of games reaching bucket"),
              ("death_rate", "death rate given reached (collision/timeout)"),
              ("steps", "mean steps spent per game")]
    for k, (key, title) in enumerate(panels):
        for i, (name, a) in enumerate(agg.items()):
            ax[k].bar(x + i * w - 0.4 + w / 2, a[key], w, color=COLORS[i % len(COLORS)], label=name)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(BUCKETS)
        ax[k].set_title(title, fontsize=9)
    ax[0].legend(fontsize=6)
    for i, (name, a) in enumerate(agg.items()):
        bottom = 0.0
        for j, rs in enumerate(REASONS):
            ax[3].bar(i, a["reasons"][rs], bottom=bottom, color=COLORS[j], label=rs if i == 0 else None)
            bottom += a["reasons"][rs]
    ax[3].set_xticks(range(len(agg)))
    ax[3].set_xticklabels([str(i + 1) for i in range(len(agg))])
    ax[3].set_xlabel("feature set (order of --sets)")
    ax[3].set_title("how games end (fraction)", fontsize=9)
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "buckets.png"), dpi=120)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved to", run_dir)
    for name, s in summary["sets"].items():
        print(f"  {name} ({s['n_states']} states): greedy score {s['final_greedy_score_mean']:.2f} "
              f"+- {s['final_greedy_score_std']:.2f}")


if __name__ == "__main__":
    main()
