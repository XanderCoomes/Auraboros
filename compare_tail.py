"""Compare feature sets across several training lengths and seeds (parallel).

For every (feature set, episodes, seed): train from scratch, then play greedy games (epsilon = 0, no learning)
and record the mean score. Epsilon decays over a fixed fraction of the run, so each episode count is a separate run.

Usage:
  python compare_tail.py --size 8 --episodes 20000 40000 60000 --seeds 10 --eval-games 100 \
      --sets "danger,apple_dir,heading" "danger,apple_dir,heading,tail"
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

import feature_library as feat
from feature_library import make_feature_set
from q_tabular import QAgent
from snake_env import SnakeEnv
from train import run_episode


def job(spec):
    names, episodes, seed, args = spec
    fs = make_feature_set(names)
    env = SnakeEnv(size=args["size"], seed=seed, features=fs)
    agent = QAgent(fs.shape, alpha=args["alpha"], gamma=args["gamma"], seed=seed)
    decay = max(1, int(episodes * args["eps_decay_frac"]))
    for ep in range(episodes):
        eps = args["eps_start"] + min(1.0, ep / decay) * (args["eps_end"] - args["eps_start"])
        run_episode(env, agent, eps)
    qdir = args["q_dir"]
    os.makedirs(qdir, exist_ok=True)  # keep every trained Q-table so metrics can be recomputed without retraining
    agent.save(os.path.join(qdir, f"q_{names.replace(',', '+')}_ep{episodes}_seed{seed}.pkl"))
    scores, reasons, total_steps = [], {}, 0
    for t in range(args["eval_games"]):  # fixed eval seeds shared by all configs (same scheme as sweep_rewards.run_one)
        ev_env = SnakeEnv(size=args["size"], seed=100000 + t, features=fs)
        sc, st, _, info = run_episode(ev_env, agent, 0.0, learn=False)
        scores.append(sc)
        total_steps += st
        reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
    rate = 100.0 * sum(scores) / max(total_steps, 1)  # apples per 100 steps over all eval games
    return names, episodes, seed, float(np.mean(scores)), int(max(scores)), reasons, fs.n_states, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, nargs="+", default=[20000, 40000, 60000])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--sets", nargs="+", default=[",".join(feat.DEFAULT_FEATURES),
                                                  ",".join(feat.DEFAULT_FEATURES + ("length_bucket",)),
                                                  ",".join(feat.DEFAULT_FEATURES + ("tail",))])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-frac", type=float, default=0.6)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--outdir", default="test_result")
    args = p.parse_args()
    run_dir = os.path.join(args.outdir, "tail_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    cfg = {k: getattr(args, k) for k in ("size", "alpha", "gamma", "eps_start", "eps_end", "eps_decay_frac", "eval_games")}
    cfg["q_dir"] = os.path.join(run_dir, "q_tables")

    specs = [(s, e, seed, cfg) for e in args.episodes for s in args.sets for seed in range(args.seeds)]
    # Longest jobs first so the pool stays busy
    specs.sort(key=lambda t: (-t[1], -len(t[0])))
    out = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for names, episodes, seed, mean, mx, reasons, n_states, rate in ex.map(job, specs):
            out.setdefault((names, episodes), []).append(
                {"seed": seed, "mean_score": mean, "max_score": mx, "end_reasons": reasons, "n_states": n_states,
                 "apples_per_100_steps": rate})
            print(f"[{names}] ep={episodes} seed={seed}: greedy mean {mean:.2f} apples/100steps {rate:.2f} max {mx} {reasons}", flush=True)

    summary = {"config": cfg, "seeds": args.seeds, "results": {}}
    print("\n=== mean greedy score over %d games (mean +- std over %d seeds) ===" % (args.eval_games, args.seeds))
    print(f"{'feature set':45s} {'states':>7s} " + " ".join(f"{e:>14d}" for e in args.episodes))
    for s in args.sets:
        row, summary["results"][s] = [], {}
        for e in args.episodes:
            runs = sorted(out[(s, e)], key=lambda r: r["seed"])
            m = [r["mean_score"] for r in runs]
            summary["results"][s][str(e)] = {"mean": float(np.mean(m)), "std": float(np.std(m)),
                                             "per_seed": m, "apples_per_100_steps_per_seed": [r["apples_per_100_steps"] for r in runs], "max_score": max(r["max_score"] for r in runs),
                                             "n_states": runs[0]["n_states"]}
            row.append(f"{np.mean(m):7.2f}+-{np.std(m):5.2f}")
        print(f"{s:45s} {runs[0]['n_states']:7d} " + " ".join(f"{c:>14s}" for c in row))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
