"""Train a tabular Q-learning agent on Snake.

Usage:  python train.py --size 8 --episodes 20000 [--features danger,apple_dir,heading,length_bucket]
Outputs: log.csv (per-episode stats), q_table.pkl, and q_table.json (training config for plot.py)
"""
import argparse
import csv
import json
from datetime import datetime

import numpy as np

import feature_library as feat
import snake_env
from feature_library import make_feature_set
from feature_library.length_bucket import BUCKET_THRESHOLDS
from q_tabular import QAgent
from snake_env import SnakeEnv


def run_episode(env, agent, epsilon, learn=True):
    # One game = repeat: observe state -> choose action -> get reward and next state -> learn
    stateBeforeAction = env.reset()
    done, totalReward, info = False, 0.0, {"score": 0}
    while not done:
        selectedAction = agent.act(stateBeforeAction, epsilon)  # choose an action for the current state (epsilon-greedy)
        stateAfterAction, reward, done, info = env.step(selectedAction)  # environment returns next state, reward, game-over flag
        if learn:
            agent.update(stateBeforeAction, selectedAction, reward, stateAfterAction, done)  # adjust the Q-table with this experience
        stateBeforeAction, totalReward = stateAfterAction, totalReward + reward  # move on to the next state
    return info["score"], env.steps, totalReward, info


def evaluate(env, agent, n=100):
    # Pure skill check: epsilon = 0 (no exploration) and no learning; returns mean score over n games
    return float(np.mean([run_episode(env, agent, 0.0, learn=False)[0] for _ in range(n)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-frac", type=float, default=0.6,
                   help="fraction of episodes over which epsilon decays linearly")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--features", default=",".join(feat.DEFAULT_FEATURES),
                   help=f"comma-separated state features, available: {','.join(feat.REGISTRY)}")
    p.add_argument("--log", default="log.csv")
    p.add_argument("--out", default="q_table.pkl")
    args = p.parse_args()

    fs = make_feature_set(args.features)
    print(f"features: {fs} ({fs.n_states} states)")
    env = SnakeEnv(size=args.size, seed=args.seed, features=fs)
    agent = QAgent(fs.shape, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    decay_eps = max(1, int(args.episodes * args.eps_decay_frac))

    with open(args.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "epsilon", "score", "steps", "return", "max_length", "final_bucket"])
        for ep in range(args.episodes):
            # Epsilon (exploration rate) decays linearly from eps_start to eps_end, then stays constant:
            # explore a lot early on, mostly exploit what was learned later
            frac = min(1.0, ep / decay_eps)
            eps = args.eps_start + frac * (args.eps_end - args.eps_start)
            score, steps, ret, info = run_episode(env, agent, eps)
            w.writerow([ep, f"{eps:.4f}", score, steps, f"{ret:.3f}", info["length"], info["bucket"]])
            if (ep + 1) % 2000 == 0:
                print(f"episode {ep + 1}: eps={eps:.3f} greedy eval score={evaluate(env, agent):.2f}")

    agent.save(args.out)
    final = evaluate(env, agent)
    print(f"saved {args.out}, final greedy score over 100 games: {final:.2f}")

    config = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "q_file": args.out,
        "log_file": args.log,
        "board_size": args.size,
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": {"start": args.eps_start, "end": args.eps_end,
                    "linear_decay_fraction": args.eps_decay_frac},
        "seed": args.seed,
        "features": fs.names,
        "feature_details": fs.describe(),
        "n_states": fs.n_states,
        "bucket_thresholds_fraction_of_board": BUCKET_THRESHOLDS,
        "final_greedy_eval_score_100_games": final,
        "rewards": {"apple": snake_env.R_APPLE, "death": snake_env.R_DEATH,
                    "step": snake_env.R_STEP, "closer_or_farther": snake_env.R_CLOSER},
    }
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
