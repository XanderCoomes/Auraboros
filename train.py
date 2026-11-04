"""Train a tabular Q-learning agent on Snake.

Usage:  python train.py --size 8 --episodes 20000 [--features danger_straight,danger_left,...]
        python train.py --list-features
Outputs: log.csv (per-episode stats), q_table.pkl, and q_table.json (training config for plot.py)
"""
import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime

import numpy as np

import features
import rewards
from q_tabular import QAgent
from snake_env import SnakeEnv


def run_episode(env, agent, epsilon, learn=True, visited=None):
    # One game = repeat: observe state -> choose action -> get reward and next state -> learn
    # visited: optional set; every Q-table row seen this episode is added to it (state coverage)
    stateBeforeAction = env.reset()
    done, totalReward, info = False, 0.0, {"score": 0}
    while not done:
        if visited is not None:
            visited.add(agent.spec.row(stateBeforeAction))
        selectedAction = agent.act(stateBeforeAction, epsilon)  # choose an action for the current state (epsilon-greedy)
        stateAfterAction, reward, done, info = env.step(selectedAction)  # environment returns next state, reward, game-over flag
        if learn:
            agent.update(stateBeforeAction, selectedAction, reward, stateAfterAction, done)  # adjust the Q-table with this experience
        stateBeforeAction, totalReward = stateAfterAction, totalReward + reward  # move on to the next state
    return info["score"], env.steps, totalReward, info.get("reason", "unknown"), len(env.snake)


def evaluate(env, agent, n=100):
    # Pure skill check: epsilon = 0 (no exploration) and no learning; returns mean score over n games
    return float(np.mean([run_episode(env, agent, 0.0, learn=False)[0] for _ in range(n)]))


def evaluate_details(env, agent, n=100):
    # Like evaluate(), but also reports max score and why the greedy games ended
    games = [run_episode(env, agent, 0.0, learn=False) for _ in range(n)]
    scores = [g[0] for g in games]
    return {"games": n, "mean_score": float(np.mean(scores)), "max_score": int(max(scores)),
            "mean_steps": float(np.mean([g[1] for g in games])),
            "end_reasons": dict(Counter(g[3] for g in games))}


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
    p.add_argument("--log", default="log.csv")
    p.add_argument("--out", default="q_table.pkl")
    p.add_argument("--features", default=",".join(features.DEFAULT_FEATURES),
                   help="comma-separated state features from features.py (order = state tuple order), or 'all'")
    p.add_argument("--list-features", action="store_true", help="print the available features and exit")
    p.add_argument("--rewards", default=",".join(rewards.DEFAULT_REWARDS),
                   help="comma-separated reward terms from rewards.py, each optionally =weight "
                        "(e.g. apple=100,death=-1000,step,closer), or 'all'")
    p.add_argument("--list-rewards", action="store_true", help="print the available reward terms and exit")
    args = p.parse_args()

    if args.list_features:
        for name, f in features.list_features().items():
            print(f"{name:20s} {f['n_values']} values  {f['labels'] or ''}  {f['doc']}")
        return
    if args.list_rewards:
        for name, t in rewards.list_rewards().items():
            print(f"{name:20s} default weight {t['default_weight']:+g}  {t['doc']}")
        return

    names = list(features.FEATURES) if args.features == "all" else args.features.split(",")
    env = SnakeEnv(size=args.size, seed=args.seed, features=names, rewards=rewards.RewardSpec.parse(args.rewards))
    agent = QAgent(env.spec, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    print(f"state features {env.spec.names} -> shape {env.spec.shape}, {env.spec.num_states} states")
    print(f"reward terms: {env.rewards}")
    decay_eps = max(1, int(args.episodes * args.eps_decay_frac))
    started_at, t0 = datetime.now(), time.time()
    visited, end_reasons, eval_history, scores = set(), Counter(), [], []

    with open(args.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "epsilon", "score", "steps", "return", "length", "end_reason", "states_visited"])
        for ep in range(args.episodes):
            # Epsilon (exploration rate) decays linearly from eps_start to eps_end, then stays constant:
            # explore a lot early on, mostly exploit what was learned later
            frac = min(1.0, ep / decay_eps)
            eps = args.eps_start + frac * (args.eps_end - args.eps_start)
            score, steps, ret, reason, length = run_episode(env, agent, eps, visited=visited)
            end_reasons[reason] += 1
            scores.append(score)
            w.writerow([ep, f"{eps:.4f}", score, steps, f"{ret:.3f}", length, reason, len(visited)])
            if (ep + 1) % 2000 == 0:
                ev = evaluate(env, agent)
                eval_history.append({"episode": ep + 1, "epsilon": round(eps, 4), "greedy_mean_score": ev,
                                     "states_visited": len(visited)})
                print(f"episode {ep + 1}: eps={eps:.3f} greedy eval score={ev:.2f} "
                      f"states visited {len(visited)}/{env.spec.num_states}")

    duration = time.time() - t0
    agent.save(args.out)
    final_eval = evaluate_details(env, agent)
    final = final_eval["mean_score"]
    print(f"saved {args.out}, final greedy score over 100 games: {final:.2f} ({duration:.0f}s)")

    config = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_started_at": started_at.isoformat(timespec="seconds"),
        "training_duration_seconds": round(duration, 1),
        "command": "python train.py " + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items()
                                                  if k not in ("list_features", "list_rewards")),
        "q_file": args.out,
        "log_file": args.log,
        "board_size": args.size,
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": {"start": args.eps_start, "end": args.eps_end,
                    "linear_decay_fraction": args.eps_decay_frac},
        "seed": args.seed,
        "state": {"features": env.spec.names, "shape": list(env.spec.shape), "num_states": env.spec.num_states,
                  "feature_details": {f.name: {"n_values": f.n_values, "labels": f.labels, "doc": f.doc}
                                      for f in env.spec.features}},
        "final_greedy_eval_score_100_games": final,
        "final_greedy_eval": final_eval,
        "eval_history": eval_history,
        # How much of the Q-table training ever touched; unvisited rows are still all zeros
        "state_coverage": {"visited": len(visited), "total": env.spec.num_states,
                           "fraction": round(len(visited) / env.spec.num_states, 4)},
        "training_end_reasons": dict(end_reasons),
        "training_last_1000_episodes": {"mean_score": float(np.mean(scores[-1000:])),
                                        "max_score": int(max(scores[-1000:]))},
        "training_max_score": int(max(scores)),
        "rewards": env.rewards.weights,  # term -> weight; reward = sum(weight * term(env, event))
        "reward_details": {n: rewards.REWARDS[n].doc for n in env.rewards.weights},
    }
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
