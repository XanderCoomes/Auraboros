"""Train a DQN agent on Snake (models, encoders and settings are described in DQN.md).

Usage:  python train.py --model rescnn8 --encoder entities_tail --size 8 --episodes 20000
        python train.py --list-rewards
Outputs: runs/<model>-<encoder>-<timestamp>/ with model.pt, model.json (config + eval history), log.csv
"""
import argparse
from collections import Counter
import numpy as np

import rewards


def run_episode(env, agent, epsilon, learn=True):
    # One game = repeat: observe state -> choose action -> get reward and next state -> learn
    stateBeforeAction = env.reset()
    done, totalReward, info = False, 0.0, {"score": 0}
    while not done:
        selectedAction = agent.act(stateBeforeAction, epsilon)  # choose an action for the current state (epsilon-greedy)
        stateAfterAction, reward, done, info = env.step(selectedAction)  # environment returns next state, reward, game-over flag
        if learn:
            agent.update(stateBeforeAction, selectedAction, reward, stateAfterAction, done)  # learn from this experience
        stateBeforeAction, totalReward = stateAfterAction, totalReward + reward  # move on to the next state
    return info["score"], env.steps, totalReward, info.get("reason", "unknown"), len(env.snake)


def evaluate_details(env, agent, n=100):
    # Pure skill check: epsilon = 0 (no exploration) and no learning, over n games;
    # reports mean and max score and why the greedy games ended
    games = [run_episode(env, agent, 0.0, learn=False) for _ in range(n)]
    scores = [g[0] for g in games]
    return {"games": n, "mean_score": float(np.mean(scores)), "max_score": int(max(scores)),
            "mean_steps": float(np.mean([g[1] for g in games])),
            "end_reasons": dict(Counter(g[3] for g in games))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-frac", type=float, default=0.6,
                   help="fraction of episodes over which epsilon decays linearly")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--rewards", default=",".join(rewards.DEFAULT_REWARDS),
                   help="comma-separated reward terms from rewards.py, each optionally =weight "
                        "(e.g. apple=100,death=-1000,step,closer), or 'all'")
    p.add_argument("--list-rewards", action="store_true", help="print the available reward terms and exit")
    from train_dqn import add_arguments, train as train_dqn
    add_arguments(p)
    args = p.parse_args()

    if args.list_rewards:
        for name, t in rewards.list_rewards().items():
            print(f"{name:20s} default weight {t['default_weight']:+g}  {t['doc']}")
        return
    train_dqn(args)


if __name__ == "__main__":
    main()
