"""Train the DQN v2 agent with N environments stepped in lockstep.

Why N environments?
    One forward pass picks the actions of all N envs, so GPU overhead per env step drops ~N times, and
    consecutive transitions in the replay buffer come from different games (less correlated data).

Outputs in --out (default dqn/v2/runs/seed<seed>):
    config.json   all hyperparameters
    train.csv     one row per finished training episode
    eval.csv      one row per periodic greedy evaluation (fixed seeds, same games every time)
    best.pth      checkpoint with the best evaluation (win rate, then mean score, then fewer steps to win)
    last.pth      final checkpoint

Usage (from the repo root):
    python -m dqn.v2.train --episodes 20000 --seed 0
    python -m dqn.v2.train --episodes 20000 --seed 0 --shield --out dqn/v2/runs/shield_seed0
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from snake_env import RewardConfig, SnakeEnv

from .agent import DQNAgentV2
from .baselines import DQNv2Policy
from .evaluate import run_games, summarize
from .model import DuelingHybridDQN
from .replay import NStepAccumulator, PrioritizedReplay
from .shield import action_mask
from .state import N_FEATURES, N_SCALARS, board_shape, encode_state

TRAIN_FIELDS = ("episode", "env_steps", "updates", "epsilon", "beta", "lr", "score", "length", "steps",
                "reward", "reason", "loss")
EVAL_FIELDS = ("episode", "env_steps", "updates", "games", "wins", "win_rate", "win_lo", "win_hi",
               "mean_score", "max_score", "steps_per_apple", "steps_to_win", "seconds")


def linear(start, end, progress):
    return start + min(1.0, max(0.0, progress)) * (end - start)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=20000)
    # On this GPU one gradient step costs ~7 ms for any batch from 32 to 1024 (kernel-launch bound), so the
    # defaults use few, large updates: 2 x 512 samples per 32 env steps = 32 samples per env step.
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    # RL
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=5e-5)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--buffer-size", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=5_000, help="transitions before learning starts")
    ap.add_argument("--updates-per-iter", type=int, default=2,
                    help="gradient steps per lockstep iteration (= per n-envs env steps)")
    ap.add_argument("--alpha", type=float, default=0.6, help="prioritization exponent")
    ap.add_argument("--beta-start", type=float, default=0.4, help="importance-sampling exponent, annealed to 1")
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--eps-decay-frac", type=float, default=0.4)
    # model / env
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--feat-hidden", type=int, default=64)
    ap.add_argument("--shield", action="store_true", help="also mask moves that lose the escape route")
    ap.add_argument("--legacy-timeout", action="store_true", help="fixed 2*area timeout instead of length-scaled")
    # evaluation / io
    ap.add_argument("--eval-interval", type=int, default=1000, help="episodes between evaluations")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out or f"dqn/v2/runs/seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "out": str(out), "device": str(device)}
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    rewards = RewardConfig.v2(args.gamma)
    envs = [SnakeEnv(size=args.size, seed=args.seed * 1000 + i, rewards=rewards,
                     length_timeout=not args.legacy_timeout) for i in range(args.n_envs)]
    model = DuelingHybridDQN(args.size, hidden=args.hidden, feat_hidden=args.feat_hidden)
    agent = DQNAgentV2(model, device, lr=args.lr, lr_min=args.lr_min, tau=args.tau,
                       grad_clip=args.grad_clip, seed=args.seed)
    replay = PrioritizedReplay(args.buffer_size, board_shape(args.size), N_SCALARS, (3, N_FEATURES),
                               alpha=args.alpha, seed=args.seed)
    accs = [NStepAccumulator(args.n_step, args.gamma) for _ in envs]

    states = [encode_state(e) for e in envs]
    masks = [action_mask(s[2], args.shield) for s in states]
    ep_reward = [0.0] * len(envs)

    episodes = env_steps = 0
    loss = float("nan")
    best_key = None
    next_eval = args.eval_interval
    t_start = time.time()

    with open(out / "train.csv", "w", newline="") as ftrain, open(out / "eval.csv", "w", newline="") as feval:
        train_log = csv.DictWriter(ftrain, fieldnames=TRAIN_FIELDS)
        eval_log = csv.DictWriter(feval, fieldnames=EVAL_FIELDS)
        train_log.writeheader()
        eval_log.writeheader()

        while episodes < args.episodes:
            progress = episodes / args.episodes
            epsilon = linear(args.eps_start, args.eps_end, progress / args.eps_decay_frac)
            beta = linear(args.beta_start, 1.0, progress)
            lr = agent.set_lr_progress(progress)

            boards, scalars, feats = (np.stack(x) for x in zip(*states))
            actions, _ = agent.act(boards, scalars, feats, np.stack(masks), epsilon)

            for i, env in enumerate(envs):
                _, r, done, info = env.step(int(actions[i]))
                ep_reward[i] += r
                next_state = encode_state(env)
                next_mask = action_mask(next_state[2], args.shield)
                for transition in accs[i].push(states[i], int(actions[i]), r, next_state, done, next_mask):
                    replay.push(*transition)
                if done:
                    episodes += 1
                    train_log.writerow(dict(
                        episode=episodes, env_steps=env_steps, updates=agent.updates, epsilon=round(epsilon, 4),
                        beta=round(beta, 4), lr=f"{lr:.2e}", score=info["score"], length=info["length"],
                        steps=env.steps, reward=round(ep_reward[i], 3), reason=info["reason"],
                        loss=round(loss, 5)))
                    ep_reward[i] = 0.0
                    env.reset()
                    next_state = encode_state(env)
                    next_mask = action_mask(next_state[2], args.shield)
                states[i], masks[i] = next_state, next_mask
            env_steps += len(envs)

            if len(replay) >= max(args.warmup, args.batch_size):
                for _ in range(args.updates_per_iter):
                    idx, batch = replay.sample(args.batch_size, beta)
                    loss, td = agent.train_step(batch)
                    replay.update_priorities(idx, td)

            if episodes >= next_eval or episodes >= args.episodes:
                next_eval += args.eval_interval
                t0 = time.time()
                policy = DQNv2Policy(agent.model, device, shield=args.shield)
                rec = run_games(policy, args.eval_games, args.size, args.eval_seed)
                agent.model.train()
                s = summarize(rec)
                row = dict(episode=episodes, env_steps=env_steps, updates=agent.updates, games=s["games"],
                           wins=s["wins"], win_rate=s["win_rate"], win_lo=s["win_rate_ci"][0],
                           win_hi=s["win_rate_ci"][1], mean_score=s["mean_score"], max_score=s["max_score"],
                           steps_per_apple=s["steps_per_apple"], steps_to_win=s["steps_to_win"],
                           seconds=round(time.time() - t_start))
                eval_log.writerow(row)
                feval.flush()
                ftrain.flush()
                steps_to_win = s["steps_to_win"] if s["wins"] else float("inf")
                key = (s["win_rate"], s["mean_score"], -steps_to_win)
                if best_key is None or key > best_key:
                    best_key = key
                    agent.save(out / "best.pth", {**config, "episode": episodes})
                elapsed = time.time() - t_start
                print(f"ep {episodes:>6}/{args.episodes}  steps {env_steps:>9,}  eps {epsilon:.3f}  "
                      f"lr {lr:.1e}  loss {loss:.4f} | eval win {s['win_rate']:.0%} "
                      f"[{s['win_rate_ci'][0]:.0%},{s['win_rate_ci'][1]:.0%}]  score {s['mean_score']:.1f} "
                      f"max {s['max_score']}  steps/apple {s['steps_per_apple']:.1f}  "
                      f"({env_steps / elapsed:,.0f} steps/s, eval {time.time() - t0:.0f}s, {elapsed / 60:.1f} min)",
                      flush=True)

    agent.save(out / "last.pth", {**config, "episode": episodes})
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")
    return out


if __name__ == "__main__":
    train(parse_args())
