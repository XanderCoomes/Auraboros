"""Train a tabular Q-learning agent on Snake.

Usage:  python train.py --size 8 --episodes 20000 [--features danger,apple_dir,heading,length_bucket]
Outputs: log.csv (per-episode stats), q_table.pkl, and q_table.json (training config for plot.py)
"""
import argparse
import csv
import json
from collections import deque
from datetime import datetime
from time import monotonic

import numpy as np

import feature_library as feat
import snake_env
from feature_library import make_feature_set
from feature_library.length_bucket import BUCKET_THRESHOLDS
from q_tabular import QAgent
from pygame_viewer import PygameViewer, ViewerClosed
from rollout import evaluate, run_episode
from snake_env import SnakeEnv


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
    p.add_argument("--visual", action="store_true", help="watch training and evaluation games in Pygame")
    p.add_argument("--visual-eval", action="store_true", help="watch only evaluation; train at full speed")
    p.add_argument("--visual-every", type=int, default=1,
                   help="with --visual, display every Nth training episode (default: all)")
    p.add_argument("--fps", type=int, default=30, help="initial visual steps per second (1..240)")
    p.add_argument("--eval-every", type=int, default=2000,
                   help="evaluate every N training episodes; 0 disables periodic checks")
    p.add_argument("--eval-games", type=int, default=100, help="games per periodic/final evaluation")
    p.add_argument("--eval-seed", type=int, default=1000, help="independent evaluation environment seed")
    args = p.parse_args()
    if args.size < 4:
        p.error("--size must be at least 4 for the initial snake")
    if args.episodes < 1 or args.visual_every < 1 or args.eval_games < 1 or args.eval_every < 0:
        p.error("episodes, visual-every and eval-games must be positive; eval-every must be nonnegative")
    if not 1 <= args.fps <= 240:
        p.error("--fps must be between 1 and 240")

    fs = make_feature_set(args.features)
    print(f"features: {fs} ({fs.n_states} states)")
    env = SnakeEnv(size=args.size, seed=args.seed, features=fs)
    eval_env = SnakeEnv(size=args.size, seed=args.eval_seed, features=fs)
    agent = QAgent(fs.shape, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    decay_eps = max(1, int(args.episodes * args.eps_decay_frac))
    viewer = None
    completed, final, interrupted = 0, None, False
    recent_scores, best_score = deque(maxlen=100), 0
    last_return = 0.0
    last_preview = 0.0
    try:
        if args.visual or args.visual_eval:
            viewer = PygameViewer(args.size, args.fps)
    except RuntimeError as exc:
        p.exit(1, f"error: {exc}\n")
    try:
        with open(args.log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "epsilon", "score", "steps", "return", "max_length", "final_bucket"])
            for ep in range(args.episodes):
                # Explore early, then gradually favor learned actions.
                frac = min(1.0, ep / decay_eps)
                eps = args.eps_start + frac * (args.eps_end - args.eps_start)
                training_view = viewer if args.visual and ep % args.visual_every == 0 else None
                summary = ""
                if viewer:
                    viewer.poll()
                    summary = (f"Last 100 mean {np.mean(recent_scores):.2f} | Best {best_score}"
                               if recent_scores else "Learning enabled | First episode")
                    if training_view is None and (viewer.paused or monotonic() - last_preview >= 0.25):
                        viewer.show(env, f"Training | {completed}/{args.episodes} completed", eps, last_return,
                                    summary=summary + " | Waiting for next visual game", throttle=False)
                        last_preview = monotonic()
                score, steps, ret, info = run_episode(
                    env, agent, eps, viewer=training_view,
                    title=f"Training | Episode {ep + 1}/{args.episodes}", summary=summary)
                recent_scores.append(score)
                best_score = max(best_score, score)
                last_return = ret
                completed += 1
                w.writerow([ep, f"{eps:.4f}", score, steps, f"{ret:.3f}", info["length"], info["bucket"]])
                if args.eval_every and completed % args.eval_every == 0:
                    result = evaluate(eval_env, agent, args.eval_games, viewer,
                                      title=f"Evaluation after episode {completed}")
                    print(f"episode {completed}: eps={eps:.3f} greedy eval score={result:.2f}")
            final = evaluate(eval_env, agent, args.eval_games, viewer, title="Final evaluation")
    except (ViewerClosed, KeyboardInterrupt):
        interrupted = True
        print(f"Stopped after {completed} completed episodes; saving learned progress.")
    finally:
        if viewer:
            viewer.close()

    agent.save(args.out)
    print(f"saved {args.out}" + (f", final greedy score over {args.eval_games} games: {final:.2f}"
                               if final is not None else ""))

    config = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "q_file": args.out,
        "log_file": args.log,
        "board_size": args.size,
        "episodes": completed,
        "requested_episodes": args.episodes,
        "interrupted": interrupted,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": {"start": args.eps_start, "end": args.eps_end,
                    "linear_decay_fraction": args.eps_decay_frac},
        "seed": args.seed,
        "features": fs.names,
        "feature_details": fs.describe(),
        "n_states": fs.n_states,
        "bucket_thresholds_fraction_of_board": BUCKET_THRESHOLDS,
        "evaluation": {"games": args.eval_games, "seed": args.eval_seed, "mean_score": final},
        "rewards": {"apple": snake_env.R_APPLE, "death": snake_env.R_DEATH,
                    "step": snake_env.R_STEP, "closer_or_farther": snake_env.R_CLOSER},
    }
    if args.eval_games == 100:
        config["final_greedy_eval_score_100_games"] = final
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
