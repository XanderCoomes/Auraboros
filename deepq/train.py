"""Train Snake DQN. Run python3 train.py from deepq/, or python -m deepq.train."""
import argparse
from collections import deque
import csv
from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
from time import monotonic

import numpy as np
import torch

if __package__ in (None, ""):
    import model as dqn
else:
    from . import model as dqn

from pygame_viewer import PygameViewer, ViewerClosed


########################
# Command-Line Options #
########################

def main(argv=None):
    # Editable defaults live together near the top of model.py.
    learning, rewards = dqn.DQNConfig(), dqn.RewardConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--size", type=int, default=dqn.BOARD_SIZE)
    p.add_argument("--episodes", type=int, default=dqn.TRAINING_EPISODES)
    p.add_argument("--seed", type=int, default=dqn.TRAINING_SEED)
    p.add_argument("--hidden-dim", type=int, default=dqn.HIDDEN_DIM_1)
    p.add_argument("--alpha", type=float, default=learning.alpha)
    p.add_argument("--gamma", type=float, default=learning.gamma)
    p.add_argument("--epsilon", "--epsilon-start", type=float, default=learning.epsilon_start)
    p.add_argument("--epsilon-end", type=float, default=learning.epsilon_end)
    p.add_argument("--epsilon-decay-frac", type=float, default=learning.epsilon_decay_frac)
    p.add_argument("--batch-size", type=int, default=learning.batch_size)
    p.add_argument("--replay-capacity", type=int, default=learning.replay_capacity)
    p.add_argument("--learning-starts", type=int, default=learning.learning_starts)
    p.add_argument("--train-every", type=int, default=learning.train_every)
    p.add_argument("--target-update-every", type=int, default=learning.target_update_every,
                   help="optimizer updates between target-network copies")
    p.add_argument("--gradient-clip", type=float, default=learning.gradient_clip)
    p.add_argument("--reward-scale", type=float, default=learning.reward_scale,
                   help="multiply raw rewards by this for learning; CSV returns stay unscaled")
    p.add_argument("--living-reward", type=float, default=rewards.living)
    p.add_argument("--apple-reward", type=float, default=rewards.apple)
    p.add_argument("--wall-death-reward", type=float, default=rewards.wall_death)
    p.add_argument("--body-death-reward", type=float, default=rewards.body_death)
    p.add_argument("--timeout-reward", type=float, default=rewards.timeout)
    p.add_argument("--distance-shaping", type=float, default=rewards.distance_shaping)
    p.add_argument("--timeout-factor", type=float, default=dqn.TIMEOUT_FACTOR)
    p.add_argument("--eval-every", type=int, default=dqn.EVAL_EVERY, help="0 disables periodic checks")
    p.add_argument("--eval-games", type=int, default=dqn.EVAL_GAMES)
    p.add_argument("--eval-seed", type=int, default=dqn.EVAL_SEED)
    p.add_argument("--print-every", type=int, default=dqn.PRINT_EVERY, help="episodes between progress messages")
    p.add_argument("--visual", action="store_true", help="animate training and evaluation")
    p.add_argument("--visual-eval", action="store_true", help="animate only evaluation games")
    p.add_argument("--visual-every", type=int, default=1, help="animate every Nth training episode")
    p.add_argument("--fps", type=int, default=dqn.TRAINING_FPS)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=dqn.DEVICE)
    p.add_argument("--threads", type=int, default=dqn.TORCH_THREADS)
    p.add_argument("--run-dir", type=Path, help="new output folder; default: deepq/runs/<timestamp>")
    args = p.parse_args(argv)

    if args.size < 4 or min(args.episodes, args.hidden_dim, args.threads, args.print_every) < 1:
        p.error("size must be at least 4; episodes, hidden-dim, threads and print-every must be positive")
    if min(args.seed, args.eval_seed, args.eval_every) < 0 or min(args.eval_games, args.visual_every) < 1:
        p.error("seeds/eval-every must be nonnegative; eval-games/visual-every must be positive")
    if not 1 <= args.fps <= 240 or not math.isfinite(args.timeout_factor) or args.timeout_factor <= 0:
        p.error("fps must be in 1..240 and timeout-factor must be finite and positive")
    try:
        learning = dqn.DQNConfig(
            alpha=args.alpha, gamma=args.gamma, epsilon_start=args.epsilon,
            epsilon_end=args.epsilon_end, epsilon_decay_frac=args.epsilon_decay_frac,
            batch_size=args.batch_size, replay_capacity=args.replay_capacity,
            learning_starts=args.learning_starts, train_every=args.train_every,
            target_update_every=args.target_update_every, gradient_clip=args.gradient_clip,
            reward_scale=args.reward_scale)
        rewards = dqn.RewardConfig(args.apple_reward, args.wall_death_reward, args.body_death_reward,
                                   args.timeout_reward, args.living_reward, args.distance_shaping)
    except ValueError as exc:
        p.error(str(exc))

    started = datetime.now()
    run_dir = args.run_dir or Path(__file__).parent / "runs" / started.strftime("%Y%m%d_%H%M%S_%f")
    if run_dir.exists():
        p.error(f"Run directory already exists: {run_dir}; choose a new directory")
    viewer = None
    try:
        torch.set_num_threads(args.threads)
        agent = dqn.DQNAgent(args.size, args.hidden_dim, learning, args.seed, args.device)
        env = dqn.DQNSnakeEnv(args.size, args.seed, rewards, learning.gamma, args.timeout_factor)
        if args.visual or args.visual_eval:
            viewer = PygameViewer(args.size, args.fps)
        run_dir.mkdir(parents=True)
    except (OSError, ValueError, RuntimeError) as exc:
        if viewer:
            viewer.close()
        p.exit(1, f"error: {exc}\n")

    completed, interrupted, failure, last_evaluation = 0, False, None, None
    recent, best_score, last_return, last_preview = deque(maxlen=100), 0, 0.0, 0.0
    model_path = run_dir / "model.pt"

    #######################
    # Saving Run Progress #
    #######################

    def save_checkpoint():
        agent.save(model_path, {
            "created_at": started.isoformat(timespec="seconds"),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "environment": {"size": args.size, "initial_length": 3, "timeout_factor": args.timeout_factor},
            "rewards": asdict(rewards),
            "training": {"seed": args.seed, "requested_episodes": args.episodes,
                         "completed_episodes": completed, "interrupted": interrupted, "failure": failure},
            "evaluation": {"seed": args.eval_seed, "games": args.eval_games,
                           "every": args.eval_every, "last_result": last_evaluation},
            "runtime": {"device": str(agent.device), "threads": args.threads,
                        "torch_version": str(torch.__version__)},
        })

    print(f"DQN | {args.size}x{args.size} | device={agent.device} | outputs: {run_dir}")
    print(f"Network: {agent.input_dim} -> {args.hidden_dim} -> {args.hidden_dim} -> 3")
    print(f"alpha={learning.alpha}, gamma={learning.gamma}, "
          f"epsilon={learning.epsilon_start}->{learning.epsilon_end}, living={rewards.living}")
    print(f"Learning reward scale={learning.reward_scale}; scores and logged returns use raw rewards.")

    #################
    # Training Loop #
    #################

    try:
        with (run_dir / "log.csv").open("w", newline="") as log_file, \
                (run_dir / "eval.csv").open("w", newline="") as eval_file:
            log, eval_log = csv.writer(log_file), csv.writer(eval_file)
            log.writerow(["episode", "epsilon", "score", "steps", "return", "max_length",
                          "end_reason", "steps_since_food", "mean_loss", "time_steps", "update_steps"])
            eval_log.writerow(["after_episode", "game", "score", "steps", "return",
                               "length", "end_reason", "steps_since_food"])

            def check_policy(title):
                def record(game, result):
                    eval_log.writerow([completed, game, result.score, result.steps, result.total_reward,
                                       result.length, result.reason, result.steps_since_apple])
                    eval_file.flush()
                stats = dqn.evaluate(agent, rewards, args.eval_games, args.eval_seed, args.timeout_factor,
                                     viewer=viewer, title=title, on_game=record)
                return {"after_episode": completed, **stats}

            for episode in range(args.episodes):
                epsilon = learning.epsilon_at(episode, args.episodes)
                training_view = viewer if args.visual and episode % args.visual_every == 0 else None
                summary = ""
                if viewer:
                    summary = (f"Last 100 mean {np.mean(recent):.2f} | Best {best_score}"
                               if recent else "Learning enabled | First episode")
                    if not training_view and (viewer.paused or monotonic() - last_preview >= 0.25):
                        viewer.show(env, f"DQN training | {completed}/{args.episodes} completed", epsilon,
                                    last_return, summary=summary, throttle=False)
                        last_preview = monotonic()

                result = dqn.run_episode(env, agent, epsilon, viewer=training_view, controls=viewer,
                                         title=f"DQN training | Episode {episode + 1}/{args.episodes}",
                                         summary=summary)
                completed += 1
                recent.append(result.score)
                best_score = max(best_score, result.score)
                last_return = result.total_reward
                log.writerow([completed, epsilon, result.score, result.steps, result.total_reward,
                              result.length, result.reason, result.steps_since_apple, result.mean_loss,
                              agent.time_steps, agent.update_steps])
                log_file.flush()

                if completed % args.print_every == 0:
                    loss = "--" if result.mean_loss is None else f"{result.mean_loss:.4f}"
                    print(f"episode {completed}: epsilon={epsilon:.3f}, last-100 mean={np.mean(recent):.2f}, "
                          f"loss={loss}, learning updates={agent.update_steps}")
                if args.eval_every and completed % args.eval_every == 0 and completed < args.episodes:
                    save_checkpoint()
                    last_evaluation = check_policy(f"DQN eval after episode {completed}")
                    print(f"Greedy mean: {last_evaluation['mean_score']:.2f} apples; "
                          f"best {last_evaluation['max_score']}; wins {last_evaluation['win_rate']:.1%}")

            # Save before final evaluation so weights are available while viewing games.
            save_checkpoint()
            last_evaluation = check_policy("DQN final evaluation")
            print("Final evaluation:")
            dqn.print_summary(last_evaluation)
    except (ViewerClosed, KeyboardInterrupt):
        interrupted = True
        print(f"Stopped after {completed} completed episodes; saving current weights.")
    except FloatingPointError as exc:
        interrupted, failure = True, str(exc)
        print(f"Training stopped: {failure}")
    except Exception as exc:
        interrupted, failure = True, str(exc)
        raise
    finally:
        if viewer:
            viewer.close()
        save_checkpoint()
        print(f"Saved {model_path}")
    if failure:
        p.exit(1)


if __name__ == "__main__":
    main()
