"""Evaluate frozen DQN weights. Run python3 eval.py from inside deepq/."""
import argparse
from pathlib import Path

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, help="model.pt or run directory; default: latest model in deepq/runs")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=dqn.EVAL_SEED)
    p.add_argument("--start-game", type=int, default=1, help="replay earlier batch games without animation")
    p.add_argument("--visual", action="store_true")
    p.add_argument("--fps", type=int, default=dqn.EVAL_FPS)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=dqn.DEVICE)
    p.add_argument("--threads", type=int, default=dqn.TORCH_THREADS)
    args = p.parse_args(argv)
    if min(args.games, args.start_game, args.threads) < 1 or args.seed < 0 or not 1 <= args.fps <= 240:
        p.error("games/start-game/threads must be positive, seed nonnegative and fps in 1..240")

    model_path, viewer, results = args.model, None, []

    def record(game, result):
        results.append(result)
        print(f"game {game}: apples={result.score}, length={result.length}, total_steps={result.steps}, "
              f"end={result.reason}, steps_since_food={result.steps_since_apple}/{timeout_steps:g}")

    ############################
    # Load Weights & Play Games #
    ############################

    try:
        torch.set_num_threads(args.threads)
        if model_path is None:
            run_root = Path(__file__).resolve().parent / "runs"
            models = [path for path in run_root.glob("*/model.pt") if path.is_file()]
            if not models:
                p.error("No saved DQN models. Train first or pass --model <path>.")
            model_path = max(models, key=lambda path: (path.stat().st_mtime_ns, str(path)))
            print(f"Using latest saved model: {model_path}")
        elif model_path.is_dir():
            model_path = model_path / "model.pt"

        agent, metadata = dqn.DQNAgent.load(model_path, device=args.device, seed=args.seed)
        rewards = dqn.RewardConfig(**metadata["rewards"])
        timeout_factor = metadata["environment"]["timeout_factor"]
        timeout_steps = timeout_factor * agent.size * agent.size
        training = metadata["training"]
        print(f"Loaded {model_path} | {training['completed_episodes']} training episodes | "
              f"{agent.time_steps} training moves | device={agent.device}")
        if training.get("interrupted"):
            print("This checkpoint was saved from an interrupted training run.")
        print(f"Seed={args.seed} | games {args.start_game}..{args.start_game + args.games - 1} | "
              f"timeout after more than {timeout_steps:g} consecutive moves without food")
        print("Greedy evaluation: epsilon=0, weights frozen, no replay or optimizer updates.")
        if args.visual:
            viewer = PygameViewer(agent.size, args.fps)
        stats = dqn.evaluate(agent, rewards, args.games, args.seed, timeout_factor,
                             viewer=viewer, on_game=record, start_game=args.start_game)
    except (ViewerClosed, KeyboardInterrupt):
        print("Evaluation stopped.")
        stats = dqn.summarize(results) if results else None
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, FloatingPointError) as exc:
        p.exit(1, f"error: {exc}\n")
    finally:
        if viewer:
            viewer.close()
    if stats:
        dqn.print_summary(stats)


if __name__ == "__main__":
    main()
