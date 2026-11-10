"""Evaluate a saved Q-table without training or writing training artifacts.

Usage: python eval.py --q q_table.pkl --games 100 --visual
"""
import argparse
import json
from pathlib import Path

import numpy as np

from feature_library import DEFAULT_FEATURES, make_feature_set
from pygame_viewer import PygameViewer, ViewerClosed
from q_tabular import QAgent
from rollout import evaluation_scores
from snake_env import SnakeEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q", default="q_table.pkl", help="saved Q-table; matching JSON supplies size/features")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--size", type=int, help="board size (normally read from the model JSON)")
    p.add_argument("--features", help="comma-separated features (normally read from the model JSON)")
    p.add_argument("--visual", action="store_true", help="watch evaluation games in Pygame")
    p.add_argument("--fps", type=int, default=30, help="initial steps per second in visual mode (1..240)")
    args = p.parse_args()
    if args.games < 1 or not 1 <= args.fps <= 240:
        p.error("--games must be positive and --fps must be between 1 and 240")

    cfg_path = Path(args.q).with_suffix(".json")
    viewer = None
    try:
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if not config and (args.size is None or args.features is None):
            p.error("model JSON is missing; provide both --size and --features")
        size = args.size if args.size is not None else config.get("board_size", 8)
        fs = make_feature_set(args.features if args.features is not None
                              else config.get("features", DEFAULT_FEATURES))
        if size < 4:
            p.error("--size must be at least 4 for the initial snake")
        if "board_size" in config and size != config["board_size"]:
            p.error("--size differs from the saved model's board size")
        if "features" in config and fs.names != make_feature_set(config["features"]).names:
            p.error("--features differs from the saved model's feature set")
        env = SnakeEnv(size=size, seed=args.seed, features=fs)
        agent = QAgent(fs.shape, seed=args.seed)
        agent.load(args.q)
        if args.visual:
            viewer = PygameViewer(size, args.fps)
        print(f"Evaluating {args.q}: {size}x{size}, features {fs}; epsilon=0, learning disabled")
        scores = evaluation_scores(env, agent, args.games, viewer)
        print(f"Greedy evaluation over {len(scores)} games: mean score {np.mean(scores):.2f}, "
              f"max {max(scores)}")
    except (ViewerClosed, KeyboardInterrupt):
        print("Evaluation stopped. Saved model unchanged.")
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(1, f"error: {exc}\n")
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
