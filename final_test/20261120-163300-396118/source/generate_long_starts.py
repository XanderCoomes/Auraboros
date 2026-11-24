"""Generate replayable long-snake starts using a saved DQN policy."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch

from q_dqn import DQNAgent
from rewards import RewardSpec
from state_encoders import BoardEncoder, BoardSnakeEnv


def generate_pool(agent, metadata, rng, count, seed_start, min_apples, exploration,
                  survival_moves, max_games, path, seen):
    size = metadata["board_size"]
    encoder = BoardEncoder(**metadata["encoder"])
    histogram, saved_games, attempts = Counter(), 0, 0
    with path.open("x") as handle:
        while sum(histogram.values()) < count and attempts < max_games:
            seed = seed_start + attempts
            attempts += 1
            # Same initial state and apple RNG sequence as the recording loader.
            env = BoardSnakeEnv(size=size, seed=seed, encoder=encoder,
                                rewards=RewardSpec(metadata["rewards"]))
            state = env.get_state()
            actions, candidates = [], []
            remaining = count - sum(histogram.values())
            # A live game cannot exceed this bound under the no-food timeout.
            max_steps = (size * size - env.init_length + 1) * (env.timeout + 1)
            for _ in range(max_steps):
                safe = [a for a, turn in enumerate((0, -1, 1))
                        if not env._is_danger((env.heading + turn) % 4)]
                if not safe:
                    break
                if rng.random() < exploration:
                    action = rng.choice(safe)
                else:
                    values = agent.q_values(state)
                    action = max(safe, key=lambda a: values[a])
                state, _, done, info = env.step(action)
                actions.append(action)
                if done:
                    # Only genuinely live continuation moves count toward validation.
                    actions.pop()
                    break
                if info["event"]["outcome"] == "apple" and env.score >= min_apples:
                    signature = (tuple(env.snake), env.heading, env.apple)
                    candidates.append((len(actions), env.score, signature))
                if len(candidates) >= remaining and len(actions) >= candidates[remaining - 1][0] + survival_moves:
                    break
            eligible = [item for item in candidates if item[0] <= len(actions) - survival_moves]
            if not eligible:
                continue
            # Reject whole trajectories with repeated starts, since the loader consumes
            # every apple milestone in a recorded trajectory.
            if any(item[2] in seen for item in eligible):
                continue
            eligible = eligible[:remaining]
            actions = actions[:eligible[-1][0]]
            record = {"version": 1, "size": size, "seed": seed, "actions": actions}
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            saved_games += 1
            for _, score, signature in eligible:
                histogram[score] += 1
                seen.add(signature)
            if saved_games % 25 == 0:
                handle.flush()
                print(f"{path.name}: {sum(histogram.values())}/{count} starts from {saved_games} games", flush=True)
    if sum(histogram.values()) < count:
        raise RuntimeError(f"Only generated {sum(histogram.values())}/{count} starts; increase --max-games")
    return {"positions": sum(histogram.values()), "games": saved_games, "attempted_games": attempts,
            "seed_start": seed_start, "seed_end": seed_start + attempts - 1,
            "apples_histogram": dict(sorted(histogram.items())),
            "snake_length_range": [min(histogram) + 3, max(histogram) + 3],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/rescnn8-ordered-20261115-175028-075188/model.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("demos/synthetic-long"))
    parser.add_argument("--starts", type=int, default=5000)
    parser.add_argument("--validation-starts", type=int, default=1000)
    parser.add_argument("--min-apples", type=int, default=15)
    parser.add_argument("--survival-moves", type=int, default=10)
    parser.add_argument("--exploration", type=float, default=0.02)
    parser.add_argument("--seed-start", type=int, default=700000)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (min(args.starts, args.validation_starts, args.min_apples, args.survival_moves, args.max_games) < 1
            or not 0 <= args.exploration <= 1):
        parser.error("Counts and thresholds must be positive; exploration must be in [0,1]")
    torch.set_num_threads(1)
    agent, metadata = DQNAgent.load(args.checkpoint, device=args.device)
    if metadata.get("initial_length", 3) != 3 or metadata.get("timeout_factor", 2) != 2:
        parser.error("Recording format requires initial_length=3 and timeout_factor=2")
    if args.min_apples >= metadata["board_size"] ** 2 - 3:
        parser.error("min-apples must leave space for a playable non-winning board")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"checkpoint": str(args.checkpoint),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "board_size": metadata["board_size"], "min_apples": args.min_apples,
                "survival_moves_checked": args.survival_moves, "exploration": args.exploration,
                "note": "Survival verified on the generating trajectory; arbitrary agent moves can still fail.",
                "pools": {}}
    seen = set()
    for index, (name, count) in enumerate((("train", args.starts), ("validation", args.validation_starts))):
        first_seed = args.seed_start + index * args.max_games
        manifest["pools"][name] = generate_pool(agent, metadata, random.Random(first_seed),
            count, first_seed, args.min_apples, args.exploration, args.survival_moves,
            args.max_games, args.out_dir / f"{name}.jsonl", seen)
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(name, manifest["pools"][name], flush=True)


if __name__ == "__main__":
    main()
