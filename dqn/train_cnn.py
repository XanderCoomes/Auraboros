
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.encoders.board import BoardEncoder
from dqn.models.cnn import DQNCNN
from dqn.replay_buffer import ReplayBuffer
from snake_env import SnakeEnv


def get_epsilon(
    episode,
    total_episodes,
    eps_start=1.0,
    eps_end=0.01,
    decay_fraction=0.6,
):
    decay_episodes = total_episodes * decay_fraction
    progress = min(episode / decay_episodes, 1.0)

    epsilon = eps_start + progress * (eps_end - eps_start)

    return epsilon


def run_episode_dqn(
    env,
    agent,
    encoder,
    epsilon,
    batch_size=64,
    learn=True,
):
    """Run one Snake episode using a DQN agent."""

    env.reset()
    state = encoder.encode(env)

    done = False
    total_reward = 0.0
    info = {"score": 0}

    while not done:

        # Select action.
        action = agent.act(state, epsilon)

        # Execute action.
        _, reward, done, info = env.step(action)

        # Encode next state.
        next_state = encoder.encode(env)

        # Learn from experience.
        if learn:
            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step(batch_size)

            if loss is not None:
                agent.training_steps += 1

                if (
                    agent.training_steps
                    % agent.target_update_interval == 0
                ):
                    agent.update_target_network()

        state = next_state
        total_reward += reward

    return info["score"], env.steps, total_reward, info


def evaluate_dqn(agent, encoder, size=8, n=100, base_seed=10000):
    """Evaluate the DQN agent on a fixed set of reproducible games."""

    scores = []

    # Preserve the agent's random-number generator state.
    # Evaluation should not affect training randomness.
    rng_state = agent.rng.getstate()

    try:
        for i in range(n):

            # Create a fresh environment with a fixed seed.
            eval_env = SnakeEnv(
                size=size,
                seed=base_seed + i
            )

            score, steps, total_reward, info = run_episode_dqn(
                eval_env,
                agent,
                encoder,
                epsilon=0.0,
                learn=False,
            )

            scores.append(score)

    finally:
        # Restore the random-number generator state.
        agent.rng.setstate(rng_state)

    return float(np.mean(scores))


def main():

    parser = argparse.ArgumentParser(
        description="Train a CNN-DQN agent on Snake."
    )

    # Environment and training.
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)

    # DQN hyperparameters.
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--target-update-interval", type=int, default=1000)

    # Exploration.
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay-frac", type=float, default=0.6)

    # Evaluation and reproducibility.
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=21)

    # CNN-specific output files.
    parser.add_argument(
        "--out",
        type=str,
        default="dqn/cnn_snake_dqn.pth",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="dqn/cnn_training.csv",
    )

    args = parser.parse_args()

    # Current CNN architecture supports 8x8 boards.
    if args.size != 8:
        parser.error("The current DQNCNN architecture requires size=8.")

    # Select GPU if available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Initialize environment and encoder.
    env = SnakeEnv(size=args.size, seed=args.seed)
    encoder = BoardEncoder()

    # Initialize CNN.
    model = DQNCNN()

    # Initialize replay buffer.
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    # Initialize DQN agent.
    agent = DQNAgent(
        model=model,
        replay_buffer=replay_buffer,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )

    # Target-network synchronization.
    agent.training_steps = 0
    agent.target_update_interval = args.target_update_interval

    # Prepare output paths.
    model_path = Path(args.out)
    log_path = Path(args.log)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare CSV log.
    with open(log_path, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "episode",
            "epsilon",
            "score",
            "steps",
            "total_reward",
            "average_score",
        ])

        for episode in range(1, args.episodes + 1):

            epsilon = get_epsilon(
                episode,
                args.episodes,
                args.eps_start,
                args.eps_end,
                args.eps_decay_frac,
            )

            score, steps, total_reward, info = run_episode_dqn(
                env,
                agent,
                encoder,
                epsilon=epsilon,
                batch_size=args.batch_size,
                learn=True,
            )

            # Periodic evaluation.
            if episode % args.eval_interval == 0:

                average_score = evaluate_dqn(
                        agent,
                        encoder,
                        size=args.size,
                        n=args.eval_episodes,
                        base_seed=10000,
                    )

                print(
                    f"Episode: {episode}/{args.episodes} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Score: {score} | "
                    f"Average evaluation score: {average_score:.2f}"
                )

            else:
                average_score = ""

            writer.writerow([
                episode,
                epsilon,
                score,
                steps,
                total_reward,
                average_score,
            ])

    # Save trained online network.
    torch.save(agent.model.state_dict(), model_path)

    # Save training configuration.
    config = vars(args)
    config["device"] = str(device)
    config["training_steps"] = agent.training_steps

    config_path = model_path.with_suffix(".json")

    with open(config_path, "w") as jsonfile:
        json.dump(config, jsonfile, indent=4)

    print("\nCNN training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
