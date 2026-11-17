
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.encoders.board import BoardEncoderWithTime
from dqn.features.tail_features import get_feature_matrix
from dqn.models.hybrid import DQNHybrid
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


def encode_hybrid_state(env, encoder):
    """Encode the board, heading, and action-specific features."""

    board, heading = encoder.encode(env)
    features = get_feature_matrix(env)

    return board, heading, features


def run_episode_hybrid(
    env,
    agent,
    encoder,
    epsilon,
    batch_size=64,
    learn=True,
):
    """Run one Snake episode using a hybrid DQN agent."""

    env.reset()
    state = encode_hybrid_state(env, encoder)

    done = False
    total_reward = 0.0
    info = {"score": 0}

    while not done:

        # Select action.
        action = agent.act(state, epsilon)

        # Execute action.
        _, reward, done, info = env.step(action)

        # Encode next state.
        next_state = encode_hybrid_state(env, encoder)

        # Learn from experience.
        if learn:
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

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




def evaluate_hybrid(
    agent,
    encoder,
    size=8,
    n=100,
    base_seed=10000,
):
    """Evaluate the hybrid DQN on reproducible games."""

    scores = []

    max_score = 0
    wins = 0

    rng_state = agent.rng.getstate()

    try:
        for i in range(n):

            eval_env = SnakeEnv(
                size=size,
                seed=base_seed + i,
            )

            score, steps, total_reward, info = run_episode_hybrid(
                eval_env,
                agent,
                encoder,
                epsilon=0.0,
                learn=False,
            )

            scores.append(score)

            max_score = max(max_score, score)

            if score >= (size * size - 3):
                wins += 1

    finally:
        agent.rng.setstate(rng_state)

    return {
    "average_score": float(np.mean(scores)),
    "max_score": max_score,
    "wins": wins,
    "win_rate": wins / n * 100,
    }   



def main():

    parser = argparse.ArgumentParser(
        description="Train a Hybrid DQN agent on Snake."
    )

    # Environment and training.
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20000)
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

    # Hybrid-specific output files.
    parser.add_argument(
        "--out",
        type=str,
        default="dqn/hybrid/hybrid_snake_dqn.pth",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="dqn/hybrid/hybrid_training.csv",
    )

    args = parser.parse_args()

    if args.size != 8:
        parser.error("The current DQNHybrid architecture requires size=8.")

    # Select GPU if available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Initialize environment and encoder.
    env = SnakeEnv(size=args.size, seed=args.seed)
    encoder = BoardEncoderWithTime()

    # Initialize hybrid model.
    model = DQNHybrid()

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
            "eval_average_score",
            "eval_max_score",
            "eval_wins",
            "eval_win_rate",
        ])

        for episode in range(1, args.episodes + 1):

            epsilon = get_epsilon(
                episode,
                args.episodes,
                args.eps_start,
                args.eps_end,
                args.eps_decay_frac,
            )

            score, steps, total_reward, info = run_episode_hybrid(
                env,
                agent,
                encoder,
                epsilon=epsilon,
                batch_size=args.batch_size,
                learn=True,
            )

            
            # Default values when no evaluation occurs.
            average_score = ""
            max_score = ""
            wins = ""
            win_rate = ""

            # Periodic evaluation.
            if episode % args.eval_interval == 0:

                eval_results = evaluate_hybrid(
                    agent,
                    encoder,
                    size=args.size,
                    n=args.eval_episodes,
                    base_seed=10000,
                )

                average_score = eval_results["average_score"]
                max_score = eval_results["max_score"]
                wins = eval_results["wins"]
                win_rate = eval_results["win_rate"]

                print(
                    f"Episode: {episode}/{args.episodes} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Score: {score} | "
                    f"Eval Avg: {average_score:.2f} | "
                    f"Eval Max: {max_score} | "
                    f"Wins: {wins}/{args.eval_episodes} "
                    f"({win_rate:.1f}%)"
                )

            writer.writerow([
                episode,
                epsilon,
                score,
                steps,
                total_reward,
                average_score,
                max_score,
                wins,
                win_rate,
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

    print("\nHybrid training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
