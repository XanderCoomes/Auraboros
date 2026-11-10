import numpy as np

from dqn.agent import DQNAgent
from dqn.encoders.engineered import EngineeredEncoder
from dqn.models.mlp import DQNMLP
from dqn.replay_buffer import ReplayBuffer
from snake_env import SnakeEnv


import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch



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

    # Reset environment and encode initial state.
    env.reset()
    state = encoder.encode(env)

    done = False
    total_reward = 0.0
    info = {"score": 0}

    while not done:

        # 1. Select action using epsilon-greedy.
        action = agent.act(state, epsilon)

        # 2. Execute action in the environment.
        _, reward, done, info = env.step(action)

        # 3. Encode the resulting state.
        next_state = encoder.encode(env)

        # 4 Learning
        if learn:
            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step(batch_size)

            if loss is not None:
                agent.training_steps += 1

                if agent.training_steps % agent.target_update_interval == 0:    
                    agent.update_target_network()   #updates the target network

        # 6. Move to next state.
        state = next_state
        total_reward += reward

    return info["score"], env.steps, total_reward, info


def evaluate_dqn(env, agent, encoder, n=100):
    """Evaluate the DQN agent without exploration or learning."""

    scores = []

    for _ in range(n):
        score, steps, total_reward, info = run_episode_dqn(
            env,
            agent,
            encoder,
            epsilon=0.0,
            learn=False,
        )

        scores.append(score)

    return float(np.mean(scores))



def main():

    parser = argparse.ArgumentParser(
        description="Train a DQN agent on Snake."
    )

    # Environment and training
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)

    # DQN hyperparameters
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--target-update-interval", type=int, default=1000)

    # Exploration
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay-frac", type=float, default=0.6)

    # Evaluation and reproducibility
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=21)

    # Output files
    parser.add_argument("--out", type=str, default="dqn/mlp_snake_dqn.pth")
    parser.add_argument("--log", type=str, default="dqn/mlp_dqn_training.csv")

    args = parser.parse_args()

    # Select GPU if available, otherwise CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Initialize environment and state encoder.
    env = SnakeEnv(size=args.size, seed=args.seed)
    encoder = EngineeredEncoder()

    # Initialize neural network.
    model = DQNMLP(input_dim=encoder.input_shape[0])

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

    # Track training steps for target-network synchronization.
    agent.training_steps = 0
    agent.target_update_interval = args.target_update_interval

    # Prepare output paths.
    model_path = Path(args.out)
    log_path = Path(args.log)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare training log.
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

        scores = []

        for episode in range(1, args.episodes + 1):

            # Calculate exploration probability.
            epsilon = get_epsilon(
                episode,
                args.episodes,
                args.eps_start,
                args.eps_end,
                args.eps_decay_frac,
            )

            # Run one training episode.
            score, steps, total_reward, info = run_episode_dqn(
                env,
                agent,
                encoder,
                epsilon=epsilon,
                batch_size=args.batch_size,
                learn=True,
            )

            scores.append(score)

            # Evaluate periodically.
            if episode % args.eval_interval == 0:

                average_score = evaluate_dqn(
                    env,
                    agent,
                    encoder,
                    n=args.eval_episodes,
                )

                print(
                    f"Episode: {episode}/{args.episodes} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Score: {score} | "
                    f"Average evaluation score: {average_score:.2f}"
                )

            else:
                average_score = ""

            # Record episode results.
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

    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()