
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(log_path, output_dir, window=500):

    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)

    required_columns = {
        "episode",
        "score",
        "steps",
        "total_reward",
        "average_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    # Calculate moving averages.
    df["score_ma"] = df["score"].rolling(
        window=window,
        min_periods=1
    ).mean()

    df["steps_ma"] = df["steps"].rolling(
        window=window,
        min_periods=1
    ).mean()

    df["reward_ma"] = df["total_reward"].rolling(
        window=window,
        min_periods=1
    ).mean()

    # Evaluation scores exist only at evaluation checkpoints.
    evaluation = df.dropna(subset=["average_score"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training score.
    axes[0, 0].plot(
        df["episode"],
        df["score"],
        alpha=0.2,
        label="Episode score"
    )

    axes[0, 0].plot(
        df["episode"],
        df["score_ma"],
        label=f"{window}-episode moving average"
    )

    axes[0, 0].set_title("Training Score")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Apples")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Evaluation score.
    axes[0, 1].plot(
        evaluation["episode"],
        evaluation["average_score"],
        marker="o",
        label="Evaluation score"
    )

    axes[0, 1].set_title("Evaluation Performance")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Average apples")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Episode steps.
    axes[1, 0].plot(
        df["episode"],
        df["steps_ma"],
        label=f"{window}-episode moving average"
    )

    axes[1, 0].set_title("Episode Length")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Total reward.
    axes[1, 1].plot(
        df["episode"],
        df["reward_ma"],
        label=f"{window}-episode moving average"
    )

    axes[1, 1].set_title("Episode Reward")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Total reward")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("CNN-DQN Training Analysis", fontsize=16)
    fig.tight_layout()

    output_path = output_dir / "curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Learning curves saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        type=str,
        default="dqn/cnn_training.csv"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_result/cnn"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=500
    )

    args = parser.parse_args()

    plot_learning_curves(
        args.log,
        args.output_dir,
        args.window
    )
