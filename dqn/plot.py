
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(log_path, output_dir, window=500, title="Hybrid DQN Training Analysis"):

    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)

    # The MLP trainer logs its periodic evaluation as "average_score" and has no
    # max-score / win-rate columns; those panels are left empty with a note.
    if "eval_average_score" not in df.columns and "average_score" in df.columns:
        df = df.rename(columns={"average_score": "eval_average_score"})

    required_columns = {
        "episode",
        "epsilon",
        "score",
        "steps",
        "total_reward",
        "eval_average_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    def not_logged(ax, what):
        ax.text(0.5, 0.5, f"{what} not logged\nfor this run", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    # Moving averages for training metrics.
    df["score_ma"] = df["score"].rolling(
        window=window, min_periods=1
    ).mean()

    df["steps_ma"] = df["steps"].rolling(
        window=window, min_periods=1
    ).mean()

    df["reward_ma"] = df["total_reward"].rolling(
        window=window, min_periods=1
    ).mean()

    # Evaluation metrics only exist at evaluation checkpoints.
    evaluation = df.dropna(subset=["eval_average_score"])

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))

    # 1. Training score.
    axes[0, 0].plot(
        df["episode"], df["score"],
        alpha=0.2, label="Episode score"
    )
    axes[0, 0].plot(
        df["episode"], df["score_ma"],
        label=f"{window}-episode moving average"
    )
    axes[0, 0].set_title("Training Score")
    axes[0, 0].set_ylabel("Apples")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Evaluation average and maximum score.
    axes[0, 1].plot(
        evaluation["episode"], evaluation["eval_average_score"],
        marker="o", label="Average score"
    )
    if "eval_max_score" in evaluation.columns:
        axes[0, 1].plot(
            evaluation["episode"], evaluation["eval_max_score"],
            marker="s", label="Maximum score"
        )
    axes[0, 1].set_title("Evaluation Performance")
    axes[0, 1].set_ylabel("Apples")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Evaluation win rate.
    if "eval_win_rate" in evaluation.columns:
        axes[1, 0].plot(
            evaluation["episode"], evaluation["eval_win_rate"],
            marker="o", label="Win rate"
        )
        axes[1, 0].legend()
    else:
        not_logged(axes[1, 0], "Win rate")
    axes[1, 0].set_title("Evaluation Win Rate")
    axes[1, 0].set_ylabel("Win rate (%)")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].grid(alpha=0.3)

    # 4. Episode length.
    axes[1, 1].plot(
        df["episode"], df["steps_ma"],
        label=f"{window}-episode moving average"
    )
    axes[1, 1].set_title("Episode Length")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # 5. Total reward.
    axes[2, 0].plot(
        df["episode"], df["reward_ma"],
        label=f"{window}-episode moving average"
    )
    axes[2, 0].set_title("Episode Reward")
    axes[2, 0].set_ylabel("Total reward")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    # 6. Epsilon decay.
    axes[2, 1].plot(
        df["episode"], df["epsilon"],
        label="Epsilon"
    )
    axes[2, 1].set_title("Exploration Decay")
    axes[2, 1].set_ylabel("Epsilon")
    axes[2, 1].set_ylim(0, 1.05)
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Episode")

    fig.suptitle(title, fontsize=16)
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
        default="dqn/hybrid/hybrid_training.csv"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_result/hybrid"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=500
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Hybrid DQN Training Analysis"
    )

    args = parser.parse_args()

    plot_learning_curves(
        args.log,
        args.output_dir,
        args.window,
        args.title
    )
