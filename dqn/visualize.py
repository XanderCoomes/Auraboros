
import numpy as np

from snake_env import SnakeEnv

import torch

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from dqn.features.tail_features import get_feature_matrix


def record_episode(
    model,
    encoder,
    device,
    size=8,
    seed=42,
):
    """
    Run one greedy hybrid DQN episode and record gameplay,
    state representation, and action-specific features.
    """

    env = SnakeEnv(size=size, seed=seed)
    env.reset()

    frames = []
    done = False
    total_reward = 0.0

    while not done:

        # Encode the current state.
        board, heading = encoder.encode(env)
        features = get_feature_matrix(env)

        # Convert inputs to tensors.
        board_tensor = torch.as_tensor(
            board, dtype=torch.float32, device=device
        ).unsqueeze(0)

        heading_tensor = torch.as_tensor(
            heading, dtype=torch.float32, device=device
        ).unsqueeze(0)

        features_tensor = torch.as_tensor(
            features, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Obtain Q-values.
        with torch.no_grad():
            q_values = model(
                board_tensor,
                heading_tensor,
                features_tensor
            )

        q_values = q_values.squeeze(0).cpu().numpy()

        # Greedy action selection.
        action = int(np.argmax(q_values))

        # Capture state before executing the action.
        frames.append({
            "board": env.board_array().copy(),
            "state": (board.copy(), heading.copy(), features.copy()),
            "heading": env.heading,
            "action": action,
            "q_values": q_values.copy(),
            "score": env.score,
            "steps": env.steps,
            "total_reward": total_reward,
        })

        # Execute action.
        _, reward, done, info = env.step(action)

        total_reward += reward

        frames[-1]["reward"] = reward
        frames[-1]["total_reward"] = total_reward

    # Record terminal information.
    frames[-1]["done"] = True
    frames[-1]["end_reason"] = info.get(
        "reason", "unknown"
    )

    frames[-1]["final_score"] = env.score

    return frames


def record_episode_mlp(model, encoder, device, size=8, seed=42):
    """
    Run one greedy MLP DQN episode and record frames in the same format as
    record_episode, with "state" = the 13-feature vector (for save_state_gif "mlp").
    """

    env = SnakeEnv(size=size, seed=seed)
    env.reset()

    frames = []
    done = False
    total_reward = 0.0
    info = {}

    while not done:
        state = encoder.encode(env)

        with torch.no_grad():
            q_values = model(
                torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            ).squeeze(0).cpu().numpy()

        action = int(np.argmax(q_values))

        frames.append({
            "board": env.board_array().copy(),
            "state": state.copy(),
            "heading": env.heading,
            "action": action,
            "q_values": q_values.copy(),
            "score": env.score,
            "steps": env.steps,
            "total_reward": total_reward,
        })

        _, reward, done, info = env.step(action)
        total_reward += reward
        frames[-1]["reward"] = reward
        frames[-1]["total_reward"] = total_reward

    frames[-1]["done"] = True
    frames[-1]["end_reason"] = info.get("reason", "unknown")
    frames[-1]["final_score"] = env.score

    return frames




def save_gameplay_gif(frames, output_path, fps=6):
    """
    Create a GIF showing the Snake gameplay and the DQN's decisions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Board colors:
    # 0 = empty, 1 = body, 2 = head, 3 = apple
    colors = ["#F5F5F5", "#4CAF50", "#145A32", "#E53935"]
    cmap = ListedColormap(colors)

    fig, (ax_board, ax_info) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={"width_ratios": [1, 1]}
    )

    fig.suptitle("Snake DQN - Gameplay", fontsize=16)

    # Board visualization
    board_image = ax_board.imshow(
        frames[0]["board"],
        cmap=cmap,
        vmin=0,
        vmax=3
    )

    size = frames[0]["board"].shape[0]

    ax_board.set_xticks(np.arange(-0.5, size, 1))
    ax_board.set_yticks(np.arange(-0.5, size, 1))
    ax_board.grid(color="gray", linewidth=0.5)
    ax_board.set_xticklabels([])
    ax_board.set_yticklabels([])
    ax_board.set_title("Game board")

    # Information panel
    ax_info.axis("off")

    info_text = ax_info.text(
        0.05, 0.95, "",
        transform=ax_info.transAxes,
        verticalalignment="top",
        fontsize=11,
        family="monospace",
        linespacing=1.6
    )

    action_names = {
        0: "Straight",
        1: "Left",
        2: "Right"
    }

    heading_names = {
        0: "Up",
        1: "Right",
        2: "Down",
        3: "Left"
    }

    def update(frame_index):
        frame = frames[frame_index]

        board_image.set_data(frame["board"])

        q_values = frame["q_values"]

        info = (
            f"Frame: {frame_index + 1}/{len(frames)}\n\n"
            f"Score: {frame['score']}\n"
            f"Steps: {frame['steps']}\n"
            f"Heading: {heading_names[frame['heading']]}\n\n"
            f"Action: {action_names[frame['action']]}\n\n"
            f"Q-values:\n"
            # Action order matches snake_env: 0 = straight, 1 = left, 2 = right
            f"  Straight: {q_values[0]:8.3f}\n"
            f"  Left:     {q_values[1]:8.3f}\n"
            f"  Right:    {q_values[2]:8.3f}\n\n"
            f"Last reward: {frame['reward']:.2f}\n"
            f"Total reward: {frame['total_reward']:.2f}"
        )

        if frame_index == len(frames) - 1:
            info += (
                f"\n\nGAME OVER\n"
                f"Final score: {frame['final_score']}\n"
                f"Reason: {frame['end_reason']}"
            )

        info_text.set_text(info)

        return board_image, info_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False
    )

    animation.save(
        output_path,
        writer=PillowWriter(fps=fps)
    )

    plt.close(fig)

    print(f"Gameplay GIF saved to: {output_path}")


def save_state_gif(frames, model_type, output_path, fps=6):
    """
    Visualize the state representation received by the DQN.
    Supports CNN (board channels) and MLP (engineered features).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "cnn":
        fig = plt.figure(figsize=(11, 6))
        grid = fig.add_gridspec(2, 3, height_ratios=[3, 1])

        ax_snake = fig.add_subplot(grid[0, 0])
        ax_apple = fig.add_subplot(grid[0, 1])
        ax_head = fig.add_subplot(grid[0, 2])
        ax_heading = fig.add_subplot(grid[1, :])

        axes = [ax_snake, ax_apple, ax_head]

        channel_names = ["Snake occupancy", "Apple", "Head"]
        channel_images = []

        for ax, name in zip(axes, channel_names):
            image = ax.imshow(
                np.zeros((8, 8)),
                cmap="gray",
                vmin=0,
                vmax=1
            )
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            channel_images.append(image)

        heading_names = ["Up", "Right", "Down", "Left"]

        ax_heading.set_xlim(-0.5, 3.5)
        ax_heading.set_ylim(0, 1.5)
        ax_heading.set_xticks(range(4), heading_names)
        ax_heading.set_yticks([])
        ax_heading.set_title("Heading one-hot encoding")

        heading_bars = ax_heading.bar(
            range(4), [0, 0, 0, 0]
        )

        fig.suptitle("CNN State Representation")

        def update(frame_index):
            frame = frames[frame_index]
            board, heading = frame["state"]

            for channel, image in enumerate(channel_images):
                image.set_data(board[channel])

            for bar, value in zip(heading_bars, heading):
                bar.set_height(float(value))

            fig.suptitle(
                f"CNN State Representation | Frame {frame_index + 1}"
            )

            return (*channel_images, *heading_bars)

    elif model_type == "mlp":
        fig, ax = plt.subplots(figsize=(11, 5))

        feature_names = [
            "Danger straight",
            "Danger right",
            "Danger left",
            "Heading up",
            "Heading right",
            "Heading down",
            "Heading left",
            "Apple up",
            "Apple right",
            "Apple down",
            "Apple left",
            "Snake length",
            "Apple distance",
        ]

        x = np.arange(len(feature_names))

        bars = ax.bar(
            x,
            np.zeros(len(feature_names)),
            color="steelblue"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            feature_names,
            rotation=55,
            ha="right"
        )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Feature value")
        ax.set_title("MLP State Representation")

        fig.tight_layout()

        def update(frame_index):
            frame = frames[frame_index]
            state = frame["state"]

            for bar, value in zip(bars, state):
                bar.set_height(float(value))

            ax.set_title(
                f"MLP State Representation | Frame {frame_index + 1}"
            )

            return tuple(bars)


    elif model_type == "hybrid":
        fig = plt.figure(figsize=(14, 8))
        grid = fig.add_gridspec(
            2, 4,
            height_ratios=[2, 1.5]
        )

        # Four CNN input channels.
        channel_names = [
            "Snake occupancy",
            "Apple position",
            "Head position",
            "Temporal body encoding"
        ]

        channel_axes = [
            fig.add_subplot(grid[0, i])
            for i in range(4)
        ]

        channel_images = []

        for ax, name in zip(channel_axes, channel_names):
            image = ax.imshow(
                np.zeros((8, 8)),
                cmap="viridis",
                vmin=0,
                vmax=1
            )

            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            channel_images.append(image)

        # Heading one-hot vector.
        ax_heading = fig.add_subplot(grid[1, 0])

        heading_names = ["Up", "Right", "Down", "Left"]

        heading_bars = ax_heading.bar(
            range(4),
            [0, 0, 0, 0]
        )

        ax_heading.set_xticks(range(4), heading_names)
        ax_heading.set_ylim(0, 1.2)
        ax_heading.set_title("Heading")
        ax_heading.set_ylabel("Value")

        # Action-specific feature matrix: 3 actions x 11 features.
        ax_features = fig.add_subplot(grid[1, 1:])

        feature_names = [
            "Bias",
            "Apple dist.",
            "Would eat",
            "Wall collision",
            "Body collision",
            "Length frac.",
            "Reachable space",
            "Tail reachable",
            "Length × dist.",
            "Length × space",
            "Length × tail"
        ]

        action_names = ["Straight", "Left", "Right"]

        feature_image = ax_features.imshow(
            np.zeros((3, 11)),
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto"
        )

        ax_features.set_xticks(
            range(11),
            feature_names,
            rotation=45,
            ha="right"
        )

        ax_features.set_yticks(
            range(3),
            action_names
        )

        ax_features.set_title(
            "Action-specific feature matrix (3 × 11)"
        )

        # Numeric values inside each feature cell.
        feature_texts = []

        def update(frame_index):
            frame = frames[frame_index]

            board, heading, features = frame["state"]

            # Update CNN channels.
            for channel, image in enumerate(channel_images):
                image.set_data(board[channel])

            # Update heading.
            for bar, value in zip(heading_bars, heading):
                bar.set_height(float(value))

            # Update feature matrix.
            feature_image.set_data(features)

            # Remove annotations from previous frame.
            for annotation in feature_texts:
                annotation.remove()

            feature_texts.clear()

            # Add current numerical feature values.
            for row in range(3):
                for col in range(11):
                    annotation = ax_features.text(
                        col,
                        row,
                        f"{features[row, col]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white"
                    )
                    feature_texts.append(annotation)

            fig.suptitle(
                f"Hybrid DQN Inputs | Frame {frame_index + 1}/{len(frames)}"
            )

            return (
                *channel_images,
                *heading_bars,
                feature_image,
                *feature_texts
            )

    else:
        raise ValueError(
            "model_type must be either 'cnn' or 'mlp'"
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False
    )

    animation.save(
        output_path,
        writer=PillowWriter(fps=fps)
    )

    plt.close(fig)

    print(f"State representation GIF saved to: {output_path}")