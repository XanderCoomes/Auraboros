"""Replay a checkpoint with the final three layer activations and output contributions."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
import numpy as np
import torch

from q_dqn import DQNAgent
from rewards import RewardSpec
from state_encoders import BoardEncoder, BoardSnakeEnv


ACTION_NAMES = ("straight", "left", "right")


def record_game(agent, metadata, seed=1, max_steps=10000):
    encoder = BoardEncoder(**metadata["encoder"])
    env = BoardSnakeEnv(encoder=encoder, size=metadata["board_size"], seed=seed,
                        rewards=RewardSpec(metadata["rewards"]),
                        init_length=metadata["initial_length"], timeout_factor=metadata["timeout_factor"])
    observation = env.reset()
    frames, total, reason, collision = [], 0.0, "viewer step limit", None
    for _ in range(max_steps):
        activations = agent.inspect(observation)
        action = int(np.argmax(activations["q"]))
        frames.append({"board": env.board_array(), "activations": activations,
                       "score": env.score, "steps": env.steps, "return": total,
                       "action": action, "reason": ""})
        observation, reward, done, info = env.step(action)
        total += reward
        if done:
            reason = info["reason"]
            if reason == "collision":
                collision = info["event"]["head"]
                row, col = collision
                reason = "body collision" if 0 <= row < env.size and 0 <= col < env.size else "wall collision"
            break
    # The terminal frame shows the result of the last decision.
    frames.append({"board": env.board_array(), "activations": frames[-1]["activations"],
                   "score": env.score, "steps": env.steps, "return": total,
                   "action": frames[-1]["action"], "reason": reason, "collision": collision})
    return frames


class ReplayViewer:
    def __init__(self, frames, output_weights, output_bias, interval=150):
        self.frames = frames
        self.weights = output_weights
        self.bias = output_bias
        self.index = 0
        self.playing = False
        self.figure = plt.figure(figsize=(12, 8))
        layout = self.figure.add_gridspec(4, 2, left=0.08, right=0.96, bottom=0.20,
                                        top=0.88, width_ratios=(1, 1.5), hspace=0.8)
        self.board_axis = self.figure.add_subplot(layout[:, 0])
        self.board_image = self.board_axis.imshow(frames[0]["board"], vmin=0, vmax=3,
                                                  cmap=ListedColormap(["#f1f5f9", "#4caf50", "#166534", "#e53935"]))
        self.collision_marker, = self.board_axis.plot([], [], marker="x", color="#f97316",
                                                      markersize=14, markeredgewidth=3,
                                                      linestyle="none", clip_on=False)
        self.board_axis.set_xticks([])
        self.board_axis.set_yticks([])
        self.layer_images = {}
        for row, name in enumerate(("fc1", "fc2")):
            axis = self.figure.add_subplot(layout[row, 1])
            maximum = max(max(frame["activations"][name]) for frame in frames)
            values = np.asarray(frames[0]["activations"][name])[None, :]
            self.layer_images[name] = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=max(maximum, 1e-6))
            self.figure.colorbar(self.layer_images[name], ax=axis, fraction=0.025, pad=0.02)
            axis.set_title(f"{name} activations after ReLU")
            axis.set_yticks([])
            axis.set_xlabel("Neuron index")
        q_axis = self.figure.add_subplot(layout[2, 1])
        self.q_bars = q_axis.bar(ACTION_NAMES, frames[0]["activations"]["q"], color="#64748b")
        all_q = np.asarray([f["activations"]["q"] for f in frames])
        low, high = min(0, all_q.min()), max(0, all_q.max())
        margin = max((high - low) * 0.2, 0.1)
        q_axis.set_ylim(low - margin, high + margin)
        q_axis.axhline(0, color="#64748b", linewidth=0.5)
        self.q_title = q_axis.set_title("Q-values for this move (scaled reward units)")
        self.q_labels = [q_axis.text(i, 0, "", ha="center", fontsize=9) for i in range(3)]
        axis = self.figure.add_subplot(layout[3, 1])
        contributions = [self.weights * np.asarray(f["activations"]["fc2"])[None, :] for f in frames]
        limit = max(float(np.abs(contributions).max()), 1e-6)
        self.contribution_image = axis.imshow(contributions[0], aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        self.figure.colorbar(self.contribution_image, ax=axis, fraction=0.025, pad=0.02)
        axis.set_yticks(range(3), ACTION_NAMES)
        axis.set_xlabel("fc2 neuron index")
        axis.set_title("Output contribution = weight × fc2 activation")
        self.figure.text(0.08, 0.13, "Green body · dark head · red apple. Output Q = row sum of contributions + bias.", fontsize=10)
        self.slider = Slider(self.figure.add_axes([0.16, 0.075, 0.72, 0.025]), "Frame", 0, len(frames) - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.draw)
        self.play_button = Button(self.figure.add_axes([0.08, 0.015, 0.15, 0.04]), "Play / Pause")
        self.previous_button = Button(self.figure.add_axes([0.25, 0.015, 0.15, 0.04]), "Previous")
        self.next_button = Button(self.figure.add_axes([0.42, 0.015, 0.15, 0.04]), "Next")
        self.end_button = Button(self.figure.add_axes([0.59, 0.015, 0.15, 0.04]), "Ending")
        self.play_button.on_clicked(self.toggle_play)
        self.previous_button.on_clicked(lambda event: self.seek(self.index - 1))
        self.next_button.on_clicked(lambda event: self.seek(self.index + 1))
        self.end_button.on_clicked(lambda event: self.seek(len(frames) - 1))
        self.timer = self.figure.canvas.new_timer(interval=interval)
        self.timer.add_callback(self.tick)
        self.draw(0)

    def draw(self, index):
        self.index = int(index)
        frame = self.frames[self.index]
        activations = frame["activations"]
        self.board_image.set_data(frame["board"])
        collision = frame.get("collision")
        self.collision_marker.set_visible(collision is not None)
        if collision is not None:
            height, width = frame["board"].shape
            row, col = collision
            self.collision_marker.set_data([np.clip(col, -0.5, width - 0.5)],
                                           [np.clip(row, -0.5, height - 0.5)])
        for name, image in self.layer_images.items():
            image.set_data(np.asarray(activations[name])[None, :])
        for action, (bar, label, value) in enumerate(zip(self.q_bars, self.q_labels, activations["q"])):
            bar.set_height(value)
            bar.set_color("#2563eb" if action == frame["action"] else "#64748b")
            label.set_text(f"{value:.3f}")
            label.set_position((action, value))
            label.set_verticalalignment("bottom" if value >= 0 else "top")
        self.contribution_image.set_data(self.weights * np.asarray(activations["fc2"])[None, :])
        action_text = ACTION_NAMES[frame["action"]]
        if frame["reason"]:
            action_text = f"last move {action_text}"
        self.q_title.set_text("Q-values for the last move (scaled reward units)" if frame["reason"]
                              else "Q-values for this move (scaled reward units)")
        self.board_axis.set_title(f"Step {frame['steps']} · apples {frame['score']}\n"
                                  f"Return {frame['return']:.2f} · {action_text}")
        biases = ", ".join(f"{value:.3f}" for value in self.bias)
        self.figure.suptitle(f"DQN replay · {frame['reason'] or 'playing'}\n"
                             f"Output biases [{biases}]")
        self.figure.canvas.draw_idle()

    def seek(self, index):
        self.slider.set_val(int(np.clip(index, 0, len(self.frames) - 1)))

    def toggle_play(self, event=None):
        self.playing = not self.playing
        if self.playing:
            if self.index == len(self.frames) - 1:
                self.seek(0)
            self.timer.start()
        else:
            self.timer.stop()

    def tick(self):
        if self.index == len(self.frames) - 1:
            self.playing = False
            self.timer.stop()
        else:
            self.seek(self.index + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interval", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--gif", help="Export an animated replay")
    parser.add_argument("--snapshot", help="Export the final frame")
    args = parser.parse_args()
    if args.max_steps < 1 or args.interval < 1:
        parser.error("max-steps and interval must be positive")
    torch.set_num_threads(1)
    agent, metadata = DQNAgent.load(args.checkpoint, device=args.device)
    frames = record_game(agent, metadata, seed=args.seed, max_steps=args.max_steps)
    weights = agent.online.output.weight.detach().cpu().numpy()
    bias = agent.online.output.bias.detach().cpu().numpy()
    viewer = ReplayViewer(frames, weights, bias, interval=args.interval)
    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        movie = animation.FuncAnimation(viewer.figure, viewer.seek, frames=len(frames),
                                         interval=args.interval, repeat=False, init_func=lambda: [])
        movie.save(args.gif, writer="pillow", fps=1000 / args.interval)
    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        viewer.seek(len(frames) - 1)
        viewer.figure.savefig(args.snapshot)
    if not args.gif and not args.snapshot:
        viewer.toggle_play()
        plt.show()
    plt.close(viewer.figure)
    print(f"Replay ended with {frames[-1]['score']} apples: {frames[-1]['reason']}")


if __name__ == "__main__":
    main()
