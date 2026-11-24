"""Record step-by-step human games and replay them for DQN pretraining."""
import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from snake_env import LEFT, N_ACTIONS, RIGHT, STRAIGHT, SnakeEnv
from state_encoders import BoardSnakeEnv


def _replay_moves(path, size, encoder, rewards):
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                game = json.loads(line)
                if not isinstance(game, dict) or game.get("version") != 1:
                    raise ValueError("unsupported game format")
                if game.get("size") != size:
                    raise ValueError(f"recorded board size must match --size {size}")
                seed, actions = game["seed"], game["actions"]
                if type(seed) is not int or not isinstance(actions, list) or not actions:
                    raise ValueError("expected a seed and a nonempty action list")
                env = BoardSnakeEnv(size=size, seed=seed, encoder=encoder, rewards=rewards)
                state = env.get_state()
                for action in actions:
                    if type(action) is not int or not 0 <= action < N_ACTIONS or env.done:
                        raise ValueError("invalid action or move after game over")
                    next_state, reward, done, info = env.step(action)
                    yield env, (state, action, reward, next_state, done), info
                    state = next_state
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def load_demonstrations(path, size, encoder, rewards):
    """Reconstruct transitions using the current training encoder and rewards."""
    transitions = [transition for _, transition, _ in _replay_moves(path, size, encoder, rewards)]
    if not transitions:
        raise ValueError(f"No recorded moves in {path}")
    return transitions


@dataclass(frozen=True)
class LateStart:
    snake: tuple
    heading: int
    apple: tuple
    score: int

    def restore(self, env):
        env.snake = deque(self.snake)
        env.body = set(self.snake)
        env.heading = self.heading
        env.apple = self.apple
        env.score = self.score
        # Snapshots are taken right after eating. New apples use the training RNG.
        env.steps_since_apple = 0
        env.steps = 0
        env.done = False
        return env.get_state()


def load_late_starts(path, size, encoder, rewards, min_apples):
    starts = []
    for env, _, info in _replay_moves(path, size, encoder, rewards):
        if info["event"]["outcome"] == "apple" and env.score >= min_apples:
            starts.append(LateStart(tuple(env.snake), env.heading, env.apple, env.score))
    if not starts:
        raise ValueError(f"No playable positions at {min_apples} or more apples in {path}")
    return starts


class HumanRecorder:
    def __init__(self, path, size, seed=None):
        self.path = Path(path)
        self.size = size
        self.rng = random.Random(seed)
        self.actions = []
        self.restart()

    def restart(self):
        self.save()
        self.seed = self.rng.randrange(2**32)
        self.env = SnakeEnv(size=self.size, seed=self.seed)

    def move(self, action):
        _, _, done, info = self.env.step(action)
        self.actions.append(action)
        if done:
            self.save()
        return info.get("reason")

    def save(self):
        if not self.actions:
            return
        game = {"version": 1, "size": self.size, "seed": self.seed, "actions": self.actions}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(game) + "\n")
        print(f"Saved {len(self.actions)} moves ({self.env.score} apples) to {self.path}", flush=True)
        self.actions.clear()


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demos/human.jsonl"))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.size < 4:
        parser.error("--size must be at least 4")
    recorder = HumanRecorder(args.out, args.size, args.seed)
    figure, axis = plt.subplots(figsize=(6, 6))
    board = axis.imshow(recorder.env.board_array(), vmin=0, vmax=3,
                        cmap=ListedColormap(["#f1f5f9", "#4caf50", "#166534", "#e53935"]))
    axis.set_xticks([])
    axis.set_yticks([])
    figure.text(0.5, 0.04, "Arrow keys move  ·  Space goes straight  ·  R restarts  ·  Q saves and closes",
                ha="center", fontsize=9)
    figure.subplots_adjust(bottom=0.1, top=0.85)
    headings = ("up", "right", "down", "left")
    relative_actions = {0: STRAIGHT, 1: RIGHT, 3: LEFT}

    def draw(message=""):
        env = recorder.env
        board.set_data(env.board_array())
        axis.set_title(f"Apples {env.score}  ·  Move {env.steps}  ·  Facing {headings[env.heading]}\n{message}")
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            recorder.save()
            plt.close(figure)
        elif event.key == "r":
            recorder.restart()
            draw()
        elif not recorder.env.done and event.key in (*headings, " "):
            direction = recorder.env.heading if event.key == " " else headings.index(event.key)
            turn = (direction - recorder.env.heading) % 4
            if turn == 2:
                draw("The snake cannot reverse direction")
                return
            reason = recorder.move(relative_actions[turn])
            draw(f"{reason.title()} · Press R to play again" if reason else "")

    # Give Q to the recorder so saving comes before closing the window.
    figure.canvas.mpl_disconnect(figure.canvas.manager.key_press_handler_id)
    figure.canvas.mpl_connect("key_press_event", on_key)
    figure.canvas.mpl_connect("close_event", lambda event: recorder.save())
    draw()
    plt.show()


if __name__ == "__main__":
    main()
