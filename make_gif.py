"""Play one greedy game with a saved Q-table and save a GIF where the body fades from dark (head) to light (tail).

Usage:
  python make_gif.py --q test_result/tail_compare_<ts>/q_tables/q_danger+apple_dir+heading+tail_ep60000_seed5.pkl \
      --features danger,apple_dir,heading,tail --size 8 --play-seed 1 --max-steps 2000 --outdir test_result/gif_tail_seed5
Outputs: game.gif and game.json (result + per-step log) in --outdir.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgb

from feature_library import make_feature_set
from q_tabular import QAgent
from snake_env import SnakeEnv

ACTIONS = ["STRAIGHT", "LEFT", "RIGHT"]
HEAD, TAIL = np.array(to_rgb("#0b3d0b")), np.array(to_rgb("#d7f0d2"))  # darkest green (head) -> lightest (tail)
APPLE, EMPTY = np.array(to_rgb("#e53935")), np.array([1.0, 1.0, 1.0])


def render(snake, apple, size):
    """RGB board image: body colour interpolated from head (dark) to tail (light) by position along the snake."""
    img = np.tile(EMPTY, (size, size, 1))
    if apple is not None:
        img[apple] = APPLE
    n = len(snake)
    for i, cell in enumerate(snake):  # snake is ordered head first
        t = i / max(n - 1, 1)
        img[cell] = HEAD * (1 - t) + TAIL * t
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True)
    p.add_argument("--features", required=True, help="comma-separated feature names used in training")
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--play-seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--interval", type=int, default=80, help="ms per frame")
    p.add_argument("--outdir", required=True)
    a = p.parse_args()

    fs = make_feature_set(a.features)
    agent = QAgent(fs.shape)
    agent.load(a.q)
    env = SnakeEnv(size=a.size, seed=a.play_seed, features=fs, max_steps=a.max_steps)

    s, done, info = env.reset(), False, {"score": 0}
    frames, log = [], []
    while not done:
        act = agent.act(s, 0.0)
        frames.append((list(env.snake), env.apple))
        log.append({"step": env.steps, "score": env.score, "action": ACTIONS[act]})
        s, _, done, info = env.step(act)
    frames.append((list(env.snake), env.apple))
    log.append({"step": env.steps, "score": env.score, "action": None})
    reason = info.get("reason", "unknown")

    os.makedirs(a.outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(render(*frames[0], a.size), interpolation="nearest")
    title = ax.set_title("", fontsize=9, family="monospace", loc="left")

    def draw(i):
        e = log[i]
        title.set_text(f"step {e['step']}  apples {e['score']}  action {e['action']}"
                       + (f"\nEND: {reason}" if i == len(log) - 1 else ""))
        im.set_data(render(*frames[i], a.size))
        return [im, title]

    out = os.path.join(a.outdir, "game.gif")
    animation.FuncAnimation(fig, draw, frames=len(frames), interval=a.interval).save(out, writer="pillow")
    result = {"q_table": a.q, "features": fs.names, "board_size": a.size, "play_seed": a.play_seed,
              "max_steps": a.max_steps, "final_score": info["score"], "total_steps": env.steps,
              "end_reason": reason, "frames": len(frames), "per_step_log": log}
    with open(os.path.join(a.outdir, "game.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved {out}: score {info['score']}, steps {env.steps}, end: {reason}, frames {len(frames)}")


if __name__ == "__main__":
    main()
