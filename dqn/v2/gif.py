"""Gameplay GIFs for any policy (see baselines.build_policy for the specs).

Frames are drawn with Pillow directly (one small image per step), which is much faster than a matplotlib
animation and gives crisp pixels. The body is shaded from head to tail so the direction of travel is
visible in a still frame.

Usage (from the repo root):
    python -m dqn.v2.gif --policy v2shield:dqn/v2/runs/seed0/best.pth --seed 10003 --out dqn/v2/gifs/v2.gif
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from snake_env import SnakeEnv

from .baselines import build_policy

BG, GRID, APPLE, HEAD = (24, 24, 32), (40, 40, 52), (230, 70, 70), (250, 220, 90)
BODY_NEAR, BODY_FAR = (80, 200, 120), (30, 90, 70)


def _mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def draw_frame(env, cell=32, caption=""):
    s = env.size
    top = 22
    img = Image.new("RGB", (s * cell, s * cell + top), BG)
    d = ImageDraw.Draw(img)
    for i in range(s + 1):
        d.line([(i * cell, top), (i * cell, top + s * cell)], fill=GRID)
        d.line([(0, top + i * cell), (s * cell, top + i * cell)], fill=GRID)
    box = lambda r, c, pad: [c * cell + pad, top + r * cell + pad, (c + 1) * cell - pad, top + (r + 1) * cell - pad]
    if env.apple is not None:
        d.ellipse(box(*env.apple, 5), fill=APPLE)
    n = len(env.snake)
    for i, (r, c) in enumerate(env.snake):
        color = HEAD if i == 0 else _mix(BODY_NEAR, BODY_FAR, i / max(1, n - 1))
        d.rectangle(box(r, c, 2), fill=color)
    d.text((4, 4), caption or f"score {env.score}  steps {env.steps}", fill=(220, 220, 220))
    return img


def record(policy, seed, size=8, max_steps=5000):
    """Play one greedy game -> (frames, final info)."""
    env = SnakeEnv(size=size, seed=seed)
    policy.reset(seed)
    frames, done, info = [draw_frame(env)], False, {}
    while not done and env.steps < max_steps:
        _, _, done, info = env.step(int(policy.act_batch([env])[0]))
        frames.append(draw_frame(env, caption=f"{policy.name}  score {env.score}  steps {env.steps}"
                                 + (f"  [{info.get('reason')}]" if done else "")))
    return frames, info


def save_gif(frames, path, fps=15, hold_last=2.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ms = int(1000 / fps)
    durations = [ms] * (len(frames) - 1) + [int(hold_last * 1000)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    policy = build_policy(args.policy)
    frames, info = record(policy, args.seed, args.size)
    out = args.out or f"dqn/v2/gifs/{policy.name}_seed{args.seed}.gif"
    save_gif(frames, out, args.fps)
    print(f"{policy.name}: score {info.get('score')} in {len(frames) - 1} steps ({info.get('reason')}) -> {out}")


if __name__ == "__main__":
    main()
