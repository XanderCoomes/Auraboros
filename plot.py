"""Plot learning curves from log.csv and optionally save an annotated GIF + metadata JSON of the trained agent.

Usage:  python plot.py [--log log.csv] [--q q_table.pkl --size 8 --outdir test_result]
Outputs in test_result/train_<timestamp>/: curves.png, game.gif (per-frame overlay), game.json (method, state, config, result, tested_at),
plus copies of the training files (log.csv, q_table.pkl, q_table.json)
"""
import argparse
import csv
import json
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

ACTION_NAMES = ["STRAIGHT", "LEFT", "RIGHT"]
HEADING_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def moving_avg(x, k):
    k = min(k, len(x))
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_curves(path, window, out):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    score = np.array([float(r["score"]) for r in rows])
    steps = np.array([float(r["steps"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(moving_avg(score, window)); ax[0].set_title(f"Score (moving avg {window})")
    ax[1].plot(moving_avg(steps, window)); ax[1].set_title(f"Episode steps (moving avg {window})")
    for a in ax:
        a.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


def eval_greedy(agent, size, n=100, seed=1000):
    from snake_env import SnakeEnv
    env = SnakeEnv(size=size, seed=seed)
    scores = []
    for _ in range(n):
        s, done = env.reset(), False
        while not done:
            s, _, done, info = env.step(agent.act(s, 0.0))
        scores.append(info["score"])
    return {"games": n, "mean_score": float(np.mean(scores)), "max_score": int(max(scores))}


def save_gif(q_path, size, out, seed=1):
    from matplotlib import animation
    from matplotlib.colors import ListedColormap
    import snake_env
    from q_tabular import QAgent, get_row_by_state
    from snake_env import SnakeEnv

    agent = QAgent()
    agent.load(q_path)
    cfg_path = q_path.rsplit(".", 1)[0] + ".json"
    train_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None

    env = SnakeEnv(size=size, seed=seed)
    # Reward function constants actually used by the environment for this GIF game
    reward_cfg = {"apple": snake_env.R_APPLE, "death": snake_env.R_DEATH,
                  "step": snake_env.R_STEP, "closer_or_farther": snake_env.R_CLOSER}

    s, done, frames, log = env.reset(), False, [], []
    info = {"score": 0}
    total = 0.0  # cumulative reward (return) so far
    while not done and len(log) < 500:
        a = agent.act(s, 0.0)
        frames.append(env.board_array())
        entry = {"step": env.steps, "score": env.score, "return": round(total, 3), "reward": None,
                 "state": list(s), "action": ACTION_NAMES[a],
                 "q_values": [round(float(v), 3) for v in agent.q[get_row_by_state(s)]]}
        s, r, done, info = env.step(a)
        total += r
        # reward received for this step; score/return shown on the NEXT frame include it
        entry["reward"] = round(r, 3)
        log.append(entry)
    frames.append(env.board_array())
    log.append({"step": env.steps, "score": env.score, "return": round(total, 3), "reward": None,
                "state": list(s), "action": None,
                "q_values": [round(float(v), 3) for v in agent.q[get_row_by_state(s)]]})
    reason = info.get("reason", "max_frames" if not done else "unknown")

    cmap = ListedColormap(["white", "#4caf50", "#1b5e20", "#e53935"])
    fig, ax = plt.subplots(figsize=(5.5, 6.2))
    ax.set_xticks([])  # hide ticks but keep the frame so the board edge is visible
    ax.set_yticks([])
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=3)
    title = ax.set_title("", fontsize=7, family="monospace", loc="left")
    reward_line = (f"Reward fn: apple {reward_cfg['apple']:+g}, death {reward_cfg['death']:+g}, "
                   f"step {reward_cfg['step']:+g}, closer/farther {reward_cfg['closer_or_farther']:+g}/"
                   f"{-reward_cfg['closer_or_farther']:g}")

    def draw(i):
        e = log[i]
        st = e["state"]
        prev = log[i - 1]["reward"] if i > 0 else None  # reward received by the step that led to this frame
        title.set_text(
            f"step {e['step']}  score(apples) {e['score']}  return {e['return']:.2f}"
            + (f"  last reward {prev:+.2f}" if prev is not None else "") + "\n"
            f"action {e['action']}  heading {HEADING_NAMES[st[5]]}\n"
            f"state: danger S/L/R={st[0]}{st[1]}{st[2]}  apple rowSgn={st[3]} colSgn={st[4]}\n"
            f"Q [S, L, R] = {e['q_values']}\n"
            f"{reward_line}" + (f"\nEND: {reason}" if i == len(log) - 1 else ""))
        im.set_data(frames[i])
        return [im, title]

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=150)
    ani.save(out, writer="pillow")
    print(f"saved {out} (final score {info['score']}, end: {reason})")

    meta = {
        # Results first, so they are the first thing seen when opening the file
        "tested_at": datetime.now().isoformat(timespec="seconds"),
        "gif_game": {"play_seed": seed, "final_score": info["score"], "final_return": round(total, 3),
                     "total_steps": env.steps, "end_reason": reason},
        "greedy_eval": eval_greedy(agent, size),
        "reward_function_used_in_gif": reward_cfg,
        "gif_file": out,
        "method": {
            "algorithm": "tabular Q-learning",
            "update": "Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)); target = r if terminal",
            "action_selection": "epsilon-greedy while training; greedy (epsilon=0) when playing this GIF",
        },
        "state": {
            "tuple": ["danger_straight", "danger_left", "danger_right",
                      "apple_row_sign(0 up/1 same/2 down)", "apple_col_sign(0 left/1 same/2 right)", "heading(0 up,1 right,2 down,3 left)"],
            "num_states": 288,
            "actions": ACTION_NAMES,
        },
        "training_config": train_cfg,
        "environment": {"board_size": size, "initial_length": 3, "timeout_steps_without_apple": 2 * size * size},
        "files": {"q_table": q_path, "training_log": train_cfg["log_file"] if train_cfg else None},
        "per_step_log": log,
    }
    meta_path = out.rsplit(".", 1)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", meta_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="log.csv")
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--q", default=None)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--outdir", default="test_result")
    a = p.parse_args()
    # One timestamped folder per run, e.g. test_result/train_20261026_090331/ (sorts chronologically)
    run_dir = os.path.join(a.outdir, "train_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    plot_curves(a.log, a.window, os.path.join(run_dir, "curves.png"))
    if a.q:
        save_gif(a.q, a.size, os.path.join(run_dir, "game.gif"))
    # Keep a copy of the training artifacts with the results, since the next training run overwrites them
    q_cfg = a.q.rsplit(".", 1)[0] + ".json" if a.q else None
    for f in (a.log, a.q, q_cfg):
        if f and os.path.exists(f):
            shutil.copy2(f, run_dir)
    print("copied training files to", run_dir)
