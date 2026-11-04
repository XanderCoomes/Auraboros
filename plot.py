"""Plot learning curves from log.csv and optionally save an annotated GIF + metadata JSON of the trained agent.

Usage:  python plot.py [--log log.csv] [--q q_table.pkl --size 8 --outdir test_result]
Outputs in test_result/train_<trained>_test_<tested>/: curves.png, game.gif (per-frame overlay), game.json (method, state, config, result, tested_at),
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


def wrap_state(desc, width=70):
    """'name=value' pairs joined into lines of at most ~width characters."""
    lines, cur = [], "state:"
    for k, v in desc.items():
        item = f" {k}={v}"
        if len(cur) + len(item) > width:
            lines.append(cur)
            cur = "      "
        cur += item
    return "\n".join(lines + [cur])


def load_train_cfg(q_or_log_path):
    """q_table.json written by train.py next to the given .pkl, or None."""
    if not q_or_log_path:
        return None
    path = q_or_log_path.rsplit(".", 1)[0] + ".json"
    return json.load(open(path)) if os.path.exists(path) else None


def train_header(cfg):
    """One-line description of the training run for plot titles."""
    if not cfg:
        return "trained: unknown (no q_table.json)"
    when = cfg["trained_at"].replace("T", " ")
    st = cfg.get("state", {})
    return (f"trained {when}  |  {cfg['episodes']} eps, {cfg['board_size']}x{cfg['board_size']}, "
            f"{st.get('num_states', '?')} states  |  greedy {cfg['final_greedy_eval_score_100_games']:.2f}")


def train_summary(cfg):
    """Key training signals, put at the top of game.json so they are not missed."""
    if not cfg:
        return None
    keys = ["trained_at", "training_started_at", "training_duration_seconds", "command",
            "final_greedy_eval", "state_coverage", "training_end_reasons",
            "training_last_1000_episodes", "training_max_score", "eval_history"]
    out = {k: cfg[k] for k in keys if k in cfg}
    out["features"] = cfg.get("state", {}).get("features")
    return out


def reward_spec_from_cfg(cfg):
    """RewardSpec used in training, from q_table.json; old configs used fixed constants with different names."""
    from rewards import RewardSpec
    w = (cfg or {}).get("rewards")
    if not w:
        return RewardSpec()
    old_names = {"closer_or_farther": "closer"}
    return RewardSpec({old_names.get(k, k): v for k, v in w.items()})


def moving_avg(x, k):
    k = min(k, len(x))
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_curves(path, window, out, cfg=None):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    score = np.array([float(r["score"]) for r in rows])
    steps = np.array([float(r["steps"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(moving_avg(score, window)); ax[0].set_title(f"Score (moving avg {window})")
    ax[1].plot(moving_avg(steps, window)); ax[1].set_title(f"Episode steps (moving avg {window})")
    for a in ax:
        a.set_xlabel("episode")
    feats = ", ".join(cfg["state"]["features"]) if cfg and "state" in cfg else "unknown"
    fig.suptitle(f"{train_header(cfg)}\nfeatures: {feats}", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


def eval_greedy(agent, size, rewards=None, n=100, seed=1000):
    from snake_env import SnakeEnv
    env = SnakeEnv(size=size, seed=seed, features=agent.spec.names, rewards=rewards)
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
    from q_tabular import QAgent
    from snake_env import SnakeEnv

    agent = QAgent.load(q_path)  # also restores which state features the table was trained on
    spec = agent.spec
    train_cfg = load_train_cfg(q_path)

    # Replay with the reward terms/weights used in training (old configs: the default reward function)
    rewards = reward_spec_from_cfg(train_cfg)
    env = SnakeEnv(size=size, seed=seed, features=spec.names, rewards=rewards)
    reward_cfg = rewards.weights

    s, done, frames, log = env.reset(), False, [], []
    info = {"score": 0}
    total = 0.0  # cumulative reward (return) so far
    while not done and len(log) < 500:
        a = agent.act(s, 0.0)
        frames.append(env.board_array())
        entry = {"step": env.steps, "score": env.score, "return": round(total, 3), "reward": None,
                 "state": list(s), "state_named": spec.describe(s), "heading": HEADING_NAMES[env.heading],
                 "action": ACTION_NAMES[a],
                 "q_values": [round(float(v), 3) for v in agent.q[spec.row(s)]]}
        s, r, done, info = env.step(a)
        total += r
        # reward received for this step; score/return shown on the NEXT frame include it
        entry["reward"] = round(r, 3)
        entry["reward_terms"] = {k: round(v, 3) for k, v in rewards.breakdown(env, info["event"]).items() if v}
        log.append(entry)
    frames.append(env.board_array())
    log.append({"step": env.steps, "score": env.score, "return": round(total, 3), "reward": None,
                "state": list(s), "state_named": spec.describe(s), "heading": HEADING_NAMES[env.heading],
                "action": None,
                "q_values": [round(float(v), 3) for v in agent.q[spec.row(s)]]})
    reason = info.get("reason", "max_frames" if not done else "unknown")

    cmap = ListedColormap(["white", "#4caf50", "#1b5e20", "#e53935"])
    fig, ax = plt.subplots(figsize=(6, 6.6))
    ax.set_xticks([])  # hide ticks but keep the frame so the board edge is visible
    ax.set_yticks([])
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=3)
    title = ax.set_title("", fontsize=7, family="monospace", loc="left")
    reward_line = f"Reward fn: {rewards}"

    def draw(i):
        e = log[i]
        prev = log[i - 1]["reward"] if i > 0 else None  # reward received by the step that led to this frame
        title.set_text(
            f"{train_header(train_cfg)}\n"
            f"step {e['step']}  score(apples) {e['score']}  return {e['return']:.2f}"
            + (f"  last reward {prev:+.2f}" if prev is not None else "") + "\n"
            f"action {e['action']}  heading {e['heading']}\n"
            f"{wrap_state(e['state_named'])}\n"
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
        "trained_at": train_cfg["trained_at"] if train_cfg else None,
        "features": spec.names,
        "training_summary": train_summary(train_cfg),
        "gif_game": {"play_seed": seed, "final_score": info["score"], "final_return": round(total, 3),
                     "total_steps": env.steps, "end_reason": reason},
        "greedy_eval": eval_greedy(agent, size, rewards),
        "reward_function_used_in_gif": reward_cfg,
        "gif_file": out,
        "method": {
            "algorithm": "tabular Q-learning",
            "update": "Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)); target = r if terminal",
            "action_selection": "epsilon-greedy while training; greedy (epsilon=0) when playing this GIF",
        },
        "state": {
            "features": {f.name: {"n_values": f.n_values, "labels": f.labels, "doc": f.doc} for f in spec.features},
            "shape": list(spec.shape),
            "num_states": spec.num_states,
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
    cfg = load_train_cfg(a.q)
    # One folder per run, named by when the model was TRAINED (then when it was tested), e.g.
    # test_result/train_20261102_202627_test_20261102_211333/ (sorts chronologically by training)
    trained = (datetime.fromisoformat(cfg["trained_at"]).strftime("%Y%m%d_%H%M%S") if cfg else "unknown")
    run_dir = os.path.join(a.outdir, f"train_{trained}_test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    plot_curves(a.log, a.window, os.path.join(run_dir, "curves.png"), cfg)
    if a.q:
        save_gif(a.q, a.size, os.path.join(run_dir, "game.gif"))
    # Keep a copy of the training artifacts with the results, since the next training run overwrites them
    q_cfg = a.q.rsplit(".", 1)[0] + ".json" if a.q else None
    for f in (a.log, a.q, q_cfg):
        if f and os.path.exists(f):
            shutil.copy2(f, run_dir)
    print("copied training files to", run_dir)
