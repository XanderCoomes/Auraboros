"""Evaluate saved DQN runs and export final-report statistics and plots."""
import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from q_dqn import DQNAgent
from rewards import RewardSpec
from state_encoders import BoardEncoder, BoardSnakeEnv


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_game(agent, metadata, seed, masking):
    env = BoardSnakeEnv(size=metadata["board_size"], seed=seed,
                        encoder=BoardEncoder(**metadata["encoder"]),
                        rewards=RewardSpec(metadata["rewards"]),
                        init_length=metadata.get("initial_length", 3),
                        timeout_factor=metadata.get("timeout_factor", 2))
    # Match the standard evaluation/viewer reset sequence.
    state = env.reset()
    total, apple_steps, last_apple, interventions = 0.0, [], 0, 0
    started = time.perf_counter()
    while True:
        values = agent.q_values(state)
        raw_action = int(np.argmax(values))
        safe = [a for a, turn in enumerate((0, -1, 1))
                if not env._is_danger((env.heading + turn) % 4)]
        action = max(safe, key=lambda a: values[a]) if masking and safe else raw_action
        interventions += int(action != raw_action)
        avoidable = bool(safe)
        state, reward, done, info = env.step(action)
        total += reward
        outcome = info["event"]["outcome"]
        if outcome in ("apple", "win"):
            apple_steps.append(env.steps - last_apple)
            last_apple = env.steps
        if done:
            reason = info["reason"]
            if reason == "collision":
                row, col = info["event"]["head"]
                reason = "body_collision" if 0 <= row < env.size and 0 <= col < env.size else "wall_collision"
            return {"seed": seed, "apples": env.score, "steps": env.steps, "return": total,
                    "end_reason": reason, "avoidable_collision": int(outcome == "collision" and avoidable),
                    "mask_interventions": interventions,
                    "steps_to_collected_apples": sum(apple_steps),
                    "seconds": time.perf_counter() - started}


def summarize(games):
    scores = np.asarray([game["apples"] for game in games])
    total_apples = int(scores.sum())
    # Bootstrap interval describes evaluation-seed uncertainty for this checkpoint.
    rng = np.random.default_rng(0)
    bootstrap = [rng.choice(scores, len(scores), replace=True).mean() for _ in range(2000)]
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    reasons = Counter(game["end_reason"] for game in games)
    return {"games": len(games), "mean_apples": float(scores.mean()),
            "apples_ci95_low": float(low), "apples_ci95_high": float(high),
            "std_apples": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "median_apples": float(np.median(scores)), "min_apples": int(scores.min()),
            "max_apples": int(scores.max()),
            "mean_steps": float(np.mean([g["steps"] for g in games])),
            "steps_per_apple": sum(g["steps"] for g in games) / total_apples if total_apples else None,
            "collection_steps_per_apple": sum(g["steps_to_collected_apples"] for g in games) / total_apples if total_apples else None,
            "mean_return": float(np.mean([g["return"] for g in games])),
            "win_rate": reasons["win"] / len(games),
            "timeout_rate": reasons["timeout"] / len(games),
            "wall_collision_rate": reasons["wall_collision"] / len(games),
            "body_collision_rate": reasons["body_collision"] / len(games),
            "avoidable_collision_rate": sum(g["avoidable_collision"] for g in games) / len(games),
            "mask_interventions": sum(g["mask_interventions"] for g in games),
            "evaluation_seconds": sum(g["seconds"] for g in games)}


def training_data(checkpoint, metadata):
    config_path = checkpoint.with_suffix(".json")
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    # Sidecar settings may have been written after an atomic checkpoint replacement.
    metadata_matches = config.get("completed_episodes") == metadata.get("completed_episodes")
    duration = config.get("training_duration_seconds") if metadata_matches else None
    history = config.get("eval_history", [])
    completed = metadata.get("completed_episodes")
    if completed is not None:
        history = [entry for entry in history if entry["episode"] <= completed]
    training = {"checkpoint_metadata": metadata, "sidecar_config": config,
                "sidecar_matches_checkpoint_episode": metadata_matches,
                "evaluation_history": history}
    return duration, training


def plot_results(folder, summaries, details):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['id']} ({row['masking']})" for row in summaries]
    width = max(8, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(width, 5))
    means = np.array([row["mean_apples"] for row in summaries])
    errors = np.array([[row["mean_apples"] - row["apples_ci95_low"] for row in summaries],
                       [row["apples_ci95_high"] - row["mean_apples"] for row in summaries]])
    ax.bar(labels, means, yerr=errors, capsize=3)
    ax.set_ylabel("Mean apples (95% bootstrap interval)")
    ax.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(width, 5))
    ax.boxplot([[g["apples"] for g in item["games"]] for item in details], showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=65)
    ax.set_ylabel("Apples per game")
    fig.tight_layout()
    fig.savefig(folder / "score_distributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(width, 5))
    bottom = np.zeros(len(summaries))
    for reason in ("win", "timeout", "wall_collision", "body_collision"):
        values = np.array([row[f"{reason}_rate"] for row in summaries])
        ax.bar(labels, values, bottom=bottom, label=reason.replace("_", " "))
        bottom += values
    ax.set_ylabel("Fraction of games")
    ax.tick_params(axis="x", rotation=65)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "end_reasons.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = set()
    for row, item in zip(summaries, details):
        if row["id"] in plotted:
            continue
        plotted.add(row["id"])
        history = item["training"]["evaluation_history"]
        if history:
            ax.plot([entry["episode"] for entry in history],
                    [entry.get("mean_score", entry.get("greedy_mean_score")) for entry in history],
                    label=row["id"])
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Historical greedy evaluation mean apples")
    if ax.lines:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = set()
    for row, item in zip(summaries, details):
        if row["id"] in plotted:
            continue
        plotted.add(row["id"])
        history = [entry for entry in item["training"]["evaluation_history"]
                   if entry.get("training_flops") is not None]
        if history:
            ax.plot([entry["training_flops"] for entry in history],
                    [entry.get("mean_score", entry.get("greedy_mean_score")) for entry in history],
                    label=row["id"])
    ax.set_xlabel("Cumulative estimated training FLOPs")
    ax.set_ylabel("Historical greedy evaluation mean apples")
    if ax.lines:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Older runs have no recorded FLOP history", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(folder / "learning_curves_flops.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, help="Default: all runs/*/model.pt")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--masking", choices=("none", "safe", "both"), default="none",
                        help="none matches training evaluation; safe matches the collision-masked viewer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.games < 1 or args.torch_threads < 1 or args.seed_start < 0:
        parser.error("games/torch-threads must be positive; seed-start must be nonnegative")
    checkpoints = sorted(set(args.checkpoints or Path("runs").glob("*/model.pt")))
    if not checkpoints:
        parser.error("No DQN checkpoints found")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
    folder = args.out or Path("final_results") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.torch_threads)
    modes = ("none", "safe") if args.masking == "both" else (args.masking,)
    summaries, details = [], []
    protocol = {"games_per_checkpoint_per_mode": args.games, "seed_start": args.seed_start,
                "seed_end": args.seed_start + args.games - 1, "masking": args.masking,
                "device": args.device, "torch_threads": args.torch_threads,
                "torch_version": torch.__version__, "python": platform.python_version(),
                "evaluation_host": platform.platform(), "initial_states": "normal starts",
                "action_selection": "greedy, first maximum breaks ties", "learning": False}
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for index, checkpoint in enumerate(checkpoints, 1):
        # Read and fingerprint one byte snapshot so live training cannot change it mid-load.
        import tempfile
        payload = checkpoint.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "model.pt"
            snapshot.write_bytes(payload)
            agent, metadata = DQNAgent.load(snapshot, device=args.device)
        del payload
        duration, training = training_data(checkpoint, metadata)
        for mode in modes:
            run_id = f"run_{index:02d}"
            print(f"{run_id}: {checkpoint.parent.name}, masking={mode}", flush=True)
            games = []
            evaluation_flops_before = agent.compute.evaluation
            for offset in range(args.games):
                games.append(evaluate_game(agent, metadata, args.seed_start + offset, mode == "safe"))
                if (offset + 1) % 100 == 0:
                    print(f"  {offset + 1}/{args.games} games", flush=True)
            result = {"id": run_id, "run": checkpoint.parent.name, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": digest, "masking": mode,
                      "model": agent.model_config["name"], "encoder": metadata["encoder"]["name"],
                      "board_size": metadata["board_size"],
                      "parameters": sum(p.numel() for p in agent.online.parameters()),
                      "training_episodes": metadata.get("completed_episodes"),
                      "training_moves": agent.steps, "optimizer_updates": agent.updates,
                      "training_seconds": duration,
                      "training_flops": agent.compute.summary()["training_flops"],
                      "pretraining_flops": agent.compute.pretraining if agent.compute.history_complete else None,
                      "evaluation_flops": agent.compute.evaluation - evaluation_flops_before,
                      "forward_flops_per_state": agent.compute.forward,
                      "flops_convention": agent.compute.convention,
                      "partial_last_episode": metadata.get("partial_last_episode", False),
                      "stop_reason": metadata.get("stop_reason"), **summarize(games)}
            summaries.append(result)
            details.append({"summary": result, "training": training, "games": games})
            write_csv(folder / f"{run_id}_{mode}_games.csv", games)
            write_csv(folder / "summary.csv", summaries)
            (folder / "results.json").write_text(json.dumps({"protocol": protocol, "runs": details}, indent=2))
            print(f"  mean apples {result['mean_apples']:.2f}; max {result['max_apples']}", flush=True)
        del agent
    plot_results(folder, summaries, details)
    lines = ["Final DQN evaluation", "", "ID | Run | Mask | Mean apples | 95% interval | Steps/apple | Win % | Training seconds | Estimated training FLOPs",
             "--- | --- | --- | --- | --- | --- | --- | --- | ---"]
    for row in summaries:
        efficiency = f"{row['steps_per_apple']:.2f}" if row['steps_per_apple'] is not None else "undefined"
        duration = f"{row['training_seconds']:.1f}" if row['training_seconds'] is not None else "unavailable"
        flops = f"{row['training_flops']:.3e}" if row["training_flops"] is not None else "unavailable"
        lines.append(f"{row['id']} | {row['run']} | {row['masking']} | {row['mean_apples']:.2f} | "
                     f"[{row['apples_ci95_low']:.2f}, {row['apples_ci95_high']:.2f}] | "
                     f"{efficiency} | {100 * row['win_rate']:.1f} | {duration} | {flops}")
    lines += ["", "Steps/apple = total game moves / total apples, including moves after the final apple.",
              "collection_steps_per_apple excludes the terminal stretch without an apple.",
              "Intervals cover evaluation-seed variability for a single trained checkpoint, not variability across training seeds.",
              "Each run uses its recorded rewards; returns across different reward functions are not directly comparable.",
              "Historical learning curves use the original evaluation protocol, not this final test's masking or seeds.",
              "Training time comes from matching run metadata and includes original evaluation overhead. Training hardware was not recorded.",
              "FLOPs are estimates using the convention recorded in summary.csv. Historical checkpoints without counters remain unavailable.",
              "Pretraining FLOPs are included in training FLOPs; evaluation FLOPs are separate. Counter profiling is excluded.",
              "A FLOP budget may stop partway through a game; partial_last_episode identifies that final training row.",
              "Comparisons use normal starts, even for models trained using crowded boards.",
              "Training configurations, pretraining, and crowded-start settings are included in results.json."]
    (folder / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"Results saved in {folder.resolve()}")


if __name__ == "__main__":
    main()
