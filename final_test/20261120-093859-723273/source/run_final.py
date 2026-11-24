"""Retrain saved DQN configurations, then evaluate the resulting final checkpoints."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def save_manifest(folder, manifest):
    temporary = folder / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(folder / "manifest.json")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_arguments(config, episodes=None):
    model, encoder, settings = config["model"], config["encoder"], config["settings"]
    epsilon = config["epsilon"]
    arguments = ["--agent", "dqn", "--model", model["name"], "--size", str(config["board_size"]),
                 "--encoder", encoder["name"], "--encoding", encoder["encoding"],
                 "--base-channels", str(model["base_channels"]),
                 "--hidden-dims", *map(str, model["hidden_dims"]),
                 "--episodes", str(episodes or config["episodes"]),
                 "--seed", str(settings["seed"]),
                 "--eps-start", str(epsilon["start"]), "--eps-end", str(epsilon["end"]),
                 "--eps-decay-frac", str(epsilon["linear_decay_fraction"]),
                 "--rewards", ",".join(f"{name}={weight}" for name, weight in config["rewards"].items())]
    for option, key in (("learning-rate", "learning_rate"), ("gamma", "gamma"),
                        ("batch-size", "batch_size"), ("replay-capacity", "capacity"),
                        ("warmup", "warmup"), ("train-every", "train_every"),
                        ("target-every", "target_every"), ("reward-scale", "reward_scale")):
        arguments += [f"--{option}", str(settings[key])]
    return arguments


def create_plan(args, project):
    for name in ("train.py", "evaluate_final.py"):
        if not (project / name).is_file():
            raise FileNotFoundError(f"Required script is missing: {project / name}")
    paths = args.configs or sorted((project / "runs").glob("*/model.json"))
    configs = [(path.resolve(), json.loads(path.read_text())) for path in paths]
    if not configs:
        raise ValueError("No saved runs found. Use --configs to select run JSON files.")
    # Resolve every training-data dependency before launching any jobs.
    for path, config in configs:
        training_arguments(config, args.episodes)
        if config.get("initial_length", 3) != 3 or config.get("timeout_factor", 2) != 2:
            raise ValueError(f"{path}: training CLI cannot reproduce this environment configuration")
        for key in ("pretraining", "late_starts"):
            if key in config:
                source = args.late_start_file if key == "late_starts" and args.late_start_file else Path(config[key]["source"])
                source = source if source.is_absolute() else project / source
                if not source.is_file():
                    raise FileNotFoundError(f"Required {key} data is missing: {source}")

    folder = (args.out or project / "final_test" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source_dir, inputs_dir = folder / "source", folder / "inputs"
    source_dir.mkdir()
    inputs_dir.mkdir()
    files = sorted(project.glob("*.py")) + sorted(project.glob("requirements*.txt"))
    for path in files:
        shutil.copy2(path, source_dir / path.name)
    # Include this exact orchestrator when planning from a staging location.
    shutil.copy2(Path(__file__), source_dir / "run_final.py")
    manifest = {"created_at": datetime.now().isoformat(), "project": str(project),
                "python": sys.executable, "platform": platform.platform(),
                "training_device": args.device, "evaluation_device": args.eval_device,
                "torch_threads": args.torch_threads, "training_eval_games": args.train_eval_games,
                "evaluation_games": args.games, "evaluation_seed_start": args.seed_start,
                "masking": args.masking, "episode_override": args.episodes,
                "late_start_file_override": str(args.late_start_file) if args.late_start_file else None,
                "jobs": [], "files_sha256": {}}
    for index, (path, config) in enumerate(configs, 1):
        job_id = f"{index:02d}-{path.parent.name}"
        job_dir = folder / "models" / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "original_config.json").write_text(json.dumps(config, indent=2))
        arguments = training_arguments(config, args.episodes)
        for key, option in (("pretraining", "--demo-file"), ("late_starts", "--late-start-file")):
            if key not in config:
                continue
            original = args.late_start_file if key == "late_starts" and args.late_start_file else Path(config[key]["source"])
            original = original if original.is_absolute() else project / original
            data_name = f"{fingerprint(original)[:16]}-{original.name}"
            destination = inputs_dir / data_name
            if not destination.exists():
                shutil.copy2(original, destination)
            arguments += [option, str(destination)]
            if key == "pretraining":
                arguments += ["--pretrain-updates", str(config[key]["updates"])]
            else:
                arguments += ["--late-start-min-apples", str(config[key]["min_apples"]),
                              "--late-start-prob", str(config[key]["probability"])]
        arguments += ["--device", args.device, "--torch-threads", str(args.torch_threads),
                      "--eval-games", str(args.train_eval_games)]
        manifest["jobs"].append({"id": job_id, "original_config": str(path),
                                 "arguments": arguments, "status": "pending", "attempts": []})
    for path in list(source_dir.iterdir()) + list(inputs_dir.iterdir()):
        manifest["files_sha256"][str(path.relative_to(folder))] = fingerprint(path)
    save_manifest(folder, manifest)
    # Capture installed package versions for reproducibility.
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (folder / "environment.txt").write_text(f"Python {sys.version}\n{platform.platform()}\n"
                                             + (result.stdout if result.returncode == 0 else result.stderr))
    return folder, manifest


def run_command(command, cwd, log_path):
    print("Running " + " ".join(command), flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    environment.setdefault("MPLCONFIGDIR", str(log_path.parent / "matplotlib-cache"))
    with log_path.open("w") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Process exited with code {code}. See {log_path}")


def execute(folder, manifest):
    for relative, expected in manifest["files_sha256"].items():
        if fingerprint(folder / relative) != expected:
            raise ValueError(f"Saved experiment source/data changed: {relative}")
    source_dir = folder / "source"
    for job in manifest["jobs"]:
        if job["status"] == "complete":
            checkpoint = Path(job["checkpoint"])
            if fingerprint(checkpoint) != job["checkpoint_sha256"]:
                raise ValueError(f"Completed checkpoint changed: {checkpoint}")
            continue
        # Each retry gets a new directory to preserve partial training artifacts.
        attempt = folder / "models" / job["id"] / f"{job['id']}-attempt-{len(job['attempts']) + 1:02d}"
        attempt.mkdir()
        checkpoint = attempt / "model.pt"
        command = [sys.executable, str(source_dir / "train.py"), *job["arguments"], "--out", str(checkpoint)]
        job["attempts"].append({"directory": str(attempt), "command": command})
        job["status"] = "training"
        manifest["evaluation_status"] = "pending"
        save_manifest(folder, manifest)
        try:
            run_command(command, source_dir, attempt / "console.log")
        except (RuntimeError, KeyboardInterrupt):
            job["status"] = "interrupted_or_failed"
            save_manifest(folder, manifest)
            raise
        config = json.loads(checkpoint.with_suffix(".json").read_text())
        expected_episodes = int(job["arguments"][job["arguments"].index("--episodes") + 1])
        if config.get("completed_episodes") != expected_episodes:
            raise RuntimeError(f"Training did not reach {expected_episodes} episodes: {job['id']}")
        job.update(status="complete", checkpoint=str(checkpoint), checkpoint_sha256=fingerprint(checkpoint))
        save_manifest(folder, manifest)

    if (manifest.get("evaluation_status") == "complete"
            and (Path(manifest["results"]) / "report.txt").is_file()):
        print(f"Already complete. Results are in {manifest['results']}")
        return
    results = folder / f"results-{datetime.now():%Y%m%d-%H%M%S-%f}"
    command = [sys.executable, str(source_dir / "evaluate_final.py"), "--checkpoints",
               *[job["checkpoint"] for job in manifest["jobs"]],
               "--games", str(manifest["evaluation_games"]), "--seed-start", str(manifest["evaluation_seed_start"]),
               "--masking", manifest["masking"], "--device", manifest["evaluation_device"],
               "--torch-threads", str(manifest["torch_threads"]), "--out", str(results)]
    manifest.update(evaluation_status="running", results=str(results), evaluation_command=command)
    save_manifest(folder, manifest)
    try:
        run_command(command, source_dir, folder / f"{results.name}.log")
    except (RuntimeError, KeyboardInterrupt):
        manifest["evaluation_status"] = "interrupted_or_failed"
        save_manifest(folder, manifest)
        raise
    manifest["evaluation_status"] = "complete"
    save_manifest(folder, manifest)
    print(f"Final models and results saved in {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--configs", nargs="+", type=Path, help="Original run JSON files; default all runs/*/model.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--episodes", type=int, help="Override every run's original planned episode count")
    parser.add_argument("--late-start-file", type=Path,
                        help="Replace the dataset for configurations that already use crowded starts")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. mps")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--masking", choices=("none", "safe", "both"), default="both")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--train-eval-games", type=int, default=100)
    parser.add_argument("--plan-only", action="store_true", help="Save the experiment plan without starting training")
    parser.add_argument("--resume", type=Path, help="Continue a saved plan; restart incomplete jobs from scratch")
    args = parser.parse_args()
    if (min(args.games, args.torch_threads, args.train_eval_games) < 1 or args.seed_start < 0
            or (args.episodes is not None and args.episodes < 1)):
        parser.error("Episode/game/thread counts must be positive, and seed-start nonnegative")
    if args.resume:
        folder = args.resume.resolve()
        manifest = json.loads((folder / "manifest.json").read_text())
    else:
        folder, manifest = create_plan(args, args.project.resolve())
    print(f"Experiment folder: {folder}\nTraining jobs: {len(manifest['jobs'])}", flush=True)
    if not args.plan_only:
        execute(folder, manifest)


if __name__ == "__main__":
    main()
