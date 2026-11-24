"""DQN training, dispatched by train.py --agent dqn."""
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def add_arguments(parser):
    parser.add_argument("--agent", choices=("tabular", "dqn"), default="tabular")
    parser.add_argument("--encoder", default="entities_tail")
    parser.add_argument("--encoding", choices=("auto", "onehot", "binary"), default="auto")
    parser.add_argument("--model", default="rescnn8", help="rescnn4 or rescnn8, or a registered model")
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[64, 32])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-every", type=int, default=250, help="Optimizer updates between target copies")
    parser.add_argument("--reward-scale", type=float, default=0.01, help="Scale rewards inside DQN only")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--demo-file", type=Path, help="Human games recorded with human_play.py")
    parser.add_argument("--pretrain-updates", type=int, help="Demonstration updates (default 1000 with --demo-file)")
    parser.add_argument("--late-start-file", type=Path, help="Start some training games from recorded crowded boards")
    parser.add_argument("--late-start-min-apples", type=int, default=15)
    parser.add_argument("--late-start-prob", type=float, default=0.2)


def train(args):
    import torch
    from q_dqn import DQNAgent
    from rewards import RewardSpec
    from state_encoders import BoardEncoder, BoardSnakeEnv
    from train import evaluate_details, run_episode

    if args.size < 4:
        raise ValueError("Use size >= 4 for the environment's initial three-segment snake")
    if min(args.episodes, args.eval_games, args.eval_every, args.checkpoint_every, args.torch_threads) < 1:
        raise ValueError("Episode counts, intervals, and thread count must be positive")
    if not 0 <= args.eps_end <= args.eps_start <= 1 or args.eps_decay_frac <= 0:
        raise ValueError("Use 0 <= eps-end <= eps-start <= 1 and a positive decay fraction")
    if args.pretrain_updates is not None and (args.demo_file is None or args.pretrain_updates < 1):
        raise ValueError("--pretrain-updates requires --demo-file and a positive count")
    if args.late_start_file is not None and (args.late_start_min_apples < 1 or not 0 < args.late_start_prob <= 1):
        raise ValueError("Crowded starts require a positive apple threshold and a probability in (0, 1]")
    torch.set_num_threads(args.torch_threads)
    encoder = BoardEncoder(args.encoder, args.encoding)
    reward_spec = RewardSpec.parse(args.rewards)
    if args.demo_file is not None:
        from human_play import load_demonstrations
        demonstrations = load_demonstrations(args.demo_file, args.size, encoder, reward_spec)
    late_starts = []
    if args.late_start_file is not None:
        from human_play import load_late_starts
        late_starts = load_late_starts(args.late_start_file, args.size, encoder, reward_spec,
                                      args.late_start_min_apples)
    env = BoardSnakeEnv(encoder=encoder, size=args.size, seed=args.seed, rewards=reward_spec)
    model_config = dict(name=args.model, input_shape=encoder.shape(args.size),
                        context_dim=len(encoder.context_names), base_channels=args.base_channels,
                        hidden_dims=args.hidden_dims)
    agent = DQNAgent(model_config, learning_rate=args.learning_rate, gamma=args.gamma,
                     batch_size=args.batch_size, capacity=args.replay_capacity, warmup=args.warmup,
                     train_every=args.train_every, target_every=args.target_every,
                     reward_scale=args.reward_scale, seed=args.seed, device=args.device)
    run_dir = Path("runs") / f"{args.model}-{args.encoder}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    checkpoint = Path(args.out) if args.out else run_dir / "model.pt"
    log_path = Path(args.log) if args.log else checkpoint.parent / "log.csv"
    config_path = checkpoint.with_suffix(".json")
    if len({p.resolve() for p in (checkpoint, log_path, config_path)}) != 3:
        raise ValueError("Checkpoint, JSON config, and training log must have distinct paths")
    for path in (checkpoint, log_path, config_path):
        if path.exists():
            raise FileExistsError(f"Choose a fresh output path; already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {"board_size": args.size, "encoder": encoder.config(), "model": model_config,
                "rewards": reward_spec.weights, "reward_scale": args.reward_scale,
                "seed": args.seed, "initial_length": env.init_length, "timeout_factor": 2}
    config = {**metadata, "algorithm": "dqn", "training_started_at": datetime.now().isoformat(),
              "q_file": str(checkpoint), "log_file": str(log_path), "episodes": args.episodes,
              "state": {"features": list(encoder.channel_names), "shape": list(encoder.shape(args.size)),
                        "context": list(encoder.context_names)},
              "settings": agent.settings, "epsilon": {"start": args.eps_start, "end": args.eps_end,
                                                       "linear_decay_fraction": args.eps_decay_frac}}
    print(f"{args.model}: input {encoder.shape(args.size)} + {len(encoder.context_names)} context values")
    print(f"Checkpoint {checkpoint}\nTraining log {log_path}", flush=True)
    started = time.monotonic()
    if late_starts:
        start_rng = random.Random(args.seed)
        late_config = {"source": str(args.late_start_file), "positions": len(late_starts),
                       "min_apples": args.late_start_min_apples, "probability": args.late_start_prob,
                       "episodes": 0}
        metadata["late_starts"] = config["late_starts"] = late_config
        print(f"Loaded {len(late_starts)} crowded starts for {args.late_start_prob:.0%} of training games",
              flush=True)
    if args.demo_file is not None:
        pretrain_updates = args.pretrain_updates or 1000
        print(f"Pretraining on {len(demonstrations)} human moves for {pretrain_updates} updates", flush=True)
        loss = agent.pretrain(demonstrations, pretrain_updates)
        metadata["pretraining"] = {"source": str(args.demo_file), "moves": len(demonstrations),
                                   "updates": pretrain_updates, "last_loss": loss}
        config["pretraining"] = metadata["pretraining"]
        del demonstrations
        print(f"Pretraining complete (loss {loss:.4f}). Starting agent games.", flush=True)
    decay_episodes = max(1, int(args.episodes * args.eps_decay_frac))
    history, scores = [], []
    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        start_columns = ("start_apples",) if late_starts else ()
        writer.writerow(("episode", "epsilon", "score", "steps", "return", "length", "end_reason",
                         "loss", "updates", "replay_size") + start_columns)
        for episode in range(args.episodes):
            fraction = min(1.0, episode / decay_episodes)
            epsilon = args.eps_start + fraction * (args.eps_end - args.eps_start)
            initial_state, start_apples = None, 0
            if late_starts and start_rng.random() < args.late_start_prob:
                start = start_rng.choice(late_starts)
                initial_state, start_apples = start.restore(env), start.score
                late_config["episodes"] += 1
            score, steps, total, reason, length = run_episode(env, agent, epsilon, initial_state=initial_state)
            scores.append(score)
            start_values = (start_apples,) if late_starts else ()
            writer.writerow((episode, epsilon, score, steps, total, length, reason,
                             agent.last_loss, agent.updates, len(agent.memory)) + start_values)
            completed = episode + 1
            if completed % args.eval_every == 0 or completed == args.episodes:
                # Separate environment and restored RNG keep evaluation out of training trajectories.
                eval_env = BoardSnakeEnv(encoder=encoder, size=args.size, seed=args.seed + 10000,
                                        rewards=reward_spec)
                rng_state = agent.rng.getstate()
                evaluation = evaluate_details(eval_env, agent, n=args.eval_games)
                agent.rng.setstate(rng_state)
                history.append({"episode": completed, **evaluation})
                print(f"Episode {completed}: mean apples {evaluation['mean_score']:.2f}, "
                      f"updates {agent.updates}, loss {agent.last_loss}", flush=True)
            if completed % args.checkpoint_every == 0 or completed == args.episodes:
                metadata["completed_episodes"] = completed
                agent.save(checkpoint, metadata)
                config.update(trained_at=datetime.now().isoformat(), completed_episodes=completed,
                              training_duration_seconds=time.monotonic() - started,
                              eval_history=history, training_max_score=max(scores),
                              training_last_1000_episodes={"mean_score": float(np.mean(scores[-1000:]))})
                if history:
                    config["final_greedy_eval"] = history[-1]
                    config["final_greedy_eval_score_100_games"] = history[-1]["mean_score"]
                config_path.write_text(json.dumps(config, indent=2))
                handle.flush()
    print(f"Saved {checkpoint}. View with python view_dqn.py --checkpoint {checkpoint}")
