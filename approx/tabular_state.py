"""Linear Q-learning using the same six observed values as tabular Q-learning.

The 38 features encode danger, apple direction, heading, and heading/apple
combinations. No coordinates, distances, lengths, or board searches are used.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import snake_env
from feature_library import make_feature_set
from q_tabular import QAgent
from rollout import run_episode
from snake_env import N_ACTIONS, SnakeEnv


STATE_NAMES = ("danger_straight", "danger_left", "danger_right",
               "apple_row", "apple_col", "heading")
STATE_SHAPE = (2, 2, 2, 3, 3, 4)
APPLE_ROW_FEATURES = ("apple_above", "apple_same_row", "apple_below")
APPLE_COL_FEATURES = ("apple_left", "apple_same_col", "apple_right")
HEADING_FEATURES = ("heading_up", "heading_right", "heading_down", "heading_left")
FEATURE_NAMES = (
    "bias", "danger_straight", "danger_left", "danger_right",
) + APPLE_ROW_FEATURES + APPLE_COL_FEATURES + HEADING_FEATURES + tuple(
    f"{heading}_x_{apple}"
    for apple_categories in (APPLE_ROW_FEATURES, APPLE_COL_FEATURES)
    for heading in HEADING_FEATURES
    for apple in apple_categories
)


class OriginalTabularState:
    """Unpack today's equivalent compact observation into the original six values."""

    names = STATE_NAMES
    shape = STATE_SHAPE
    n_states = 288

    def __init__(self):
        self.compact = make_feature_set("danger,apple_dir,heading")

    def encode(self, env):
        if env.apple is None:
            return (0, 0, 0, 1, 1, env.heading)
        danger, apple_dir, heading = self.compact.encode(env)
        row, col = divmod(apple_dir, 3)
        return ((danger >> 2) & 1, (danger >> 1) & 1, danger & 1,
                row, col, heading)

    def format_state(self, state):
        return " ".join(f"{name}={value}" for name, value in zip(self.names, state))


def state_features(state):
    """Encode only the six observed values, never inspect an environment."""
    if len(state) != len(STATE_SHAPE):
        raise ValueError("Expected the original six-value tabular state")
    if any(not isinstance(v, (int, np.integer)) or not 0 <= v < size
           for v, size in zip(state, STATE_SHAPE)):
        raise ValueError(f"Invalid tabular state: {state}")
    straight, left, right, row, col, heading = state
    features = np.zeros(len(FEATURE_NAMES), dtype=float)
    features[:4] = (1.0, straight, left, right)
    features[4 + row] = 1.0
    features[7 + col] = 1.0
    features[10 + heading] = 1.0
    # Each heading/apple block has 4 headings x 3 directions = 12 indicators.
    features[14 + heading * 3 + row] = 1.0
    features[26 + heading * 3 + col] = 1.0
    return features


class LinearQAgent:
    """Q(s,a) = weights[a] dot features(s), compatible with rollout.run_episode."""

    def __init__(self, alpha=0.01, gamma=0.95, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.weights = np.zeros((N_ACTIONS, len(FEATURE_NAMES)), dtype=float)

    def q_values(self, state):
        return self.weights @ state_features(state)

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        values = self.q_values(state)
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(list(best)))

    def update(self, state, action, reward, next_state, done):
        features = state_features(state)
        prediction = float(self.weights[action] @ features)
        target = reward if done else reward + self.gamma * self.q_values(next_state).max()
        updated = self.weights[action] + self.alpha * (target - prediction) * features
        if not np.isfinite(updated).all():
            raise FloatingPointError("Non-finite linear weights; reduce --alpha")
        self.weights[action] = updated


def make_agent(kind, alpha, gamma, seed):
    if kind == "approximate":
        return LinearQAgent(alpha, gamma, seed)
    if kind == "tabular":
        return QAgent(STATE_SHAPE, alpha=alpha, gamma=gamma, seed=seed)
    raise ValueError(f"Unknown agent type: {kind}")


def load_model(path):
    """Load weights only when their saved features match this implementation."""
    payload = json.loads(Path(path).read_text())
    if (payload.get("state_names") != list(STATE_NAMES)
            or payload.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Model is not compatible with this state/feature encoding")
    size = payload["environment"]["size"]
    if payload["environment"] != environment_config(size):
        raise ValueError("Shared environment/rewards changed since training")
    agent = make_agent(payload["agent"], payload["alpha"], payload["gamma"], 0)
    rng_state = payload["rng_state"]
    agent.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
    parameters = np.asarray(payload["parameters"], dtype=float)
    destination = agent.weights if isinstance(agent, LinearQAgent) else agent.q
    if parameters.shape != destination.shape or not np.isfinite(parameters).all():
        raise ValueError("Model parameter shape/values are invalid")
    destination[:] = parameters
    return agent, payload


def evaluate(agent, size, games, seed):
    """Reset evaluation seeds on every check; preserve training RNG and weights."""
    env = SnakeEnv(size=size, seed=seed, features=OriginalTabularState())
    rng_state = agent.rng.getstate()
    rows = []
    try:
        for game in range(1, games + 1):
            score, steps, total, info = run_episode(env, agent, 0.0, learn=False)
            rows.append(dict(game=game, score=score, steps=steps,
                             total_reward=total, end_reason=info["reason"]))
    finally:
        agent.rng.setstate(rng_state)
    apples = sum(row["score"] for row in rows)
    reasons = Counter(row["end_reason"] for row in rows)
    ratios = [row["steps"] / row["score"] for row in rows if row["score"]]
    stats = dict(
        games=games, mean_apples=apples / games, win_rate=reasons["win"] / games,
        mean_steps=sum(row["steps"] for row in rows) / games,
        pooled_steps_per_apple=sum(row["steps"] for row in rows) / apples if apples else None,
        mean_game_steps_per_apple=float(np.mean(ratios)) if ratios else None,
        zero_apple_games=sum(row["score"] == 0 for row in rows),
        end_reasons=dict(reasons),
    )
    return stats, rows


def environment_config(size):
    return dict(size=size, initial_length=3, timeout_factor=2,
                rewards=dict(apple=snake_env.R_APPLE, death=snake_env.R_DEATH,
                             step=snake_env.R_STEP, closer_or_farther=snake_env.R_CLOSER))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--agent", choices=("approximate", "tabular"), default="approximate")
    train.add_argument("--size", type=int, default=8)
    train.add_argument("--episodes", type=int, default=20000)
    train.add_argument("--alpha", type=float, default=0.01)
    train.add_argument("--gamma", type=float, default=0.95)
    train.add_argument("--eps-start", type=float, default=1.0)
    train.add_argument("--eps-end", type=float, default=0.01)
    train.add_argument("--eps-decay-frac", type=float, default=0.6)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--eval-seed", type=int, default=1000)
    train.add_argument("--eval-games", type=int, default=100)
    train.add_argument("--eval-every", type=int, default=2000)
    train.add_argument("--run-dir", type=Path)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--games", type=int, default=100)
    evaluation.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.command == "eval":
        if args.games < 1:
            parser.error("--games must be positive")
        try:
            agent, payload = load_model(args.model)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps(evaluate(agent, payload["environment"]["size"],
                                  args.games, args.seed)[0], indent=2))
        return

    if args.size < 4 or args.episodes < 1 or args.eval_games < 1 or args.eval_every < 0:
        parser.error("size >= 4; episodes/eval-games > 0; eval-every >= 0 required")
    if not np.isfinite(args.alpha) or args.alpha <= 0 or not 0 <= args.gamma <= 1:
        parser.error("alpha must be finite and positive; gamma must be in [0,1]")
    if not all(0 <= v <= 1 for v in (args.eps_start, args.eps_end, args.eps_decay_frac)):
        parser.error("epsilon settings must be in [0,1]")
    run_dir = args.run_dir or Path(__file__).parent / "tabular_state_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.agent)
    # An explicit fresh directory prevents overwriting previous experiments.
    run_dir.mkdir(parents=True, exist_ok=False)
    agent = make_agent(args.agent, args.alpha, args.gamma, args.seed)
    if isinstance(agent, LinearQAgent):
        print(f"{len(FEATURE_NAMES)} inputs per action; "
              f"{agent.weights.size} weights; original six-value observation only", flush=True)
    env = SnakeEnv(size=args.size, seed=args.seed, features=OriginalTabularState())
    completed = total_steps = 0
    training_seconds = 0.0
    last_stats = None

    with (run_dir / "log.csv").open("w", newline="") as log_file, \
            (run_dir / "eval.csv").open("w", newline="") as eval_file:
        log = csv.writer(log_file)
        log.writerow(("episode", "epsilon", "score", "steps", "return", "end_reason", "total_steps"))
        eval_log = csv.DictWriter(eval_file, fieldnames=(
            "after_episode", "game", "score", "steps", "total_reward", "end_reason"))
        eval_log.writeheader()
        for episode in range(args.episodes):
            fraction = min(1.0, episode / max(1, int(args.episodes * args.eps_decay_frac)))
            epsilon = args.eps_start + fraction * (args.eps_end - args.eps_start)
            started = perf_counter()
            score, steps, total, info = run_episode(env, agent, epsilon)
            training_seconds += perf_counter() - started
            completed += 1
            total_steps += steps
            log.writerow((completed, epsilon, score, steps, total, info["reason"], total_steps))
            log_file.flush()
            if completed == args.episodes or (args.eval_every and completed % args.eval_every == 0):
                last_stats, rows = evaluate(agent, args.size, args.eval_games, args.eval_seed)
                for row in rows:
                    eval_log.writerow(dict(after_episode=completed, **row))
                eval_file.flush()
                print(f"Episode {completed}: {json.dumps(last_stats)}", flush=True)

    payload = dict(
        format="original_tabular_state", agent=args.agent,
        state_names=list(STATE_NAMES), state_shape=list(STATE_SHAPE),
        feature_names=list(FEATURE_NAMES), environment=environment_config(args.size),
        alpha=args.alpha, gamma=args.gamma, seed=args.seed,
        rng_state=agent.rng.getstate(),
        epsilon=dict(start=args.eps_start, end=args.eps_end, decay_fraction=args.eps_decay_frac),
        episodes=completed, training_steps=total_steps, training_seconds=training_seconds,
        evaluation=dict(seed=args.eval_seed, **last_stats),
        parameters=(agent.weights if isinstance(agent, LinearQAgent) else agent.q).tolist(),
    )
    (run_dir / "model.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {run_dir / 'model.json'}")


if __name__ == "__main__":
    main()
