"""Approximate Q-learning for Snake: edit settings, rewards and features here.

From the project root:
    python -m approx.snake train --run-dir approx/runs/first
    python -m approx.snake eval
    python -m approx.snake eval --model approx/runs/first --games 100
    python -m approx.snake eval --model approx/runs/first --games 1 --visual

"""
import argparse
from collections import Counter, deque
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
from time import monotonic

import numpy as np

# Also support `python approx/snake.py ...` by locating the shared game/viewer.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pygame_viewer import PygameViewer, ViewerClosed
from snake_env import DIRS, LEFT, RIGHT, N_ACTIONS, SnakeEnv


#############################
# Hyperparameters & Rewards #
#############################

# EDIT THESE DEFAULTS for new training runs. CLI flags can override each value.
# Evaluation loads the settings saved with the weights instead of these defaults.
ALPHA = 0.001                  # Learning Rate: how much the temporal difference update changes the weights.
GAMMA = 0.9                    # How much future reward counts vs .
EPSILON = 0.2                  # Initial chance of choosing a random action.
EPSILON_END = 0.01             # Set to EPSILON for constant exploration.
EPSILON_DECAY_FRAC = 0.6       # Decay linearly over the first 60% of training.

LIVING_REWARD = -0.2           # reward just for surviving.
APPLE_REWARD = 100.0           # reward when the snake eats an apple.
WALL_DEATH_REWARD = -6350.0    # penalize wall collision.
BODY_DEATH_REWARD = -6350.0    # penalize running into body.
TIMEOUT_REWARD = -6350.0       # penalize taking too long to eat.
DISTANCE_SHAPING = 0.0         # Optional potential-based distance reward


##################
# Game Constants #
##################

BOARD_SIZE = 8
TRAINING_EPISODES = 30000
TRAINING_SEED = 0
TIMEOUT_FACTOR = 3.0           # Completely end the game after MORE THAN factor * board area steps without eating.
EVAL_EVERY = 2000              # Training episodes between evaluations; 0 disables periodic checks.
EVAL_GAMES = 100
EVAL_SEED = 1000
TRAINING_FPS = 30
EVAL_FPS = 10


FEATURE_VERSION = 1
FEATURE_NAMES_V1 = (
    "bias", "apple_distance", "would_eat", "would_hit_wall", "would_hit_body",
    "length_fraction", "reachable_space", "tail_reachable",
    "length_x_distance", "length_x_space", "length_x_tail",
)
FEATURE_NAMES_V2 = FEATURE_NAMES_V1 + ("apple_reachable",)
FEATURE_SCHEMAS = {1: FEATURE_NAMES_V1, 2: FEATURE_NAMES_V2}
FEATURE_NAMES = FEATURE_SCHEMAS[FEATURE_VERSION]
ACTION_NAMES = ("straight", "left", "right")


#########################
# Settings & Validation #
#########################

@dataclass(frozen=True)
class LearningConfig:
    alpha: float = ALPHA
    gamma: float = GAMMA
    epsilon_start: float = EPSILON
    epsilon_end: float = EPSILON_END
    epsilon_decay_frac: float = EPSILON_DECAY_FRAC

    def __post_init__(self):
        values = (self.alpha, self.gamma, self.epsilon_start,
                  self.epsilon_end, self.epsilon_decay_frac)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Learning settings must be finite")
        if self.alpha <= 0 or not 0 <= self.gamma <= 1:
            raise ValueError("alpha must be positive and gamma must be in [0, 1]")
        if not all(0 <= value <= 1 for value in values[2:]):
            raise ValueError("Epsilon values and epsilon-decay-frac must be in [0, 1]")

    def epsilon_at(self, episode, episodes):
        if self.epsilon_decay_frac == 0:
            return self.epsilon_end
        fraction = min(1.0, episode / max(1, int(episodes * self.epsilon_decay_frac)))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


@dataclass(frozen=True)
class RewardConfig:
    apple: float = APPLE_REWARD
    wall_death: float = WALL_DEATH_REWARD
    body_death: float = BODY_DEATH_REWARD
    timeout: float = TIMEOUT_REWARD
    living: float = LIVING_REWARD
    distance_shaping: float = DISTANCE_SHAPING

    def __post_init__(self):
        values = (self.apple, self.wall_death, self.body_death,
                  self.timeout, self.living, self.distance_shaping)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Reward settings must be finite")
        if self.distance_shaping < 0:
            raise ValueError("distance-shaping must be nonnegative")


##############################
# Game State & Move Previews #
##############################

@dataclass(frozen=True)
class Observation:
    size: int
    snake: tuple
    apple: tuple | None
    heading: int
    done: bool = False


def normalized_distance(head, apple, size):
    if apple is None:
        return 0.0
    distance = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
    return min(1.0, distance / (2 * (size - 1)))


def preview(observation, action):
    """Return new head, eating, wall/body collision, and post-move snake cells."""
    if action not in range(N_ACTIONS):
        raise ValueError("Action must be 0 (straight), 1 (left), or 2 (right)")
    heading = observation.heading
    if action == LEFT:
        heading = (heading - 1) % 4
    elif action == RIGHT:
        heading = (heading + 1) % 4
    dr, dc = DIRS[heading]
    old_head = observation.snake[0]
    head = (old_head[0] + dr, old_head[1] + dc)
    eating = head == observation.apple
    wall = not (0 <= head[0] < observation.size and 0 <= head[1] < observation.size)
    occupied = observation.snake if eating else observation.snake[:-1]
    body = not wall and head in occupied
    snake = observation.snake
    if not wall and not body:
        snake = (head,) + (snake if eating else snake[:-1])
    return head, eating, wall, body, snake


def _neighbors(cell, size):
    for dr, dc in DIRS:
        neighbor = (cell[0] + dr, cell[1] + dc)
        if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
            yield neighbor


def _connectivity(snake, size, apple=None):
    # Body INCLUDING the post-move tail blocks the free-space flood fill.
    # The head counts as accessible space; normalize against the whole board.
    blocked = set(snake[1:])
    reachable, queue = {snake[0]}, deque([snake[0]])
    while queue:
        for cell in _neighbors(queue.popleft(), size):
            if cell not in blocked and cell not in reachable:
                reachable.add(cell)
                queue.append(cell)
    # For the tail-path check alone, allow the tail as the destination. It may
    # eventually vacate; the rest of the body is still considered blocked.
    tail = snake[-1]
    tail_reachable = tail in reachable or any(cell in reachable for cell in _neighbors(tail, size))
    # Unlike the tail, the apple must be reachable through the currently open
    # cells. The same flood fill measures space and checks the path to food.
    apple_reachable = apple is not None and apple in reachable
    return len(reachable) / (size * size), float(tail_reachable), float(apple_reachable)


class DisplayFeatures:
    """Current-state summary for the shared viewer; action values appear above it."""

    def format_state(self, state):
        distance = normalized_distance(state.snake[0], state.apple, state.size)
        length = len(state.snake) / (state.size * state.size)
        return f"Current state: apple distance {distance:.3f} | length fraction {length:.3f}"


################################
# Approximate Q-Learning Agent #
################################

class ApproxQAgent:
    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, seed=None,
                 feature_version=FEATURE_VERSION):
        LearningConfig(alpha=alpha, gamma=gamma, epsilon_start=epsilon, epsilon_end=epsilon)
        if feature_version not in FEATURE_SCHEMAS:
            raise ValueError(f"Unsupported feature version: {feature_version}")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.time_steps = 0
        self.rng = random.Random(seed)
        self.feature_version = feature_version
        self.feature_names = FEATURE_SCHEMAS[feature_version]

        # One weight per feature for EACH action; all weights begin at zero.
        self.weights = np.zeros((N_ACTIONS, len(self.feature_names)), dtype=float)

    ######################
    # Feature Extraction #
    ######################

    def getFeatures(self, state, action):
        # Preview this action without changing the real game or drawing a new apple.
        head, eating, wall, body, snake = preview(state, action)
        features = {}

        # 1. BIAS: a constant baseline value for this action.
        features["bias"] = 1.0

        # 2. FOOD DISTANCE: head to the current apple, manhattan distance, normalized to [0,1].
        features["apple_distance"] = normalized_distance(head, state.apple, state.size)

        # 3. EATING: does this action immediately collect the apple? bool
        features["would_eat"] = float(eating)

        # 4-5. COLLISIONS: separate features for wall and body deaths.
        # The tail is considered free only when this action does not eat.
        features["would_hit_wall"] = float(wall)
        features["would_hit_body"] = float(body)

        # 6. LENGTH: fraction of the board occupied after selecting a move.
        length = len(snake) / (state.size * state.size)
        features["length_fraction"] = length

        # 7-8. SPACE / ESCAPE: static connectivity of the post-move board.
        # Reachable Space: How much empty space is accessible to the head in a given state,
        # e.g. how much empty room the head can reach that isn't blocked by the body or a wall
        space, tail, apple_reachable = ((0.0, 0.0, 0.0) if wall or body else
                                        _connectivity(snake, state.size, state.apple))
        features["reachable_space"] = space
        features["tail_reachable"] = tail # Bool on whether the body has blocked the snake from the head

        # 9-11. LENGTH INTERACTIONS: let the linear model learn different priorities for food and maneuvering space as the snake gets longer.
        
        # as the snake gets longer, it becomes less concerned about being further from the food, it offsets the distance penalty as length grows
        # during eval, our weight is multiplied by the length, making this feature more important later on
        features["length_x_distance"] = length * features["apple_distance"]

        # this allows the snake to choose the move that allows the head more "space" as it grows: "preserving extra space matters more when im long"
        features["length_x_space"] = length * space

        # this feature is 0 if the tail is unreachable (bool from above), and scales with the length
        # model learns to maintain tail access as it grows
        features["length_x_tail"] = length * tail
        return features

    def getFeatureMatrix(self, state):
        # Cache one row per action so choosing/updating does not repeat flood fills.
        if state.done:
            return np.zeros((N_ACTIONS, len(self.feature_names)), dtype=float)
        rows = []
        for action in range(N_ACTIONS):
            features = self.getFeatures(state, action)
            rows.append([features[name] for name in self.feature_names])
        return np.asarray(rows, dtype=float)

    ####################
    # Action Selection #
    ####################

    def getQValue(self, state, action):
        # Q(s,a) = sum of each feature multiplied by its learned action weight.
        features = self.getFeatures(state, action)
        return float(sum(self.weights[action, i] * features[name]
                         for i, name in enumerate(self.feature_names)))

    def q_values(self, features):
        # Same dot products for all three actions using already-computed features.
        return np.sum(self.weights * features, axis=1)

    def chooseUsingPolicy(self, features, epsilon=None):
        epsilon = self.epsilon if epsilon is None else epsilon

        # Explore: choose any of the three actions randomly, independent of Q.
        # Evaluation explicitly passes epsilon=0 and never updates weights.
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)

        # Exploit: choose a highest-valued action, breaking ties randomly.
        values = self.q_values(features)
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(list(best)))

    def act(self, features, epsilon=None):
        return self.chooseUsingPolicy(features, epsilon)

    ##################
    # Weight Updates #
    ##################

    def update(self, features, action, reward, next_features, done):
        # Prediction: what we thought this action would be worth.
        prediction = self.q_values(features)[action]

        # Target: actual reward plus the discounted best next action.
        # Terminal transitions have no future reward to bootstrap from.
        target = reward if done else reward + self.gamma * self.q_values(next_features).max()
        error = target - prediction

        # Semi-gradient update: adjust ONLY this action's feature weights.
        updated = self.weights[action] + self.alpha * error * features[action]
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError("Non-finite weights: reduce --alpha or reward magnitudes")
        self.weights[action] = updated
        self.time_steps += 1
        return float(error)

    ############################
    # Saving & Loading Weights #
    ############################

    def save(self, path, metadata):
        path = Path(path)
        payload = {
            "format_version": 1, "algorithm": "linear_approximate_q_learning",
            "feature_version": self.feature_version, "features": list(self.feature_names),
            "actions": list(ACTION_NAMES), "weights": self.weights.tolist(),
            "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon,
            "time_steps": self.time_steps, "metadata": metadata,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path, seed=EVAL_SEED):
        payload = json.loads(Path(path).read_text())
        if payload.get("format_version") != 1 or payload.get("algorithm") != "linear_approximate_q_learning":
            raise ValueError("This is not a supported approximate Q-learning model")
        version = payload.get("feature_version")
        if version not in FEATURE_SCHEMAS or payload.get("features") != list(FEATURE_SCHEMAS[version]):
            raise ValueError("Saved model's feature definitions do not match this implementation")
        if payload.get("actions") != list(ACTION_NAMES):
            raise ValueError("Saved model's action order does not match this implementation")
        agent = cls(payload["alpha"], payload["gamma"], payload["epsilon"], seed,
                    feature_version=version)
        weights = np.asarray(payload["weights"], dtype=float)
        if weights.shape != agent.weights.shape or not np.all(np.isfinite(weights)):
            raise ValueError("Saved weights have an invalid shape or non-finite values")
        agent.weights = weights
        agent.time_steps = int(payload["time_steps"])
        return agent, payload["metadata"]


#################################
# Environment & Reward Function #
#################################

class ApproxSnakeEnv(SnakeEnv):
    def __init__(self, size=BOARD_SIZE, seed=None, rewards=None, gamma=GAMMA,
                 timeout_factor=TIMEOUT_FACTOR):
        self.rewards = rewards if rewards is not None else RewardConfig()
        self.gamma = gamma
        super().__init__(size=size, seed=seed, timeout_factor=timeout_factor,
                         features=DisplayFeatures())

    def get_state(self):
        # Freeze positions so later game moves cannot change earlier observations.
        return Observation(self.size, tuple(self.snake), self.apple, self.heading, self.done)

    def computeReward(self, lastState, currentState, reason, eating):
        # COLLISIONS / TIMEOUT: terminal failure replaces living and food rewards.
        if reason == "wall":
            reward = self.rewards.wall_death
        elif reason == "body":
            reward = self.rewards.body_death
        elif reason == "timeout":
            reward = self.rewards.timeout
        else:
            # SURVIVAL: reward every surviving step, including necessary detours.
            reward = self.rewards.living

            # EAT APPLE: add the food bonus, including the final apple in a win.
            if eating:
                reward += self.rewards.apple

        # OPTIONAL DISTANCE SHAPING: gamma*Phi(next)-Phi(current), Phi=-distance.
        # It is off by default. Terminal Phi is zero. After eating, next Phi uses
        # the newly placed apple so this remains a potential over actual states.
        previous_potential = -normalized_distance(lastState.snake[0], lastState.apple, self.size)
        next_potential = (0.0 if currentState.done else
                          -normalized_distance(currentState.snake[0], currentState.apple, self.size))
        shaping = self.rewards.distance_shaping * (self.gamma * next_potential - previous_potential)
        return reward + shaping, {"task_reward": reward, "shaping_reward": shaping}

    def step(self, action):
        lastState = self.get_state()
        _, eating, wall, _, _ = preview(lastState, action)

        # Reuse baseline movement/food/win/timeout rules; replace its reward.
        currentState, _, done, info = super().step(action)
        reason = info.get("reason")
        if reason == "collision":
            reason = "wall" if wall else "body"
            info["reason"] = reason
        reward, parts = self.computeReward(lastState, currentState, reason, eating)
        info.update(parts)
        return currentState, reward, done, info


##############################
# Playing & Evaluating Games #
##############################

@dataclass(frozen=True)
class EpisodeResult:
    score: int
    steps: int
    total_reward: float
    length: int
    reason: str
    steps_since_apple: int = 0


def run_episode(env, agent, epsilon, learn=True, viewer=None, controls=None,
                title="Approximate Q-learning", summary=""):
    # Start a fresh game and compute the candidate-action features.
    state = env.reset()
    features = agent.getFeatureMatrix(state)
    total = 0.0
    if viewer:
        viewer.show(env, title, epsilon, total, summary=summary)
    while True:
        if controls and not viewer:
            controls.poll()
            if controls.paused:
                controls.show(env, title, epsilon, total, summary=summary, throttle=False)
        # Choose -> advance the game -> optionally learn from the transition.
        values = agent.q_values(features)
        action = agent.act(features, epsilon)
        state, reward, done, info = env.step(action)
        next_features = None if done else agent.getFeatureMatrix(state)
        if learn:
            agent.update(features, action, reward, next_features, done)
        total += reward
        if viewer:
            q_text = f"Q [S,L,R]: {values[0]:.1f}, {values[1]:.1f}, {values[2]:.1f}"
            viewer.show(env, title, epsilon, total, reward, action,
                        summary=q_text, reason=info.get("reason"))
        if done:
            return EpisodeResult(info["score"], env.steps, total, info["length"], info["reason"],
                                 env.steps_since_apple)
        features = next_features


def summarize(results):
    reasons = Counter(result.reason for result in results)
    scores = [result.score for result in results]
    winning_steps = [result.steps for result in results if result.reason == "win"]
    return {
        "games": len(results), "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)), "max_score": max(scores),
        "mean_steps": float(np.mean([result.steps for result in results])),
        "winning_games": len(winning_steps),
        "mean_win_steps": float(np.mean(winning_steps)) if winning_steps else None,
        "min_win_steps": min(winning_steps) if winning_steps else None,
        "max_win_steps": max(winning_steps) if winning_steps else None,
        "win_rate": reasons["win"] / len(results),
        "end_reasons": {reason: reasons[reason] for reason in ("wall", "body", "timeout", "win")},
    }


def evaluate(agent, size, rewards, games=EVAL_GAMES, seed=EVAL_SEED, timeout_factor=TIMEOUT_FACTOR,
             viewer=None, title="Approximate evaluation", on_game=None, start_game=1):
    """Fresh evaluation environment and fixed seed; no learning or training RNG drift."""
    env = ApproxSnakeEnv(size=size, seed=seed, rewards=rewards, gamma=agent.gamma,
                         timeout_factor=timeout_factor)
    results = []
    rng_state = agent.rng.getstate()
    # A reproducible evaluation RNG independent of training tie-break history.
    agent.rng.seed(seed)
    try:
        # Earlier games must actually advance both RNGs to reproduce the exact
        # selected batch game. Changing just the seed would create a different game.
        if viewer and start_game > 1:
            viewer.show(env, f"Preparing batch game {start_game}", 0.0, 0.0,
                        summary=f"Replaying games 1..{start_game - 1} without animation", throttle=False)
        last_game = start_game + games - 1
        for game in range(1, last_game + 1):
            selected = game >= start_game
            result = run_episode(env, agent, 0.0, learn=False, viewer=viewer if selected else None,
                                 controls=viewer,
                                 title=f"{title} | Game {game}/{last_game}",
                                 summary="Greedy policy | Learning disabled")
            if not selected:
                continue
            results.append(result)
            if on_game:
                on_game(game, result)
    finally:
        agent.rng.setstate(rng_state)
    return summarize(results)


####################
# Training Command #
####################

def train_main(argv=None):
    learning_default, reward_default = LearningConfig(), RewardConfig()
    p = argparse.ArgumentParser(prog="python -m approx.snake train", description="Train approximate Q-learning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--size", type=int, default=BOARD_SIZE)
    p.add_argument("--episodes", type=int, default=TRAINING_EPISODES)
    p.add_argument("--seed", type=int, default=TRAINING_SEED)
    p.add_argument("--alpha", type=float, default=learning_default.alpha)
    p.add_argument("--gamma", type=float, default=learning_default.gamma)
    p.add_argument("--epsilon", "--epsilon-start", type=float, default=learning_default.epsilon_start,
                   help="initial probability of a random action")
    p.add_argument("--epsilon-end", type=float, default=learning_default.epsilon_end,
                   help="set equal to --epsilon for constant exploration")
    p.add_argument("--epsilon-decay-frac", type=float, default=learning_default.epsilon_decay_frac,
                   help="fraction of training used for linear epsilon decay; 0 uses epsilon-end immediately")
    p.add_argument("--apple-reward", type=float, default=reward_default.apple)
    p.add_argument("--wall-death-reward", type=float, default=reward_default.wall_death)
    p.add_argument("--body-death-reward", type=float, default=reward_default.body_death)
    p.add_argument("--timeout-reward", type=float, default=reward_default.timeout)
    p.add_argument("--living-reward", type=float, default=reward_default.living,
                   help="reward on every surviving move, including moves that eat")
    p.add_argument("--distance-shaping", type=float, default=reward_default.distance_shaping,
                   help="coefficient for gamma*Phi(next)-Phi(current), Phi=-normalized distance")
    p.add_argument("--timeout-factor", type=float, default=TIMEOUT_FACTOR,
                   help="timeout after more than factor*size*size steps without eating")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="0 disables periodic evaluation")
    p.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    p.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    p.add_argument("--visual", action="store_true", help="animate training and evaluation")
    p.add_argument("--visual-eval", action="store_true", help="animate evaluation; train between checks at full speed")
    p.add_argument("--visual-every", type=int, default=1, help="animate every Nth training episode")
    p.add_argument("--fps", type=int, default=TRAINING_FPS)
    p.add_argument("--run-dir", type=Path, help="new output directory; default: approx/runs/<timestamp>")
    args = p.parse_args(argv)
    if args.size < 4 or args.episodes < 1:
        p.error("size must be at least 4 and episodes must be positive")
    if args.eval_every < 0 or args.eval_games < 1 or args.visual_every < 1:
        p.error("eval-every must be nonnegative; eval-games and visual-every must be positive")
    if not 1 <= args.fps <= 240 or not math.isfinite(args.timeout_factor) or args.timeout_factor <= 0:
        p.error("fps must be in 1..240 and timeout-factor must be finite and positive")
    try:
        learning = LearningConfig(args.alpha, args.gamma, args.epsilon, args.epsilon_end, args.epsilon_decay_frac)
        rewards = RewardConfig(args.apple_reward, args.wall_death_reward, args.body_death_reward,
                               args.timeout_reward, args.living_reward, args.distance_shaping)
    except ValueError as exc:
        p.error(str(exc))

    started = datetime.now()
    run_dir = args.run_dir or Path(__file__).parent / "runs" / started.strftime("%Y%m%d_%H%M%S_%f")
    if run_dir.exists():
        p.error(f"run directory already exists: {run_dir}; choose a new directory")
    viewer = None
    try:
        if args.visual or args.visual_eval:
            viewer = PygameViewer(args.size, args.fps)
        run_dir.mkdir(parents=True)
    except (OSError, RuntimeError) as exc:
        if viewer:
            viewer.close()
        p.exit(1, f"error: {exc}\n")

    agent = ApproxQAgent(learning.alpha, learning.gamma, learning.epsilon_start, args.seed)
    env = ApproxSnakeEnv(args.size, args.seed, rewards, learning.gamma, args.timeout_factor)
    completed, interrupted, failure, last_evaluation = 0, False, None, None
    recent, best_score, last_return, last_preview = deque(maxlen=100), 0, 0.0, 0.0
    model_path = run_dir / "model.json"

    def save_checkpoint():
        agent.save(model_path, {
            "created_at": started.isoformat(timespec="seconds"),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "environment": {"size": args.size, "initial_length": 3, "timeout_factor": args.timeout_factor},
            "learning": asdict(learning), "rewards": asdict(rewards),
            "training": {"seed": args.seed, "requested_episodes": args.episodes,
                         "completed_episodes": completed, "interrupted": interrupted,
                         "failure": failure},
            "evaluation": {"seed": args.eval_seed, "games": args.eval_games,
                           "every": args.eval_every, "last_result": last_evaluation},
        })

    print(f"Approximate Q-learning | {args.size}x{args.size} | outputs: {run_dir}")
    print(f"Feature version {agent.feature_version}: {len(agent.feature_names)} features per action")
    print(f"alpha={learning.alpha}, gamma={learning.gamma}, "
          f"epsilon={learning.epsilon_start}->{learning.epsilon_end}, living={rewards.living}")
    try:
        with (run_dir / "log.csv").open("w", newline="") as log_file, \
                (run_dir / "eval.csv").open("w", newline="") as eval_file:
            log = csv.writer(log_file)
            eval_log = csv.writer(eval_file)
            log.writerow(["episode", "epsilon", "score", "steps", "return", "max_length",
                          "length_fraction", "end_reason", "time_steps"])
            eval_log.writerow(["after_episode", "game", "score", "steps", "return",
                               "length", "length_fraction", "end_reason"])

            def check_policy(label):
                def record(game, result):
                    eval_log.writerow([completed, game, result.score, result.steps, result.total_reward,
                                       result.length, result.length / (args.size * args.size), result.reason])
                    eval_file.flush()
                stats = evaluate(agent, args.size, rewards, games=args.eval_games, seed=args.eval_seed,
                                 timeout_factor=args.timeout_factor, viewer=viewer, title=label, on_game=record)
                return {"after_episode": completed, **stats}

            for episode in range(args.episodes):
                agent.epsilon = learning.epsilon_at(episode, args.episodes)
                training_view = viewer if args.visual and episode % args.visual_every == 0 else None
                summary = (f"Last 100 mean {np.mean(recent):.2f} | Best {best_score}"
                           if recent else "Learning enabled | First episode")
                if viewer and not training_view and (viewer.paused or monotonic() - last_preview >= 0.25):
                    viewer.show(env, f"Approx training | {completed}/{args.episodes} completed",
                                agent.epsilon, last_return, summary=summary, throttle=False)
                    last_preview = monotonic()
                result = run_episode(env, agent, agent.epsilon, viewer=training_view, controls=viewer,
                                     title=f"Approx training | Episode {episode + 1}/{args.episodes}",
                                     summary=summary)
                completed += 1
                recent.append(result.score)
                best_score = max(best_score, result.score)
                last_return = result.total_reward
                log.writerow([completed, agent.epsilon, result.score, result.steps, result.total_reward,
                              result.length, result.length / (args.size * args.size), result.reason,
                              agent.time_steps])
                log_file.flush()
                if args.eval_every and completed % args.eval_every == 0 and completed < args.episodes:
                    save_checkpoint()
                    last_evaluation = check_policy(f"Approx eval after episode {completed}")
                    print(f"episode {completed}: epsilon={agent.epsilon:.3f}, "
                          f"greedy mean={last_evaluation['mean_score']:.2f}, "
                          f"best={last_evaluation['max_score']}, wins={last_evaluation['win_rate']:.1%}")
            # A checkpoint exists before potentially slow visual final evaluation.
            save_checkpoint()
            last_evaluation = check_policy("Approx final evaluation")
            print(f"Final greedy mean: {last_evaluation['mean_score']:.2f} apples; "
                  f"best {last_evaluation['max_score']}; wins {last_evaluation['win_rate']:.1%}")
    except (ViewerClosed, KeyboardInterrupt):
        interrupted = True
        print(f"Stopped after {completed} completed episodes; saving current weights.")
    except FloatingPointError as exc:
        interrupted, failure = True, str(exc)
        print(f"Training stopped: {failure}. Saving last finite weights.")
    finally:
        if viewer:
            viewer.close()
        save_checkpoint()
        print(f"Saved {model_path}")
    if failure:
        p.exit(1)


######################
# Evaluation Command #
######################

def eval_main(argv=None):
    p = argparse.ArgumentParser(prog="python -m approx.snake eval", description="Evaluate saved approximate weights")
    p.add_argument("--model", type=Path,
                   help="model.json path or run directory; default: most recently saved model in approx/runs")
    p.add_argument("--games", type=int, default=1, help="default: play one complete game")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--start-game", type=int, default=1,
                   help="start at this 1-based game in the seeded batch; earlier games replay without animation")
    p.add_argument("--visual", action="store_true", help="watch frozen weights in Pygame")
    p.add_argument("--fps", type=int, default=EVAL_FPS)
    args = p.parse_args(argv)
    if args.games < 1 or args.start_game < 1 or not 1 <= args.fps <= 240:
        p.error("games and start-game must be positive and fps must be in 1..240")
    model_path = args.model
    viewer, results = None, []

    def record(game, result):
        results.append(result)
        print(f"game {game}: apples={result.score}, length={result.length}, "
              f"total_steps={result.steps}, end={result.reason}, "
              f"steps_since_food={result.steps_since_apple}/{timeout_steps:g}")

    try:
        if model_path is None:
            # Find the newest saved checkpoint independently of the working directory.
            run_root = Path(__file__).resolve().parent / "runs"
            models = [path for path in run_root.glob("*/model.json") if path.is_file()]
            if not models:
                p.error("No saved models in approx/runs. Train first or pass --model <path>.")
            model_path = max(models, key=lambda path: (path.stat().st_mtime_ns, str(path)))
            print(f"Using latest saved model: {model_path}")
        elif model_path.is_dir():
            model_path = model_path / "model.json"
        agent, metadata = ApproxQAgent.load(model_path, seed=args.seed)
        environment = metadata["environment"]
        rewards = RewardConfig(**metadata["rewards"])
        size = environment["size"]
        if not isinstance(size, int) or size < 4:
            raise ValueError("Saved board size must be an integer of at least 4")
        if args.visual:
            viewer = PygameViewer(size, args.fps)
        training = metadata["training"]
        print(f"Loaded {model_path} | {training['completed_episodes']} completed training episodes | "
              f"{agent.time_steps} learning steps")
        print(f"Feature version {agent.feature_version}: {len(agent.feature_names)} features per action")
        if "apple_reachable" in agent.feature_names:
            print("This is an experimental apple-reachability model; new training uses the original 11 features.")
        if training.get("interrupted"):
            print("This checkpoint was saved from an interrupted training run.")
        timeout_steps = environment["timeout_factor"] * size * size
        print(f"Seed={args.seed} | games {args.start_game}..{args.start_game + args.games - 1} | "
              f"timeout after more than {timeout_steps:g} consecutive moves without food")
        print("Greedy evaluation: epsilon=0, learning disabled, saved weights frozen.")
        stats = evaluate(agent, size, rewards, args.games, args.seed, environment["timeout_factor"],
                         viewer=viewer, on_game=record, start_game=args.start_game)
    except (ViewerClosed, KeyboardInterrupt):
        print("Evaluation stopped.")
        stats = summarize(results) if results else None
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        p.exit(1, f"error: {exc}\n")
    finally:
        if viewer:
            viewer.close()
    if stats:
        print(f"{stats['games']} games | mean apples {stats['mean_score']:.2f} | "
              f"best {stats['max_score']} | win rate {stats['win_rate']:.1%}")
        print(f"Mean total steps (all games): {stats['mean_steps']:.2f}")
        if stats["mean_win_steps"] is None:
            print("Mean total steps (winning games): N/A (no wins)")
        else:
            print(f"Mean total steps ({stats['winning_games']} winning games): {stats['mean_win_steps']:.2f} "
                  f"| min {stats['min_win_steps']} | max {stats['max_win_steps']}")
        print(f"End reasons: {stats['end_reasons']}")


############################
# Command-Line Entry Point #
############################

def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Snake approximate Q-learning: train or evaluate.")
    parser.add_argument("mode", choices=("train", "eval"), help="use train --help or eval --help for options")
    # Parse only the mode so each command handles its own flags and --help.
    mode = parser.parse_args(arguments[:1]).mode
    if mode == "train":
        train_main(arguments[1:])
    else:
        eval_main(arguments[1:])


if __name__ == "__main__":
    main()
