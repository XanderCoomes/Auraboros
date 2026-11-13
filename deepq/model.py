"""Snake DQN: settings, state encoding, network, rewards and learning."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Find the shared game and viewer when running directly from inside deepq/.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snake_env import DIRS, LEFT, RIGHT, N_ACTIONS, SnakeEnv


#############################
# Hyperparameters & Rewards #
#############################

# Edit these defaults for new runs. CLI flags can override them.
ALPHA = 0.0001                 # Adam learning rate.
GAMMA = 0.995                   # How much future reward counts.
EPSILON = 1.0                  # Initial chance of choosing a random action.
EPSILON_END = 0.05             # Final exploration rate during training.
EPSILON_DECAY_FRAC = 0.6       # Decay over the first 60% of training episodes.

BATCH_SIZE = 64               # Stored moves sampled for each weight update.
REPLAY_CAPACITY = 50000        # Maximum number of stored moves.
LEARNING_STARTS = 1000         # Collect this many moves before learning.
TRAIN_EVERY = 4                # Learn once every N training moves.
TARGET_UPDATE_EVERY = 1000     # Copy learned weights every N optimizer updates.
GRADIENT_CLIP = 10.0           # Limit the gradient norm for an update.
REWARD_SCALE = 0.01            # Scale ALL rewards for learning; logs keep raw rewards.

LIVING_REWARD = 0.0            # Every surviving move, including eating.
APPLE_REWARD = 100.0           # Added when eating an apple.
WALL_DEATH_REWARD = -1500.0    # Terminal wall collision reward.
BODY_DEATH_REWARD = -1500.0    # Terminal self-collision reward.
TIMEOUT_REWARD = -1500.0       # Terminal failure after too long without food.
DISTANCE_SHAPING = 0.0         # Optional potential-based distance reward; off.


##################
# Game Constants #
##################

BOARD_SIZE = 8
INPUT_DIM = 3 * BOARD_SIZE * BOARD_SIZE + 11  # 203 inputs on an 8x8 board.
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 96
HIDDEN_DIM_3 = 256
TRAINING_EPISODES = 30000
TIMEOUT_FACTOR = 2.0
TRAINING_SEED = 0
EVAL_SEED = 1000
EVAL_EVERY = 2000
EVAL_GAMES = 100
PRINT_EVERY = 1000
TRAINING_FPS = 30
EVAL_FPS = 10
DEVICE = "auto"                # auto, cpu, cuda or mps.
TORCH_THREADS = 1              # CPU threads used by PyTorch.
ENCODING_VERSION = 2
ACTION_NAMES = ("straight", "left", "right")


#########################
# Settings & Validation #
#########################

@dataclass(frozen=True)
class DQNConfig:
    alpha: float = ALPHA
    gamma: float = GAMMA
    epsilon_start: float = EPSILON
    epsilon_end: float = EPSILON_END
    epsilon_decay_frac: float = EPSILON_DECAY_FRAC
    batch_size: int = BATCH_SIZE
    replay_capacity: int = REPLAY_CAPACITY
    learning_starts: int = LEARNING_STARTS
    train_every: int = TRAIN_EVERY
    target_update_every: int = TARGET_UPDATE_EVERY
    gradient_clip: float = GRADIENT_CLIP
    reward_scale: float = REWARD_SCALE

    def __post_init__(self):
        floats = (self.alpha, self.gamma, self.epsilon_start, self.epsilon_end,
                  self.epsilon_decay_frac, self.gradient_clip, self.reward_scale)
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("Learning settings must be finite")
        if self.alpha <= 0 or self.gradient_clip <= 0 or self.reward_scale <= 0:
            raise ValueError("alpha, gradient-clip and reward-scale must be positive")
        if not all(0 <= value <= 1 for value in floats[1:5]):
            raise ValueError("gamma, epsilon values and epsilon-decay-frac must be in [0, 1]")
        if min(self.batch_size, self.replay_capacity, self.train_every, self.target_update_every) < 1:
            raise ValueError("Batch size, replay capacity and update intervals must be positive")
        if not 0 <= self.learning_starts <= self.replay_capacity or self.batch_size > self.replay_capacity:
            raise ValueError("Replay capacity must cover both batch-size and learning-starts")

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
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Reward settings must be finite")
        if self.distance_shaping < 0:
            raise ValueError("distance-shaping must be nonnegative")


##################
# State Encoding #
##################

def state_dimension(size):
    return 3 * size * size + 11


def encode_state(env):
    # Channel 0: head. Channel 1: apple. Channel 2: ordered body.
    area = env.size * env.size
    grids = np.zeros((3, env.size, env.size), dtype=np.float32)
    head_row, head_col = env.snake[0]
    grids[0, head_row, head_col] = 1.0
    if env.apple is not None:
        grids[1, env.apple[0], env.apple[1]] = 1.0

    # Tail=1/area, next segment=2/area, ..., head=length/area.
    # This preserves body order instead of just marking occupied cells.
    for order, (row, col) in enumerate(reversed(env.snake), start=1):
        grids[2, row, col] = order / area

    heading = np.zeros(4, dtype=np.float32)
    heading[env.heading] = 1.0
    hunger = np.array([env.steps_since_apple / env.timeout], dtype=np.float32)

    collision_flags = []
    tail = env.snake[-1]

    # Append wall/body flags for straight, left, then right without moving.
    for turn in (0, -1, 1):
        direction = (env.heading + turn) % 4
        dr, dc = DIRS[direction]
        candidate = (head_row + dr, head_col + dc)

        hits_wall = not env._in_bounds(candidate)
        eating = candidate == env.apple
        # The tail cell is safe when the tail moves away.
        hits_body = candidate in env.body and (
            candidate != tail or eating
        )

        collision_flags.extend((float(hits_wall), float(hits_body)))

    collision_flags = np.array(collision_flags, dtype=np.float32)
    return np.concatenate((grids.reshape(-1), heading, hunger, collision_flags))


class StateDisplay:
    def format_state(self, state):
        # Hunger comes immediately before the six collision flags.
        return f"Input: head, apple, body order, heading, collisions | No-food fraction {state[-7]:.3f}"


##################
# Neural Network #
##################

class SnakeNN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim_1=HIDDEN_DIM_1, hidden_dim_2=HIDDEN_DIM_2, hidden_dim_3=HIDDEN_DIM_3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim_1)
        self.linear2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.linear3 = nn.Linear(hidden_dim_2, N_ACTIONS)
        self.linear4 = nn.Linear(hidden_dim_3, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        return self.linear3(x)


############################
# Environment & Rewards #
############################

class DQNSnakeEnv(SnakeEnv):
    def __init__(self, size=BOARD_SIZE, seed=None, rewards=None, gamma=GAMMA,
                 timeout_factor=TIMEOUT_FACTOR):
        if size < 4 or not math.isfinite(timeout_factor) or timeout_factor <= 0:
            raise ValueError("size must be at least 4 and timeout-factor must be finite and positive")
        self.rewards = rewards if rewards is not None else RewardConfig()
        self.gamma = gamma
        super().__init__(size=size, seed=seed, timeout_factor=timeout_factor, features=StateDisplay())

    def get_state(self):
        return encode_state(self)

    def computeReward(self, reason, eating, previous_potential):
        # Terminal failures replace living and food rewards.
        if reason == "wall":
            reward = self.rewards.wall_death
        elif reason == "body":
            reward = self.rewards.body_death
        elif reason == "timeout":
            reward = self.rewards.timeout
        else:
            reward = self.rewards.living
            if eating:
                reward += self.rewards.apple

        # Optional shaping uses actual states, including a newly placed apple.
        next_potential = 0.0 if self.done else self._potential()
        shaping = self.rewards.distance_shaping * (self.gamma * next_potential - previous_potential)
        return reward + shaping

    def _potential(self):
        if self.apple is None:
            return 0.0
        head = self.snake[0]
        distance = abs(head[0] - self.apple[0]) + abs(head[1] - self.apple[1])
        return -distance / (2 * (self.size - 1))

    def step(self, action):
        if action not in range(N_ACTIONS):
            raise ValueError("Action must be straight, left or right (0..2)")
        heading = self.heading
        if action == LEFT:
            heading = (heading - 1) % 4
        elif action == RIGHT:
            heading = (heading + 1) % 4
        dr, dc = DIRS[heading]
        head = (self.snake[0][0] + dr, self.snake[0][1] + dc)
        wall = not self._in_bounds(head)
        eating = head == self.apple
        previous_potential = self._potential()

        # Reuse game rules, but replace the baseline's hard-coded rewards.
        state, _, done, info = super().step(action)
        reason = info.get("reason")
        if reason == "collision":
            reason = "wall" if wall else "body"
            info["reason"] = reason
        reward = self.computeReward(reason, eating, previous_potential)
        return state, reward, done, info


#################
# Replay Memory #
#################

class ReplayMemory:
    def __init__(self, capacity, input_dim, seed):
        # Fixed-size arrays keep memory bounded. Adding a move copies its states.
        self.states = np.empty((capacity, input_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.capacity, self.position, self.length = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.position = (i + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.length, batch_size, replace=False)
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.dones[indices])


#########################
# Device & DQN Agent #
#########################

def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda or mps")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu or auto")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use --device cpu or auto")
    return torch.device(name)


class DQNAgent:
    def __init__(self, size=BOARD_SIZE, hidden_dim=HIDDEN_DIM_1, config=None,
                 seed=TRAINING_SEED, device=DEVICE, training=True):
        self.config = config if config is not None else DQNConfig()
        self.size, self.hidden_dim = size, hidden_dim
        self.input_dim = state_dimension(size)
        self.device = select_device(device)
        self.rng = random.Random(seed)
        self.time_steps, self.update_steps = 0, 0
        torch.manual_seed(seed)
        self.policy_net = SnakeNN(self.input_dim, hidden_dim).to(self.device)
        self.target_net = self.optimizer = self.memory = None
        if training:
            self.target_net = deepcopy(self.policy_net).eval()
            self.target_net.requires_grad_(False)
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.alpha)
            self.memory = ReplayMemory(self.config.replay_capacity, self.input_dim, seed)

    def getQValues(self, state):
        with torch.inference_mode():
            inputs = torch.from_numpy(state).unsqueeze(0).to(self.device)
            values = self.policy_net(inputs)[0].cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Network produced non-finite Q-values")
        return values

    def chooseUsingPolicy(self, state, epsilon=0.0, values=None):
        # Exploration can choose any action, including one that collides.
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        values = self.getQValues(state) if values is None else values
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(list(best)))

    def remember(self, state, action, reward, next_state, done):
        if self.memory is None:
            raise RuntimeError("An evaluation-only agent cannot learn")
        self.memory.add(state, action, reward, next_state, done)
        self.time_steps += 1

    def learn(self):
        cfg = self.config
        if self.memory is None or self.time_steps < cfg.learning_starts:
            return None
        if self.memory.length < cfg.batch_size or self.time_steps % cfg.train_every:
            return None

        # Revisit a random batch of past moves, not just the most recent move.
        arrays = self.memory.sample(cfg.batch_size)
        states, actions, rewards, next_states, dones = [torch.from_numpy(a).to(self.device) for a in arrays]
        predictions = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # The target network supplies future Q-values. Terminal moves get no future value.
        with torch.no_grad():
            next_values = self.target_net(next_states).max(dim=1).values
            targets = rewards * cfg.reward_scale + cfg.gamma * next_values * (~dones).float()
        loss = F.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite loss; reduce alpha or reward magnitudes")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        try:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
        except RuntimeError as exc:
            raise FloatingPointError("Non-finite gradients") from exc
        self.optimizer.step()
        self.update_steps += 1

        # Keep targets fixed between copies so they do not chase every update.
        if self.update_steps % cfg.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    ############################
    # Saving & Loading Weights #
    ############################

    def save(self, path, metadata):
        path = Path(path)
        details = {
            "format_version": 1, "algorithm": "dqn", "encoding_version": ENCODING_VERSION,
            "architecture": {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                             "hidden_layers": 2, "actions": list(ACTION_NAMES)},
            "learning": asdict(self.config), "time_steps": self.time_steps,
            "update_steps": self.update_steps, "metadata": metadata,
        }
        # Store tensors, not a pickled network object. Load settings from the same file.
        weights = {key: value.detach().cpu() for key, value in self.policy_net.state_dict().items()}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({**details, "state_dict": weights}, temporary)
        temporary.replace(path)
        config_path = path.parent / "config.json"
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
        temporary.replace(config_path)

    @classmethod
    def load(cls, path, device=DEVICE, seed=EVAL_SEED):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1 or saved.get("algorithm") != "dqn":
            raise ValueError("Not a supported DQN checkpoint")
        architecture, metadata = saved["architecture"], saved["metadata"]
        size = metadata["environment"]["size"]
        if saved.get("encoding_version") != ENCODING_VERSION or architecture["input_dim"] != state_dimension(size):
            raise ValueError("Saved input encoding does not match this implementation")
        if architecture["hidden_layers"] != 2 or architecture["actions"] != list(ACTION_NAMES):
            raise ValueError("Saved network architecture does not match this implementation")
        if size < 4 or architecture["hidden_dim"] < 1:
            raise ValueError("Saved board/network dimensions are invalid")
        agent = cls(size, architecture["hidden_dim"], DQNConfig(**saved["learning"]), seed, device, training=False)
        agent.policy_net.load_state_dict(saved["state_dict"])
        if not all(torch.isfinite(value).all().item() for value in saved["state_dict"].values()):
            raise ValueError("Saved weights contain non-finite values")
        agent.policy_net.eval()
        agent.time_steps, agent.update_steps = int(saved["time_steps"]), int(saved["update_steps"])
        return agent, metadata


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
    steps_since_apple: int
    mean_loss: float | None


def run_episode(env, agent, epsilon, learn=True, viewer=None, controls=None,
                title="Deep Q-learning", summary=""):
    state = env.reset()
    total_reward, loss_total, updates = 0.0, 0.0, 0
    if viewer:
        viewer.show(env, title, epsilon, total_reward, summary=summary)
    while True:
        if controls and not viewer:
            controls.poll()
            if controls.paused:
                controls.show(env, title, epsilon, total_reward, summary=summary, throttle=False)
        values = agent.getQValues(state) if viewer else None
        action = agent.chooseUsingPolicy(state, epsilon, values)
        next_state, reward, done, info = env.step(action)
        if learn:
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                loss_total += loss
                updates += 1
        total_reward += reward
        if viewer:
            q_text = f"Q [S,L,R] = {values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f} (scaled rewards)"
            viewer.show(env, title, epsilon, total_reward, reward, action,
                        summary=q_text, reason=info.get("reason"))
        if done:
            return EpisodeResult(env.score, env.steps, total_reward, len(env.snake), info["reason"],
                                 env.steps_since_apple, loss_total / updates if updates else None)
        state = next_state


def summarize(results):
    reasons = Counter(result.reason for result in results)
    wins = [result.steps for result in results if result.reason == "win"]
    return {
        "games": len(results), "mean_score": float(np.mean([r.score for r in results])),
        "max_score": max(r.score for r in results),
        "mean_steps": float(np.mean([r.steps for r in results])),
        "winning_games": len(wins), "win_rate": len(wins) / len(results),
        "mean_win_steps": float(np.mean(wins)) if wins else None,
        "min_win_steps": min(wins) if wins else None, "max_win_steps": max(wins) if wins else None,
        "end_reasons": {reason: reasons[reason] for reason in ("wall", "body", "timeout", "win")},
    }


def print_summary(stats):
    print(f"{stats['games']} games | mean apples {stats['mean_score']:.2f} | "
          f"best {stats['max_score']} | win rate {stats['win_rate']:.1%}")
    print(f"Mean total steps (all games): {stats['mean_steps']:.2f}")
    if stats["mean_win_steps"] is None:
        print("Mean total steps (winning games): N/A (no wins)")
    else:
        print(f"Mean total steps ({stats['winning_games']} winning games): {stats['mean_win_steps']:.2f} "
              f"| min {stats['min_win_steps']} | max {stats['max_win_steps']}")
    print(f"End reasons: {stats['end_reasons']}")


def evaluate(agent, rewards, games=EVAL_GAMES, seed=EVAL_SEED, timeout_factor=TIMEOUT_FACTOR,
             viewer=None, title="DQN evaluation", on_game=None, start_game=1):
    env = DQNSnakeEnv(agent.size, seed, rewards, agent.config.gamma, timeout_factor)
    rng_state, was_training = agent.rng.getstate(), agent.policy_net.training
    agent.rng.seed(seed)
    agent.policy_net.eval()
    results = []
    try:
        if viewer and start_game > 1:
            viewer.show(env, f"Preparing batch game {start_game}", 0.0, 0.0,
                        summary=f"Replaying games 1..{start_game - 1} without animation", throttle=False)
        last_game = start_game + games - 1
        for game in range(1, last_game + 1):
            selected = game >= start_game
            result = run_episode(env, agent, 0.0, learn=False, viewer=viewer if selected else None,
                                 controls=viewer, title=f"{title} | Game {game}/{last_game}",
                                 summary="Greedy policy | Learning disabled")
            if selected:
                results.append(result)
                if on_game:
                    on_game(game, result)
    finally:
        agent.rng.setstate(rng_state)
        agent.policy_net.train(was_training)
    return summarize(results)
