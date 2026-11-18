"""Basic Snake environment (pure logic, no rendering dependencies).

Actions are relative to the current heading: 0 = straight, 1 = turn left, 2 = turn right.
Headings: 0 = up, 1 = right, 2 = down, 3 = left.
"""
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from feature_library import make_feature_set
from feature_library.length_bucket import length_bucket

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # (d_row, d_col) for up, right, down, left
STRAIGHT, LEFT, RIGHT = 0, 1, 2
N_ACTIONS = 3

R_APPLE = 10.0
R_DEATH = -10.0
R_STEP = -0.01
R_CLOSER = 0.1   # shaping: moved closer to / farther from the apple


@dataclass(frozen=True)
class RewardConfig:
    """Reward scheme of the environment.

    The defaults reproduce the original (legacy) rewards exactly, so tabular / MLP / CNN / hybrid
    agents keep working unchanged. The DQN v2 agent uses `RewardConfig.v2(gamma)` instead.

    shaping:
      "distance"  legacy: +closer if the head moved closer to the apple, -closer otherwise.
                  NOT potential-based -> it can change the optimal policy (it pushes the snake to
                  beeline to the apple, which is what traps long snakes in the endgame).
      "potential" F = gamma * phi(s') - phi(s), phi(s) = -closer * w(len) * dist(head, apple) / max_dist.
                  Potential-based shaping (Ng et al., 1999) provably keeps the optimal policy unchanged;
                  w(len) = 1 - len / area fades the hint out as the snake grows (hint is useful early,
                  harmful late when the safe path is rarely the shortest one).
      "none"      no shaping.
    """
    apple: float = R_APPLE
    death: float = R_DEATH
    step: float = R_STEP
    closer: float = R_CLOSER
    win: float = 0.0            # extra bonus for filling the board (legacy: none)
    shaping: str = "distance"
    gamma: float = 0.99         # only used by potential shaping (must match the agent's gamma)

    @staticmethod
    def v2(gamma=0.99):
        return RewardConfig(apple=1.0, death=-1.0, step=-0.005, closer=0.2,
                            win=10.0, shaping="potential", gamma=gamma)


class SnakeEnv:
    def __init__(self, size=15, seed=None, init_length=3, timeout_factor=2, features=None, rewards=None,
                 length_timeout=False):
        """length_timeout=False (legacy): fixed timeout = timeout_factor * area steps without eating.
        length_timeout=True: timeout = area / 2 + timeout_factor * length. A short snake can always reach the
        apple in under `size` steps, so long loops early on are pure waste and end the episode sooner; a long
        snake may need ~area steps of tail-chasing before the apple becomes reachable, so its budget grows.
        """
        self.size = size
        self.features = features if features is not None else make_feature_set()  # which state features to use
        self.init_length = init_length
        self.timeout_factor = timeout_factor
        self.length_timeout = length_timeout
        self.rewards = rewards if rewards is not None else RewardConfig()
        self.rng = random.Random(seed)
        self.reset()

    @property
    def timeout(self):
        """Max steps without eating before the episode ends (see __init__)."""
        area = self.size * self.size
        if self.length_timeout:
            return area // 2 + self.timeout_factor * len(self.snake)
        return self.timeout_factor * area

    # ------------------------------------------------------------------ core
    def reset(self):
        mid = self.size // 2
        self.heading = 1
        self.snake = deque((mid, mid - i) for i in range(self.init_length))  # head first
        self.body = set(self.snake)
        self.score = 0
        self.steps = 0
        self.steps_since_apple = 0
        self.done = False
        self._place_apple()
        return self.get_state()

    def _place_apple(self):
        free = [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.body]
        self.apple = self.rng.choice(free) if free else None

    def _dist_to_apple(self, head):
        return abs(head[0] - self.apple[0]) + abs(head[1] - self.apple[1])

    def step(self, action):
        assert not self.done, "call reset() first"
        if action == LEFT:
            self.heading = (self.heading - 1) % 4
        elif action == RIGHT:
            self.heading = (self.heading + 1) % 4

        dr, dc = DIRS[self.heading]
        old_head = self.snake[0]
        head = (old_head[0] + dr, old_head[1] + dc)
        self.steps += 1
        self.steps_since_apple += 1

        # Wall / self collision (the tail cell frees up unless an apple is eaten)
        eating = head == self.apple
        tail = self.snake[-1]
        cfg = self.rewards
        if not self._in_bounds(head) or (head in self.body and (head != tail or eating)):
            self.done = True
            # Potential of a terminal state is 0, so potential shaping adds -phi(s) here.
            return self.get_state(), cfg.death + self._terminal_shaping(), True, self._info(reason="collision")

        # Reward: small step penalty by default; +apple if eating (below);
        # shaping for moving toward / away from the apple; death reward on collision or timeout
        reward = cfg.step
        prev_dist = self._dist_to_apple(old_head)
        prev_potential = self._potential()
        self.snake.appendleft(head)
        self.body.add(head)

        if eating:
            self.score += 1
            reward = cfg.apple
            self.steps_since_apple = 0
            self._place_apple()
            if self.apple is None:  # board full -> win
                self.done = True
                return self.get_state(), reward + cfg.win - prev_potential, True, self._info(reason="win")
            if cfg.shaping == "potential":
                reward += cfg.gamma * self._potential() - prev_potential
        else:
            self.body.discard(self.snake.pop())
            self.body.add(head)  # head may equal the freed tail cell
            if cfg.shaping == "distance":
                reward += cfg.closer if self._dist_to_apple(head) < prev_dist else -cfg.closer
            elif cfg.shaping == "potential":
                reward += cfg.gamma * self._potential() - prev_potential

        if self.steps_since_apple > self.timeout:
            self.done = True
            return self.get_state(), cfg.death - prev_potential, True, self._info(reason="timeout")
        return self.get_state(), reward, False, self._info()

    def _potential(self):
        """phi(s) for potential-based shaping (0 when shaping is not 'potential')."""
        cfg = self.rewards
        if cfg.shaping != "potential" or self.apple is None:
            return 0.0
        area = self.size * self.size
        weight = 1.0 - len(self.snake) / area
        return -cfg.closer * weight * self._dist_to_apple(self.snake[0]) / (2 * (self.size - 1))

    def _terminal_shaping(self):
        # Called before the snake moves: F = gamma * 0 - phi(s)
        return -self._potential()

    def _info(self, **extra):
        # length/bucket are reported at the moment of the call (i.e. at death for terminal steps)
        return {"score": self.score, "length": len(self.snake),
                "bucket": int(length_bucket(len(self.snake), self.size)), **extra}

    def _in_bounds(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    # ----------------------------------------------------------------- state
    def is_danger(self, heading):
        """1 if moving one cell in `heading` hits a wall or the body (the tail cell counts as free)."""
        dr, dc = DIRS[heading]
        p = (self.snake[0][0] + dr, self.snake[0][1] + dc)
        return int(not self._in_bounds(p) or (p in self.body and p != self.snake[-1]))

    def get_state(self):
        """Discrete state tuple made of the selected features (see features.py)."""
        return self.features.encode(self)

    # --------------------------------------------------------------- display
    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for r, c in self.snake:
            grid[r][c] = "o"
        hr, hc = self.snake[0]
        grid[hr][hc] = "@"
        if self.apple:
            grid[self.apple[0]][self.apple[1]] = "*"
        return "\n".join(" ".join(row) for row in grid)

    def board_array(self):
        """int array for matplotlib: 0 empty, 1 body, 2 head, 3 apple."""
        a = np.zeros((self.size, self.size), dtype=int)
        for r, c in self.snake:
            a[r, c] = 1
        a[self.snake[0]] = 2
        if self.apple:
            a[self.apple] = 3
        return a


if __name__ == "__main__":
    # Sanity check with a random agent
    env = SnakeEnv(size=8, seed=0)
    scores, lengths = [], []
    for _ in range(1000):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(random.randrange(N_ACTIONS))
        scores.append(info["score"])
        lengths.append(env.steps)
    print(f"random agent: mean score {np.mean(scores):.2f}, max {max(scores)}, mean steps {np.mean(lengths):.1f}")
    env.reset()
    print(env.render())
    print("state:", env.get_state())
