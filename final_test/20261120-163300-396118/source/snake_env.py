"""Basic Snake environment (pure logic, no rendering dependencies).

Actions are relative to the current heading: 0 = straight, 1 = turn left, 2 = turn right.
Headings: 0 = up, 1 = right, 2 = down, 3 = left.
"""
import random
from collections import deque

import numpy as np

from features import StateSpec
from rewards import RewardSpec

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # (d_row, d_col) for up, right, down, left
STRAIGHT, LEFT, RIGHT = 0, 1, 2
N_ACTIONS = 3


class SnakeEnv:
    def __init__(self, size=15, seed=None, init_length=3, timeout_factor=2, features=None, rewards=None):
        self.size = size
        self.spec = StateSpec(features)  # which state features get_state() returns; None = features.DEFAULT_FEATURES
        # reward terms + weights (see rewards.py); None = rewards.DEFAULT_REWARDS at default weights
        self.rewards = rewards if isinstance(rewards, RewardSpec) else RewardSpec(rewards)
        self.init_length = init_length
        self.timeout = timeout_factor * size * size  # max steps without eating
        self.rng = random.Random(seed)
        self.reset()

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
        # Event describing what happened this step; the reward terms in rewards.py read it
        ev = {"outcome": "move", "prev_head": old_head, "head": head,
              "prev_dist": self._dist_to_apple(old_head), "dist": None}
        if not self._in_bounds(head) or (head in self.body and (head != tail or eating)):
            ev["outcome"] = "collision"
            return self._finish(ev)

        ev["dist"] = self._dist_to_apple(head)  # distance to the apple being chased, before any respawn
        self.snake.appendleft(head)
        self.body.add(head)

        if eating:
            self.score += 1
            ev["outcome"] = "apple"
            self.steps_since_apple = 0
            self._place_apple()
            if self.apple is None:  # board full -> win
                ev["outcome"] = "win"
        else:
            self.body.discard(self.snake.pop())
            self.body.add(head)  # head may equal the freed tail cell
            if self.steps_since_apple > self.timeout:
                ev["outcome"] = "timeout"
        return self._finish(ev)

    def _finish(self, ev):
        """Score the step with the reward terms and package (state, reward, done, info)."""
        done = ev["outcome"] in ("collision", "timeout", "win")
        self.done = done
        info = {"score": self.score, "event": ev}
        if done:
            info["reason"] = ev["outcome"]
        return self.get_state(), self.rewards.total(self, ev), done, info

    def _in_bounds(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    # ----------------------------------------------------------------- state
    def _is_danger(self, heading):
        dr, dc = DIRS[heading]
        p = (self.snake[0][0] + dr, self.snake[0][1] + dc)
        return int(not self._in_bounds(p) or (p in self.body and p != self.snake[-1]))

    def get_state(self):
        """State tuple for tabular Q-learning: one int per feature selected in self.spec (see features.py)."""
        return self.spec.encode(self)

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
