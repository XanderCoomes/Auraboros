"""Basic Snake environment (pure logic, no rendering dependencies).

Actions are relative to the current heading: 0 = straight, 1 = turn left, 2 = turn right.
Headings: 0 = up, 1 = right, 2 = down, 3 = left.
"""
import random
from collections import deque

import numpy as np

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # (d_row, d_col) for up, right, down, left
STRAIGHT, LEFT, RIGHT = 0, 1, 2
N_ACTIONS = 3

R_APPLE = 100.0
R_DEATH = -6350.0
R_STEP = -0.01
R_CLOSER = 0.1   # shaping: moved closer to / farther from the apple


class SnakeEnv:
    def __init__(self, size=15, seed=None, init_length=3, timeout_factor=2):
        self.size = size
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
        if not self._in_bounds(head) or (head in self.body and (head != tail or eating)):
            self.done = True
            return self.get_state(), R_DEATH, True, {"score": self.score, "reason": "collision"}

        # Reward: small step penalty by default; +R_APPLE if eating (below);
        # +/-R_CLOSER shaping for moving toward / away from the apple; R_DEATH on collision or timeout
        reward = R_STEP
        prev_dist = self._dist_to_apple(old_head)
        self.snake.appendleft(head)
        self.body.add(head)

        if eating:
            self.score += 1
            reward = R_APPLE
            self.steps_since_apple = 0
            self._place_apple()
            if self.apple is None:  # board full -> win
                self.done = True
                return self.get_state(), reward, True, {"score": self.score, "reason": "win"}
        else:
            self.body.discard(self.snake.pop())
            self.body.add(head)  # head may equal the freed tail cell
            reward += R_CLOSER if self._dist_to_apple(head) < prev_dist else -R_CLOSER

        if self.steps_since_apple > self.timeout:
            self.done = True
            return self.get_state(), R_DEATH, True, {"score": self.score, "reason": "timeout"}
        return self.get_state(), reward, False, {"score": self.score}

    def _in_bounds(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    # ----------------------------------------------------------------- state
    def _is_danger(self, heading):
        dr, dc = DIRS[heading]
        p = (self.snake[0][0] + dr, self.snake[0][1] + dc)
        return int(not self._in_bounds(p) or (p in self.body and p != self.snake[-1]))

    def get_state(self):
        """Compact tuple state for tabular Q-learning (288 states).

        (danger straight, danger left, danger right, apple row sign, apple col sign, heading)
        """
        if self.apple is None:  # board is full, no apple left
            return (0, 0, 0, 1, 1, self.heading)
        h = self.heading
        sgn = lambda x: (x > 0) - (x < 0) + 1  # -1/0/+1 -> 0/1/2
        return (self._is_danger(h),                 # wall/body directly ahead?
                self._is_danger((h - 1) % 4),       # wall/body to the left?
                self._is_danger((h + 1) % 4),       # wall/body to the right?
                sgn(self.apple[0] - self.snake[0][0]),  # apple row vs head: 0 above, 1 same row, 2 below
                sgn(self.apple[1] - self.snake[0][1]),  # apple col vs head: 0 left, 1 same col, 2 right
                h)                                  # current heading (0 up, 1 right, 2 down, 3 left)

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
