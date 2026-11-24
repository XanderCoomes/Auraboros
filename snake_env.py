"""Snake on an N x N grid, set up for DQN (simplified copy of complex_snake/snake_env.py).

What is the same as complex_snake
---------------------------------
* Safety mask: an action is "safe" if, after taking it, the head can still reach
  the tail through free cells. Starting from a safe state there is always at least
  one safe action, so an agent that only picks safe actions can never crash. The
  only way left to lose is the hunger limit (looping without eating).
* Observation: 4 x N x N uint8 planes
      0 head              (1 at the head cell, 0 elsewhere)
      1 body time-to-free (each body cell = number of steps until it is vacated;
                           head = L, ..., tail = 1, empty = 0)
      2 apple             (1 at the apple cell, 0 elsewhere)
      3 hunger            (steps since the last apple, written into every cell)
  The network divides by `SnakeEnv.obs_scale` to bring everything into [0, 1].
* Rewards: apple +1, win +10, crash -10, hunger timeout -10, no per-step penalty.

What was removed
----------------
* The 8 board symmetries (used for data augmentation), `clone()` (lookahead),
  alternative observation modes, and mid-game start helpers.
"""
from collections import deque
import random

import numpy as np

# Absolute actions. The index of each action matches the row offsets in DIRS.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # (d_row, d_col) for each action
N_ACTIONS = 4
N_CHANNELS = 4
MASK_MODES = ("safe", "legal", "none")


class Bitboard:
    """Flood fill on an N x N board, using one Python int as a set of cells.

    Cell (r, c) is bit number r * n + c. Python ints have unlimited precision, so a
    64-cell board fits in one int and "expand the reachable region by one step" becomes
    a handful of shifts and ANDs. That is much faster than a BFS over Python tuples,
    which matters because the safety mask runs a flood fill for every action, every step.
    """

    def __init__(self, n):
        self.n = n
        self.full = (1 << (n * n)) - 1                     # every cell on the board
        col0 = sum(1 << (r * n) for r in range(n))         # bits of column 0
        self.not_col0 = self.full & ~col0                  # stops wrap-around when shifting right
        self.not_last_col = self.full & ~(col0 << (n - 1))  # stops wrap-around when shifting left

    def bit(self, cell):
        return 1 << (cell[0] * self.n + cell[1])

    def expand(self, x):
        """All cells that are in x or 4-adjacent to a cell in x."""
        n = self.n
        return ((x << n) | (x >> n)                    # down / up one row
                | ((x << 1) & self.not_col0)           # right one column (no wrap)
                | ((x >> 1) & self.not_last_col)       # left one column (no wrap)
                ) & self.full

    def reachable(self, start, goal, free):
        """Can `goal` be reached from `start` moving only through `free` cells?"""
        free |= goal  # the goal itself counts as walkable
        reach = start
        while True:
            if reach & goal:
                return True
            new = reach | (self.expand(reach) & free)
            if new == reach:  # region stopped growing: goal is cut off
                return False
            reach = new


class SnakeEnv:
    def __init__(self, n=8, hunger_limit=None, r_apple=1.0, r_death=-10.0, r_win=10.0,
                 r_hunger=-10.0, r_step=0.0, mask_mode="safe", seed=None):
        assert n >= 4, "board must be at least 4x4"
        assert mask_mode in MASK_MODES
        self.n = n
        self.cells = n * n
        # Steps allowed without eating before the episode ends. Default 2 * cells (128 on 8x8).
        self.hunger_limit = hunger_limit or 2 * self.cells
        assert self.hunger_limit <= 255, "hunger plane is stored as uint8"
        self.r_apple, self.r_death, self.r_win = r_apple, r_death, r_win
        self.r_hunger, self.r_step = r_hunger, r_step
        self.mask_mode = mask_mode
        self.bb = Bitboard(n)
        self.rng = random.Random(seed)  # only used for apple placement
        # Per-channel divisors that map each plane into [0, 1]; the model applies them.
        # head: 0/1, body: 0..cells, apple: 0/1, hunger: 0..hunger_limit
        self.obs_scale = np.array([1, self.cells, 1, self.hunger_limit], dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, seed=None):
        """Start a new game and return the first observation."""
        if seed is not None:
            self.rng.seed(seed)
        # Length-3 snake in the top row, head at (0, 3), facing right.
        # `body` is a deque ordered head first: body[0] = head, body[-1] = tail.
        self.body = deque([(0, 3), (0, 2), (0, 1)])
        self.occ = set(self.body)  # fast "is this cell occupied?" lookups
        self.occ_bits = 0          # the same set, as a bitboard (for the flood fill)
        for c in self.body:
            self.occ_bits |= self.bb.bit(c)
        self.hunger = 0            # steps since the last apple
        self.steps = 0
        self.apples = 0
        self.done = False
        self.won = False
        self.reason = None         # why the episode ended: 'wall', 'body', 'hunger' or 'win'
        self.apple = self._spawn_apple()
        self._mask = self._compute_mask()
        return self.obs()

    @property
    def length(self):
        return len(self.body)

    @property
    def head(self):
        return self.body[0]

    @property
    def mask(self):
        """Boolean array (4,) of allowed actions in the current state. Never all False."""
        return self._mask

    def next_cell(self, action):
        dr, dc = DIRS[action]
        return (self.body[0][0] + dr, self.body[0][1] + dc)

    def _spawn_apple(self):
        free = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in self.occ]
        return self.rng.choice(free) if free else None

    # ------------------------------------------------------------ observation
    def obs(self):
        """Return the current state as a (4, n, n) uint8 array (see the module docstring)."""
        n = self.n
        o = np.zeros((N_CHANNELS, n, n), dtype=np.uint8)

        head = self.body[0]
        o[0, head[0], head[1]] = 1
        for ttf, (r, c) in enumerate(reversed(self.body), start=1):  # tail = 1, ..., head = L
            o[1, r, c] = ttf
        if self.apple is not None:
            o[2, self.apple[0], self.apple[1]] = 1
        o[3] = min(self.hunger, 255)

        return o

    # ------------------------------------------------------------------- mask
    def _compute_mask(self):
        """Which of the 4 actions are allowed right now.

        legal: the move does not hit a wall or the body (moving into the cell the
               tail is about to leave is fine, unless we eat and the tail stays).
        safe:  legal, AND after the move the new head can still reach the new tail.
               Following your own tail is always possible, so a safe state stays safe.
        Falls back safe -> legal -> all four so the mask is never empty.
        """
        if self.mask_mode == "none":
            return np.ones(N_ACTIONS, dtype=bool)
        n, bb = self.n, self.bb
        legal = np.zeros(N_ACTIONS, dtype=bool)
        safe = np.zeros(N_ACTIONS, dtype=bool)
        hr, hc = self.body[0]
        tail = self.body[-1]
        L = len(self.body)
        for a, (dr, dc) in enumerate(DIRS):
            r, c = hr + dr, hc + dc
            if not (0 <= r < n and 0 <= c < n):
                continue                                  # wall
            nh = (r, c)
            eat = nh == self.apple
            if nh in self.occ and not (nh == tail and not eat):
                continue                                  # body (the tail moves away unless we eat)
            legal[a] = True
            if self.mask_mode != "safe":
                continue
            if eat and L + 1 == self.cells:
                safe[a] = True                            # this move fills the board: a win
                continue
            # Simulate the body after the move, then flood fill from new head to new tail.
            if eat:   # snake grows, tail stays put
                new_tail = tail
                occ = self.occ_bits | bb.bit(nh)
            else:     # tail moves forward one cell
                new_tail = self.body[-2]
                occ = (self.occ_bits & ~bb.bit(tail)) | bb.bit(nh)
            safe[a] = bb.reachable(bb.bit(nh), bb.bit(new_tail), bb.full & ~occ)
        if safe.any():
            return safe
        if legal.any():
            return legal
        return np.ones(N_ACTIONS, dtype=bool)  # every move loses, so let the agent pick

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Apply one action. Returns (obs, reward, done, info).

        info['reason'] is None while the game runs, then one of 'wall', 'body', 'hunger', 'win'.
        """
        assert not self.done, "episode is over, call reset()"
        self.steps += 1
        nh = self.next_cell(action)
        r, c = nh
        if not (0 <= r < self.n and 0 <= c < self.n):
            return self._end(self.r_death, "wall")
        eat = nh == self.apple
        if not eat:                       # move the tail first so the head may enter its cell
            tail = self.body.pop()
            self.occ.discard(tail)
            self.occ_bits &= ~self.bb.bit(tail)
        if nh in self.occ:
            return self._end(self.r_death, "body")
        self.body.appendleft(nh)
        self.occ.add(nh)
        self.occ_bits |= self.bb.bit(nh)
        reward = self.r_step
        if eat:
            self.apples += 1
            self.hunger = 0
            reward += self.r_apple
            if len(self.body) == self.cells:  # board full: game won
                self.won = True
                self.apple = None
                return self._end(reward + self.r_win, "win")
            self.apple = self._spawn_apple()
        else:
            self.hunger += 1
            if self.hunger >= self.hunger_limit:
                return self._end(reward + self.r_hunger, "hunger")
        self._mask = self._compute_mask()
        return self.obs(), reward, False, self._info()

    def _end(self, reward, reason):
        self.done = True
        self.reason = reason
        # The mask of a terminal state is never used for bootstrapping (done = 1 zeroes it
        # out), but it must not be all False or a masked max would give -inf.
        self._mask = np.ones(N_ACTIONS, dtype=bool)
        return self.obs(), reward, True, self._info()

    def _info(self):
        return {"reason": self.reason, "length": len(self.body), "steps": self.steps,
                "won": self.won, "apples": self.apples}

    # ----------------------------------------------------------------- render
    def render(self):
        """Text picture of the board: @ head, o body, * apple."""
        g = [["." for _ in range(self.n)] for _ in range(self.n)]
        for (r, c) in self.body:
            g[r][c] = "o"
        hr, hc = self.body[0]
        g[hr][hc] = "@"
        if self.apple is not None:
            g[self.apple[0]][self.apple[1]] = "*"
        return "\n".join(" ".join(row) for row in g) + \
            f"\nlen={len(self.body)} steps={self.steps} hunger={self.hunger}"
