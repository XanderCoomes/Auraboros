"""Uniform experience replay: a fixed-size ring buffer of 1-step transitions.

Each stored transition is (s, a, r, s', mask(s'), done):
    obs        s, the state the action was taken in          (4, n, n) uint8
    action     a                                              int
    reward     r, the reward received for that step           float
    next_obs   s', the state after the step                   (4, n, n) uint8
    next_mask  allowed actions in s' (used in the masked max) (4,) bool
    done       1.0 if s' is terminal, else 0.0                float

Compared with complex_snake/replay.py there is no n-step accumulation, no
prioritized sampling (sum tree) and no symmetry augmentation: every stored
transition is equally likely to be sampled.
"""
import numpy as np

from snake_env import N_ACTIONS


class ReplayBuffer:
    FIELDS = ("obs", "action", "reward", "next_obs", "next_mask", "done")

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        # Pre-allocate one array per field. Observations are uint8, so 300k transitions
        # take about 150 MB.
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0     # where the next transition is written
        self.count = 0   # how many slots are filled (stops growing at capacity)

    def __len__(self):
        return self.count

    def add(self, obs, action, reward, next_obs, next_mask, done):
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.next_mask[i] = next_mask
        self.done[i] = float(done)
        # Once full, the oldest transition is overwritten first.
        self.ptr = (i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Draw `batch_size` transitions uniformly at random (with replacement).

        Returns a dict of numpy arrays keyed by FIELDS.
        """
        idx = rng.integers(0, self.count, batch_size)
        return {f: getattr(self, f)[idx] for f in self.FIELDS}
