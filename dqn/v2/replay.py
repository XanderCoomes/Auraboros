"""Prioritized replay buffer with n-step returns, stored in preallocated numpy arrays.

Why preallocated arrays instead of a deque of tuples?
    The old buffer stored Python tuples and ran np.stack on 64 tuples every train step. Preallocated
    arrays let us gather a batch with one fancy-indexing call, which is much faster. Boards are stored as
    float16 (values are 0/1 or k/64, exactly representable) to halve memory.

Why prioritized replay (Schaul et al., 2016)?
    Long-snake states are rare (most episodes die early during training), but they decide whether the agent
    wins. Sampling transitions in proportion to their TD error replays the surprising (usually late-game or
    fatal) transitions more often. Importance-sampling weights w_i = (N * P(i))^-beta correct the bias.

    Sampling uses cumsum + searchsorted over the priority array: O(N) but fully vectorized, simple, and fast
    enough for N <= a few hundred thousand.

Why n-step returns?
    R = r_t + g r_{t+1} + ... + g^{n-1} r_{t+n-1} + g^n max Q(s_{t+n}). Rewards (apple, death) propagate n
    steps per update instead of 1, which speeds up learning of long-horizon consequences.
"""
from collections import deque

import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity, board_shape, n_scalars, feat_shape, n_actions=3,
                 alpha=0.6, eps=1e-3, seed=0):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.boards = np.zeros((capacity, *board_shape), dtype=np.float16)
        self.next_boards = np.zeros((capacity, *board_shape), dtype=np.float16)
        self.scalars = np.zeros((capacity, n_scalars), dtype=np.float32)
        self.next_scalars = np.zeros((capacity, n_scalars), dtype=np.float32)
        self.feats = np.zeros((capacity, *feat_shape), dtype=np.float32)
        self.next_feats = np.zeros((capacity, *feat_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)   # gamma^k, 0 if the episode ended
        self.next_masks = np.ones((capacity, n_actions), dtype=bool)
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.max_priority = 1.0
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, discount, next_mask):
        i = self.pos
        self.boards[i], self.scalars[i], self.feats[i] = state
        self.next_boards[i], self.next_scalars[i], self.next_feats[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.discounts[i] = discount
        self.next_masks[i] = next_mask
        self.priorities[i] = self.max_priority  # new data is replayed at least once soon
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        p = self.priorities[:self.size]
        cdf = np.cumsum(p)
        total = cdf[-1]
        targets = (np.arange(batch_size) + self.rng.random(batch_size)) * (total / batch_size)  # stratified
        idx = np.minimum(np.searchsorted(cdf, targets), self.size - 1)
        probs = p[idx] / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()
        batch = dict(
            boards=self.boards[idx], scalars=self.scalars[idx], feats=self.feats[idx],
            actions=self.actions[idx], rewards=self.rewards[idx], discounts=self.discounts[idx],
            next_boards=self.next_boards[idx], next_scalars=self.next_scalars[idx],
            next_feats=self.next_feats[idx], next_masks=self.next_masks[idx],
            weights=weights.astype(np.float32),
        )
        return idx, batch

    def update_priorities(self, idx, td_errors):
        pr = (np.abs(td_errors) + self.eps) ** self.alpha
        self.priorities[idx] = pr
        self.max_priority = max(self.max_priority, float(pr.max()))


class NStepAccumulator:
    """Turns 1-step transitions of ONE environment into n-step transitions.

    push() returns the list of finished n-step transitions:
        (state, action, n_step_return, state_after_k_steps, discount = gamma^k or 0, mask_after)
    When an episode ends, all pending transitions are flushed with discount 0 (no bootstrap).
    """

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = deque()

    def push(self, state, action, reward, next_state, done, next_mask):
        self.queue.append((state, action, reward))
        out = []
        if done:
            while self.queue:
                out.append(self._make(next_state, next_mask, terminal=True))
                self.queue.popleft()
        elif len(self.queue) == self.n:
            out.append(self._make(next_state, next_mask, terminal=False))
            self.queue.popleft()
        return out

    def _make(self, next_state, next_mask, terminal):
        ret = 0.0
        for k, (_, _, r) in enumerate(self.queue):
            ret += (self.gamma ** k) * r
        state, action, _ = self.queue[0]
        discount = 0.0 if terminal else self.gamma ** len(self.queue)
        return state, action, ret, next_state, discount, next_mask
