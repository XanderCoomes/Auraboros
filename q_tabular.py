"""Tabular Q-learning agent over the discrete state tuple from SnakeEnv.get_state() (features in features.py)."""
import pickle
import random

import numpy as np

from snake_env import N_ACTIONS


def get_row_by_state(state, shape):
    # Flatten the state tuple into one integer = the row index into the Q-table.
    # `shape` is FeatureSet.shape (number of possible values per feature).
    return int(np.ravel_multi_index(state, shape))


class QAgent:
    def __init__(self, shape, alpha=0.1, gamma=0.95, seed=None):
        self.shape = tuple(shape)  # per-feature cardinalities, from FeatureSet.shape
        self.alpha = alpha  # learning rate: how far one update moves Q toward the target
        self.gamma = gamma  # discount factor: how much future reward counts
        self.rng = random.Random(seed)
        # Q-table: one row per state, one column per action (straight / left / right).
        # Starts at all zeros, i.e. the agent knows nothing yet.
        self.q = np.zeros((int(np.prod(self.shape)), N_ACTIONS))

    def act(self, state, epsilon=0.0):
        """epsilon-greedy; ties broken randomly."""
        # Explore: with probability epsilon take a random action
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        # Exploit: take the action with the highest Q value in this state's row
        actionValues = self.q[get_row_by_state(state, self.shape)]
        bestActions = np.flatnonzero(actionValues == actionValues.max())  # all actions tied for the max
        return int(self.rng.choice(list(bestActions)))

    def update(self, stateBeforeAction, selectedAction, reward, stateAfterAction, done):
        # Q-learning update:
        #   Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        rowBefore = get_row_by_state(stateBeforeAction, self.shape)
        rowAfter = get_row_by_state(stateAfterAction, self.shape)
        # Target = reward received + discounted best value of the next state.
        # If the episode ended there is no next state, so the target is just the reward.
        target = reward if done else reward + self.gamma * self.q[rowAfter].max()
        # Move Q(s,a) a fraction alpha of the way toward the target (target - Q is the error)
        self.q[rowBefore, selectedAction] += self.alpha * (target - self.q[rowBefore, selectedAction])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path):
        with open(path, "rb") as f:
            q = pickle.load(f)
        if q.shape[0] != int(np.prod(self.shape)):
            raise ValueError(f"Q-table has {q.shape[0]} states but the feature set needs {int(np.prod(self.shape))}")
        self.q = q
