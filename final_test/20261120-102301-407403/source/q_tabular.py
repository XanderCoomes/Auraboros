"""Tabular Q-learning agent over the state tuple from SnakeEnv.get_state()."""
import pickle
import random

import numpy as np

from features import DEFAULT_FEATURES, StateSpec
from snake_env import N_ACTIONS


class QAgent:
    def __init__(self, spec=None, alpha=0.1, gamma=0.95, seed=None):
        # spec: the StateSpec the env uses (env.spec); fixes the Q-table size and state -> row mapping
        self.spec = spec or StateSpec()
        self.alpha = alpha  # learning rate: how far one update moves Q toward the target
        self.gamma = gamma  # discount factor: how much future reward counts
        self.rng = random.Random(seed)
        # Q-table: one row per state, one column per action (straight / left / right).
        # Starts at all zeros, i.e. the agent knows nothing yet.
        self.q = np.zeros((self.spec.num_states, N_ACTIONS))

    def act(self, state, epsilon=0.0):
        """epsilon-greedy; ties broken randomly."""
        # Explore: with probability epsilon take a random action
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        # Exploit: take the action with the highest Q value in this state's row
        actionValues = self.q[self.spec.row(state)].tolist()  # plain floats: much faster than numpy on 3 values
        best = max(actionValues)
        bestActions = [a for a, v in enumerate(actionValues) if v == best]  # all actions tied for the max
        return self.rng.choice(bestActions)  # always draw, even with no tie, so the RNG stream (and results) stay reproducible

    def update(self, stateBeforeAction, selectedAction, reward, stateAfterAction, done):
        # Q-learning update:
        #   Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        rowBefore, rowAfter = self.spec.row(stateBeforeAction), self.spec.row(stateAfterAction)
        # Target = reward received + discounted best value of the next state.
        # If the episode ended there is no next state, so the target is just the reward.
        target = reward if done else reward + self.gamma * max(self.q[rowAfter].tolist())
        # Move Q(s,a) a fraction alpha of the way toward the target (target - Q is the error)
        q = self.q.item(rowBefore, selectedAction)
        self.q[rowBefore, selectedAction] = q + self.alpha * (target - q)

    def save(self, path):
        # Store the feature names with the table: a Q-table is meaningless without its state definition
        with open(path, "wb") as f:
            pickle.dump({"features": self.spec.names, "q": self.q}, f)

    @staticmethod
    def load(path, **kwargs):
        """Rebuild an agent (and its StateSpec) from a saved Q-table."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, np.ndarray):  # old format: bare array, always the default 288-state features
            data = {"features": DEFAULT_FEATURES, "q": data}
        agent = QAgent(StateSpec(data["features"]), **kwargs)
        assert agent.q.shape == data["q"].shape, f"Q-table shape {data['q'].shape} != spec {agent.q.shape}"
        agent.q = data["q"]
        return agent
