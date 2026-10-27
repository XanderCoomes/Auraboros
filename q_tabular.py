"""Tabular Q-learning agent over the compact state tuple from SnakeEnv.get_state()."""
import pickle
import random

import numpy as np

from snake_env import N_ACTIONS

# Number of possible values for each element of the state tuple from SnakeEnv.get_state():
# (danger straight, danger left, danger right, apple row sign, apple col sign, heading)
# Must stay in sync with get_state(); 2*2*2*3*3*4 = 288 states in total.
STATE_SHAPE = (2, 2, 2, 3, 3, 4)


def get_row_by_state(state):
    # Flatten the state tuple into one integer 0..287 = the row index into the Q-table
    return int(np.ravel_multi_index(state, STATE_SHAPE))


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95, seed=None):
        self.alpha = alpha  # learning rate: how far one update moves Q toward the target
        self.gamma = gamma  # discount factor: how much future reward counts
        self.rng = random.Random(seed)
        # Q-table: one row per state, one column per action (straight / left / right).
        # Starts at all zeros, i.e. the agent knows nothing yet.
        self.q = np.zeros((int(np.prod(STATE_SHAPE)), N_ACTIONS))

    def act(self, state, epsilon=0.0):
        """epsilon-greedy; ties broken randomly."""
        # Explore: with probability epsilon take a random action
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        # Exploit: take the action with the highest Q value in this state's rowwhgd
        actionValues = self.q[get_row_by_state(state)]
        bestActions = np.flatnonzero(actionValues == actionValues.max())  # all actions tied for the max
        return int(self.rng.choice(list(bestActions)))

    def update(self, stateBeforeAction, selectedAction, reward, stateAfterAction, done):
        # Q-learning update:
        #   Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        rowBefore, rowAfter = get_row_by_state(stateBeforeAction), get_row_by_state(stateAfterAction)
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
            self.q = pickle.load(f)
