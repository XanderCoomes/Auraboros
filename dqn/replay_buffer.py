
import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    #store a new experience
    def push(self, state, action, reward, next_state, done):    
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    #Randomly retrieves a batch of experiences
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    #Return the num of stored experiences
    def __len__(self):
        return len(self.buffer)


# MLP
# experience = (
#     state,       # Current state (13 features)
#     action,      # 0, 1, or 2
#     reward,      # e.g. -0.1
#     next_state,  # Next state (13 features)
#     done         # True or False
# )

#CNN
# experience = (
#     (board, heading),       # Current state
#     action,
#     reward,
#     (next_board, next_heading),
#     done
# )