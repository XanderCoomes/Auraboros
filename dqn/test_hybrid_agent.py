
import numpy as np
import torch

from snake_env import SnakeEnv
from dqn.agent import DQNAgent
from dqn.models.hybrid import DQNHybrid
from dqn.encoders.board import BoardEncoderWithTime
from dqn.features.tail_features import get_feature_matrix
from dqn.replay_buffer import ReplayBuffer


def test_hybrid_agent():

    print("\nTesting Hybrid DQN Agent")

    env = SnakeEnv(size=8, seed=42)

    encoder = BoardEncoderWithTime()
    board, heading = encoder.encode(env)
    features = get_feature_matrix(env)

    state = (board, heading, features)

    model = DQNHybrid()

    replay_buffer = ReplayBuffer(capacity=1000)

    agent = DQNAgent(
        model=model,
        replay_buffer=replay_buffer,
        seed=21,
    )

    # Test action selection.
    action = agent.act(state, epsilon=0.0)
    q_values = agent.get_q_values(state)

    print("Selected action:", action)
    print("Q-values:", q_values)
    print("Q-values shape:", q_values.shape)

    assert action in range(3)
    assert q_values.shape == (3,)
    assert np.isfinite(q_values).all()

    # Test random action.
    random_action = agent.act(state, epsilon=1.0)
    assert random_action in range(3)

    # Test experience replay.
    next_state = state

    for _ in range(64):
        replay_buffer.push(
            state,
            action,
            1.0,
            next_state,
            False,
        )

    loss = agent.train_step(batch_size=64)

    print("Training loss:", loss)

    assert loss is not None
    assert np.isfinite(loss)

    print("Hybrid agent test passed!")


if __name__ == "__main__":
    test_hybrid_agent()
