
import numpy as np
import torch

from snake_env import SnakeEnv

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer

from dqn.encoders.engineered import EngineeredEncoder
from dqn.encoders.board import BoardEncoder

from dqn.models.mlp import DQNMLP
from dqn.models.cnn import DQNCNN


def test_agent(model_type="mlp"):
    print(f"\nTesting {model_type.upper()} DQN Agent")

    # 1. Create environment
    env = SnakeEnv(size=8, seed=42)
    env.reset()

    # 2. Select encoder and model
    if model_type == "mlp":
        encoder = EngineeredEncoder()

        model = DQNMLP(
            input_dim=encoder.input_shape[0]
        )

    elif model_type == "cnn":
        encoder = BoardEncoder()

        model = DQNCNN()

    else:
        raise ValueError("model_type must be 'mlp' or 'cnn'")

    # 3. Create replay buffer and agent
    replay_buffer = ReplayBuffer(capacity=10)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    agent = DQNAgent(
        model=model,
        replay_buffer=replay_buffer,
        gamma=0.95,
        seed=42,
        device=device
    )

    # 4. Encode current environment state
    state = encoder.encode(env)

    # 5. Test action selection
    action = agent.act(state, epsilon=0.0)

    print("Selected action:", action)

    assert action in [0, 1, 2]

    # 6. Test Q-value inspection
    q_values = agent.get_q_values(state)

    print("Q-values:", q_values)
    print("Q-values shape:", q_values.shape)

    assert q_values.shape == (3,)
    assert np.all(np.isfinite(q_values))

    # 7. Test exploratory action selection
    random_action = agent.act(state, epsilon=1.0)

    print("Random action:", random_action)

    assert random_action in [0, 1, 2]

    print(f"{model_type.upper()} test passed!")


if __name__ == "__main__":
    test_agent("mlp")
    test_agent("cnn")

    print("\nAll agent tests passed!")
