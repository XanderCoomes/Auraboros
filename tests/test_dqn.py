"""Regression checks for observations, DQN targets, and checkpoint replay."""
from collections import deque

import numpy as np
import pytest
import torch

from models import build_model
from q_dqn import DQNAgent
from state_encoders import BoardEncoder, BoardSnakeEnv


torch.set_num_threads(1)


def make_agent(env, **settings):
    config = {"name": "rescnn4", "input_shape": env.encoder.shape(env.size),
              "context_dim": len(env.encoder.context_names)}
    return DQNAgent(config, batch_size=1, warmup=1, train_every=1, **settings)


@pytest.mark.parametrize("encoder_name", ["entities_tail", "entities", "directional", "ordered"])
@pytest.mark.parametrize("model_name", ["rescnn4", "rescnn8"])
@pytest.mark.parametrize("size", [4, 8, 9])
def test_encoders_and_models_roundtrip(encoder_name, model_name, size):
    encoder = BoardEncoder(encoder_name)
    env = BoardSnakeEnv(encoder=encoder, size=size, seed=0)
    observation = env.reset()
    assert observation.board.shape == encoder.shape(size)
    if encoder.encoding == "onehot":
        np.testing.assert_array_equal(observation.board.sum(axis=0), 1)
    model = build_model(model_name, input_shape=encoder.shape(size), context_dim=7)
    board = torch.from_numpy(observation.board).unsqueeze(0).repeat(2, *([1] * observation.board.ndim))
    context = torch.from_numpy(observation.context).unsqueeze(0).repeat(2, 1)
    q, activations = model(board, context, return_activations=True)
    assert q.shape == (2, 3)
    assert activations["features"].shape == (2, int(model_name[-1]))
    q.sum().backward()
    assert model.encoder[0].conv1.weight.grad is not None


def test_directional_turns_and_ordered_body():
    env = BoardSnakeEnv(size=8, seed=0)
    env.snake = deque([(2, 3), (2, 2), (3, 2), (3, 1)])
    env.body = set(env.snake)
    env.heading = 0
    env.apple = (1, 1)
    direction = BoardEncoder("directional", "binary").encode(env).board
    assert direction[2, 3, 0] == 1  # Head heading up.
    assert direction[2, 2, 1] == 1  # Body points right toward head.
    assert direction[3, 2, 0] == 1
    assert direction[3, 1, 1] == 1
    assert direction[1, 1, 4] == 1
    ordered = BoardEncoder("ordered").encode(env).board
    np.testing.assert_allclose([ordered[position + (0,)] for position in env.snake], np.array([4, 3, 2, 1]) / 64)


def test_observations_are_snapshots_and_include_heading_hunger():
    env = BoardSnakeEnv(size=8, seed=0)
    before = env.reset()
    saved = before.board.copy()
    env.step(1)
    after = env.get_state()
    np.testing.assert_array_equal(before.board, saved)
    assert before.context[1] == 1
    assert after.context[0] == 1
    assert after.context[-1] > before.context[-1]


def test_terminal_target_masks_bootstrap_and_output_can_be_negative():
    env = BoardSnakeEnv(size=8, seed=0)
    state = env.reset()
    agent = make_agent(env, reward_scale=1, target_every=2)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.target.output.bias.fill_(100)
    # Huber(0, -2) = 1.5. Any bootstrapping would make this much larger.
    assert agent.update(state, 0, -2, state, True) == pytest.approx(1.5)
    assert agent.q_values(state)[0] < 0
    assert torch.all(agent.target.output.bias == 100)
    agent.update(state, 0, -2, state, True)
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        torch.testing.assert_close(online, target)


def test_nonterminal_target_uses_discount():
    env = BoardSnakeEnv(size=8, seed=0)
    state = env.reset()
    agent = make_agent(env, gamma=0.5, reward_scale=1)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.target.output.bias.fill_(4)
    # Target = 1 + 0.5 * 4 = 3. Huber(0, 3) = 2.5.
    assert agent.update(state, 0, 1, state, False) == pytest.approx(2.5)


def test_checkpoint_roundtrip_and_replay_ownership(tmp_path):
    env = BoardSnakeEnv(size=8, seed=0)
    state = env.reset()
    agent = make_agent(env)
    next_state, reward, done, _ = env.step(0)
    agent.update(state, 0, reward, next_state, done)
    saved = state.board.copy()
    state.board[:] = 0
    np.testing.assert_array_equal(agent.memory[0][0].board, saved)
    expected = agent.q_values(next_state)
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint, {"encoder": env.encoder.config()})
    loaded, metadata = DQNAgent.load(checkpoint)
    np.testing.assert_array_equal(loaded.q_values(next_state), expected)
    assert metadata["encoder"] == env.encoder.config()
    assert loaded.updates == agent.updates


@pytest.mark.parametrize("encoder_name", ["entities_tail", "entities", "directional", "ordered"])
def test_full_board_win_and_collision_are_encodable(encoder_name):
    env = BoardSnakeEnv(encoder=BoardEncoder(encoder_name), size=4, seed=0)
    # Serpentine body ending with a single free cell immediately ahead.
    env.snake = deque([(0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (1, 2),
                       (1, 3), (2, 3), (2, 2), (2, 1), (2, 0), (3, 0),
                       (3, 1), (3, 2), (3, 3)])
    env.body = set(env.snake)
    env.apple, env.heading = (0, 3), 1
    observation, _, done, info = env.step(0)
    assert done and info["reason"] == "win" and env.apple is None
    assert np.isfinite(observation.board).all()
    env.reset()
    for _ in range(4):
        observation, _, done, info = env.step(0)
        if done:
            break
    assert done and info["reason"] == "collision"
    assert np.isfinite(observation.board).all()


def test_ordered_onehot_rejected():
    with pytest.raises(ValueError, match="Ordered"):
        BoardEncoder("ordered", "onehot")
