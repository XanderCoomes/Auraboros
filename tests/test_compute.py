import numpy as np
import pytest
import torch
from torch import nn

from flop_accounting import FlopBudgetReached, FlopCounter, forward_flops
from q_dqn import DQNAgent
from state_encoders import BoardSnakeEnv
from train import evaluate_details, run_episode


torch.set_num_threads(1)


def make_agent():
    env = BoardSnakeEnv(size=8, seed=0)
    agent = DQNAgent({"name": "rescnn8", "input_shape": env.encoder.shape(8), "context_dim": 7},
                     batch_size=2, warmup=2, train_every=1)
    return env, agent


def test_forward_count_matches_hand_calculation():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 3)

        def forward(self, board, context):
            return self.fc(board)

    model = Tiny()
    assert forward_flops(model, {"input_shape": (2,), "context_dim": 0}) == 12
    assert model.training
    assert not model.fc._forward_hooks
    meter = FlopCounter(12, 9, budget=222)
    assert meter.update_cost(2) == 222
    meter.require(222)
    meter.record_update(2, pretraining=True)
    assert meter.total == meter.pretraining == 222
    with pytest.raises(FlopBudgetReached):
        meter.require(1)


def test_evaluation_and_random_actions_do_not_spend_training_budget():
    env, agent = make_agent()
    agent.compute.training = True
    agent.act(env.get_state(), 1.0)
    assert agent.compute.total == 0
    agent.act(env.get_state(), 0.0)
    assert agent.compute.total == agent.compute.forward
    before = agent.compute.total
    evaluate_details(env, agent, n=2)
    assert agent.compute.total == before
    assert agent.compute.evaluation > 0


def test_replay_budget_refusal_preserves_weights_and_updates():
    env, agent = make_agent()
    state = env.get_state()
    agent.compute.budget = agent.compute.update_cost(2) - 1
    before = {name: value.clone() for name, value in agent.online.state_dict().items()}
    agent.update(state, 0, 0, state, False)
    with pytest.raises(FlopBudgetReached):
        agent.update(state, 0, 0, state, False)
    assert agent.updates == agent.compute.total == 0
    for name, value in agent.online.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_update_and_pretraining_charge_expected_cost(tmp_path):
    env, agent = make_agent()
    state = env.get_state()
    agent.update(state, 0, 0, state, False)
    agent.update(state, 0, 0, state, False)
    assert agent.compute.total == agent.compute.update_cost(2)
    agent.pretrain([(state, 0, 0, state, False)], 1)
    assert agent.compute.pretraining == agent.compute.update_cost(1)
    path = tmp_path / "model.pt"
    agent.save(path)
    restored, _ = DQNAgent.load(path)
    assert restored.compute.summary() == agent.compute.summary()
    # Older checkpoint format loads, but its unrecorded training cost stays unknown.
    payload = torch.load(path, weights_only=True)
    payload.pop("compute")
    torch.save(payload, path)
    legacy, _ = DQNAgent.load(path)
    assert legacy.compute.summary()["training_flops"] is None
    np.testing.assert_allclose(agent.q_values(state), legacy.q_values(state))


def test_budget_can_stop_before_first_action_without_negative_score():
    env, agent = make_agent()
    env.score = 20
    state = env.get_state()
    agent.compute.budget = 1
    score, steps, total, reason, _ = run_episode(env, agent, 0, initial_state=state)
    assert (score, steps, total, reason) == (0, 0, 0.0, "flops_budget")
