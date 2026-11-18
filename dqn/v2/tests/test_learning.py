"""n-step returns, prioritized replay, the Double DQN update, stats helpers and baseline policies."""
import numpy as np
import pytest
import torch

from snake_env import SnakeEnv

from dqn.v2 import stats
from dqn.v2.agent import DQNAgentV2
from dqn.v2.baselines import (GreedyBFSPolicy, HamiltonianPolicy, hamiltonian_cycle, relative_action)
from dqn.v2.evaluate import run_games, summarize
from dqn.v2.model import DuelingHybridDQN
from dqn.v2.replay import NStepAccumulator, PrioritizedReplay
from dqn.v2.state import N_FEATURES, N_SCALARS, board_shape, encode_state


# ------------------------------------------------------------------------------------------- n-step / replay
def test_nstep_returns_and_flush():
    acc = NStepAccumulator(n=3, gamma=0.5)
    m = np.ones(3, bool)
    assert acc.push("s0", 0, 1.0, "s1", False, m) == []
    assert acc.push("s1", 1, 2.0, "s2", False, m) == []
    (s, a, ret, nxt, disc, _), = acc.push("s2", 2, 4.0, "s3", False, m)
    assert (s, a, nxt) == ("s0", 0, "s3")
    assert ret == pytest.approx(1 + 0.5 * 2 + 0.25 * 4) and disc == pytest.approx(0.125)
    # Episode ends: all three pending transitions flush with discount 0 (no bootstrap past the end).
    out = acc.push("s3", 0, 8.0, "end", True, m)
    assert [o[0] for o in out] == ["s1", "s2", "s3"]
    assert [o[2] for o in out] == pytest.approx([2 + 0.5 * 4 + 0.25 * 8, 4 + 0.5 * 8, 8])
    assert all(o[4] == 0.0 for o in out)
    assert acc.push("s0", 0, 1.0, "s1", False, m) == []  # queue was emptied


def _state():
    return (np.zeros(board_shape(8), np.float32), np.zeros(N_SCALARS, np.float32),
            np.zeros((3, N_FEATURES), np.float32))


def test_prioritized_sampling_follows_priorities():
    rb = PrioritizedReplay(100, board_shape(8), N_SCALARS, (3, N_FEATURES), alpha=1.0, seed=0)
    for i in range(10):
        rb.push(_state(), 0, float(i), _state(), 0.9, np.ones(3, bool))
    pr = np.full(10, 1e-6)
    pr[7] = 1.0
    rb.update_priorities(np.arange(10), pr - rb.eps)
    idx, batch = rb.sample(64, beta=0.4)
    assert (idx == 7).mean() > 0.9
    assert batch["weights"].max() == pytest.approx(1.0)
    assert (batch["rewards"][idx == 7] == 7).all()


def test_replay_wraps_around():
    rb = PrioritizedReplay(5, board_shape(8), N_SCALARS, (3, N_FEATURES))
    for i in range(12):
        rb.push(_state(), 0, float(i), _state(), 0.9, np.ones(3, bool))
    assert len(rb) == 5 and sorted(rb.rewards.tolist()) == [7, 8, 9, 10, 11]


# ------------------------------------------------------------------------------------------------- agent
def test_train_step_fits_terminal_rewards():
    """With discount 0 the target is just the reward, so repeated updates on one batch must fit it."""
    torch.manual_seed(0)
    agent = DQNAgentV2(DuelingHybridDQN(8, hidden=64, feat_hidden=16), torch.device("cpu"), lr=1e-3)
    env = SnakeEnv(size=8, seed=0)
    s = encode_state(env)
    n = 8
    batch = dict(
        boards=np.repeat(s[0][None], n, 0).astype(np.float16), scalars=np.repeat(s[1][None], n, 0),
        feats=np.repeat(s[2][None], n, 0), actions=np.array([0, 1, 2] * 2 + [0, 1]),
        rewards=np.array([1, -1, 0.5] * 2 + [1, -1], np.float32), discounts=np.zeros(n, np.float32),
        next_boards=np.repeat(s[0][None], n, 0).astype(np.float16), next_scalars=np.repeat(s[1][None], n, 0),
        next_feats=np.repeat(s[2][None], n, 0), next_masks=np.ones((n, 3), bool),
        weights=np.ones(n, np.float32))
    for _ in range(300):
        loss, td = agent.train_step(batch)
    q = agent.q_values(s[0][None], s[1][None], s[2][None])[0]
    assert q == pytest.approx([1, -1, 0.5], abs=0.05)


def test_lr_schedule():
    agent = DQNAgentV2(DuelingHybridDQN(8, hidden=32, feat_hidden=8), torch.device("cpu"), lr=1e-3, lr_min=1e-5)
    assert agent.set_lr_progress(0) == pytest.approx(1e-3)
    assert agent.set_lr_progress(0.5) == pytest.approx((1e-3 + 1e-5) / 2)
    assert agent.set_lr_progress(1) == pytest.approx(1e-5)


# ------------------------------------------------------------------------------------------------- stats
def test_wilson_ci_known_values():
    assert stats.wilson_ci(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert stats.wilson_ci(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    lo, hi = stats.wilson_ci(100, 100)
    assert hi == 1.0 and lo == pytest.approx(0.9630, abs=1e-4)


def test_bootstrap_and_tests():
    est, lo, hi = stats.bootstrap_ci([3.0] * 20)
    assert est == lo == hi == 3.0
    est, lo, hi = stats.bootstrap_ci(np.arange(100))
    assert lo < est < hi
    assert stats.two_proportion_test(5, 10, 5, 10) == (0.0, 1.0)
    z, p = stats.two_proportion_test(90, 100, 50, 100)
    assert z > 0 and p < 1e-6
    assert stats.fisher_exact(10, 10, 0, 10) < 1e-4


# --------------------------------------------------------------------------------------------- baselines
def test_relative_action():
    assert relative_action(0, 0) == 0 and relative_action(0, 3) == 1 and relative_action(0, 1) == 2
    assert relative_action(0, 2) is None


@pytest.mark.parametrize("size", [4, 6, 8])
def test_hamiltonian_cycle_is_valid(size):
    nxt = hamiltonian_cycle(size)
    assert len(nxt) == size * size
    cell, seen = (0, 0), set()
    for _ in range(size * size):
        seen.add(cell)
        n = nxt[cell]
        assert abs(n[0] - cell[0]) + abs(n[1] - cell[1]) == 1
        cell = n
    assert cell == (0, 0) and len(seen) == size * size


def test_hamiltonian_policy_always_wins():
    rec = run_games(HamiltonianPolicy(), 10, size=6, base_seed=0)
    assert all(r["win"] for r in rec)
    s = summarize(rec)
    assert s["win_rate"] == 1.0 and s["max_score"] == 36 - 3


def test_greedy_bfs_beats_random_floor():
    rec = run_games(GreedyBFSPolicy(), 20, size=8, base_seed=0)
    assert summarize(rec)["mean_score"] > 10
