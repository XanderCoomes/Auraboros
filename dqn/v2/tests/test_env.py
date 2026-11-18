"""Environment changes made for DQN v2: reward config, potential shaping, length-scaled timeout."""
import random

import pytest

from snake_env import LEFT, R_APPLE, R_CLOSER, R_STEP, RIGHT, STRAIGHT, RewardConfig, SnakeEnv

from .helpers import make_env


def test_legacy_rewards_unchanged():
    # Apple straight ahead -> eating gives exactly the legacy apple reward.
    env = make_env(8, [(4, 4), (4, 3), (4, 2)], heading=1, apple=(4, 5))
    _, r, done, _ = env.step(STRAIGHT)
    assert r == R_APPLE and not done
    # Moving away from the apple -> step penalty minus the legacy distance shaping.
    env = make_env(8, [(4, 4), (4, 3), (4, 2)], heading=1, apple=(4, 7))
    _, r, _, _ = env.step(LEFT)
    assert r == pytest.approx(R_STEP - R_CLOSER)


def test_potential_shaping_telescopes():
    """With gamma = 1 and every other reward 0, the shaping terms of a whole episode sum to -phi(s0).

    This is the property that makes potential-based shaping policy-invariant: the total shaping reward does
    not depend on which actions were taken, only on the start state."""
    cfg = RewardConfig(apple=0, death=0, step=0, win=0, closer=0.2, shaping="potential", gamma=1.0)
    rng = random.Random(1)
    for seed in range(20):
        env = SnakeEnv(size=6, seed=seed, rewards=cfg)
        phi0 = env._potential()
        total, done = 0.0, False
        while not done:
            _, r, done, _ = env.step(rng.randrange(3))
            total += r
        assert total == pytest.approx(-phi0, abs=1e-9)


def test_v2_win_bonus():
    # 2x2 board, snake of length 3, apple on the last free cell -> eating wins.
    env = make_env(2, [(0, 1), (0, 0), (1, 0)], heading=1, apple=(1, 1), rewards=RewardConfig.v2())
    phi = env._potential()  # -0.2 * (1 - 3/4) * dist 1 / max_dist 2 = -0.025
    assert phi == pytest.approx(-0.025)
    _, r, done, info = env.step(RIGHT)  # heading right -> turn right = down -> (1, 1)
    assert done and info["reason"] == "win"
    assert r == pytest.approx(1.0 + 10.0 - phi)  # apple + win bonus + terminal shaping (phi(terminal) = 0)


def test_length_timeout():
    legacy = SnakeEnv(size=8, seed=0)
    scaled = SnakeEnv(size=8, seed=0, length_timeout=True)
    assert legacy.timeout == 2 * 64
    assert scaled.timeout == 32 + 2 * 3
    scaled.snake.extend([(0, 0)] * 10)  # length 13
    assert scaled.timeout == 32 + 2 * 13


def test_timeout_ends_episode():
    # A length-3 snake turning right forever circles a 2x2 square and never dies or eats.
    env = make_env(8, [(4, 4), (4, 3), (4, 2)], heading=1, apple=(0, 0), length_timeout=True)
    steps, done = 0, False
    while not done:
        _, _, done, info = env.step(RIGHT)
        steps += 1
    assert info["reason"] == "timeout"
    assert steps == env.timeout + 1  # ends on the first step with steps_since_apple > timeout
