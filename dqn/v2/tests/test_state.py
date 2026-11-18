"""Encoder rotation, wall ring, lookahead features, time-aware BFS and the action masks."""
import numpy as np
import pytest

from snake_env import LEFT, RIGHT, STRAIGHT, SnakeEnv

from dqn.v2.shield import death_mask, shield_mask
from dqn.v2.state import (F, N_CHANNELS, analyze_space, encode_board, encode_features, encode_scalars,
                          encode_state, preview)

from .helpers import make_env

APPLE_CH, HEAD_CH, WALL_CH, TTF_CH = 4, 3, 0, 2


@pytest.mark.parametrize("heading", [0, 1, 2, 3])
def test_board_is_egocentric(heading):
    """Whatever the heading, the cell straight ahead of the head is the cell ABOVE it in the encoding,
    and the cell to the snake's left is to the LEFT in the encoding."""
    size, head = 8, (4, 4)
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    back = dirs[(heading + 2) % 4]
    snake = [head, (head[0] + back[0], head[1] + back[1])]
    ahead = (head[0] + dirs[heading][0], head[1] + dirs[heading][1])
    left = dirs[(heading - 1) % 4]
    left_cell = (head[0] + left[0], head[1] + left[1])

    board = encode_board(make_env(size, snake, heading, apple=ahead))
    hr, hc = np.argwhere(board[HEAD_CH] == 1)[0]
    assert board[APPLE_CH, hr - 1, hc] == 1

    board = encode_board(make_env(size, snake, heading, apple=left_cell))
    assert board[APPLE_CH, hr, hc - 1] == 1


def test_wall_ring_and_normalized_time():
    env = SnakeEnv(size=8, seed=0)
    board = encode_board(env)
    assert board.shape == (N_CHANNELS, 10, 10)
    ring = np.ones((10, 10), bool)
    ring[1:-1, 1:-1] = False
    assert (board[WALL_CH][ring] == 1).all() and (board[WALL_CH][~ring] == 0).all()
    assert 0 < board[TTF_CH].max() <= 1  # old encoder fed raw values up to 64


def test_scalars_include_hunger():
    env = SnakeEnv(size=8, seed=0)
    env.steps_since_apple = env.timeout // 2
    length_frac, hunger = encode_scalars(env)
    assert length_frac == pytest.approx(3 / 64)
    assert hunger == pytest.approx(0.5, abs=0.02)


def test_features_die_and_eat():
    # Head in the top-left corner heading up: straight and left hit the wall, right eats the apple.
    env = make_env(8, [(0, 0), (1, 0), (2, 0)], heading=0, apple=(0, 1))
    f = encode_features(env)
    assert f[STRAIGHT, F["would_die"]] == 1 and f[LEFT, F["would_die"]] == 1
    assert f[RIGHT, F["would_die"]] == 0 and f[RIGHT, F["would_eat"]] == 1
    assert death_mask(f).tolist() == [False, False, True]


def test_preview_can_enter_tail_cell_but_not_when_eating():
    # 2x2 loop: head (0,0), tail (0,1) right of it. Heading up -> turn right moves into the tail cell.
    snake = [(0, 0), (1, 0), (1, 1), (0, 1)]
    env = make_env(4, snake, heading=0, apple=(3, 3))
    assert preview(env, RIGHT)[2] is False   # the tail moves away this turn
    env = make_env(4, snake, heading=0, apple=(0, 1))
    env.body = set(snake)
    # Apple can never be on the body in a real game; this only checks the growth rule of preview.
    assert preview(env, RIGHT)[2] is True


def test_time_aware_bfs_trapped():
    """Head in a pocket walled by segments that free up too late -> no escape, nothing reachable."""
    snake = ((0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0))
    reachable, _, escape = analyze_space(snake, 5)
    assert reachable == 0 and escape is False


def test_time_aware_bfs_tail_adjacent():
    """Same pocket, but the blocking segment is the tail: it frees after 1 move -> escape."""
    snake = ((0, 0), (0, 1), (1, 1), (1, 0))
    reachable, _, escape = analyze_space(snake, 5)
    assert escape is True and reachable > 0


def test_shield_fallback_keeps_most_space():
    f = np.zeros((3, len(F)), np.float32)
    f[:, F["can_escape"]] = 0
    f[:, F["reachable_fraction"]] = [0.2, 0.5, 0.1]
    assert shield_mask(f).tolist() == [False, True, False]
    f[2, F["can_escape"]] = 1
    assert shield_mask(f).tolist() == [False, False, True]
    f[:, F["would_die"]] = 1
    assert death_mask(f).tolist() == [True, True, True]  # everything dies -> allow all


def test_encode_state_shapes():
    board, scalars, feats = encode_state(SnakeEnv(size=8, seed=3))
    assert board.dtype == scalars.dtype == feats.dtype == np.float32
    assert feats.shape == (3, len(F))
