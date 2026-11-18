"""Virtual-snake lookahead (shield level 3)."""
import numpy as np

from snake_env import LEFT, RIGHT, STRAIGHT

from dqn.v2.shield import lookahead_mask, shield_mask, shortest_path, virtual_snake_safe
from dqn.v2.state import encode_features

from .helpers import make_env


def test_shortest_path_open_board():
    snake = ((4, 4), (4, 3), (4, 2))
    path = shortest_path(snake, 8, (1, 6))
    assert len(path) == 3 + 2 and path[-1] == (1, 6)
    for a, b in zip(((4, 4),) + tuple(path), path):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_shortest_path_goes_through_freed_body():
    """Row 0 is walled off by row 1 of the body, but segments (2,0) and (1,0) move away before the head
    gets there (free after 2 and 3 steps), so the time-aware path squeezes through them."""
    snake = ((2, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0))
    path = shortest_path(snake, 4, (0, 0))
    assert path is not None and (2, 0) in path and (1, 0) in path
    assert path.index((2, 0)) + 1 >= 2 and path.index((1, 0)) + 1 >= 3


def test_shortest_path_blocked():
    # Head walled in by segments that free up too late (same pocket as test_time_aware_bfs_trapped).
    snake = ((0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0))
    assert shortest_path(snake, 5, (4, 4)) is None


def test_virtual_snake_safe_on_open_board():
    env = make_env(8, [(4, 4), (4, 3), (4, 2)], heading=1, apple=(1, 6))
    assert all(virtual_snake_safe(env, a) for a in (STRAIGHT, LEFT, RIGHT))


def test_eating_into_sealed_corner_is_unsafe():
    """Eating the corner apple leaves the head walled in by segments that free up too late."""
    snake = [(0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0)]
    env = make_env(5, snake, heading=3, apple=(0, 0))   # heading left: straight eats (0, 0)
    assert virtual_snake_safe(env, STRAIGHT) is False


def test_route_into_sealed_corner_is_unsafe():
    """Same trap one step earlier: the move itself is fine, the apple route it starts is not."""
    snake = [(0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0)]
    env = make_env(5, snake, heading=3, apple=(0, 0))
    assert virtual_snake_safe(env, STRAIGHT) is False


def test_lookahead_only_when_hungry():
    snake = [(0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0)]
    env = make_env(5, snake, heading=3, apple=(0, 0))
    f = encode_features(env)
    env.steps_since_apple = 0
    assert (lookahead_mask(env, f, 0.25) == shield_mask(f)).all()
    # Hungry, but no move starts a safe route -> fall back to the shield mask instead of masking everything.
    env.steps_since_apple = env.timeout
    m = lookahead_mask(env, f, 0.25)
    assert m.any() and (m == shield_mask(f)).all()


def test_desperation_takes_unsafe_route_near_timeout():
    """No safe route + timeout imminent -> take the shortest (unsafe) apple route instead of looping."""
    snake = [(0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0)]
    env = make_env(5, snake, heading=3, apple=(0, 0))
    f = encode_features(env)
    env.steps_since_apple = env.timeout - 1     # 1 step left; the apple is 2 steps away
    m = lookahead_mask(env, f, 0.25, desperation_slack=4)
    assert m.tolist() == [True, False, False]   # straight toward the apple
    env.steps_since_apple = env.timeout // 2    # plenty of time left -> keep the shield's choice
    assert (lookahead_mask(env, f, 0.25, desperation_slack=4) == shield_mask(f)).all()


def test_lookahead_narrows_to_apple_routes():
    # Open board, hungry: only moves that start a safe route to the apple survive.
    env = make_env(8, [(4, 4), (4, 3), (4, 2)], heading=1, apple=(4, 7))
    env.steps_since_apple = env.timeout
    m = lookahead_mask(env, encode_features(env), 0.25)
    assert m.any() and m.dtype == np.bool_
