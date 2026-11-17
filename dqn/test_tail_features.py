
import numpy as np

from snake_env import SnakeEnv
from dqn.features.tail_features import (
    FEATURE_NAMES,
    get_feature_matrix,
    preview,
)


def test_feature_matrix():
    env = SnakeEnv(size=8, seed=42)

    original_snake = tuple(env.snake)
    original_heading = env.heading
    original_apple = env.apple

    features = get_feature_matrix(env)

    assert features.shape == (3, 11)
    assert features.dtype == np.float32

    # The extractor must not modify the environment.
    assert tuple(env.snake) == original_snake
    assert env.heading == original_heading
    assert env.apple == original_apple

    # Bias should always be 1.
    assert np.all(features[:, 0] == 1.0)

    # All features should be finite.
    assert np.all(np.isfinite(features))

    # Normalized features should be between 0 and 1.
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)

    print("Feature names:", FEATURE_NAMES)
    print("Feature matrix:\n", features)
    print("Feature extractor test passed!")


def test_tail_cell_exception():
    """
    Verify that entering the current tail cell is allowed
    when the snake does not eat.
    """
    env = SnakeEnv(size=8, seed=42)

    # Construct a simple snake configuration.
    env.snake.clear()
    env.snake.extend([
        (3, 3),  # Head
        (3, 2),
        (4, 2),
        (4, 3),  # Tail
    ])

    env.body = set(env.snake)
    env.heading = 1  # Facing right
    env.apple = (0, 0)

    # Arrange the tail directly in front of the head.
    # For this test, use a configuration with the tail cell
    # as the candidate destination.
    env.snake.clear()
    env.snake.extend([
        (3, 3),
        (2, 3),
        (2, 2),
        (3, 2),
    ])
    env.body = set(env.snake)
    env.heading = 3  # Facing left: next cell is (3, 2), the tail.

    head, eating, wall, body, resulting_snake = preview(env, 0)

    assert head == (3, 2)
    assert not eating
    assert not wall
    assert not body
    assert len(resulting_snake) == len(env.snake)

    print("Tail-cell exception test passed!")



from dqn.features.tail_features import connectivity


def test_connectivity():
    # Snake forms a U-shaped obstacle.
    snake = (
        (2, 2),  # Head
        (2, 3),
        (2, 4),
        (3, 4),
        (4, 4),
        (4, 3),
        (4, 2),  # Tail
    )

    space, tail, apple = connectivity(
        snake,
        size=8,
        apple=(0, 0)
    )

    print("\nConnectivity test:")
    print("Reachable space:", space)
    print("Tail reachable:", tail)
    print("Apple reachable:", apple)

    assert 0.0 <= space <= 1.0
    assert tail in (0.0, 1.0)
    assert apple in (0.0, 1.0)

    print("Connectivity test passed!")

def test_blocked_apple():
    # The snake completely surrounds its head.
    snake = (
        (3, 3),
        (3, 4),
        (4, 4),
        (4, 3),
        (4, 2),
        (3, 2),
        (2, 2),
        (2, 3),
    )

    space, tail, apple = connectivity(
        snake,
        size=8,
        apple=(0, 0)
    )

    print("Blocked apple test:")
    print("Reachable space:", space)
    print("Tail reachable:", tail)
    print("Apple reachable:", apple)

    assert apple == 0.0
    assert space == 1 / 64


if __name__ == "__main__":
    test_feature_matrix()
    test_tail_cell_exception()
    test_connectivity()
    test_blocked_apple()
