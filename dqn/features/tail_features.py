
from collections import deque

import numpy as np

from snake_env import DIRS, LEFT, RIGHT, N_ACTIONS


FEATURE_NAMES = (
    "bias",
    "apple_distance",
    "would_eat",
    "would_hit_wall",
    "would_hit_body",
    "length_fraction",
    "reachable_space",
    "tail_reachable",
    "length_x_distance",
    "length_x_space",
    "length_x_tail",
)

N_FEATURES = len(FEATURE_NAMES)


def normalized_distance(head, apple, size):
    """Manhattan distance normalized to [0, 1]."""
    if apple is None:
        return 0.0

    distance = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
    return min(1.0, distance / (2 * (size - 1)))


def preview(env, action):
    """
    Simulate a candidate action without modifying the real environment.

    Returns:
        head, eating, wall, body, resulting_snake
    """
    if action not in range(N_ACTIONS):
        raise ValueError("Invalid action")

    heading = env.heading

    if action == LEFT:
        heading = (heading - 1) % 4
    elif action == RIGHT:
        heading = (heading + 1) % 4

    dr, dc = DIRS[heading]
    old_head = env.snake[0]
    head = (old_head[0] + dr, old_head[1] + dc)

    eating = head == env.apple

    wall = not (
        0 <= head[0] < env.size
        and 0 <= head[1] < env.size
    )

    # The tail can be entered only if it moves away.
    occupied = env.snake if eating else list(env.snake)[:-1]
    body = not wall and head in occupied

    snake = tuple(env.snake)

    if not wall and not body:
        snake = (head,) + (
            snake if eating else snake[:-1]
        )

    return head, eating, wall, body, snake


def neighbors(cell, size):
    """Yield valid neighboring cells."""
    for dr, dc in DIRS:
        neighbor = (cell[0] + dr, cell[1] + dc)

        if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
            yield neighbor


def connectivity(snake, size, apple=None):
    """
    Calculate reachable space, tail reachability, and apple reachability.

    Reproduces the static flood-fill implementation from the
    approximate Q-learning reference.
    """
    blocked = set(snake[1:])

    reachable = {snake[0]}
    queue = deque([snake[0]])

    while queue:
        current = queue.popleft()

        for cell in neighbors(current, size):
            if cell not in blocked and cell not in reachable:
                reachable.add(cell)
                queue.append(cell)

    tail = snake[-1]

    tail_reachable = (
        tail in reachable
        or any(
            cell in reachable
            for cell in neighbors(tail, size)
        )
    )

    apple_reachable = (
        apple is not None and apple in reachable
    )

    return (
        len(reachable) / (size * size),
        float(tail_reachable),
        float(apple_reachable),
    )


def get_features(env, action):
    """Calculate the 11 engineered features for one candidate action."""

    head, eating, wall, body, snake = preview(env, action)

    features = {}

    # 1. Bias
    features["bias"] = 1.0

    # 2. Normalized distance to apple after the candidate move
    features["apple_distance"] = normalized_distance(
        head, env.apple, env.size
    )

    # 3–5. Immediate consequences
    features["would_eat"] = float(eating)
    features["would_hit_wall"] = float(wall)
    features["would_hit_body"] = float(body)

    # 6. Post-move snake length
    length = len(snake) / (env.size * env.size)
    features["length_fraction"] = length

    # 7–8. Reachable space and tail reachability
    if wall or body:
        space, tail, apple_reachable = 0.0, 0.0, 0.0
    else:
        space, tail, apple_reachable = connectivity(
            snake, env.size, env.apple
        )

    features["reachable_space"] = space
    features["tail_reachable"] = tail

    # 9–11. Length-dependent features
    features["length_x_distance"] = (
        length * features["apple_distance"]
    )

    features["length_x_space"] = length * space

    features["length_x_tail"] = length * tail

    return features


def get_feature_matrix(env):
    """
    Calculate features for all three candidate actions.

    Returns:
        float32 array with shape (3, 11).
    """
    if env.done:
        return np.zeros((N_ACTIONS, N_FEATURES), dtype=np.float32)

    rows = []

    for action in range(N_ACTIONS):
        features = get_features(env, action)

        rows.append([
            features[name] for name in FEATURE_NAMES
        ])

    return np.asarray(rows, dtype=np.float32)
