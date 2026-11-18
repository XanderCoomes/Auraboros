"""State representation for DQN v2.

A state is a tuple of three numpy arrays:

    board    (C, S+2, S+2)  egocentric image: rotated so the snake always faces "up", padded with a wall ring
    scalars  (N_SCALARS,)   global numbers the CNN cannot easily read (length, hunger)
    features (3, N_FEATURES) one row of lookahead features per relative action (straight, left, right)

Why egocentric?
    Actions are relative (straight / left / right) but the original board was absolute, so the network had to
    learn the same situation 4 times (once per heading). Rotating the board so the heading is always "up"
    makes "straight" always mean "the cell above the head", "left" the cell to its left, etc.
    This removes the heading input entirely and gives ~4x more data per situation.

Why a wall ring?
    Convolutions without padding cannot see the border, so a wall looked like empty space. Padding the board
    with a ring of 1s in a "wall" channel lets the CNN treat walls like body: obstacles.

Why time-to-free?
    A body segment i (0 = head) disappears after (L - i) moves if no apple is eaten. That number tells whether
    a cell that is blocked now will be free by the time the head gets there. It is normalized by the board
    area so it stays in [0, 1] (the old encoder fed raw values up to 64 next to 0/1 channels).
"""
from collections import deque

import numpy as np

from snake_env import DIRS, LEFT, RIGHT, N_ACTIONS

# ---------------------------------------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------------------------------------
BOARD_CHANNELS = ("wall", "body", "time_to_free", "head", "apple")
N_CHANNELS = len(BOARD_CHANNELS)

SCALAR_NAMES = ("length_fraction", "hunger")
N_SCALARS = len(SCALAR_NAMES)

FEATURE_NAMES = (
    "would_die",          # move hits a wall or the body (instant death)
    "would_eat",          # move eats the apple
    "apple_distance",     # Manhattan distance to the apple after the move, / max distance
    "apple_closer",       # +1 closer to the apple, -1 farther, 0 if it ate / died
    "reachable_fraction", # cells reachable after the move / free cells (1 = nothing is cut off)
    "length_vs_space",    # min(1, reachable cells / snake length): is there room for the whole body?
    "can_escape",         # time-aware BFS: the head can reach a body cell after it frees -> can follow the tail
    "apple_reachable",    # apple is in the reachable region after the move
    "length_fraction",    # snake length / area (repeated so the per-action branch can condition on it)
    "hunger",             # steps since last apple / timeout
)
N_FEATURES = len(FEATURE_NAMES)
F = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ---------------------------------------------------------------------------------------------------------
# Grid helpers (flat indices are faster than tuples in the BFS hot loop)
# ---------------------------------------------------------------------------------------------------------
_NEIGHBOR_CACHE = {}


def neighbor_table(size):
    """neighbors[cell] = list of in-bounds neighbor flat indices, cached per board size."""
    if size not in _NEIGHBOR_CACHE:
        table = []
        for r in range(size):
            for c in range(size):
                table.append([(r + dr) * size + (c + dc) for dr, dc in DIRS
                              if 0 <= r + dr < size and 0 <= c + dc < size])
        _NEIGHBOR_CACHE[size] = table
    return _NEIGHBOR_CACHE[size]


def turn(heading, action):
    if action == LEFT:
        return (heading - 1) % 4
    if action == RIGHT:
        return (heading + 1) % 4
    return heading


def preview(env, action):
    """Simulate `action` without touching env.

    Returns (head, eating, died, snake_after) with snake_after a tuple of (r, c), head first
    (the unchanged snake if the move dies).
    """
    dr, dc = DIRS[turn(env.heading, action)]
    old_head = env.snake[0]
    head = (old_head[0] + dr, old_head[1] + dc)
    eating = head == env.apple
    if not (0 <= head[0] < env.size and 0 <= head[1] < env.size):
        return head, eating, True, tuple(env.snake)
    # The tail cell is free on this move unless the snake grows (same rule as SnakeEnv.step).
    hits_body = head in env.body and (head != env.snake[-1] or eating)
    if hits_body:
        return head, eating, True, tuple(env.snake)
    snake = tuple(env.snake)
    snake_after = (head,) + (snake if eating else snake[:-1])
    return head, eating, False, snake_after


# ---------------------------------------------------------------------------------------------------------
# Space analysis
# ---------------------------------------------------------------------------------------------------------
def analyze_space(snake, size, apple=None):
    """Static + time-aware flood fill from the head of `snake` (tuple of (r, c), head first).

    Static BFS: body cells are walls, except the tail which frees on the next move.
        -> reachable cell count, whether the apple is reachable.

    Time-aware BFS: body segment i frees after (L - i) moves. A cell occupied by segment i can be entered at
        BFS depth d if d >= L - i. If the head can reach any body cell this way, it can keep following its own
        body toward the tail forever, so the snake cannot be trapped (ignoring future apples).

    Returns (reachable_count, apple_reachable, can_escape).
    """
    nbrs = neighbor_table(size)
    L = len(snake)
    flat = [r * size + c for r, c in snake]
    head = flat[0]
    # free_time[cell] = number of moves until the cell is empty (0 = already empty)
    free_time = {cell: L - i for i, cell in enumerate(flat)}
    free_time[head] = L  # never re-enter the current head cell through BFS
    apple_flat = apple[0] * size + apple[1] if apple is not None else -1

    # Static BFS (tail counts as free, like SnakeEnv.is_danger)
    tail = flat[-1]
    seen = {head}
    queue = deque([head])
    while queue:
        cur = queue.popleft()
        for n in nbrs[cur]:
            if n in seen:
                continue
            ft = free_time.get(n, 0)
            if ft == 0 or n == tail:
                seen.add(n)
                queue.append(n)
    reachable = len(seen) - 1  # exclude the head itself
    apple_reachable = apple_flat in seen

    # Time-aware BFS
    can_escape = False
    depth = {head: 0}
    queue = deque([head])
    while queue and not can_escape:
        cur = queue.popleft()
        d = depth[cur] + 1
        for n in nbrs[cur]:
            if n in depth:
                continue
            ft = free_time.get(n, 0)
            if ft == 0:
                depth[n] = d
                queue.append(n)
            elif d >= ft:
                can_escape = True  # reached a body cell after it freed -> can chase the tail
                break
    return reachable, apple_reachable, can_escape


# ---------------------------------------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------------------------------------
def encode_board(env):
    """Egocentric, wall-padded board of shape (N_CHANNELS, size+2, size+2)."""
    s = env.size
    area = s * s
    p = s + 2
    board = np.zeros((N_CHANNELS, p, p), dtype=np.float32)
    board[0, 0, :] = board[0, -1, :] = board[0, :, 0] = board[0, :, -1] = 1.0  # wall ring
    L = len(env.snake)
    for i, (r, c) in enumerate(env.snake):
        board[1, r + 1, c + 1] = 1.0
        board[2, r + 1, c + 1] = (L - i) / area
    hr, hc = env.snake[0]
    board[3, hr + 1, hc + 1] = 1.0
    if env.apple is not None:
        board[4, env.apple[0] + 1, env.apple[1] + 1] = 1.0
    # np.rot90 rotates counter-clockwise; heading h needs h quarter turns to point "up":
    # right (1) -> 1 CCW turn, down (2) -> 2 turns, left (3) -> 3 turns.
    return np.ascontiguousarray(np.rot90(board, k=env.heading, axes=(1, 2)))


def encode_scalars(env):
    return np.array([len(env.snake) / (env.size * env.size),
                     min(1.0, env.steps_since_apple / env.timeout)], dtype=np.float32)


def encode_features(env):
    """(3, N_FEATURES) lookahead features, one row per relative action."""
    rows = np.zeros((N_ACTIONS, N_FEATURES), dtype=np.float32)
    if env.done:
        return rows
    size = env.size
    area = size * size
    max_dist = 2 * (size - 1)
    length_frac = len(env.snake) / area
    hunger = min(1.0, env.steps_since_apple / env.timeout)
    old_head = env.snake[0]
    old_dist = (abs(old_head[0] - env.apple[0]) + abs(old_head[1] - env.apple[1])) if env.apple else 0

    for a in range(N_ACTIONS):
        row = rows[a]
        row[F["length_fraction"]] = length_frac
        row[F["hunger"]] = hunger
        head, eating, died, snake = preview(env, a)
        if died:
            row[F["would_die"]] = 1.0
            continue
        row[F["would_eat"]] = float(eating)
        # After eating, the next apple position is random: use the current one only when not eating.
        if env.apple is not None and not eating:
            dist = abs(head[0] - env.apple[0]) + abs(head[1] - env.apple[1])
            row[F["apple_distance"]] = dist / max_dist
            row[F["apple_closer"]] = 1.0 if dist < old_dist else -1.0
        apple = None if eating else env.apple
        reachable, apple_ok, escape = analyze_space(snake, size, apple)
        free_cells = area - len(snake) + 1  # +1: tail cell frees on the next move
        row[F["reachable_fraction"]] = min(1.0, reachable / max(1, free_cells))
        row[F["length_vs_space"]] = min(1.0, reachable / len(snake))
        row[F["can_escape"]] = float(escape)
        row[F["apple_reachable"]] = float(apple_ok or eating)
    return rows


def encode_state(env):
    """Full DQN v2 state: (board, scalars, features)."""
    return encode_board(env), encode_scalars(env), encode_features(env)


def board_shape(size):
    return (N_CHANNELS, size + 2, size + 2)
