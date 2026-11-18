"""Action masks built from the per-action lookahead features.

Levels 1-2 are computed only from the feature matrix (no extra simulation); level 3 (virtual-snake
lookahead, bottom of this file) simulates whole apple routes and is used at inference only:

1. Death mask (always on in v2): never pick a move that dies immediately. The env already tells us this
   (feature `would_die`), so letting the agent choose it only wastes episodes and adds noise to the targets.

2. Safety shield (OPTIONAL, --shield): among the non-lethal moves, keep only those after which the snake can
   still escape (time-aware BFS reaches a body cell after it frees up, i.e. the head can chase its tail).
   If no move is safe, fall back to the non-lethal moves with the largest reachable area (the "least bad"
   ones). The Q-network still chooses *which* safe move to take; the shield only removes moves that lead to a
   provable trap. This is a form of "shielded RL" (Alshiekh et al., 2018) and is reported as an ablation.
"""
from collections import deque

import numpy as np

from .state import F, analyze_space, neighbor_table, preview


def death_mask(features):
    """bool (3,): True = allowed. If every move dies, allow all (the episode ends anyway)."""
    mask = features[:, F["would_die"]] < 0.5
    if not mask.any():
        mask[:] = True
    return mask


def shield_mask(features):
    """bool (3,): safety-shield mask (subset of death_mask)."""
    alive = death_mask(features)
    if features[:, F["would_die"]].min() > 0.5:
        return alive
    safe = alive & (features[:, F["can_escape"]] > 0.5)
    if safe.any():
        return safe
    # No escape route anywhere: keep the moves with the most room.
    space = np.where(alive, features[:, F["reachable_fraction"]], -1.0)
    return space >= space.max() - 1e-6


def action_mask(features, use_shield):
    return shield_mask(features) if use_shield else death_mask(features)


# ---------------------------------------------------------------------------------------------------------
# 3. Virtual-snake lookahead (inference only)
# ---------------------------------------------------------------------------------------------------------
# Why: with the safety shield, most lost games were TIMEOUTS - the snake stayed safe by chasing its tail but
# never committed to the apple, because a one-move escape test cannot tell which apple routes stay safe.
# The lookahead answers that question directly: walk a "virtual snake" along the shortest path to the apple,
# let it grow when it eats, and check that the grown snake can still reach its tail. If so, the route is
# provably not a trap (for the current apple), so committing to it is safe.
#
# It is used only when the snake is hungry (steps_since_apple / timeout >= hunger_threshold). Below that
# threshold the Q-network chooses freely among shield-safe moves; above it, the choice is restricted to moves
# that start a safe apple route. The Q-network still picks among those moves.

def shortest_path(snake, size, goal):
    """Time-aware BFS path (list of (r, c), excluding the head) from snake[0] to goal, or None.

    Body segment i (0 = head) moves away after L - i steps, so its cell can be entered at path step
    d >= L - i. Treating the body as permanent walls instead fails in the endgame: with 1-2 free cells the
    apple is almost never reachable through free cells only, the lookahead finds no route, and the snake
    tail-chases around a loop that never passes over the apple until it times out. Valid because the
    virtual snake does not grow before it reaches the apple (the goal)."""
    nbrs = neighbor_table(size)
    L = len(snake)
    flat = [r * size + c for r, c in snake]
    head = flat[0]
    free_time = {cell: L - i for i, cell in enumerate(flat)}
    target = goal[0] * size + goal[1]
    parent = {head: None}
    depth = {head: 0}
    queue = deque([head])
    while queue:
        cur = queue.popleft()
        if cur == target:
            path = []
            while cur != head:
                path.append(divmod(cur, size))
                cur = parent[cur]
            return path[::-1]
        d = depth[cur] + 1
        for n in nbrs[cur]:
            if n not in parent and d >= free_time.get(n, 0) and n != head:
                parent[n] = cur
                depth[n] = d
                queue.append(n)
    return None


def apple_route(env, action):
    """(steps to the apple, safe) for the route that starts with `action`; (None, False) if the move dies or
    the apple is unreachable. safe = after following the shortest path and eating (growing by one), the
    snake can still escape by chasing its tail (time-aware BFS)."""
    head, eating, died, snake = preview(env, action)
    if died:
        return None, False
    steps = 1
    if not eating:
        path = shortest_path(snake, env.size, env.apple)
        if path is None:
            return None, False
        steps += len(path)
        body = deque(snake)
        for cell in path:
            body.appendleft(cell)
            if cell != env.apple:
                body.pop()          # the last cell is the apple: the snake grows, the tail stays
        snake = tuple(body)
    if len(snake) >= env.size * env.size:   # eating fills the board: winning is always safe
        return steps, True
    return steps, analyze_space(snake, env.size)[2]


def virtual_snake_safe(env, action):
    return apple_route(env, action)[1]


def lookahead_mask(env, features, hunger_threshold=0.0, desperation_slack=None):
    """Shield mask, narrowed to safe apple routes once the snake is hungry (see above).

    hunger_threshold=0 (always on) was best on the validation seeds (77.0% vs 74-75% for 0.1-0.5).

    Desperation rule (OFF by default, desperation_slack=None): a timeout loses just like a crash, so when NO
    route is provably safe and the steps left barely cover the shortest apple route
    (remaining <= steps + slack), take that route anyway. Negative result on the validation seeds: it
    turned timeouts into collisions one-for-one (slack 0-16: 77.0-76.7% wins, slack 64: 70.4%) - by then
    the position is already lost, so the remaining losses must be fixed earlier, i.e. in training.
    """
    base = shield_mask(features)
    if env.apple is None or env.steps_since_apple < hunger_threshold * env.timeout:
        return base
    routes = [apple_route(env, a) if not features[a, F["would_die"]] else (None, False)
              for a in range(len(base))]
    safe = np.array([base[a] and routes[a][1] for a in range(len(base))])
    if safe.any():
        return safe
    lengths = np.array([r[0] if r[0] is not None else np.inf for r in routes])
    remaining = env.timeout - env.steps_since_apple
    if (desperation_slack is not None and np.isfinite(lengths.min())
            and remaining <= lengths.min() + desperation_slack):
        return lengths == lengths.min()
    return base
