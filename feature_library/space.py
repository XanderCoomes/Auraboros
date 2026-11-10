"""Space feature (replaces danger): per direction (straight/left/right), how much room is there after stepping that way?

Per direction: 0 = blocked (wall/body), 1 = tight (reachable free cells < snake length), 2 = roomy (>= length).
Reachable cells are counted by flood fill from the cell ahead; the tail cell counts as free (it moves away).
"""
from collections import deque

from features import FeatureBase

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def reachable_cells(env, start):
    """Number of free cells reachable from `start` (including it); 0 if `start` is blocked."""
    tail = env.snake[-1]
    blocked = env.body - {tail}
    if not env._in_bounds(start) or start in blocked:
        return 0
    seen, q = {start}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            p = (r + dr, c + dc)
            if p not in seen and env._in_bounds(p) and p not in blocked:
                seen.add(p)
                q.append(p)
    return len(seen)


class SpaceFeature(FeatureBase):
    name = "space"
    cardinality = 27
    description = ("room per direction (straight*9 + left*3 + right), each 0 blocked / 1 tight (reachable < length) "
                   "/ 2 roomy; replaces danger")

    def extract(self, env):
        head, length, v = env.snake[0], len(env.snake), 0
        for d in (env.heading, (env.heading - 1) % 4, (env.heading + 1) % 4):
            p = (head[0] + DIRS[d][0], head[1] + DIRS[d][1])
            n = reachable_cells(env, p)
            v = v * 3 + (0 if n == 0 else 1 if n < length else 2)
        return v

    def format_value(self, value):
        return str(value)
