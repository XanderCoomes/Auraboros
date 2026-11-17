"""Tail feature: per direction (straight/left/right), can the head still reach the tail after stepping that way?

Per direction: 0 = blocked (wall/body), 1 = open but tail unreachable, 2 = tail reachable.
"""
from collections import deque

from features import FeatureBase

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def tail_reachable(env, start):
    """BFS from `start` through free cells. Returns None if `start` is blocked, else True/False (tail touched)."""
    tail = env.snake[-1]
    blocked = env.body - {tail}  # the tail cell moves away, so it is a valid target
    if not env._in_bounds(start) or start in blocked:
        return None
    if start == tail:
        return True
    seen, q = {start}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            p = (r + dr, c + dc)
            if p == tail:
                return True
            if p not in seen and env._in_bounds(p) and p not in blocked:
                seen.add(p)
                q.append(p)
    return False


class TailFeature(FeatureBase):
    name = "tail"
    cardinality = 27
    description = "per direction (straight*9 + left*3 + right): 0 blocked / 1 tail unreachable / 2 tail reachable"

    def extract(self, env):
        head, v = env.snake[0], 0
        for d in (env.heading, (env.heading - 1) % 4, (env.heading + 1) % 4):
            p = (head[0] + DIRS[d][0], head[1] + DIRS[d][1])
            r = tail_reachable(env, p)
            v = v * 3 + (0 if r is None else 2 if r else 1)
        return v

    def format_value(self, value):
        return str(value)
