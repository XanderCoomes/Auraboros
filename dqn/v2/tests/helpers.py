from collections import deque

from snake_env import SnakeEnv


def make_env(size, snake, heading, apple, **kwargs):
    """SnakeEnv placed in an exact position. `snake` is a list of (r, c), head first."""
    env = SnakeEnv(size=size, seed=0, **kwargs)
    env.snake = deque(snake)
    env.body = set(snake)
    env.heading = heading
    env.apple = apple
    env.steps_since_apple = 0
    return env
