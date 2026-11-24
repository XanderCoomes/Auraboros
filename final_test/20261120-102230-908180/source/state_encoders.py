"""Board observations. Encoders describe game state; the CNN learns features."""
from dataclasses import dataclass

import numpy as np

from snake_env import DIRS, SnakeEnv


ENCODERS = {}


def register_encoder(name, channels, binary=True):
    def register(function):
        ENCODERS[name] = (function, tuple(channels), binary)
        return function
    return register


@register_encoder("entities", ("body", "head", "apple"))
def entities(env):
    board = env.board_array()
    return np.stack([board == value for value in (1, 2, 3)], axis=-1)


@register_encoder("entities_tail", ("body", "head", "apple", "tail"))
def entities_tail(env):
    tail = np.zeros((env.size, env.size, 1), dtype=np.float32)
    tail[env.snake[-1]] = 1
    return np.concatenate((entities(env), tail), axis=-1)


@register_encoder("directional", ("up", "right", "down", "left", "apple"))
def directional(env):
    board = np.zeros((env.size, env.size, 5), dtype=np.float32)
    segments = list(env.snake)  # Head first.
    board[segments[0] + (env.heading,)] = 1
    for previous, current in zip(segments, segments[1:]):
        direction = (previous[0] - current[0], previous[1] - current[1])
        board[current + (DIRS.index(direction),)] = 1
    if env.apple is not None:
        board[env.apple + (4,)] = 1
    return board


@register_encoder("ordered", ("body_order", "head", "apple"), binary=False)
def ordered(env):
    board = entities(env).astype(np.float32)
    # Tail = 1 / board area, head = length / board area. Empty = 0.
    for index, position in enumerate(env.snake):
        board[position + (0,)] = (len(env.snake) - index) / (env.size ** 2)
    return board


@dataclass
class Observation:
    board: np.ndarray
    context: np.ndarray


class BoardEncoder:
    """Binary boards may use [absent/present, row, column, channel]."""
    context_names = ("up", "right", "down", "left", "head_row", "head_col", "hunger")

    def __init__(self, name="entities_tail", encoding="auto"):
        if name not in ENCODERS:
            raise ValueError(f"Unknown encoder {name!r}; choose from {list(ENCODERS)}")
        self.function, self.channel_names, binary = ENCODERS[name]
        if encoding not in ("auto", "binary", "onehot"):
            raise ValueError("encoding must be auto, binary, or onehot")
        self.name = name
        self.encoding = ("onehot" if binary else "binary") if encoding == "auto" else encoding
        if self.encoding == "onehot" and not binary:
            raise ValueError("Ordered values require --encoding binary or auto")

    def shape(self, size):
        shape = (size, size, len(self.channel_names))
        return (2,) + shape if self.encoding == "onehot" else shape

    def config(self):
        return {"name": self.name, "encoding": self.encoding}

    def encode(self, env):
        board = self.function(env).astype(np.float32)
        if self.encoding == "onehot":
            board = np.stack((1 - board, board))
        heading = np.eye(4, dtype=np.float32)[env.heading]
        head = np.asarray(env.snake[0], dtype=np.float32) / max(1, env.size - 1)
        hunger = min(1.0, env.steps_since_apple / max(1, env.timeout))
        context = np.concatenate((heading, head, [hunger])).astype(np.float32)
        return Observation(board, context)


class BoardSnakeEnv(SnakeEnv):
    """Reuse SnakeEnv rules and rewards with a pluggable observation method."""
    def __init__(self, encoder=None, **kwargs):
        self.encoder = encoder or BoardEncoder()
        super().__init__(**kwargs)

    def get_state(self):
        return self.encoder.encode(self)
