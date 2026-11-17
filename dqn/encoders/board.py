
import numpy as np

from .base import StateEncoder


class BoardEncoder(StateEncoder):

    def __init__(self):
        self._input_shape = (3, 8, 8)
        self._heading_shape = (4,)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def heading_shape(self):
        return self._heading_shape

    def encode(self, env) -> tuple[np.ndarray, np.ndarray]:

        # --------------------------------------------------
        # Board representation: 3 x size x size
        # --------------------------------------------------

        board = np.zeros(
            (3, env.size, env.size),
            dtype=np.float32
        )

        # Channel 0: Snake occupancy (including head)
        for r, c in env.snake:
            board[0, r, c] = 1.0

        # Channel 1: Apple position
        if env.apple is not None:
            apple_r, apple_c = env.apple
            board[1, apple_r, apple_c] = 1.0

        # Channel 2: Snake head position
        head_r, head_c = env.snake[0]
        board[2, head_r, head_c] = 1.0

        # --------------------------------------------------
        # Heading representation: 4-dimensional one-hot
        # --------------------------------------------------

        heading = np.zeros(4, dtype=np.float32)
        heading[env.heading] = 1.0

        return board, heading


class BoardEncoderWithTime(BoardEncoder):
    def __init__(self):
        self._input_shape = (4, 8, 8)
        self._heading_shape = (4,)

    def encode(self, env):
        board = np.zeros((4, env.size, env.size), dtype=np.float32)

        snake_length = len(env.snake)

        for index, (r, c) in enumerate(env.snake):
            board[0, r, c] = 1.0
            board[3, r, c] = snake_length - index

        if env.apple is not None:
            apple_r, apple_c = env.apple
            board[1, apple_r, apple_c] = 1.0

        head_r, head_c = env.snake[0]
        board[2, head_r, head_c] = 1.0

        heading = np.zeros(4, dtype=np.float32)
        heading[env.heading] = 1.0

        return board, heading

