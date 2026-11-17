
import numpy as np

from snake_env import SnakeEnv
from dqn.encoders.board import BoardEncoderWithTime


def test_board_encoder_with_time():
    env = SnakeEnv(size=8, seed=42)
    env.reset()

    encoder = BoardEncoderWithTime()
    board, heading = encoder.encode(env)

    assert board.shape == (4, 8, 8)
    assert heading.shape == (4,)

    # Verify snake occupancy and time channel.
    for index, (r, c) in enumerate(env.snake):
        assert board[0, r, c] == 1.0
        assert board[3, r, c] == len(env.snake) - index

    # Empty cells must have zero time value.
    assert np.all(board[3][board[0] == 0] == 0)

    # Verify apple and heading.
    if env.apple is not None:
        r, c = env.apple
        assert board[1, r, c] == 1.0

    assert np.sum(heading) == 1.0

    print("Board shape:", board.shape)
    print("Snake length:", len(env.snake))
    print("Time channel:")
    print(board[3])
    print("4-channel encoder test passed!")


if __name__ == "__main__":
    test_board_encoder_with_time()
