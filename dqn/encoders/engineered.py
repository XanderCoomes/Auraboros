
import numpy as np

from .base import StateEncoder

#Add more features, currently stay with this ones


class EngineeredEncoder(StateEncoder):

    def __init__(self):
        self._input_shape = (13,)

    @property
    def input_shape(self):
        return self._input_shape

    def encode(self, env) -> np.ndarray:

        # --------------------------------------------------
        # Danger features (relative to current heading)
        # --------------------------------------------------

        heading = env.heading

        danger_straight = env.is_danger(heading)
        danger_right = env.is_danger((heading + 1) % 4)
        danger_left = env.is_danger((heading - 1) % 4)

        # --------------------------------------------------
        # Heading features (one-hot encoding)
        # --------------------------------------------------

        #This is better than a single integer (0-3), because  the NN might interpretate
        #the differences as distances, instead one-hot encoding.
        heading_features = [
            int(heading == 0),  # Up
            int(heading == 1),  # Right
            int(heading == 2),  # Down
            int(heading == 3),  # Left
        ]

        # --------------------------------------------------
        # Apple direction features
        # --------------------------------------------------

        head_r, head_c = env.snake[0]

        if env.apple is not None:

            apple_r, apple_c = env.apple

            apple_features = [
                int(apple_r < head_r),  # Apple up
                int(apple_c > head_c),  # Apple right
                int(apple_r > head_r),  # Apple down
                int(apple_c < head_c),  # Apple left
            ]

            # Manhattan distance
            distance = (
                abs(apple_r - head_r) +
                abs(apple_c - head_c)
            )

            max_distance = max(1, 2 * (env.size - 1))
            normalized_distance = distance / max_distance

        else:
            # No apple: board is full
            apple_features = [0, 0, 0, 0]
            normalized_distance = 0.0

        # --------------------------------------------------
        # Snake length
        # --------------------------------------------------

        normalized_length = len(env.snake) / (env.size ** 2)

        # --------------------------------------------------
        # Combine all features
        # --------------------------------------------------

        state = [
            danger_straight,
            danger_right,
            danger_left,
            *heading_features,
            *apple_features,
            normalized_length,
            normalized_distance,
        ]

        return np.asarray(state, dtype=np.float32)
