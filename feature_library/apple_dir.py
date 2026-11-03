"""Apple direction feature: sign of the apple's row / column offset from the head."""
from features import FeatureBase


def _sign3(x):
    return (x > 0) - (x < 0) + 1  # -1/0/+1 -> 0/1/2


class AppleDirFeature(FeatureBase):
    name = "apple_dir"
    cardinality = 9
    description = ("apple relative to head: rowSign*3 + colSign "
                   "(row: 0 above/1 same/2 below; col: 0 left/1 same/2 right)")

    def extract(self, env):
        if env.apple is None:  # board is full, no apple left
            return 4
        head = env.snake[0]
        return _sign3(env.apple[0] - head[0]) * 3 + _sign3(env.apple[1] - head[1])

    def format_value(self, value):
        return str(value)
