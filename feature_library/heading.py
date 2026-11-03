"""Heading feature: the direction the snake is currently moving."""
from features import FeatureBase


class HeadingFeature(FeatureBase):
    name = "heading"
    cardinality = 4
    description = "current heading (0 up, 1 right, 2 down, 3 left)"

    def extract(self, env):
        return env.heading

    def format_value(self, value):
        return str(value)
