"""Danger feature: is there a wall/body one cell straight / left / right of the head?"""
from features import FeatureBase


class DangerFeature(FeatureBase):
    name = "danger"
    cardinality = 8
    description = "wall/body danger as 3 bits: straight*4 + left*2 + right (0..7)"

    def extract(self, env):
        h = env.heading
        return (env.is_danger(h) << 2) | (env.is_danger((h - 1) % 4) << 1) | env.is_danger((h + 1) % 4)

    def format_value(self, value):
        return str(value)
