"""Length bucket feature: coarse snake length class (4 buckets, thresholds relative to the board area)."""
from enum import IntEnum

from features import FeatureBase


class LengthBucket(IntEnum):
    SHORT = 0
    MEDIUM = 1
    LONG = 2
    XLONG = 3


# Upper bounds (exclusive) as a fraction of the board area; anything above the last bound is XLONG.
# On an 8x8 board: SHORT 3-6, MEDIUM 7-15, LONG 16-31, XLONG 32-64.
BUCKET_THRESHOLDS = (0.10, 0.25, 0.50)


def length_bucket(length, size):
    ratio = length / (size * size)
    for bucket, bound in zip(LengthBucket, BUCKET_THRESHOLDS):
        if ratio < bound:
            return bucket
    return LengthBucket.XLONG


class LengthBucketFeature(FeatureBase):
    name = "length_bucket"
    cardinality = len(LengthBucket)
    description = ("snake length bucket (0 SHORT, 1 MEDIUM, 2 LONG, 3 XLONG); thresholds = fraction of board area "
                   + str(BUCKET_THRESHOLDS))

    def extract(self, env):
        return int(length_bucket(len(env.snake), env.size))

    def format_value(self, value):
        return LengthBucket(value).name
