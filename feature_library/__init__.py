"""Feature library: one file per feature, each a subclass of features.FeatureBase.

Use directly:   FeatureSet([DangerFeature(), LengthBucketFeature()])
Or by name:     make_feature_set("danger,apple_dir,heading,length_bucket")

To add a feature: create feature_library/<name>.py with a FeatureBase subclass and add it to REGISTRY below.
"""
from features import FeatureSet

from .apple_dir import AppleDirFeature
from .danger import DangerFeature
from .heading import HeadingFeature
from .length_bucket import LengthBucketFeature
from .space import SpaceFeature

# The order here fixes the order of the state tuple (and Q-table layout) for make_feature_set().
REGISTRY = {cls.name: cls for cls in (DangerFeature, SpaceFeature, AppleDirFeature, HeadingFeature, LengthBucketFeature)}

DEFAULT_FEATURES = ("danger", "apple_dir", "heading")


def make_feature_set(names=DEFAULT_FEATURES):
    """Build a FeatureSet from names (list or comma-separated string); input order does not matter."""
    if isinstance(names, str):
        names = [s.strip() for s in names.split(",") if s.strip()]
    names = list(names)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown feature(s) {unknown}; available: {list(REGISTRY)}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature(s) in {names}")
    return FeatureSet([cls() for n, cls in REGISTRY.items() if n in names])
