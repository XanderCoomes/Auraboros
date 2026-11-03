"""Pluggable discrete state features for tabular Q-learning.

A feature is a subclass of FeatureBase that maps the current SnakeEnv to one integer in [0, cardinality).
A FeatureSet bundles several feature instances into a state tuple, so features can be switched on/off
without touching the environment or the agent.

Concrete features live in feature_library/ (one file per feature). To add one: subclass FeatureBase in a
new file there and add it to REGISTRY in feature_library/__init__.py.
"""
from abc import ABC, abstractmethod


class FeatureBase(ABC):
    """Interface every feature implements. Subclasses define `name`, `cardinality`, `description`
    (plain class attributes are enough) and implement `extract` and `format_value`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique short name, used by --features and in json files."""

    @property
    @abstractmethod
    def cardinality(self) -> int:
        """Number of possible values: extract() returns an int in [0, cardinality)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable meaning of the values (used in plot / json)."""

    @abstractmethod
    def extract(self, env) -> int:
        """Return this feature's value for the current state of `env`."""

    @abstractmethod
    def format_value(self, value) -> str:
        """Readable form of a value, e.g. for the GIF overlay (enum-like features return the enum name)."""


class FeatureSet:
    """Ordered collection of features. The order fixes the layout of the Q-table, so a Q-table is only
    valid together with the same features in the same order (stored as names in q_table.json)."""

    def __init__(self, features):
        self.features = list(features)
        if not self.features:
            raise ValueError("at least one feature is required")
        self.names = [f.name for f in self.features]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate feature(s) in {self.names}")
        self.shape = tuple(f.cardinality for f in self.features)
        self.n_states = 1
        for c in self.shape:
            self.n_states *= c

    def encode(self, env):
        return tuple(f.extract(env) for f in self.features)

    def describe(self):
        return [{"name": f.name, "cardinality": f.cardinality, "meaning": f.description}
                for f in self.features]

    def format_state(self, state):
        """Readable 'name=value' string, e.g. for the GIF overlay."""
        return " ".join(f"{f.name}={f.format_value(v)}" for f, v in zip(self.features, state))

    def __str__(self):
        return ",".join(self.names)
