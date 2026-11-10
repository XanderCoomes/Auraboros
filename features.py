"""Plug-and-play state features for tabular Q-learning.

Adding a feature = writing one function decorated with @feature:

    @feature("apple_adjacent", n_values=2, labels=["no", "yes"])
    def apple_adjacent(env):
        return int(_dist_to_apple(env) == 1)

The function takes the SnakeEnv and returns an int in [0, n_values).
Then train with it:  python train.py --features danger_straight,danger_left,danger_right,apple_adjacent,heading

The state is the tuple of the selected features' values; StateSpec computes the
state-space shape / size and the flattening into a Q-table row automatically.
"""
import numpy as np


# name -> Feature; filled by the @feature decorator below
FEATURES = {}

# Feature set used when none is given; this is the original 288-state representation
DEFAULT_FEATURES = ["danger_straight", "danger_left", "danger_right",
                    "apple_row_sign", "apple_col_sign", "heading"]


class Feature:
    def __init__(self, name, n_values, fn, labels=None, doc=""):
        self.name = name
        self.n_values = n_values  # how many distinct values fn can return (0 .. n_values-1)
        self.fn = fn
        self.labels = labels      # optional human-readable name for each value
        self.doc = doc

    def __call__(self, env):
        v = int(self.fn(env))
        assert 0 <= v < self.n_values, f"feature {self.name} returned {v}, expected 0..{self.n_values - 1}"
        return v

    def label(self, v):
        return self.labels[v] if self.labels else str(v)


def feature(name, n_values, labels=None):
    """Decorator: register fn(env) -> int in [0, n_values) as a state feature called `name`."""
    def register(fn):
        assert name not in FEATURES, f"duplicate feature name {name}"
        assert labels is None or len(labels) == n_values, f"{name}: need one label per value"
        FEATURES[name] = Feature(name, n_values, fn, labels, (fn.__doc__ or "").strip())
        return fn
    return register


class StateSpec:
    """An ordered selection of features -> state tuple, state-space shape, and Q-table row index."""

    def __init__(self, names=None):
        names = list(names or DEFAULT_FEATURES)
        unknown = [n for n in names if n not in FEATURES]
        if unknown:
            raise ValueError(f"unknown features {unknown}; available: {sorted(FEATURES)}")
        self.names = names
        self.features = [FEATURES[n] for n in names]
        self.shape = tuple(f.n_values for f in self.features)
        self.num_states = int(np.prod(self.shape))
        self._fns = [f.fn for f in self.features]  # raw functions: skip Feature.__call__ overhead in the hot loop
        self._rows = {}                            # state tuple -> row cache (at most num_states entries)

    def encode(self, env):
        """Current env -> state tuple (one int per feature)."""
        return tuple([fn(env) for fn in self._fns])

    def row(self, state):
        """Flatten a state tuple into one integer 0..num_states-1 = the row index into the Q-table."""
        r = self._rows.get(state)
        if r is None:  # first time this state is seen: range-check every feature value, then cache
            for f, v in zip(self.features, state):
                assert 0 <= v < f.n_values and v == int(v), f"feature {f.name} returned {v}, expected 0..{f.n_values - 1}"
            r = self._rows[state] = int(np.ravel_multi_index(tuple(int(v) for v in state), self.shape))
        return r

    def describe(self, state):
        """{feature name: readable value} for logging / plotting."""
        return {f.name: f.label(v) for f, v in zip(self.features, state)}


def list_features():
    return {n: {"n_values": f.n_values, "labels": f.labels, "doc": f.doc} for n, f in FEATURES.items()}


# ------------------------------------------------------------------ helpers
def _sgn(x):
    return (x > 0) - (x < 0) + 1  # -1/0/+1 -> 0/1/2


# ----------------------------------------------------------------- features
@feature("danger_straight", 2, labels=["safe", "danger"])
def danger_straight(env):
    """Wall/body directly ahead?"""
    return env._is_danger(env.heading)


@feature("danger_left", 2, labels=["safe", "danger"])
def danger_left(env):
    """Wall/body to the left?"""
    return env._is_danger((env.heading - 1) % 4)


@feature("danger_right", 2, labels=["safe", "danger"])
def danger_right(env):
    """Wall/body to the right?"""
    return env._is_danger((env.heading + 1) % 4)


@feature("apple_row_sign", 3, labels=["above", "same_row", "below"])
def apple_row_sign(env):
    """Apple row vs head (absolute frame)."""
    return 1 if env.apple is None else _sgn(env.apple[0] - env.snake[0][0])


@feature("apple_col_sign", 3, labels=["left", "same_col", "right"])
def apple_col_sign(env):
    """Apple column vs head (absolute frame)."""
    return 1 if env.apple is None else _sgn(env.apple[1] - env.snake[0][1])


@feature("heading", 4, labels=["up", "right", "down", "left"])
def heading(env):
    """Current heading."""
    return env.heading


LENGTH_LABELS = ["perfect", "micro", "short", "average", "long", "gargantuan", "huge", "packin"]


@feature("length", len(LENGTH_LABELS), labels=LENGTH_LABELS)
def length(env):
    """Snake length bucket: board area split into equal slices (8x8 board -> 1-8, 9-16, ..., 57-64)."""
    n = len(LENGTH_LABELS)
    return min(n - 1, (len(env.snake) - 1) * n // (env.size * env.size))
