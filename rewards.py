"""Plug-and-play reward terms.

The reward for one step is a weighted sum of terms:  reward = sum(weight_i * term_i(env, ev)).
Adding a term = writing one function decorated with @reward:

    @reward("length_bonus", weight=1.0)
    def length_bonus(env, ev):
        \"\"\"+len(snake) on every apple: bigger snake, bigger payoff.\"\"\"
        return len(env.snake) if ev["outcome"] in ("apple", "win") else 0

The function gets the env (AFTER the move) and an event dict from SnakeEnv.step():
    outcome    "move" | "apple" | "win" | "collision" | "timeout"
    prev_dist  Manhattan distance head -> apple before the move
    dist       distance from the new head to the apple the snake was chasing (None on collision)
    prev_head, head
Then pick terms / weights at train time:  python train.py --rewards apple=100,death=-1000,length_bonus=5
"""

# name -> RewardTerm; filled by the @reward decorator below
REWARDS = {}


class RewardTerm:
    def __init__(self, name, weight, fn, doc=""):
        self.name = name
        self.weight = weight  # default weight, used when the term is selected without one
        self.fn = fn
        self.doc = doc


def reward(name, weight):
    """Decorator: register fn(env, ev) -> float as a reward term called `name` with a default weight."""
    def register(fn):
        assert name not in REWARDS, f"duplicate reward name {name}"
        REWARDS[name] = RewardTerm(name, weight, fn, (fn.__doc__ or "").strip())
        return fn
    return register


class RewardSpec:
    """Selected reward terms and their weights -> scalar reward per step."""

    def __init__(self, weights=None):
        # weights: {name: weight or None (= default weight)}; None selects DEFAULT_REWARDS
        weights = dict(weights) if weights is not None else dict.fromkeys(DEFAULT_REWARDS)
        unknown = [n for n in weights if n not in REWARDS]
        if unknown:
            raise ValueError(f"unknown reward terms {unknown}; available: {sorted(REWARDS)}")
        self.weights = {n: float(REWARDS[n].weight if w is None else w) for n, w in weights.items()}
        self._terms = [(REWARDS[n].fn, w) for n, w in self.weights.items()]

    @staticmethod
    def parse(text):
        """'apple=100,death,step=-0.01' -> RewardSpec ('all' = every term at its default weight)."""
        if text == "all":
            return RewardSpec(dict.fromkeys(REWARDS))
        weights = {}
        for item in filter(None, text.split(",")):
            name, _, w = item.partition("=")
            weights[name.strip()] = float(w) if w else None
        return RewardSpec(weights)

    def total(self, env, ev):
        r = 0.0
        for fn, w in self._terms:
            v = fn(env, ev)
            if v:
                r += w * v
        return r

    def breakdown(self, env, ev):
        """{term: weighted contribution} for logging / plotting."""
        return {n: w * REWARDS[n].fn(env, ev) for n, w in self.weights.items()}

    def __str__(self):
        return ", ".join(f"{n} {w:+g}" for n, w in self.weights.items())


def list_rewards():
    return {n: {"default_weight": t.weight, "doc": t.doc} for n, t in REWARDS.items()}


# ------------------------------------------------------------------- terms
@reward("apple", 100.0)
def apple(env, ev):
    """1 when an apple is eaten (including the last one, which wins)."""
    return ev["outcome"] in ("apple", "win")


@reward("death", -6350.0)
def death(env, ev):
    """1 on collision or timeout (too long without eating)."""
    return ev["outcome"] in ("collision", "timeout")


@reward("step", -0.01)
def step(env, ev):
    """1 on every ordinary move (no apple, no death): a small cost for wasting time."""
    return ev["outcome"] == "move"


@reward("closer", 0.1)
def closer(env, ev):
    """Shaping: +1 if an ordinary move got closer to the apple, -1 if it got farther."""
    if ev["outcome"] != "move":
        return 0
    return 1 if ev["dist"] < ev["prev_dist"] else -1


# Terms used when none are given; with their default weights this is the original reward function
DEFAULT_REWARDS = ["apple", "death", "step", "closer"]
