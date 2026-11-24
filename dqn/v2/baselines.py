"""Policies that can be evaluated side by side: DQN v2, random, greedy BFS, Hamiltonian cycle, tabular Q,
and the old hybrid DQN.

Every policy implements the same tiny interface:

    policy.reset(seed)           called at the start of every game (lets stochastic policies be reproducible)
    policy.act_batch(envs)       one relative action (0 straight, 1 left, 2 right) per live env

act_batch exists so neural policies can run ONE forward pass for all games played in lockstep
(evaluate.run_games); simple policies just loop over act().
"""
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from snake_env import DIRS, LEFT, RIGHT, STRAIGHT, N_ACTIONS

from .shield import action_mask, lookahead_mask
from .state import analyze_space, encode_state, neighbor_table, preview

ROOT = Path(__file__).resolve().parents[2]


def relative_action(heading, target_heading):
    """Relative action that turns `heading` into `target_heading` (None for a 180 degree turn)."""
    diff = (target_heading - heading) % 4
    return {0: STRAIGHT, 3: LEFT, 1: RIGHT}.get(diff)


def heading_to(src, dst):
    return DIRS.index((dst[0] - src[0], dst[1] - src[1]))


class Policy:
    name = "policy"

    def reset(self, seed):
        pass

    def act(self, env):
        raise NotImplementedError

    def act_batch(self, envs):
        return [self.act(env) for env in envs]


# --------------------------------------------------------------------------------------------------- simple
class RandomPolicy(Policy):
    """Uniformly random action (includes suicidal ones): the floor of every comparison."""
    name = "random"

    def __init__(self):
        self.rng = random.Random(0)

    def reset(self, seed):
        self.rng.seed(seed)

    def act(self, env):
        return self.rng.randrange(N_ACTIONS)


class GreedyBFSPolicy(Policy):
    """Shortest path to the apple (BFS, body = walls, tail free). If the apple is unreachable, take the
    non-lethal move with the largest reachable area. Fast, but it walks into traps in the endgame."""
    name = "greedy_bfs"

    def act(self, env):
        size = env.size
        nbrs = neighbor_table(size)
        head = env.snake[0][0] * size + env.snake[0][1]
        tail = env.snake[-1][0] * size + env.snake[-1][1]
        blocked = {r * size + c for r, c in env.body} - {tail}
        goal = env.apple[0] * size + env.apple[1]
        parent = {head: None}
        queue = deque([head])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for n in nbrs[cur]:
                if n not in parent and n not in blocked:
                    parent[n] = cur
                    queue.append(n)
        if goal in parent:
            step = goal
            while parent[step] != head:
                step = parent[step]
            action = relative_action(env.heading, heading_to(env.snake[0], divmod(step, size)))
            if action is not None:
                return action
        # Fallback: most space among non-lethal moves
        best, best_space = STRAIGHT, -1
        for a in range(N_ACTIONS):
            _, _, died, snake = preview(env, a)
            if died:
                continue
            space = analyze_space(snake, size)[0]
            if space > best_space:
                best, best_space = a, space
        return best


def hamiltonian_cycle(size):
    """next_cell[(r, c)] for a Hamiltonian cycle on an even-sized board.

    Column 0 is the way back up; rows 0..size-1 sweep columns 1..size-1 in a zig-zag (even rows go right,
    odd rows go left). The snake starts on row size//2 heading right, so that row must be a right-going
    one; when size//2 is odd the cycle is mirrored top-to-bottom (row r -> size-1-r), which turns the
    right-going rows into odd rows.
    """
    if size % 2:
        raise ValueError("a Hamiltonian cycle on a grid needs an even side length")
    order = [(0, 0)]
    for r in range(size):
        cols = range(1, size) if r % 2 == 0 else range(size - 1, 0, -1)
        order += [(r, c) for c in cols]
    order += [(r, 0) for r in range(size - 1, 0, -1)]
    if (size // 2) % 2:
        order = [(size - 1 - r, c) for r, c in order]
    return {cell: order[(i + 1) % len(order)] for i, cell in enumerate(order)}


class HamiltonianPolicy(Policy):
    """Follow a fixed Hamiltonian cycle: always wins, but needs up to `area` steps per apple.
    This is the reference for the step count a learned policy should beat."""
    name = "hamiltonian"

    def __init__(self):
        self._cycles = {}

    def act(self, env):
        if env.size not in self._cycles:
            self._cycles[env.size] = hamiltonian_cycle(env.size)
        head = env.snake[0]
        nxt = self._cycles[env.size][head]
        return relative_action(env.heading, heading_to(head, nxt))


# --------------------------------------------------------------------------------------------------- learned
class TabularPolicy(Policy):
    """Greedy policy of the tabular Q-learning agent (q_tabular.py) on its discrete feature state."""

    def __init__(self, q_path=ROOT / "q_default.pkl", features="danger,apple_dir,heading", name="tabular"):
        from feature_library import make_feature_set
        from q_tabular import QAgent
        self.name = name
        self.features = make_feature_set(features)
        self.agent = QAgent(self.features.shape, seed=0)
        self.agent.load(q_path)

    def reset(self, seed):
        self.agent.rng.seed(seed)

    def act(self, env):
        return self.agent.act(self.features.encode(env), epsilon=0.0)


class MLPPolicy(Policy):
    """Greedy policy of the basic Deep-Q agent (dqn/train_mlp.py): MLP over 13 engineered features."""
    name = "mlp"

    def __init__(self, model_path=ROOT / "dqn" / "snake_dqn.pth"):
        from dqn.model_loader import load_trained_model
        self.model, self.encoder, self.device, _ = load_trained_model("mlp", model_path)

    def act_batch(self, envs):
        x = torch.as_tensor(np.stack([self.encoder.encode(env) for env in envs]), dtype=torch.float32,
                            device=self.device)
        with torch.no_grad():
            return list(self.model(x).cpu().numpy().argmax(axis=1))


class OldHybridPolicy(Policy):
    """Greedy policy of the previous hybrid DQN (dqn/train_hybrid.py), the model DQN v2 replaces."""
    name = "old_hybrid"

    def __init__(self, model_path=ROOT / "dqn" / "hybrid" / "hybrid_snake_dqn.pth"):
        from dqn.features.tail_features import get_feature_matrix
        from dqn.model_loader import load_trained_model
        self.model, self.encoder, self.device, _ = load_trained_model("hybrid", model_path)
        self._features = get_feature_matrix

    def act_batch(self, envs):
        boards, headings, feats = [], [], []
        for env in envs:
            b, h = self.encoder.encode(env)
            boards.append(b)
            headings.append(h)
            feats.append(self._features(env))
        t = lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.model(t(boards), t(headings), t(feats)).cpu().numpy()
        return list(q.argmax(axis=1))


class DQNv2Policy(Policy):
    """Greedy DQN v2 policy. Immediate-death moves are always masked; `shield=True` also masks moves that
    lose the escape route; `lookahead=<hunger threshold>` adds the virtual-snake apple-route check on top of
    the shield (see shield.py)."""

    def __init__(self, model, device, shield=False, lookahead=None, name=None):
        self.model = model.eval()
        self.device = device
        self.shield = shield or lookahead is not None
        self.lookahead = lookahead
        default = ("dqn_v2+lookahead" if lookahead is not None else "dqn_v2+shield" if shield else "dqn_v2")
        self.name = name or default

    @classmethod
    def from_checkpoint(cls, path, shield=False, lookahead=None, name=None):
        from .agent import load_model
        model, _, device = load_model(path)
        return cls(model, device, shield=shield, lookahead=lookahead, name=name)

    def _mask(self, env, features):
        if self.lookahead is not None:
            return lookahead_mask(env, features, self.lookahead)
        return action_mask(features, self.shield)

    def act_batch(self, envs):
        states = [encode_state(env) for env in envs]
        boards, scalars, feats = (np.stack(x) for x in zip(*states))
        masks = np.stack([self._mask(env, f) for env, f in zip(envs, feats)])
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.model(t(boards), t(scalars), t(feats)).cpu().numpy()
        return list(np.where(masks, q, -np.inf).argmax(axis=1))


def build_policy(spec):
    """Policy from a CLI spec: random | greedy_bfs | hamiltonian | tabular | tabular_bucket | mlp | old_hybrid |
    v2:<checkpoint> | v2shield:<checkpoint> | v2look[threshold]:<checkpoint>   (optionally '=<name>' to rename, e.g. v2:runs/a/best.pth=seed0)."""
    spec, _, rename = spec.partition("=")
    kind, _, arg = spec.partition(":")
    if kind == "random":
        p = RandomPolicy()
    elif kind == "greedy_bfs":
        p = GreedyBFSPolicy()
    elif kind == "hamiltonian":
        p = HamiltonianPolicy()
    elif kind == "tabular":
        p = TabularPolicy()
    elif kind == "tabular_bucket":
        p = TabularPolicy(ROOT / "q_bucket.pkl", "danger,apple_dir,heading,length_bucket", "tabular_bucket")
    elif kind == "old_hybrid":
        p = OldHybridPolicy()
    elif kind == "mlp":
        p = MLPPolicy()
    elif kind in ("v2", "v2shield"):
        p = DQNv2Policy.from_checkpoint(arg, shield=kind == "v2shield")
    elif kind.startswith("v2look"):   # v2look:<ckpt> (hunger threshold 0 = always) or v2look0.5:<ckpt>
        p = DQNv2Policy.from_checkpoint(arg, lookahead=float(kind[len("v2look"):] or 0.0))
    else:
        raise ValueError(f"unknown policy spec {spec!r}")
    if rename:
        p.name = rename
    return p
