"""Evaluate a trained agent greedily (epsilon = 0) on a fixed set of game seeds.

    python evaluate.py runs/default/best.pt               # 500 games
    python evaluate.py runs/default/best.pt --games 100
    python evaluate.py --random                           # random safe-masked policy, no checkpoint

Eval seeds start at 10000, the same as complex_snake, so the numbers are directly
comparable. Reference points from complex_snake/README.md (8x8):
    Hamiltonian cycle        100% wins, ~977 steps per win
    Hamiltonian + shortcuts  100% wins, ~641 steps per win
    Greedy among safe moves  ~9% wins (the rest loop until the hunger limit)
"""
import argparse
import statistics as st
import time

import numpy as np
import torch

from model import QNet
from snake_env import SnakeEnv


def pick_device(name="auto"):
    """CUDA if available, else Apple-silicon MPS, else CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def env_kwargs_from(args):
    """The SnakeEnv settings stored in a run's args (dict), so eval matches training."""
    keys = ("n", "hunger_limit", "r_apple", "r_death", "r_win", "r_hunger", "r_step", "mask_mode")
    return {k: args[k] for k in keys}


@torch.no_grad()
def greedy_actions(net, device, envs):
    """argmax_a Q(s, a) over the *allowed* actions, for a list of envs at once."""
    obs = torch.from_numpy(np.stack([e.obs() for e in envs])).to(device)
    mask = torch.from_numpy(np.stack([e.mask for e in envs])).to(device)
    q = net(obs).masked_fill(~mask, float("-inf"))  # disallowed actions can never win the argmax
    return q.argmax(dim=1).cpu().numpy()


def random_actions(envs, rng):
    """Uniformly random allowed action for each env (a policy with no learning at all)."""
    return np.array([rng.choice(np.flatnonzero(e.mask)) for e in envs])


def summarize(envs):
    """Win rate, apple and steps-to-win statistics over finished games."""
    wins = [e.steps for e in envs if e.won]
    outcomes = {}
    for e in envs:
        outcomes[e.reason] = outcomes.get(e.reason, 0) + 1
    total_apples = sum(e.apples for e in envs)
    return {
        "games": len(envs),
        "win_rate": len(wins) / len(envs),
        "apples_mean": st.mean(e.apples for e in envs),
        # Pooled over all games: total steps / total apples. Includes the steps a game
        # spends after its last apple (e.g. wandering until the hunger limit).
        "steps_per_apple": (sum(e.steps for e in envs) / total_apples) if total_apples else float("nan"),
        "steps_mean": st.mean(wins) if wins else float("nan"),
        "steps_p90": float(np.percentile(wins, 90)) if wins else float("nan"),
        "outcomes": outcomes,
    }


def format_summary(s):
    return (f"win {s['win_rate']:6.1%} | apples {s['apples_mean']:5.1f} | "
            f"steps/apple {s['steps_per_apple']:5.1f} | steps/win mean "
            f"{s['steps_mean']:6.1f} p90 {s['steps_p90']:6.1f} | outcomes {s['outcomes']}")


def evaluate_net(net, device, env_kwargs, games=200, seed0=10_000, policy="greedy"):
    """Play `games` games in parallel from the standard opening. Returns summarize(...).

    All unfinished games are batched into one forward pass per step. Every game ends
    eventually because of the hunger limit.
    """
    if net is not None:
        net.eval()
    envs = [SnakeEnv(seed=seed0 + i, **env_kwargs) for i in range(games)]
    rng = np.random.default_rng(seed0)
    active = list(range(games))
    while active:
        batch = [envs[i] for i in active]
        acts = greedy_actions(net, device, batch) if policy == "greedy" else random_actions(batch, rng)
        for e, a in zip(batch, acts):
            e.step(int(a))
        active = [i for i in active if not envs[i].done]
    if net is not None:
        net.train()
    return summarize(envs)


def load(path, device):
    """Rebuild the network from a checkpoint written by train.py. Returns (net, args)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    env = SnakeEnv(**env_kwargs_from(a))
    net = QNet(n=a["n"], width=a["width"], hidden=a["hidden"], obs_scale=env.obs_scale).to(device)
    net.load_state_dict(ck["model"])
    return net, a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="path to best.pt / last.pt")
    ap.add_argument("--random", action="store_true", help="evaluate a random safe-masked policy instead")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    t = time.time()
    if args.random:
        kw = dict(n=8, hunger_limit=128, r_apple=1.0, r_death=-10.0, r_win=10.0,
                  r_hunger=-10.0, r_step=0.0, mask_mode="safe")
        s = evaluate_net(None, device, kw, args.games, args.seed0, policy="random")
        name = "random+mask"
    else:
        assert args.checkpoint, "give a checkpoint path, or use --random"
        net, a = load(args.checkpoint, device)
        s = evaluate_net(net, device, env_kwargs_from(a), args.games, args.seed0)
        name = "agent"
    print(f"{name:12s} {format_summary(s)} | {time.time() - t:.0f}s")
