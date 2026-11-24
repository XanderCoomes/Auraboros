"""Train a plain DQN agent for 8x8 Snake.

    python train.py                                   # defaults below
    python train.py --run-name quick --total-steps 300000 --eval-every 50000
    python train.py --run-name main2 --init-from runs/main/best.pt --eps-start 0.05   # warm start

Algorithm (Mnih et al. 2015, "Human-level control through deep RL"), plus the safety mask:

    for each environment step:
        pick a with epsilon-greedy over the *allowed* actions of s
        take a, observe r, s', done; store (s, a, r, s', mask(s'), done) in replay
        sample a minibatch uniformly from replay
        target y = r + gamma * (1 - done) * max_{a' allowed in s'} Q_target(s', a')
        minimise Huber(Q(s, a) - y) by a gradient step on Q
        every C updates: copy Q's weights into Q_target

Compared with complex_snake/train_dqn.py this drops Double DQN, dueling heads, n-step
returns, prioritized replay, symmetry augmentation, demonstrations (DQfD) and mid-game
starts. It keeps the same network trunk, observation, rewards and safety mask.

Outputs go to runs/<run-name>/: args.json, log.csv, best.pt (highest win rate, then
fewest steps) and last.pt.
"""
import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import env_kwargs_from, evaluate_net, format_summary, pick_device
from model import QNet
from replay import ReplayBuffer
from snake_env import SnakeEnv


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = p.add_argument
    a("--run-name", default="default")
    a("--seed", type=int, default=0)
    a("--device", default="auto")
    a("--init-from", default=None, metavar="CKPT",
      help="warm start: load network weights from a train.py checkpoint (e.g. runs/main/best.pt). "
           "Optimizer, replay buffer and step counters start fresh; pair with a low --eps-start.")
    # environment (same defaults as complex_snake)
    a("--n", type=int, default=8)
    a("--hunger-limit", type=int, default=128)
    a("--r-apple", type=float, default=1.0)
    a("--r-death", type=float, default=-10.0)
    a("--r-win", type=float, default=10.0)
    a("--r-hunger", type=float, default=-10.0)
    a("--r-step", type=float, default=0.0)
    a("--mask-mode", choices=["safe", "legal", "none"], default="safe")
    # DQN
    a("--total-steps", type=int, default=2_000_000, help="environment steps (summed over all envs)")
    a("--num-envs", type=int, default=32, help="games played in parallel to collect experience")
    a("--gamma", type=float, default=0.99)
    a("--lr", type=float, default=1e-4)
    a("--batch-size", type=int, default=256)
    a("--buffer-size", type=int, default=300_000)
    a("--learning-starts", type=int, default=20_000, help="env steps of experience before learning")
    a("--target-update", type=int, default=2_000, help="gradient updates between target-net syncs")
    a("--grad-clip", type=float, default=10.0)
    a("--width", type=int, default=64, help="conv channels")
    a("--hidden", type=int, default=512, help="units in the fully connected layer")
    a("--eps-start", type=float, default=1.0)
    a("--eps-end", type=float, default=0.01)
    a("--eps-frac", type=float, default=0.3, help="fraction of training over which epsilon decays")
    # evaluation
    a("--eval-every", type=int, default=100_000, help="env steps between evaluations")
    a("--eval-games", type=int, default=200)
    return p.parse_args(argv)


# =============================================================================
# The core pieces of DQN
# =============================================================================

def select_actions(q, masks, epsilon, rng):
    """Epsilon-greedy action selection restricted to allowed actions.
    Args:
        q:       (B, 4) float numpy array, Q(s, a) for each of B states.
        masks:   (B, 4) bool numpy array, True where the action is allowed.
        epsilon: probability of acting randomly, per state.
        rng:     np.random.Generator, use it for every random draw.
    Returns:
        (B,) int numpy array of chosen actions, each one allowed by its mask.
    """
    B = len(q)
    greedy = np.where(masks, q, -np.inf).argmax(axis=1)
    explore = rng.random(B) < epsilon
    actions = greedy.copy()
    for i in np.flatnonzero(explore):
        actions[i] = rng.choice(np.flatnonzero(masks[i]))
    return actions


def compute_td_loss(net, target_net, batch, gamma):
    """One-step DQN loss on a minibatch.

    Regresses Q(s, a) toward the bootstrapped target
        y = r + gamma * (1 - done) * max_{a' allowed in s'} Q_target(s', a')
    with the Huber loss. y is computed without gradients (semi-gradient TD).

    Args:
        net:        online Q-network, the one being trained.
        target_net: frozen copy of net, used only to compute bootstrap targets.
        batch:      dict of torch tensors on the right device:
                      obs (B,4,n,n) uint8, action (B,) int64, reward (B,) float32,
                      next_obs (B,4,n,n) uint8, next_mask (B,4) bool, done (B,) float32
        gamma:      discount factor.
    Returns:
        scalar loss tensor (gradients flow only into `net`).
    """
    # 1. Prediction: Q(s, a) for the action actually taken in each row.
    q_all = net(batch["obs"])                                               # (B, 4)
    q_sa = q_all.gather(1, batch["action"].unsqueeze(1)).squeeze(1)         # (B,)

    # 2. Target: no gradients, target_net, max only over actions allowed in s'.
    with torch.no_grad():
        q_next = target_net(batch["next_obs"])                              # (B, 4)
        q_next = q_next.masked_fill(~batch["next_mask"], float("-inf"))
        max_q_next = q_next.max(dim=1).values                               # (B,)
        # A terminal s' may have no allowed actions, which makes the max -inf. The
        # (1 - done) factor zeroes that term anyway, but 0 * -inf is NaN, so clear it first.
        max_q_next = torch.where(batch["done"] > 0, torch.zeros_like(max_q_next), max_q_next)
        y = batch["reward"] + gamma * (1.0 - batch["done"]) * max_q_next

    # 3. Huber loss, averaged over the batch.
    return F.smooth_l1_loss(q_sa, y)


def sync_target_network(net, target_net):
    """Copy the online network's weights into the target network."""
    target_net.load_state_dict(net.state_dict())


# =============================================================================
# Training loop (plumbing: collection, logging, evaluation, checkpoints)
# =============================================================================

def linear_schedule(start, end, frac):
    """Interpolate from start to end as frac goes 0 -> 1, then hold at end."""
    return start + (end - start) * min(max(frac, 0.0), 1.0)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = pick_device(args.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)

        self.env_kwargs = env_kwargs_from(vars(args))
        probe = SnakeEnv(**self.env_kwargs)
        obs_shape = probe.obs().shape
        net_kw = dict(n=args.n, width=args.width, hidden=args.hidden, obs_scale=probe.obs_scale)
        self.net = QNet(**net_kw).to(self.device)         # online network (trained)
        self.target_net = QNet(**net_kw).to(self.device)  # target network (periodic copy)
        self.out_dir = os.path.join("runs", args.run_name)
        if args.init_from:
            self.warm_start(args.init_from)
        sync_target_network(self.net, self.target_net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(args.buffer_size, obs_shape)

        self.env_steps = 0
        self.updates = 0
        self.best_key = None
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    def warm_start(self, path):
        """Load online-network weights from a checkpoint (the caller then syncs the target net)."""
        # Writing into the source run's folder would overwrite its best.pt, last.pt and log.csv.
        if os.path.abspath(os.path.dirname(path)) == os.path.abspath(self.out_dir):
            raise SystemExit(f"--init-from {path} is inside runs/{self.args.run_name}/, which this "
                             f"run would overwrite. Pick a different --run-name.")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        src = ck["args"]
        for k in ("n", "width", "hidden"):
            if src[k] != getattr(self.args, k):
                raise SystemExit(f"--init-from: checkpoint has --{k} {src[k]}, this run has "
                                 f"--{k} {getattr(self.args, k)}; the networks are incompatible.")
        for k in ("hunger_limit", "mask_mode"):
            if src.get(k) != getattr(self.args, k):
                print(f"[warm start] note: checkpoint trained with --{k.replace('_', '-')} "
                      f"{src.get(k)}, this run uses {getattr(self.args, k)}")
        self.net.load_state_dict(ck["model"])
        ev = ck.get("eval", {})
        print(f"[warm start] loaded {path} (trained {ck.get('env_steps', 0):,} env steps, "
              f"eval win rate {ev.get('win_rate', float('nan')):.1%})")

    def epsilon(self):
        a = self.args
        return linear_schedule(a.eps_start, a.eps_end, self.env_steps / (a.eps_frac * a.total_steps))

    def update(self):
        """Sample a minibatch, take one gradient step, sync the target net when due."""
        a = self.args
        b = self.buffer.sample(a.batch_size, self.rng)
        batch = {k: torch.from_numpy(v).to(self.device) for k, v in b.items()}
        loss = compute_td_loss(self.net, self.target_net, batch, a.gamma)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), a.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % a.target_update == 0:
            sync_target_network(self.net, self.target_net)
        return loss.item()

    def evaluate(self):
        """Greedy eval on fixed seeds; save last.pt, and best.pt if it is the best so far."""
        a = self.args
        s = evaluate_net(self.net, self.device, self.env_kwargs, a.eval_games)
        # Rank by win rate first, then by fewer steps per win.
        key = (s["win_rate"], -s["steps_mean"] if s["win_rate"] > 0 else -1e9)
        ck = {"model": self.net.state_dict(), "args": vars(a), "eval": s,
              "env_steps": self.env_steps, "updates": self.updates}
        torch.save(ck, os.path.join(self.out_dir, "last.pt"))
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            torch.save(ck, os.path.join(self.out_dir, "best.pt"))
        print(f"[eval] steps {self.env_steps:,} | {format_summary(s)}")
        return s

    def train(self):
        a = self.args
        # Several independent games run side by side, so one forward pass picks actions for
        # all of them. Each step of the loop adds num_envs transitions to the buffer.
        envs = [SnakeEnv(seed=a.seed * 1000 + i, **self.env_kwargs) for i in range(a.num_envs)]
        obs = [e.reset() for e in envs]
        recent = deque(maxlen=200)  # (won, length) of recently finished training games

        log_f = open(os.path.join(self.out_dir, "log.csv"), "w", newline="")
        log = csv.writer(log_f)
        log.writerow(["env_steps", "updates", "epsilon", "loss", "train_win_rate", "train_mean_len",
                      "eval_win_rate", "eval_steps_mean", "eval_steps_p90", "eval_apples_mean",
                      "eval_steps_per_apple", "fps"])
        t0, steps0, last_eval, loss = time.time(), 0, 0, float("nan")

        while self.env_steps < a.total_steps:
            # ---- act: one batched forward pass, then epsilon-greedy over allowed moves
            masks = np.stack([e.mask for e in envs])
            with torch.no_grad():
                q = self.net(torch.from_numpy(np.stack(obs)).to(self.device)).cpu().numpy()
            eps = self.epsilon()
            actions = select_actions(q, masks, eps, self.rng)

            # ---- step every env and store the transitions
            for i, env in enumerate(envs):
                next_obs, r, done, info = env.step(int(actions[i]))
                self.buffer.add(obs[i], int(actions[i]), r, next_obs, env.mask.copy(), done)
                if done:
                    recent.append((info["won"], info["length"]))
                    obs[i] = env.reset()
                else:
                    obs[i] = next_obs
            self.env_steps += len(envs)

            # ---- learn: one gradient step per batch of num_envs env steps, once the
            # buffer holds --learning-starts transitions
            if self.env_steps >= a.learning_starts and len(self.buffer) >= a.batch_size:
                loss = self.update()

            # ---- evaluate and log
            if self.env_steps - last_eval >= a.eval_every or self.env_steps >= a.total_steps:
                last_eval = self.env_steps
                fps = (self.env_steps - steps0) / (time.time() - t0)
                s = self.evaluate()
                tw = np.mean([x[0] for x in recent]) if recent else float("nan")
                tl = np.mean([x[1] for x in recent]) if recent else float("nan")
                log.writerow([self.env_steps, self.updates, round(eps, 4), round(loss, 5), round(tw, 4),
                              round(tl, 2), s["win_rate"], s["steps_mean"], s["steps_p90"],
                              s["apples_mean"], round(s["steps_per_apple"], 2), round(fps)])
                log_f.flush()
                print(f"       eps {eps:.3f} | loss {loss:.4f} | train win {tw:.1%} len {tl:.1f} | "
                      f"{fps:.0f} env steps/s")
                t0, steps0 = time.time(), self.env_steps
        log_f.close()
        print("done. best checkpoint:", os.path.join(self.out_dir, "best.pt"))


if __name__ == "__main__":
    Trainer(parse_args()).train()
