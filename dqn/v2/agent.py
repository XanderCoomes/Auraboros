"""Double DQN agent with action masking, prioritized n-step replay, soft target updates and grad clipping.

Changes vs. dqn/agent.py and why:
- Double DQN target (van Hasselt et al., 2016): the online net picks a* = argmax Q_online(s'), the target
  net evaluates it. Plain max over the target net systematically overestimates Q, which made the old
  evaluation score bounce between 30 and 48.
- Masked targets: the argmax over s' only considers allowed actions (no instant deaths / shield), so the
  target matches the policy that will actually be executed.
- Soft (Polyak) target update, tau=0.005 per train step, instead of a hard copy every 1000 steps: smoother
  targets, no periodic jumps.
- Gradient-norm clipping (10) and a cosine-decaying learning rate (set_lr_progress) for stability late in
  training.
- Batched act(): one forward pass for all parallel environments.
"""
import copy

import numpy as np
import torch
import torch.nn as nn


class DQNAgentV2:
    def __init__(self, model, device, lr=3e-4, lr_min=5e-5, tau=0.005, grad_clip=10.0, seed=0):
        self.device = device
        self.model = model.to(device)
        self.target = copy.deepcopy(self.model).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_max, self.lr_min = lr, lr_min
        self.tau = tau
        self.grad_clip = grad_clip
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    # ------------------------------------------------------------------ helpers
    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def q_values(self, boards, scalars, feats):
        """Batched Q-values (numpy) for stacked states."""
        with torch.no_grad():
            q = self.model(self._t(boards), self._t(scalars), self._t(feats))
        return q.cpu().numpy()

    def act(self, boards, scalars, feats, masks, epsilon=0.0):
        """Epsilon-greedy over ALLOWED actions only. Inputs are batches (one row per env)."""
        q = self.q_values(boards, scalars, feats)
        q = np.where(masks, q, -np.inf)
        actions = q.argmax(axis=1)
        explore = self.rng.random(len(actions)) < epsilon
        for i in np.flatnonzero(explore):
            actions[i] = self.rng.choice(np.flatnonzero(masks[i]))
        return actions, q

    # ------------------------------------------------------------------ learning
    def train_step(self, batch):
        b = batch
        n = len(b["actions"])
        # Copy raw arrays (float16 boards) to the GPU first and convert there: fewer/lighter host copies.
        dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device, non_blocking=True)
        boards = torch.cat((dev(b["boards"]), dev(b["next_boards"]))).float()
        scalars = torch.cat((dev(b["scalars"]), dev(b["next_scalars"])))
        feats = torch.cat((dev(b["feats"]), dev(b["next_feats"])))
        actions = dev(b["actions"])

        # One online forward over [s, s'] (2n rows) instead of two separate passes: half the kernel launches.
        q_all = self.model(boards, scalars, feats)
        q_sa = q_all[:n].gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_masks = dev(b["next_masks"])
            q_online = q_all[n:].detach().masked_fill(~next_masks, -1e9)
            a_star = q_online.argmax(dim=1, keepdim=True)
            q_next = self.target(boards[n:], scalars[n:], feats[n:]).gather(1, a_star).squeeze(1)
            target = dev(b["rewards"]) + dev(b["discounts"]) * q_next

        td = target - q_sa
        loss = (dev(b["weights"]) * nn.functional.smooth_l1_loss(q_sa, target, reduction="none")).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self._soft_update()
        self.updates += 1
        return float(loss.item()), td.detach().abs().cpu().numpy()

    def set_lr_progress(self, progress):
        """Cosine decay from lr to lr_min as training progress goes 0 -> 1.

        Driven by episode progress rather than a fixed update count, because the number of updates per
        episode grows as the agent survives longer and is not known in advance."""
        progress = min(1.0, max(0.0, progress))
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _soft_update(self):
        # target <- target + tau * (online - target), as one fused multi-tensor kernel
        with torch.no_grad():
            torch._foreach_lerp_(list(self.target.parameters()), list(self.model.parameters()), self.tau)

    # ------------------------------------------------------------------ io
    def save(self, path, config):
        torch.save({"model": self.model.state_dict(), "config": config}, path)


def load_model(path, device=None):
    """Load a checkpoint written by DQNAgentV2.save -> (model in eval mode, config dict, device)."""
    from .model import DuelingHybridDQN
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DuelingHybridDQN(board_size=cfg["size"], hidden=cfg.get("hidden", 256),
                             feat_hidden=cfg.get("feat_hidden", 64))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, device
