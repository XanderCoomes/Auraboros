"""DQN agent compatible with train.run_episode's act/update interface."""
import copy
import os
import random
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models import build_model
from snake_env import N_ACTIONS
from state_encoders import Observation


class DQNAgent:
    def __init__(self, model_config, learning_rate=0.001, gamma=0.95,
                 batch_size=64, capacity=10000, warmup=1000, train_every=4,
                 target_every=250, reward_scale=0.01, seed=0, device="cpu"):
        if not 0 <= gamma <= 1 or learning_rate <= 0 or reward_scale <= 0:
            raise ValueError("Invalid discount, learning rate, or reward scale")
        if min(batch_size, capacity, train_every, target_every) < 1 or warmup < 0:
            raise ValueError("Invalid replay or training intervals")
        if capacity < max(batch_size, warmup):
            raise ValueError("Replay capacity must cover both batch size and warmup")
        self.model_config = copy.deepcopy(model_config)
        self.settings = dict(learning_rate=learning_rate, gamma=gamma, batch_size=batch_size,
                             capacity=capacity, warmup=warmup, train_every=train_every,
                             target_every=target_every, reward_scale=reward_scale, seed=seed)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.online = build_model(**model_config).to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.steps = 0
        self.updates = 0
        self.last_loss = None

    def _batch(self, observations):
        boards = torch.as_tensor(np.stack([s.board for s in observations]), device=self.device)
        contexts = torch.as_tensor(np.stack([s.context for s in observations]), device=self.device)
        return boards, contexts

    def q_values(self, state):
        self.online.eval()
        with torch.no_grad():
            return self.online(*self._batch([state]))[0].cpu().numpy()

    def inspect(self, state):
        self.online.eval()
        with torch.no_grad():
            _, activations = self.online(*self._batch([state]), return_activations=True)
        return {name: values[0].cpu().tolist() for name, values in activations.items()}

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        values = self.q_values(state)
        return int(self.rng.choice(np.flatnonzero(values == values.max()).tolist()))

    def update(self, state, action, reward, next_state, done):
        # Own snapshots even if a caller reuses its observation arrays.
        before = Observation(state.board.copy(), state.context.copy())
        after = Observation(next_state.board.copy(), next_state.context.copy())
        self.memory.append((before, action, reward, after, done))
        self.steps += 1
        ready = len(self.memory) >= max(self.settings["batch_size"], self.settings["warmup"])
        if not ready or self.steps % self.settings["train_every"]:
            return None
        transitions = self.rng.sample(list(self.memory), self.settings["batch_size"])
        states, actions, rewards, next_states, dones = zip(*transitions)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = rewards * self.settings["reward_scale"]
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            future = self.target(*self._batch(next_states)).max(dim=1).values
            targets = rewards + self.settings["gamma"] * future.masked_fill(dones, 0)
        self.online.train()
        predicted = self.online(*self._batch(states)).gather(1, actions).squeeze(1)
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.settings["target_every"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.detach())
        return self.last_loss

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "algorithm": "dqn", "model_config": self.model_config,
                   "settings": self.settings, "online": self.online.state_dict(),
                   "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "steps": self.steps, "updates": self.updates, "metadata": metadata or {}}
        # Write beside the destination so replacing a checkpoint is atomic.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temporary = temp.name
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("version") != 1 or payload.get("algorithm") != "dqn":
            raise ValueError("Unsupported DQN checkpoint")
        agent = cls(payload["model_config"], device=device, **payload["settings"])
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.steps, agent.updates = payload["steps"], payload["updates"]
        return agent, payload["metadata"]
