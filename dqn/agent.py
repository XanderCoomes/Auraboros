
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_env import N_ACTIONS


class DQNAgent:

    def __init__(
        self,
        model,
        replay_buffer,
        gamma=0.95,
        learning_rate=0.001,
        seed=21,
        device=None,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.gamma = gamma

        self.rng = random.Random(seed)

        self.device = (
            device
            if device is not None
            else next(model.parameters()).device
        )

        # Move online network to the selected device.
        self.model = self.model.to(self.device)

        # Target network starts with identical weights.
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        for parameter in self.target_model.parameters():
            parameter.requires_grad = False

        # Optimizer updates the online network only.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.SmoothL1Loss()

    # Convert one state into tensors.
    
    def _state_to_tensors(self, state):

        if isinstance(state, tuple):

            tensors = tuple(
                torch.as_tensor(
                    component,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                for component in state
            )

            return tensors

        return torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)



    # Convert a batch of states into tensors.
    
    def _batch_to_tensors(self, states):

        if isinstance(states[0], tuple):

            components = []

            for index in range(len(states[0])):

                batch_component = np.stack(
                    [state[index] for state in states]
                )

                tensor = torch.as_tensor(
                    batch_component,
                    dtype=torch.float32,
                    device=self.device,
                )

                components.append(tensor)

            return tuple(components)

        return torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device,
        )


    # Forward pass supporting both model architectures.
    def _forward(self, model, state_tensors):

        if isinstance(state_tensors, tuple):
            return model(*state_tensors)

        return model(state_tensors)


    def act(self, state, epsilon=0.0):
        """Select an action using epsilon-greedy."""

        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)

        state_tensors = self._state_to_tensors(state)

        with torch.no_grad():
            q_values = self._forward(self.model, state_tensors)

        action_values = q_values.squeeze(0).cpu().numpy()

        best_actions = np.flatnonzero(
            action_values == action_values.max()
        )

        return int(self.rng.choice(list(best_actions)))

    def remember(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer."""

        self.replay_buffer.push(
            state, action, reward, next_state, done
        )

    
    def train_step(self, batch_size=64):
        """Train the online network using a random batch."""

        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states into tensors.
        states = self._batch_to_tensors(states)
        next_states = self._batch_to_tensors(next_states)

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # Q(s, a): predicted Q-value for the selected action.
        predicted_q = self._forward(
            self.model, states
        ).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate Bellman targets using the target network.
        with torch.no_grad():

            next_q = self._forward(
                self.target_model, next_states
            ).max(dim=1)[0]

            target_q = rewards + self.gamma * (1 - dones) * next_q

        # Calculate loss and update online network.
        loss = self.loss_fn(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def update_target_network(self):
        """Copy online network weights into the target network."""

        self.target_model.load_state_dict(self.model.state_dict())

    def get_q_values(self, state):
        """Return the network's Q-values for a given state."""

        state_tensors = self._state_to_tensors(state)

        self.model.eval()

        with torch.no_grad():
            q_values = self._forward(
                self.model,
                state_tensors,
            )

        return q_values.squeeze(0).cpu().numpy()