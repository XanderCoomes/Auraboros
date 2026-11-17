
import torch
import torch.nn as nn


class DQNHybrid(nn.Module):
    def __init__(
        self,
        board_channels=4,
        n_actions=3,
        n_features=11,
        hidden_dim=128,
    ):
        super().__init__()

        # Visual representation.
        self.cnn = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

        # Engineered features: one shared encoder per action.
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combine visual, heading, and action-specific information.
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 4 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.n_actions = n_actions

    def forward(self, board, heading, features):
        board_features = self.cnn(board)

        # Encode each candidate action separately.
        action_features = self.feature_encoder(features)

        # Repeat global state representation for each action.
        board_features = board_features.unsqueeze(1).expand(
            -1, self.n_actions, -1
        )

        heading_features = heading.unsqueeze(1).expand(
            -1, self.n_actions, -1
        )

        combined = torch.cat(
            (board_features, heading_features, action_features),
            dim=2,
        )

        # One Q-value for each candidate action.
        q_values = self.fusion(combined).squeeze(-1)

        return q_values
