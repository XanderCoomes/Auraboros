
import torch
import torch.nn as nn


class DQNCNN(nn.Module):

    def __init__(
        self,
        board_channels: int = 3,
        n_actions: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Extract spatial features from the board
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=board_channels,
                out_channels=32,
                kernel_size=3,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

        # Process board features + heading
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, n_actions),
        )

    def forward(
        self,
        board: torch.Tensor,
        heading: torch.Tensor,
    ) -> torch.Tensor:

        board_features = self.cnn(board)

        combined = torch.cat(
            (board_features, heading),
            dim=1,
        )

        return self.network(combined)
