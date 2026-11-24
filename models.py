"""Residual CNN Q-networks with optional spatial feature preservation."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.relu = nn.ReLU()

    def forward(self, inputs):
        hidden = self.relu(self.conv1(inputs))
        return self.relu(self.conv2(hidden) + self.shortcut(inputs))


class ResidualQNetwork(nn.Module):
    def __init__(
        self,
        input_shape,
        context_dim,
        feature_dim=8,
        base_channels=8,
        hidden_dims=(64, 32),
        n_actions=3,
        preserve_spatial=False,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)

        if len(input_shape) not in (3, 4) or (
            len(input_shape) == 4 and input_shape[0] != 2
        ):
            raise ValueError("Expected [H,W,C] or [2,H,W,C] observations")

        if (
            len(hidden_dims) != 2
            or min(*hidden_dims, base_channels, feature_dim) < 1
            or min(input_shape) < 1
        ):
            raise ValueError("Use positive dimensions and two hidden widths")

        channels = input_shape[-1] * (2 if len(input_shape) == 4 else 1)
        height, width = input_shape[-3:-1]

        if preserve_spatial:
            # For an 8×8 board:
            # C×8×8 → 32×8×8 → 32×4×4 → 64×4×4 → 64×4×4
            self.encoder = nn.Sequential(
                ResidualBlock(channels, base_channels),
                nn.MaxPool2d(2, ceil_mode=True),
                ResidualBlock(base_channels, base_channels * 2),
                ResidualBlock(base_channels * 2, feature_dim),
                nn.Flatten(),
            )
            encoded_dim = feature_dim * ((height + 1) // 2) * (
                (width + 1) // 2
            )
        else:
            # Original architecture, retained for old checkpoints.
            self.encoder = nn.Sequential(
                ResidualBlock(channels, base_channels),
                nn.MaxPool2d(2, ceil_mode=True),
                ResidualBlock(base_channels, base_channels * 2),
                nn.MaxPool2d(2, ceil_mode=True),
                ResidualBlock(base_channels * 2, base_channels * 4),
                nn.Conv2d(base_channels * 4, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            encoded_dim = feature_dim

        self.fc1 = nn.Linear(encoded_dim + context_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output = nn.Linear(hidden_dims[1], n_actions)

    def forward(self, board, context, return_activations=False):
        if tuple(board.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected batch + {self.input_shape}, "
                f"got {tuple(board.shape)}"
            )

        if len(self.input_shape) == 4:
            # [B,2,H,W,C] → [B,2*C,H,W].
            # The leading 2 is a value axis.
            batch, _, height, width, channels = board.shape
            board = board.permute(0, 1, 4, 2, 3).reshape(
                batch, 2 * channels, height, width
            )
        else:
            # [B,H,W,C] → [B,C,H,W].
            board = board.permute(0, 3, 1, 2)

        features = self.encoder(board)
        hidden1 = torch.relu(
            self.fc1(torch.cat((features, context), dim=1))
        )
        hidden2 = torch.relu(self.fc2(hidden1))
        q_values = self.output(hidden2)

        if return_activations:
            return q_values, {
                "features": features,
                "fc1": hidden1,
                "fc2": hidden2,
                "q": q_values,
            }
        return q_values


def residual4(**kwargs):
    return ResidualQNetwork(feature_dim=4, **kwargs)


def residual8(**kwargs):
    return ResidualQNetwork(feature_dim=8, **kwargs)


def residual64(**kwargs):
    # Explicit training configuration can override these defaults.
    kwargs.setdefault("base_channels", 32)
    kwargs.setdefault("hidden_dims", (256, 128))
    return ResidualQNetwork(
        feature_dim=64,
        preserve_spatial=True,
        **kwargs,
    )


MODELS = {
    "rescnn4": residual4,
    "rescnn8": residual8,
    "rescnn64": residual64,
}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)