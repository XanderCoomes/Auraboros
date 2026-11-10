"""Residual CNNs with 4- or 8-value learned representations and raw Q outputs."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        hidden = self.relu(self.conv1(inputs))
        return self.relu(self.conv2(hidden) + self.shortcut(inputs))


class ResidualQNetwork(nn.Module):
    def __init__(self, input_shape, context_dim, feature_dim=8,
                 base_channels=8, hidden_dims=(64, 32), n_actions=3):
        super().__init__()
        self.input_shape = tuple(input_shape)
        if len(input_shape) not in (3, 4) or (len(input_shape) == 4 and input_shape[0] != 2):
            raise ValueError("Expected [H,W,C] or [2,H,W,C] observations")
        if len(hidden_dims) != 2 or min(*hidden_dims, base_channels, feature_dim) < 1:
            raise ValueError("Use two positive hidden widths and positive channel counts")
        channels = input_shape[-1] * (2 if len(input_shape) == 4 else 1)
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
        self.fc1 = nn.Linear(feature_dim + context_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output = nn.Linear(hidden_dims[1], n_actions)

    def forward(self, board, context, return_activations=False):
        if tuple(board.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected batch + {self.input_shape}, got {tuple(board.shape)}")
        if len(self.input_shape) == 4:
            # [B,2,H,W,C] -> [B,2*C,H,W]. The leading 2 is a value axis.
            batch, _, height, width, channels = board.shape
            board = board.permute(0, 1, 4, 2, 3).reshape(batch, 2 * channels, height, width)
        else:
            board = board.permute(0, 3, 1, 2)
        features = self.encoder(board)
        hidden1 = torch.relu(self.fc1(torch.cat((features, context), dim=1)))
        hidden2 = torch.relu(self.fc2(hidden1))
        q_values = self.output(hidden2)
        if return_activations:
            return q_values, {"features": features, "fc1": hidden1, "fc2": hidden2, "q": q_values}
        return q_values


def residual4(**kwargs):
    return ResidualQNetwork(feature_dim=4, **kwargs)


def residual8(**kwargs):
    return ResidualQNetwork(feature_dim=8, **kwargs)


# A replacement constructor takes the same input metadata and returns an nn.Module.
MODELS = {"rescnn4": residual4, "rescnn8": residual8}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)
