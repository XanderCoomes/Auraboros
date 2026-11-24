"""Convolutional Q-network over the full board.

Same trunk as complex_snake/model.py (4 conv layers, 3x3 kernels, "same" padding),
but with a single Q head instead of separate value / advantage (dueling) streams.

    input  (B, 4, n, n) uint8 planes
    -> divide by obs_scale, append a constant plane of ones   (B, 5, n, n)
    -> 4 x [Conv 3x3 + ReLU]                                  (B, width, n, n)
    -> flatten                                                (B, width * n * n)
    -> Linear + ReLU -> Linear                                (B, n_actions)  = Q(s, .)
"""
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, n=8, in_channels=4, n_actions=4, width=64, hidden=512, obs_scale=None):
        super().__init__()
        if obs_scale is None:
            obs_scale = [1.0] * in_channels
        # Stored as a buffer (not a parameter) so it moves with .to(device) and is saved in
        # the state_dict, but is never trained.
        self.register_buffer("scale", torch.tensor(obs_scale, dtype=torch.float32).view(1, -1, 1, 1))

        # +1 input channel: a plane of all ones. Conv layers zero-pad the border, so at the
        # edge the ones plane "drops off" to zero. That tells the first layer where the walls
        # are, which it could not otherwise distinguish from empty cells.
        c = in_channels + 1
        self.conv = nn.Sequential(
            nn.Conv2d(c, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        )
        # After 4 layers of 3x3 convs each cell "sees" a 9x9 window, i.e. the whole 8x8 board
        # from anywhere. The fully connected head then combines all positions.
        self.head = nn.Sequential(
            nn.Linear(width * n * n, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        x = obs.float() / self.scale                          # normalise each plane to [0, 1]
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)  # append the constant plane
        h = self.conv(x).flatten(1)
        return self.head(h)                                   # (B, n_actions) Q-values
