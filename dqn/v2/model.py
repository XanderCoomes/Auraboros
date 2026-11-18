"""Dueling hybrid Q-network: CNN over the egocentric board + per-action feature branch.

                board (C,10,10) --CNN--> h_board (hidden)
                scalars (2)  ----------> concatenated -> h_state ----> V(s)            (state value)
   features (3, F) -- shared MLP per action --> h_a  ;  [h_state, h_a] -> A(s, a)    (advantage per action)

   Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

Why dueling? In most Snake states the three moves have nearly the same value (only one cell differs);
what matters is how good the *situation* is. Learning V(s) once instead of three times makes value
estimates more stable (Wang et al., 2016).

Why padding=1? With the wall ring already in the input, padded 3x3 convs keep the 10x10 resolution so
cells next to the walls are processed like any other cell.
"""
import torch
import torch.nn as nn

from .state import N_CHANNELS, N_FEATURES, N_SCALARS


class DuelingHybridDQN(nn.Module):
    def __init__(self, board_size=8, hidden=256, feat_hidden=64, n_actions=3):
        super().__init__()
        side = board_size + 2
        self.n_actions = n_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 10x10 -> 5x5
            nn.Flatten(),
            nn.Linear(64 * ((side + 1) // 2) ** 2, hidden), nn.ReLU(),
        )
        self.state_mlp = nn.Sequential(nn.Linear(hidden + N_SCALARS, hidden), nn.ReLU())
        self.feat_mlp = nn.Sequential(
            nn.Linear(N_FEATURES, feat_hidden), nn.ReLU(),
            nn.Linear(feat_hidden, feat_hidden), nn.ReLU(),
        )
        # Action embedding: tells the advantage head which relative move a row belongs to
        # (straight / left / right are not symmetric once the board is egocentric).
        self.action_embed = nn.Embedding(n_actions, 16)
        self.value_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv_head = nn.Sequential(
            nn.Linear(hidden + feat_hidden + 16, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, board, scalars, features):
        h = self.state_mlp(torch.cat((self.cnn(board), scalars), dim=1))          # (B, hidden)
        v = self.value_head(h)                                                     # (B, 1)
        hf = self.feat_mlp(features)                                               # (B, A, fh)
        emb = self.action_embed.weight.unsqueeze(0).expand(h.shape[0], -1, -1)     # (B, A, 16)
        ha = torch.cat((h.unsqueeze(1).expand(-1, self.n_actions, -1), hf, emb), dim=2)
        adv = self.adv_head(ha).squeeze(-1)                                        # (B, A)
        return v + adv - adv.mean(dim=1, keepdim=True)
