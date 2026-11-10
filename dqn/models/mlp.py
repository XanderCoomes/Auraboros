
import torch
import torch.nn as nn

class DQNMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #Linear instead of softmax
            #We want to compute qvalues for estimate future rewards
            #We dont need probabilities
            nn.Linear(hidden_dim, n_actions)        
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
