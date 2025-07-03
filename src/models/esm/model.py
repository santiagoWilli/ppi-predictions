import torch
import torch.nn as nn

class SiameseMLP(nn.Module):
    def __init__(self, input_dim: int = 320, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x).squeeze(1)
