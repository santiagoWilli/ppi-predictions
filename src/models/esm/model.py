import torch
import torch.nn as nn

class SiameseMLP(nn.Module):
    def __init__(self, input_dim: int = 320, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        combined_dim = input_dim * 4  # x1, x2, |x1−x2|, x1*x2

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        abs_diff = torch.abs(x1 - x2)
        prod = x1 * x2
        combined = torch.cat([x1, x2, abs_diff, prod], dim=1)
        return self.mlp(combined).squeeze(1)
