import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualRCNNBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bi_gru  = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv    = nn.Conv1d(in_channels=2 * hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.project = nn.Conv1d(in_channels=2 * hidden_dim, out_channels=hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim, seq_len)
        x_ = x.transpose(1, 2)               # (batch, seq_len, hidden_dim)
        out, _ = self.bi_gru(x_)             # (batch, seq_len, 2*hidden_dim)
        out = out.transpose(1, 2)            # (batch, 2*hidden_dim, seq_len)

        residual = self.project(out)         # (batch, hidden_dim, seq_len)
        out = self.conv(out)                 # (batch, hidden_dim, seq_len)
        return F.relu(out + residual)


class PIPRModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_blocks: int = 3):
        super().__init__()
        self.embedding  = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.projection = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=1)

        self.blocks = nn.Sequential(
            *[ResidualRCNNBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # No sigmoid here; we use BCEWithLogitsLoss
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)                # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)               # (batch, embed_dim, seq_len)
        x = self.projection(x)              # (batch, hidden_dim, seq_len)
        x = self.blocks(x)                  # (batch, hidden_dim, seq_len)
        x = self.pool(x).squeeze(-1)        # (batch, hidden_dim)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.encode(x1)
        h2 = self.encode(x2)
        out = torch.cat([h1, h2], dim=1)     # (batch, 2*hidden_dim)
        return self.classifier(out).squeeze(-1)  # (batch,)
