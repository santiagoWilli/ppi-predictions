import torch
import torch.nn as nn

class ConvolutiveSiameseModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)               # (batch, seq_len, embed)
        embedded = embedded.transpose(1, 2)        # (batch, embed, seq_len)
        encoded = self.encoder(embedded)           # (batch, hidden_dim, 1)
        return encoded.squeeze(-1)                 # (batch, hidden_dim)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        merged = torch.cat([out1, out2], dim=1)    # (batch, 2*hidden_dim)
        return self.classifier(merged).squeeze(-1) # (batch,)
