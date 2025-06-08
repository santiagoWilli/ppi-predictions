import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class ProteinDataset(Dataset):
    def __init__(self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray):
        assert X1.shape == X2.shape, "Input shapes do not match"
        assert X1.shape[0] == y.shape[0], "Number of samples does not match labels"

        self.X1 = X1
        self.X2 = X2
        self.y = y

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = torch.tensor(self.X1[idx], dtype=torch.long)
        x2 = torch.tensor(self.X2[idx], dtype=torch.long)
        y  = torch.tensor(self.y[idx],  dtype=torch.float32)
        return x1, x2, y

    def __repr__(self) -> str:
        return f"<ProteinDatasetInMemory: {len(self)} samples>"
