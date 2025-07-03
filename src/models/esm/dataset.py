import torch
from torch.utils.data import Dataset
from pathlib import Path

class ESMDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        Load ESM-2 embeddings.
        """
        path = Path(data_dir)
        self.X1 = torch.load(path / "esm_protein1.pt")
        self.X2 = torch.load(path / "esm_protein2.pt")
        self.y  = torch.load(path / "esm_labels.pt").float()

        assert len(self.X1) == len(self.X2) == len(self.y), "Dataset inconsistente"

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple:
        return self.X1[idx], self.X2[idx], self.y[idx]
