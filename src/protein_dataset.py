import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

class ProteinDataset(Dataset):
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.chunk_idx = 0
        self._load_chunk(0)

        metadata_path = self.folder / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            self.length = metadata["total_rows"]

    def _load_chunk(self, idx: int) -> None:
        self.input1 = np.load(self.folder / f"chunk_{idx}_input1.npy")
        self.input2 = np.load(self.folder / f"chunk_{idx}_input2.npy")
        self.labels = np.load(self.folder / f"chunk_{idx}_labels.npy")
        self.chunk_size = len(self.labels)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_id = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        if chunk_id != self.chunk_idx:
            self._load_chunk(chunk_id)
            self.chunk_idx = chunk_id
        x1 = torch.tensor(self.input1[local_idx], dtype=torch.long)
        x2 = torch.tensor(self.input2[local_idx], dtype=torch.long)
        y  = torch.tensor(self.labels[local_idx], dtype=torch.float32)
        return x1, x2, y
    
    def __repr__(self) -> str:
        return f"<ProteinDataset: {self.length} samples, chunk size {self.chunk_size}>"
