import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List
import numpy as np
import os
import gc
import time

class ESMEmbedder:
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _clean(self, seq: str) -> str:
        """
        Clean sequence: remove whitespace and enforce uppercase.
        """
        return seq.replace(" ", "").replace("\n", "").upper()

    def _embed_batch(self, sequences: List[str]) -> Tensor:
        try:
            tokens = self.tokenizer(
                sequences, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**tokens)
            return output.last_hidden_state[:, 0, :].cpu()  # CLS token
        except Exception as e:
            print(f"!! Error embedding batch: {e}")
            raise

    def embed_all(self, seqs1: List[str], seqs2: List[str]) -> tuple[Tensor, Tensor]:
        assert len(seqs1) == len(seqs2), "Sequence lists must be the same length"

        all_embeddings1, all_embeddings2 = [], []

        for i in tqdm(range(0, len(seqs1), self.batch_size), desc="Encoding with ESM-2"):
            batch1 = [self._clean(s) for s in seqs1[i : i + self.batch_size]]
            batch2 = [self._clean(s) for s in seqs2[i : i + self.batch_size]]

            try:
                emb1 = self._embed_batch(batch1)
                emb2 = self._embed_batch(batch2)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"!! OOM en batch {i // self.batch_size}, bajando a batch_size=16 y reintentando...")
                    torch.cuda.empty_cache()
                    time.sleep(3)
                    self.batch_size = 16
                    continue
                else:
                    raise

            all_embeddings1.append(emb1)
            all_embeddings2.append(emb2)

            del emb1, emb2, batch1, batch2
            gc.collect()
            torch.cuda.empty_cache()

        X1 = torch.cat(all_embeddings1)
        X2 = torch.cat(all_embeddings2)
        return X1, X2

    def save(self, X1: Tensor, X2: Tensor, Y: np.ndarray, output_dir: str) -> None:
        """
        Save the encoded tensors to disk.
        """
        os.makedirs(output_dir, exist_ok=True)

        files = {
            "esm_protein1.pt": X1,
            "esm_protein2.pt": X2,
            "esm_labels.pt": torch.tensor(Y),
        }

        for name, tensor in files.items():
            path = os.path.join(output_dir, name)
            torch.save(tensor, path)
            print(f"---Saved: {path}")
