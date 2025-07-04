import gc
import math
import json
from pathlib import Path
import numpy as np
from pandas import DataFrame
from encoders.protein_encoder import ProteinEncoder

class ProteinsPreprocessor:
    """
    Preprocessor of protein sequences: encodes, pads and saves them in .npy format by chunks.
    """
    def __init__(
        self,
        encoder: ProteinEncoder,
        max_length: int,
        output_dir: str,
        padding_value: int = 0,
        chunk_size = 100000
    ):
        self.encoder       = encoder
        self.max_length    = max_length
        self.padding_value = padding_value
        self.chunk_size    = chunk_size
        self.output_dir    = Path(output_dir)

        if self.output_dir.exists():
            # Delete the folder if it exists
            for file in self.output_dir.glob("*"):
                file.unlink()
            self.output_dir.rmdir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_dataframe(self, df: DataFrame, col_seq1: str = "sequence1", col_seq2: str = "sequence2", col_label: str = "label") -> None:
        """
        Processes the DataFrame in chunks and saves encoded sequences and labels to .npy files.
        """
        total_chunks = math.ceil(len(df) / self.chunk_size)
        for i in range(total_chunks):
            start    = i * self.chunk_size
            end      = min(start + self.chunk_size, len(df))
            df_chunk = df.iloc[start:end]

            print(f"Processing chunk {i + 1}/{total_chunks}...")

            input1 = [self._encode_and_pad(seq) for seq in df_chunk[col_seq1].values]
            input2 = [self._encode_and_pad(seq) for seq in df_chunk[col_seq2].values]
            labels = df_chunk[col_label].to_numpy(dtype=np.int8)

            np.save(self.output_dir / f"chunk_{i}_input1.npy", np.array(input1, dtype=np.int16))
            np.save(self.output_dir / f"chunk_{i}_input2.npy", np.array(input2, dtype=np.int16))
            np.save(self.output_dir / f"chunk_{i}_labels.npy", labels)

            del df_chunk, input1, input2, labels
            gc.collect()
        self._save_metadata(len(df))

    def _encode_and_pad(self, sequence: str) -> list[int]:
        """
        Encode a sequence of amino acids and pad it to max_length.
        """
        encoded = self.encoder.encode(sequence)
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        return encoded + [self.padding_value] * (self.max_length - len(encoded))
    
    def _save_metadata(self, total_rows: int) -> None:
        """
        Save metadata (e.g., total rows) to a JSON file in the output directory.
        """
        metadata = {
            "total_rows": total_rows
        }
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
