import gc
from typing import Optional
from pandas import DataFrame, concat
from encoders.sequence_encoder import SequenceEncoder

class SequencePreprocessor:
    """
    Preprocessor of sequences that applies encoding + padding
    """

    def __init__(self, encoder: SequenceEncoder, max_length: int, padding_value: int = 0, chunk_size: Optional[int] = None):
        self.encoder       = encoder
        self.max_length    = max_length
        self.padding_value = padding_value
        self.chunk_size    = chunk_size

    def process_dataframe(self, df: DataFrame, col_seq1: str = "sequence1", col_seq2: str = "sequence2") -> DataFrame:
        """
        Process a DataFrame with two columns of sequences and apply encoding + padding.
        """
        if self.chunk_size and len(df) > self.chunk_size:
            print(f"Huge DataFrame: processing by chunks of {self.chunk_size} rows")
            chunks = []
            for i in range(0, len(df), self.chunk_size):
                df_chunk = df.iloc[i : i + self.chunk_size]
                processed = self._process_chunk(df_chunk, col_seq1, col_seq2)
                chunks.append(processed)
                print(f" - Processed chunk {i // self.chunk_size + 1}")
                del df_chunk
                del processed
                gc.collect()
            return concat(chunks, ignore_index=True)
        else:
            return self._process_chunk(df, col_seq1, col_seq2)
    
    def _process_chunk(self, df: DataFrame, col_seq1: str, col_seq2: str) -> DataFrame:
        df_meta = df.copy()
        df_meta["input1"] = df_meta[col_seq1].apply(self._encode_and_pad)
        df_meta["input2"] = df_meta[col_seq2].apply(self._encode_and_pad)
        return df_meta

    def _encode_and_pad(self, sequence: str) -> list:
        """
        Encode a sequence of aminoacids and pad it.
        """
        encoded = self.encoder.encode(sequence)
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        padded = encoded + [self.padding_value] * (self.max_length - len(encoded))
        return padded
