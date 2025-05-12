from pandas import DataFrame
from src.encoders.sequence_encoder import SequenceEncoder

class SequencePreprocessor:
    """
    Preprocessor of sequences that applies encoding + padding
    """

    def __init__(self, encoder: SequenceEncoder, max_length: int, padding_value: int = 0):
        self.encoder = encoder
        self.max_length = max_length
        self.padding_value = padding_value

    def process_dataframe(self, df: DataFrame, col_seq1: str = "sequence1", col_seq2: str = "sequence2") -> DataFrame:
        """
        Processes a DataFrame with two columns of sequences and applies encoding + padding.
        """
        df = df.copy()
        df["input1"] = df[col_seq1].apply(self._encode_and_pad)
        df["input2"] = df[col_seq2].apply(self._encode_and_pad)
        return df

    def _encode_and_pad(self, sequence: str) -> list:
        """
        Encodes a sequence of aminoacids and pads it.
        """
        encoded = self.encoder.encode(sequence)
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        padded = encoded + [self.padding_value] * (self.max_length - len(encoded))
        return padded
