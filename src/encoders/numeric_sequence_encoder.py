from encoders.sequence_encoder import SequenceEncoder

class NumericSequenceEncoder(SequenceEncoder):
    """
    Encoder for protein sequences using numeric representation.
    """

    def __init__(self):
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_index = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.unknown_index = len(self.aa_to_index)

    def encode(self, sequence: str) -> list:
        return [self.aa_to_index.get(aa, self.unknown_index) for aa in sequence]

    def decode(self, indices: list) -> str:
        index_to_aa = {v: k for k, v in self.aa_to_index.items()}
        return "".join(index_to_aa.get(i, "X") for i in indices)

    def vocabulary_size(self) -> int:
        return len(self.aa_to_index) + 1  # include unknown index
