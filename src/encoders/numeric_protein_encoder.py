from encoders.protein_encoder import ProteinEncoder

class NumericProteinEncoder(ProteinEncoder):
    """
    Encoder for protein sequences using numeric representation.
    """

    def __init__(self):
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_index = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.unknown_index = len(self.aa_to_index)

    def encode(self, sequence: str) -> list:
        return [self.aa_to_index.get(aa, self.unknown_index) for aa in sequence]
