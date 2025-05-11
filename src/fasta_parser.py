import re

class FastaParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def to_dict(self) -> dict:
        """
        Return the following dictionary:
        {ENSP_ID (str): sequence (str)}
        """
        protein_seqs = dict()
        current_id = None
        current_seq = []

        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        protein_seqs[current_id] = "".join(current_seq)
                    current_id = self.extract_ensembl_id(line)
                    current_seq = []
                else:
                    current_seq.append(line)

            # Add the last sequence
            if current_id and current_seq:
                protein_seqs[current_id] = "".join(current_seq)

        return protein_seqs

    def extract_ensembl_id(self, text: str) -> str:
        """
        Extract the Ensembl identifier (ENSPxxxx) from the given text.
        """
        match = re.search(r"ENSP\d+", text)
        return match.group(0) if match else None
    