from pathlib import Path
import numpy as np
from typing import Tuple
from fasta_parser import FastaParser

def analyze_missing_proteins(df, protein_seqs: dict) -> None:
    """
    Analyze missing protein sequences in the provided DataFrame.
    """
    missing_ids = set()
    id1_missing = []
    id2_missing = []

    for idx, row in df.iterrows():
        id1 = FastaParser("").extract_ensembl_id(row["protein1"])
        id2 = FastaParser("").extract_ensembl_id(row["protein2"])

        found1 = id1 in protein_seqs
        found2 = id2 in protein_seqs

        if not found1:
            missing_ids.add(id1)
            id1_missing.append(idx)
        if not found2:
            missing_ids.add(id2)
            id2_missing.append(idx)

    total_rows = len(set(id1_missing + id2_missing))
    print(f"🔍 Pares con al menos un ID faltante: {total_rows}")
    print(f"🔍 IDs únicos no encontrados: {len(missing_ids)}")

    print("\n🔍 Ejemplos de IDs no encontrados:")
    print(list(missing_ids)[:10])

def load_numpy_dataset(folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all input1, input2 and labels .npy chunks from a folder and concatenate them.
    
    Returns:
        X1: np.ndarray of shape (N, max_length)
        X2: np.ndarray of shape (N, max_length)
        y:  np.ndarray of shape (N,)
    """
    path = Path(folder)
    input1_files = sorted(path.glob("chunk_*_input1.npy"))
    input2_files = sorted(path.glob("chunk_*_input2.npy"))
    label_files  = sorted(path.glob("chunk_*_labels.npy"))

    X1 = np.concatenate([np.load(f) for f in input1_files], axis=0)
    X2 = np.concatenate([np.load(f) for f in input2_files], axis=0)
    y  = np.concatenate([np.load(f) for f in label_files],  axis=0)

    return X1, X2, y
