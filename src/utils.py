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
