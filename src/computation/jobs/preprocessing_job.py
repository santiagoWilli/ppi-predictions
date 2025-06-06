import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
from sequence_preprocessor import SequencePreprocessor
from encoders.numeric_sequence_encoder import NumericSequenceEncoder
import mltable
import pandas as pd

def main(args):
    print("-- Preprocessing job started --")
    path = {'folder': args.input_path}

    print("-- Reading input --")
    table = mltable.from_parquet_files(paths=[path])
    df = table.to_pandas_dataframe()

    print("-- Preprocess started --")
    encoder = NumericSequenceEncoder()
    preprocessor = SequencePreprocessor(encoder=encoder, max_length=1024, chunk_size=256)
    df_processed = preprocessor.process_dataframe(df)

    print("-- Saving processed data --")
    df_processed.to_parquet(Path(args.output_dir) / "processed.parquet", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args)
