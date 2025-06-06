import argparse
from computation.azure_compute_platform import AzureComputePlatform
from sequence_preprocessor import SequencePreprocessor
from encoders.sequence_encoder import SequenceEncoder
import mltable
import pandas as pd

def main(args):
    platform = AzureComputePlatform()

    data_asset = platform.get_resource(name=args.input_name, version=args.input_version)
    path = {'folder': data_asset.path}

    table = mltable.from_delimited_files(paths=[path])
    df = table.to_pandas_dataframe()

    encoder = SequenceEncoder()
    preprocessor = SequencePreprocessor(encoder=encoder, max_length=1024)
    df_processed = preprocessor.process_dataframe(df)

    df_processed.to_parquet("./outputs/processed.parquet", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-name",    required=True)
    parser.add_argument("--input-version", required=True)
    parser.add_argument("--output-name",   required=True)
    args = parser.parse_args()
    main(args)
