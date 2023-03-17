# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import logging
import pathlib

import pandas as pd
from azureml.core import Dataset, Run

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--tabular_dataset_name", type=str, help="Target TabularDataset"
    )
    parser.add_argument("--dataset_output_path", type=str, help="Path for Parquet file")

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    current_run = Run.get_context()
    ws = current_run.experiment.workspace

    _logger.info("Fetching TabularDataset")
    dataset = Dataset.get_by_name(ws, name=args.tabular_dataset_name)

    _logger.info("Loading into DataFrame")
    df: pd.DataFrame = dataset.to_pandas_dataframe()

    _logger.info("Writing to output")
    output_dir = pathlib.Path(args.dataset_output_path)
    output_file = output_dir / "from_tabular.parquet"
    df.to_parquet(str(output_file), index=False)


# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
