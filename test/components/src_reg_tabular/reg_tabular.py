# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import logging
import time

import pandas as pd

from azureml.core import Dataset, Run

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--dataset_input_path", type=str, help="Path to input model")
    parser.add_argument(
        "--dataset_base_name", type=str, help="Name of the registered tabular dataset"
    )
    parser.add_argument(
        "--dataset_name_suffix", type=int, help="Set negative to use epoch_secs"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


def load_dataset(parquet_path: str):
    _logger.info("Loading parquet file: {0}".format(parquet_path))
    df = pd.read_parquet(parquet_path)
    print(df.dtypes)
    print(df.head(10))
    return df

def main(args):
    current_run = Run.get_context()
    ws = current_run.experiment.workspace

    datastore = ws.get_default_datastore()

    print("Loading dataset")
    df = load_dataset(args.dataset_input_path)

    if args.dataset_name_suffix < 0:
        suffix = int(time.time())
    else:
        suffix = args.model_name_suffix
    registered_name = "{0}_{1}".format(args.dataset_base_name, suffix)
    print(f"Registering dataset as {registered_name}")

    datastore_path = (datastore, registered_name)
    tabular_dataset = Dataset.Tabular.register_pandas_dataframe(df, datastore_path, registered_name)
    _logger.info("Registered Dataset: {}".format(tabular_dataset))




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
