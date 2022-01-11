# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential

from azure.ml import MLClient
from azure.ml.entities import Environment


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


REG_CONFIG_FILENAME = "registration_config.json"
ENV_KEY = "environments"
COMP_KEY = "components"
DATA_KEY = "data"
SUBDIR_KEY = "nested_directories"


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--workspace_config", type=str, help="Path to workspace config.json"
    )
    parser.add_argument(
        "--component_config", type=str, help="Path to component_config.json"
    )
    parser.add_argument("--base_directory", type=str, help="Path to base directory")

    # parse args
    args = parser.parse_args()

    # return args
    return args


def read_json_path(path: str) -> Any:
    _logger.info("Reading JSON file {0}".format(path))
    with open(path, "r") as f:
        result = json.load(f)
    return result


def process_file(input_file, output_file, replacements):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            for f, r in replacements.items():
                line = line.replace(f, r)
            outfile.write(line)


def process_directory(directory: Path, ml_client: MLClient, version: int):
    _logger.info("Processing: {0}".format(directory))

    registration_file = directory / REG_CONFIG_FILENAME
    reg_config = read_json_path(registration_file)

    replacements = {"VERSION_REPLACEMENT_STRING": str(version)}

    _logger.info("Changing directory")
    os.chdir(directory)

    if ENV_KEY in reg_config.keys():
        for e in reg_config[ENV_KEY]:
            _logger.info("Registering environment: {0}".format(e))
            processed_file = e + ".processed"
            process_file(e, processed_file, replacements)
            curr_env = Environment.load(processed_file)
            ml_client.environments.create_or_update(curr_env)


def main(args):
    ws_config = read_json_path(args.workspace_config)
    component_config = read_json_path(args.component_config)

    ml_client = MLClient(
        credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
        subscription_id=ws_config["subscription_id"],
        resource_group_name=ws_config["resource_group"],
        workspace_name=ws_config["workspace_name"],
        logging_enable=True,
    )

    version: int = component_config["version"]

    process_directory(Path(args.base_directory), ml_client, version)


if __name__ == "__main__":
    args = parse_args()
    main(args)
