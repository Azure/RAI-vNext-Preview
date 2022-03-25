# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging
import os
import subprocess

from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential

from azure.ml import MLClient
from azure.ml.entities import Component, Data, Environment


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


def process_file(input_file, output_file, replacements) -> None:
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            for f, r in replacements.items():
                line = line.replace(f, r)
            outfile.write(line)


def process_directory(directory: Path, ml_client: MLClient, version: int) -> None:
    _logger.info("Processing: {0}".format(directory))
    assert directory.is_absolute()

    registration_file = directory / REG_CONFIG_FILENAME
    reg_config = read_json_path(registration_file.resolve())

    replacements = {"VERSION_REPLACEMENT_STRING": str(version)}

    _logger.info("Changing directory")
    os.chdir(directory)

    if ENV_KEY in reg_config.keys():
        for e in reg_config[ENV_KEY]:
            _logger.info("Registering environment: {0}".format(e))
            processed_file = e + ".processed"
            process_file(e, processed_file, replacements)
            curr_env: Environment = Environment.load(processed_file)
            ml_client.environments.create_or_update(curr_env)
            _logger.info("Registered {0}".format(curr_env.name))
    else:
        _logger.info("No key for environments")

    if COMP_KEY in reg_config.keys():
        for c in reg_config[COMP_KEY]:
            _logger.info("Registering component: {0}".format(c))
            processed_file = c + ".processed"
            process_file(c, processed_file, replacements)
            curr_component = Component.load(path=processed_file)
            ml_client.components.create_or_update(curr_component)
            _logger.info("Registered {0}".format(curr_component.name))
    else:
        _logger.info("No key for components")

    if DATA_KEY in reg_config.keys():
        _logger.info("Working through data entries")
        for data_info in reg_config[DATA_KEY]:
            script_file = data_info["script"]
            _logger.info("Running script {0}".format(script_file))
            subprocess.run(["python", script_file], check=True)
            for d in data_info["data_yamls"]:
                _logger.info("Processing {0}".format(d))
                processed_file = d + ".processed"
                process_file(d, processed_file, replacements)
                curr_dataset: Data = Data.load(processed_file)
                ml_client.data.create_or_update(curr_dataset)
                _logger.info("Registered {0}".format(curr_dataset.name))
    else:
        _logger.info("No key for datasets")

    if SUBDIR_KEY in reg_config.keys():
        _logger.info("Working through nested directories")
        for d in reg_config[SUBDIR_KEY]:
            next_dir = directory / d
            process_directory(next_dir.resolve(), ml_client, version)
            os.chdir(directory)
    else:
        _logger.info("No subdirectories found for {0}".format(directory))


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

    process_directory(Path(args.base_directory).resolve(), ml_client, version)


if __name__ == "__main__":
    args = parse_args()
    main(args)
