# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging
from typing import Any


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


REG_CONFIG_FILENAME = 'registration_config.json'

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--workspace_config", type=str, help="Path to workspace config.json")
    parser.add_argument(
        "--component_config", type=str, help="Path to component_config.json"
    )
    parser.add_argument(
        "--base_directory", type=str, help="Path to base directory"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


def read_json_path(path: str) -> Any:
    _logger.info("Reading JSON file {0}".format(path))
    with open(path, 'r') as f:
        result = json.load(f)
    return result


def main(args):
    ws_config = read_json_path(args.workspace_config)
    component_config = read_json_path(args.component_config)


if __name__ == "__main__":
    args = parse_args()
    main(args)