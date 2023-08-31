# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import pathlib
import platform
import subprocess
import sys

import yaml
from _telemetry._loggerfactory import _LoggerFactory, track
from arg_helpers import boolean_parser
from constants import DashboardInfo
from rai_component_utilities import (fetch_model_id,
                                     get_mlflow_model_conda_dependency_path,
                                     load_dashboard_info_file)

_logger = logging.getLogger(__file__)
_ai_logger = None


def _get_logger():
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = _LoggerFactory.get_logger(__file__)
    return _ai_logger


_get_logger()


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--rai_insights_dashboard", type=str)

    parser.add_argument("--model_info_path", type=str, help="name:version")

    parser.add_argument("--model_input", type=str, help="model local path on remote")

    parser.add_argument("--model_info", type=str, help="name:version")

    parser.add_argument(
        "--use_model_dependency", type=boolean_parser, help="Use model dependency"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


def update_conda_env(conda_env_path, out="./conda_dep.yaml"):
    with open(conda_env_path, "r") as f:
        content = yaml.safe_load(f)

    for i, d in enumerate(content["dependencies"]):
        if isinstance(d, str) and d.startswith("python"):
            content["dependencies"][i] = f"python={platform.python_version()}"

    with open(out, "w") as f:
        print(f"conda env yaml: {content}")
        yaml.dump(content, f)

    return out


def get_model_id(dashboard, model_info_path, model_input, model_info):
    if dashboard:
        config = load_dashboard_info_file(args.rai_insights_dashboard)
        return config[DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY]

    if args.model_info_path:
        return fetch_model_id(args.model_info_path)

    if args.model_input and args.model_info:
        return args.model_info


@track(_get_logger)
def main(args):
    use_model_dependency = args.use_model_dependency
    if args.rai_insights_dashboard:
        config = load_dashboard_info_file(args.rai_insights_dashboard)
        input_args = config[DashboardInfo.RAI_INSIGHTS_INPUT_ARGS_KEY]
        use_model_dependency = input_args["use_model_dependency"]

    if not use_model_dependency:
        _logger.info("Skipping model dependency installation.")
        return

    model_id = get_model_id(
        dashboard=args.rai_insights_dashboard,
        model_info_path=args.model_info_path,
        model_input=args.model_input,
        model_info=args.model_info
    )

    conda_file = get_mlflow_model_conda_dependency_path(model_id)

    local_conda_dep = "./conda_dep.yaml"
    update_conda_env(conda_file, local_conda_dep)
    conda_prefix = str(pathlib.Path(sys.executable).parents[1])

    try:
        subprocess.check_output(
            [
                "conda",
                "env",
                "update",
                "--prefix",
                conda_prefix,
                "-f",
                local_conda_dep,
            ]
        )
    except subprocess.CalledProcessError as e:
        _logger.error(
            "Installing dependency using conda.yaml from mlflow model failed: {}".format(
                e.output
            )
        )
        with open(local_conda_dep, "r") as f:
            print(f.read())

    _logger.info("Model dependency installation successful")


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
