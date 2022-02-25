# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging

from pathlib import Path
from typing import Dict

from azureml.core import Run

from responsibleai import RAIInsights, __version__ as responsibleai_version
from responsibleai.serialization_utilities import serialize_json_safe


from constants import DashboardInfo, RAIToolType
from arg_helpers import get_from_args, json_empty_is_none_parser
from rai_component_utilities import (
    add_properties_to_gather_run,
    load_dataset,
    load_mlflow_model,
)


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--title", type=str, required=True)

    parser.add_argument(
        "--task_type", type=str, required=True, choices=["classification", "regression"]
    )

    parser.add_argument("--model_id", type=str, help="name:version", required=True)

    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)

    parser.add_argument("--target_column_name", type=str, required=True)

    parser.add_argument("--maximum_rows_for_test_dataset", type=int, default=5000)
    parser.add_argument(
        "--categorical_column_names", type=str, help="Optional[List[str]]"
    )

    parser.add_argument("--classes", type=str, help="Optional[List[str]]")

    parser.add_argument("--enable_explanation", type=bool, required=True)

    parser.add_argument("--dashboard", type=str, required=True)
    parser.add_argument("--ux_json", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def create_constructor_arg_dict(args):
    """Create a kwarg dict for RAIInsights constructor

    Does not handle the model or dataframes
    """
    result = dict()

    cat_col_names = get_from_args(
        args, "categorical_column_names", custom_parser=json.loads, allow_none=True
    )
    class_names = get_from_args(
        args, "classes", custom_parser=json_empty_is_none_parser, allow_none=True
    )

    result["target_column"] = args.target_column_name
    result["task_type"] = args.task_type
    result["categorical_features"] = cat_col_names
    result["classes"] = class_names
    result["maximum_rows_for_test"] = args.maximum_rows_for_test_dataset

    return result


def main(args):
    _logger.info(f"responsibleai=={responsibleai_version}")
    my_run = Run.get_context()

    _logger.info("Dealing with initialization dataset")
    train_df = load_dataset(args.train_dataset)

    _logger.info("Dealing with evaluation dataset")
    test_df = load_dataset(args.test_dataset)

    _logger.info("Loading model: {0}".format(args.model_id))
    model_estimator = load_mlflow_model(my_run.experiment.workspace, args.model_id)

    constructor_args = create_constructor_arg_dict(args)

    # Create the RAIInsights object
    _logger.info("Creating RAIInsights object")
    rai_i = RAIInsights(
        model=model_estimator, train=train_df, test=test_df, **constructor_args
    )

    included_tools: Dict[str, bool] = {
        RAIToolType.CAUSAL: False,
        RAIToolType.COUNTERFACTUAL: False,
        RAIToolType.ERROR_ANALYSIS: False,
        RAIToolType.EXPLANATION: False,
    }

    if args.enable_explanation:
        _logger.info("Adding explanation")
        rai_i.explainer.add()
        included_tools[RAIToolType.EXPLANATION] = True

    _logger.info("Triggering computation")
    rai_i.compute()

    _logger.info("Saving binary output")
    rai_i.save(args.dashboard)

    _logger.info("Saving UX JSON")
    rai_data = rai_i.get_data()
    rai_dict = serialize_json_safe(rai_data)
    json_filename = "dashboard.json"
    output_path = Path(args.ux_json) / json_filename
    with open(output_path, "w") as json_file:
        json.dump(rai_dict, json_file)

    _logger.info("Adding properties to run")
    dashboard_info = {
        DashboardInfo.RAI_INSIGHTS_RUN_ID_KEY: str(my_run.id),
        DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY: args.model_id,
        DashboardInfo.RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY: constructor_args,
    }

    add_properties_to_gather_run(dashboard_info, included_tools)
    _logger.info("Processing completed")


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
