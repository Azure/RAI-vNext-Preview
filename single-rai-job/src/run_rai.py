# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging

from pathlib import Path
from typing import Any, Dict, List, Union

import mlflow
import pandas as pd

from azureml.core import Dataset, Model, Run, Workspace

from responsibleai import RAIInsights, __version__ as responsibleai_version
from responsibleai.serialization_utilities import serialize_json_safe


class DashboardInfo:
    MODEL_ID_KEY = "id"  # To match Model schema
    MODEL_INFO_FILENAME = "model_info.json"
    TRAIN_FILES_DIR = "train"
    TEST_FILES_DIR = "test"

    RAI_INSIGHTS_MODEL_ID_KEY = "model_id"
    RAI_INSIGHTS_RUN_ID_KEY = "rai_insights_parent_run_id"
    RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY = "constructor_args"
    RAI_INSIGHTS_PARENT_FILENAME = "rai_insights.json"


class RAIToolType:
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    ERROR_ANALYSIS = "error_analysis"
    EXPLANATION = "explanation"


def get_from_args(args, arg_name: str, custom_parser, allow_none: bool) -> Any:
    _logger.info("Looking for command line argument '{0}'".format(arg_name))
    result = None

    extracted = getattr(args, arg_name)
    if extracted is None and not allow_none:
        raise ValueError("Required argument {0} missing".format(arg_name))

    if custom_parser:
        if extracted is not None:
            result = custom_parser(extracted)
    else:
        result = extracted

    _logger.info("{0}: {1}".format(arg_name, result))

    return result


def boolean_parser(target: str) -> bool:
    true_values = ["True", "true"]
    false_values = ["False", "false"]
    if target in true_values:
        return True
    if target in false_values:
        return False
    raise ValueError("Failed to parse to boolean: {target}")


def float_or_json_parser(target: str) -> Union[float, Any]:
    try:
        return json.loads(target)
    except json.JSONDecodeError:
        return float(target.strip('"').strip("'"))


def json_empty_is_none_parser(target: str) -> Union[Dict, List]:
    parsed = json.loads(target)
    if len(parsed) == 0:
        return None
    else:
        return parsed


def int_or_none_parser(target: str) -> Union[None, int]:
    try:
        return int(target.strip('"').strip("'"))
    except ValueError:
        if "None" in target:
            return None
        raise ValueError("int_or_none_parser failed on: {0}".format(target))


def str_or_int_parser(target: str) -> Union[str, int]:
    try:
        return int(target.strip('"').strip("'"))
    except ValueError:
        return target


def str_or_list_parser(target: str) -> Union[str, list]:
    try:
        decoded = json.loads(target)
        if not isinstance(decoded, list):
            raise ValueError("Supplied JSON string not list: {0}".format(target))
        return decoded
    except json.JSONDecodeError:
        # String, but need to get rid of quotes
        return target.strip('"').strip("'")


class PropertyKeyValues:
    # The property to indicate the type of Run
    RAI_INSIGHTS_TYPE_KEY = "_azureml.responsibleai.rai_insights.type"
    RAI_INSIGHTS_TYPE_CONSTRUCT = "construction"
    RAI_INSIGHTS_TYPE_CAUSAL = "causal"
    RAI_INSIGHTS_TYPE_COUNTERFACTUAL = "counterfactual"
    RAI_INSIGHTS_TYPE_EXPLANATION = "explanation"
    RAI_INSIGHTS_TYPE_ERROR_ANALYSIS = "error_analysis"
    RAI_INSIGHTS_TYPE_GATHER = "gather"

    # Property to point at the model under examination
    RAI_INSIGHTS_MODEL_ID_KEY = "_azureml.responsibleai.rai_insights.model_id"

    # Property for tool runs to point at their constructor run
    RAI_INSIGHTS_CONSTRUCTOR_RUN_ID_KEY = (
        "_azureml.responsibleai.rai_insights.constructor_run"
    )

    # Property to record responsibleai version
    RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY = (
        "_azureml.responsibleai.rai_insights.responsibleai_version"
    )

    # Property format to indicate presence of a tool
    RAI_INSIGHTS_TOOL_KEY_FORMAT = "_azureml.responsibleai.rai_insights.has_{0}"


def add_properties_to_gather_run(
    dashboard_info: Dict[str, str], tool_present_dict: Dict[str, str]
):
    _logger.info("Adding properties to the gather run")
    gather_run = Run.get_context()

    run_properties = {
        PropertyKeyValues.RAI_INSIGHTS_TYPE_KEY: PropertyKeyValues.RAI_INSIGHTS_TYPE_GATHER,
        PropertyKeyValues.RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY: responsibleai_version,
        PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY
        ],
    }

    _logger.info("Appending tool present information")
    for k, v in tool_present_dict.items():
        key = PropertyKeyValues.RAI_INSIGHTS_TOOL_KEY_FORMAT.format(k)
        run_properties[key] = str(v)

    _logger.info("Making service call")
    gather_run.add_properties(run_properties)
    _logger.info("Properties added to gather run")


def load_tabular_dataset(tabular_ds_id: str, ws: Workspace):
    _logger.info("Loading Tabular dataset: {0}".format(tabular_ds_id))
    dataset = Dataset.get_by_name(ws, name=args.tabular_dataset_name)
    _logger.info("Loading into DataFrame")
    df: pd.DataFrame = dataset.to_pandas_dataframe()
    return df


def load_mlflow_model(workspace: Workspace, model_id: str) -> Any:
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())

    model = Model._get(workspace, id=model_id)
    model_uri = "models:/{}/{}".format(model.name, model.version)
    return mlflow.pyfunc.load_model(model_uri)._model_impl


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # Constructor arguments
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument(
        "--task_type", type=str, required=True, choices=["classification", "regression"]
    )
    parser.add_argument("--model_id", type=str, help="name:version", required=True)
    parser.add_argument("--train_dataset_id", type=str, required=True)
    parser.add_argument("--test_dataset_id", type=str, required=True)
    parser.add_argument("--target_column_name", type=str, required=True)
    parser.add_argument("--maximum_rows_for_test_dataset", type=int, default=5000)
    parser.add_argument(
        "--categorical_column_names", type=str, help="Optional[List[str]]"
    )
    parser.add_argument("--classes", type=str, help="Optional[List[str]]")

    # Causal arguments
    parser.add_argument("--enable_causal", type=bool, required=True)
    parser.add_argument(
        "--causal_treatment_features", type=json.loads, help="List[str]"
    )
    parser.add_argument(
        "--causal_heterogeneity_features",
        type=json.loads,
        help="Optional[List[str]] use 'null' to skip",
    )
    parser.add_argument(
        "--causal_nuisance_model", type=str, choices=["linear", "automl"]
    )
    parser.add_argument(
        "--causal_heterogeneity_model", type=str, choices=["linear", "forest"]
    )
    parser.add_argument("--causal_alpha", type=float)
    parser.add_argument("--causal_upper_bound_on_cat_expansion", type=int)
    parser.add_argument(
        "--causal_treatment_cost",
        type=float_or_json_parser,
        help="Union[float, List[Union[float, np.ndarray]]]",
    )
    parser.add_argument("--causal_min_tree_leaf_samples", type=int)
    parser.add_argument("--causal_max_tree_depth", type=int)
    parser.add_argument("--causal_skip_cat_limit_checks", type=boolean_parser)
    parser.add_argument("--causal_categories", type=str_or_list_parser)
    parser.add_argument("--causal_n_jobs", type=int)
    parser.add_argument("--causal_verbose", type=int)
    parser.add_argument("--causal_random_state", type=int_or_none_parser)

    # Counterfactual arguments
    parser.add_argument("--enable_counterfactual", type=bool, required=True)
    parser.add_argument("--counterfactual_total_CFs", type=int, required=True)
    parser.add_argument("--counterfactual_method", type=str)
    parser.add_argument("--counterfactual_desired_class", type=str_or_int_parser)
    parser.add_argument(
        "--counterfactual_desired_range", type=json_empty_is_none_parser, help="List"
    )
    parser.add_argument(
        "--counterfactual_permitted_range", type=json_empty_is_none_parser, help="Dict"
    )
    parser.add_argument("--counterfactual_features_to_vary", type=str_or_list_parser)
    parser.add_argument("--counterfactual_feature_importance", type=boolean_parser)

    # Error analysis arguments
    parser.add_argument("--enable_error_analysis", type=bool, required=True)
    parser.add_argument("--error_analysis_max_depth", type=int)
    parser.add_argument("--error_analysis_num_leaves", type=int)
    parser.add_argument("--error_analysis_min_child_samples", type=int)
    parser.add_argument(
        "--error_analysis_filter_features", type=json.loads, help="List"
    )

    # Explanation arguments
    parser.add_argument("--enable_explanation", type=bool, required=True)

    # Output arguments
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
    my_ws = my_run.experiment.workspace

    _logger.info("Dealing with initialization dataset")
    train_df = load_tabular_dataset(args.train_dataset_id, my_ws)

    _logger.info("Dealing with evaluation dataset")
    test_df = load_tabular_dataset(args.test_dataset_id, my_ws)

    _logger.info("Loading model: {0}".format(args.model_id))
    model_estimator = load_mlflow_model(my_ws), args.model_id)

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

    _logger.info("Checking individual tools")

    if args.enable_causal:
        _logger.info("Adding causal")
        rai_i.causal.add(
            treatment_features=args.causal_treatment_features,
            heterogeneity_features=args.causal_heterogeneity_features,
            nuisance_model=args.causal_nuisance_model,
            heterogeneity_model=args.causal_heterogeneity_model,
            alpha=args.causal_alpha,
            upper_bound_on_cat_expansion=args.causal_upper_bound_on_cat_expansion,
            treatment_cost=args.causal_treatment_cost,
            min_tree_leaf_samples=args.causal_min_tree_leaf_samples,
            max_tree_depth=args.causal_max_tree_depth,
            skip_cat_limit_checks=args.causal_skip_cat_limit_checks,
            categories=args.causal_categories,
            n_jobs=args.causal_n_jobs,
            verbose=args.causal_verbose,
            random_state=args.causal_random_state,
        )
        included_tools[RAIToolType.CAUSAL] = True

    if args.enable_counterfactual:
        _logger.info("Adding counterfactuals")
        rai_i.counterfactual.add(
            total_CFs=args.counterfactual_total_CFs,
            method=args.counterfactual_method,
            desired_class=args.counterfactual_desired_class,
            desired_range=args.counterfactual_desired_range,
            permitted_range=args.counterfactual_permitted_range,
            features_to_vary=args.counterfactual_features_to_vary,
            feature_importance=args.counterfactual_feature_importance,
        )
        included_tools[RAIToolType.COUNTERFACTUAL] = True

    if args.enable_error_analysis:
        _logger.info("Adding error analysis")
        rai_i.error_analysis.add(
            max_depth=args.error_analysis_max_depth,
            num_leaves=args.error_analysis_num_leaves,
            min_child_samples=args.error_analysis_min_child_samples,
            filter_features=args.error_analysis_filter_features,
        )
        included_tools[RAIToolType.ERROR_ANALYSIS] = True

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
