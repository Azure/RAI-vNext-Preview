# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import random
import string
import time
from typing import Dict

import nbformat as nbf
import papermill as pm
import pytest


def update_cells(input_nb_path, output_nb_path, replacement_strings: Dict[str, str]):
    notebook = nbf.read(input_nb_path, as_version=nbf.NO_CONVERT)

    for cell in notebook["cells"]:
        for original, update in replacement_strings.items():
            if cell["source"] == original:
                print(f"Replacing ---{original}--- with ---{update}---")
                cell["source"] = update

    nbf.write(notebook, output_nb_path)


def assay_one_notebook(notebook_name, test_values, replacement_strings: Dict[str, str]):
    """Test a single notebook.
    This uses nbformat to replace the contents of given cells for use in automated pipelines
    Makes certain assumptions about directory layout.
    """
    input_notebook = "examples/notebooks/" + notebook_name + ".ipynb"
    processed_notebook = "./test/notebooks/" + notebook_name + ".processed.ipynb"
    output_notebook = "./test/notebooks/" + notebook_name + ".output.ipynb"

    update_cells(input_notebook, processed_notebook, replacement_strings)
    pm.execute_notebook(processed_notebook, output_notebook)


def get_version_string():
    time_version = int(time.time())
    random_suffix = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(10)
    )
    return f"{time_version}__{random_suffix}"


@pytest.mark.notebooks
def test_responsibleaidashboard_housing_classification_model_debugging(
    component_config,
):
    nb_name = "responsibleaidashboard-housing-classification-model-debugging"

    version_string = component_config["version"]
    train_version_string = get_version_string()

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(
        os.path.join(current_file_directory, "../..", "examples/notebooks/data")
    )
    input_data_filename = "apartments-train.csv"

    replacements = dict()
    replacements["version_string = '1'"] = f"version_string = '{version_string}'"
    replacements[
        "rai_housing_example_version_string = '1'"
    ] = f"rai_housing_example_version_string = '{train_version_string}'"
    replacements[
        "data_path = 'data/apartments-train.csv'"
    ] = f'data_path = r"{os.path.join(data_dir, input_data_filename)}"'

    assay_one_notebook(nb_name, dict(), replacements)


@pytest.mark.notebooks
def test_responsibleaidashboard_housing_improvement(
    component_config,
):
    nb_name = "responsibleaidashboard-housing-decision-making"

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(
        os.path.join(current_file_directory, "../..", "examples/notebooks/data")
    )
    input_data_filename = "apartments-train.csv"

    version_string = component_config["version"]
    train_version_string = get_version_string()

    replacements = dict()
    replacements["version_string = '1'"] = f"version_string = '{version_string}'"
    replacements[
        "rai_house_improvement_version_string = '1'"
    ] = f"rai_house_improvement_version_string = '{train_version_string}'"
    replacements[
        "data_path = 'data/apartments-train.csv'"
    ] = f'data_path = r"{os.path.join(data_dir, input_data_filename)}"'

    assay_one_notebook(nb_name, dict(), replacements)


@pytest.mark.notebooks
def test_responsibleaidashboard_programmer_regression_model_debugging(
    component_config,
):
    nb_name = "responsibleaidashboard-programmer-regression-model-debugging"

    version_string = component_config["version"]
    train_version_string = get_version_string()

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(
        os.path.join(current_file_directory, "../..", "examples/notebooks")
    )
    train_path = 'data-programmer-regression/train/'
    test_path = 'data-programmer-regression/test/'

    replacements = dict()
    replacements["version_string = '1'"] = f"version_string = '{version_string}'"
    replacements[
        "rai_programmer_example_version_string = '1'"
    ] = f"rai_programmer_example_version_string = '{train_version_string}'"
    replacements[
        f"train_data_path = '{train_path}'"
    ] = f'train_data_path = r"{os.path.join(data_dir, train_path)}"'
    replacements[
        f"test_data_path = '{test_path}'"
    ] = f'test_data_path = r"{os.path.join(data_dir, test_path)}"'

    assay_one_notebook(nb_name, dict(), replacements)


@pytest.mark.notebooks
def test_responsibleaidashboard_diabetes_regression_model_debugging(
    component_config,
):
    nb_name = "responsibleaidashboard-diabetes-regression-model-debugging"

    version_string = component_config["version"]
    train_version_string = get_version_string()

    replacements = dict()
    replacements["version_string = '1'"] = f"version_string = '{version_string}'"
    replacements[
        "rai_diabetes_regression_example_version_string = '1'"
    ] = f"rai_diabetes_regression_example_version_string = '{train_version_string}'"

    assay_one_notebook(nb_name, dict(), replacements)


@pytest.mark.notebooks
def test_responsibleaidashboard_diabetes_decision_making(
    component_config,
):
    nb_name = "responsibleaidashboard-diabetes-decision-making"

    version_string = component_config["version"]
    train_version_string = get_version_string()

    replacements = dict()
    replacements["version_string = '1'"] = f"version_string = '{version_string}'"
    replacements[
        "rai_diabetes_decision_making_example_version_string = '1'"
    ] = f"rai_diabetes_decision_making_example_version_string = '{train_version_string}'"

    assay_one_notebook(nb_name, dict(), replacements)
