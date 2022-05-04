# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import nbformat as nbf
import papermill as pm
import pytest
import time

from typing import Dict


def update_cells(input_nb_path, output_nb_path, replacement_strings: Dict[str, str]):
    notebook = nbf.read(input_nb_path, as_version=nbf.NO_CONVERT)

    for cell in notebook["cells"]:
        for original, update in replacement_strings.items():
            if cell["source"] == original:
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


@pytest.mark.notebooks
def test_responsibleaidashboard_housing_classification_model_debugging(
    component_config,
):
    nb_name = "responsibleaidashboard-housing-classification-model-debugging"

    version_string = component_config["version"]
    train_version_string = int(time.time())

    replacements = dict()
    replacements["version_string = '1'"] = f"version_string = '{version_string}'"
    replacements[
        "rai_housing_example_version_string = '4'"
    ] = f"rai_housing_example_version_string = '{train_version_string}'"

    assay_one_notebook(nb_name, dict(), replacements)


@pytest.mark.notebooks
def test_responsibleaidashboard_programmer_regression_model_debugging(
    component_config,
):
    nb_name = "responsibleaidashboard-programmer-regression-model-debugging"

    version_string = component_config["version"]
    train_version_string = int(time.time())

    replacements = dict()
    replacements["version_string = '1'"] = f"version_string = '{version_string}'"
    replacements[
        "rai_programmer_example_version_string = '5'"
    ] = f"rai_programmer_example_version_string = '{train_version_string}'"
    replacements[
        "train_data_path = 'data/programmers-train.parquet'"
    ] = 'train_data_path = "examples/notebooks/data/programmers-train.parquet"'
    replacements[
        "test_data_path = 'data/programmers-test.parquet'"
    ] = 'test_data_path = "examples/notebooks/data/programmers-test.parquet"'

    assay_one_notebook(nb_name, dict(), replacements)
