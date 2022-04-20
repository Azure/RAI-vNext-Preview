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
    # append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)

    # for k, v in test_values.items():
    #    assert nb.scraps[k].data == v.expected


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
        "training_component_version_string = '4'"
    ] = f"training_component_version_string = '{train_version_string}'"

    assay_one_notebook(nb_name, dict(), replacements)
