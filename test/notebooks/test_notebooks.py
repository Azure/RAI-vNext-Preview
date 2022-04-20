# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import nbformat as nbf
import papermill as pm
import pytest
import scrapbook as sb


class ScrapSpec:
    def __init__(self, code, expected):
        self.code = code
        self.expected = expected

    @property
    def code(self):
        """The code to be inserted (string)."""  # noqa:D401
        return self._code

    @code.setter
    def code(self, value):
        self._code = value

    @property
    def expected(self):
        """The expected evaluation of the code (Python object)."""  # noqa:D401
        return self._expected

    @expected.setter
    def expected(self, value):
        self._expected = value


def append_scrapbook_commands(input_nb_path, output_nb_path, scrap_specs):
    notebook = nbf.read(input_nb_path, as_version=nbf.NO_CONVERT)

    scrapbook_cells = []
    # Always need to import nteract-scrapbook
    scrapbook_cells.append(nbf.v4.new_code_cell(source="import scrapbook as sb"))

    # Create a cell to store each key and value in the scrapbook
    for k, v in scrap_specs.items():
        source = "sb.glue(\"{0}\", {1})".format(k, v.code)
        scrapbook_cells.append(nbf.v4.new_code_cell(source=source))

    # Append the cells to the notebook
    [notebook['cells'].append(c) for c in scrapbook_cells]

    # Write out the new notebook
    nbf.write(notebook, output_nb_path)


def update_component_version(input_nb_path, output_nb_path, version_string: str):
    notebook = nbf.read(input_nb_path, as_version=nbf.NO_CONVERT)

    for cell in notebook['cells']:
        # Look for a rather specific string.....
        if cell['source'] == "version_string = '1'":
            cell['source'] = f"version_string = '{version_string}'"

    nbf.write(notebook, output_nb_path)


def assay_one_notebook(notebook_name, test_values, version_string: str):
    """Test a single notebook.
    This uses nbformat to append `nteract-scrapbook` commands to the
    specified notebook. The content of the commands and their expected
    values are stored in the `test_values` dictionary. The keys of this
    dictionary are strings to be used as scrapbook keys. They corresponding
    value is a `ScrapSpec` tuple. The `code` member of this tuple is
    the code (as a string) to be run to generate the scrapbook value. The
    `expected` member is a Python object which is checked for equality with
    the scrapbook value
    Makes certain assumptions about directory layout.
    """
    input_notebook = "examples/notebooks/" + notebook_name + ".ipynb"
    processed_notebook = "./test/notebooks/" + notebook_name + ".processed.ipynb"
    output_notebook = "./test/notebooks/" + notebook_name + ".output.ipynb"

    update_component_version(input_notebook, processed_notebook, version_string)
    # append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)

    #for k, v in test_values.items():
    #    assert nb.scraps[k].data == v.expected

@pytest.mark.notebooks
def test_responsibleaidashboard_housing_classification_model_debugging(component_config):
    nb_name = "responsibleaidashboard-housing-classification-model-debugging"

    version_string = component_config["version"]
    assay_one_notebook(nb_name, dict(), version_string)