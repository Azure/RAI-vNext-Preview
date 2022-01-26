# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import os
from pathlib import Path
import tempfile

from typing import Any

from azureml.core import Dataset, Run, Workspace
from azureml.data import FileDataset

from azure.ml import MLClient

from ._constants import OutputPortNames
from ._utilities import _get_v1_workspace_client, _get_storage_account_key


def _get_dataset_for_port(ws: Workspace, run_id: str, port_name: str) -> FileDataset:
    result = None

    my_run = Run.get(ws, run_id)

    output_datasets = my_run.get_details()["outputDatasets"]
    for od in output_datasets:
        if od["outputDetails"]["outputName"] == port_name:
            result = od["dataset"]

    return result


def download_rai_insights(ml_client: MLClient, rai_insight_id: str, path: str) -> None:
    """Download an RAIInsight dashboard from AzureML

    This is a workaround, pending implementation of the required functionality in SDKv2.

    param MLCient ml_client: Instance of MLClient to use for communicating with AzureML
    param str rai_insight_id: The id of the dashboard to be downloaded (will be the run id of the Gather component)
    param str path: Path to download the dashboard (must not exist)
    """
    v1_ws = _get_v1_workspace_client(ml_client)

    dataset = _get_dataset_for_port(
        v1_ws, rai_insight_id, OutputPortNames.RAI_INSIGHTS_GATHER_RAIINSIGHTS_PORT
    )
    dataset.download(target_path=path, overwrite=False)

    # Ensure empty directories are present
    tool_dirs = ["causal", "counterfactual", "error_analysis", "explainer"]
    for t in tool_dirs:
        os.makedirs(Path(path) / t, exist_ok=True)


def download_rai_insights_ux(
    ml_client: MLClient, rai_insight_id: str, path: str
) -> None:
    """Download the UX representation of an RAIInsight dashboard from AzureML

    This is a workaround, pending implementation of the required functionality in SDKv2.

    param MLCient ml_client: Instance of MLClient to use for communicating with AzureML
    param str rai_insight_id: The id of the dashboard to be downloaded (will be the run id of the Gather component)
    param str path: Path to download the dashboard (must not exist)
    """
    v1_ws = _get_v1_workspace_client(ml_client)

    dataset = _get_dataset_for_port(
        v1_ws, rai_insight_id, OutputPortNames.RAI_INSIGHTS_GATHER_RAIINSIGHTS_UX_PORT
    )
    dataset.download(target_path=path, overwrite=False)
