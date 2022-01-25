# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import os
from pathlib import Path
import tempfile

from typing import Any

import mlflow
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.tracking import MlflowClient

from azure.identity import ChainedTokenCredential
from azure.storage.blob import BlobServiceClient

from azure.ml import MLClient

from ._constants import OutputPortNames
from ._utilities import _get_v1_workspace_client, _get_storage_account_key


def _get_output_port_info(
    mlflow_client: MlflowClient, run_id: str, port_name: str
) -> Any:
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_client.download_artifacts(run_id, port_name, temp_dir)

        json_filename = os.path.join(temp_dir, port_name)

        with open(json_filename, "r") as json_file:
            port_info = json.load(json_file)

    return port_info


def _get_credential(ml_client: MLClient, storage_account_name: str, auth_method):
    result = None
    if auth_method == "arm_direct":
        result = ml_client._credential
    elif auth_method == "fetch_key":
        result = _get_storage_account_key(ml_client, storage_account_name)
    else:
        raise ValueError(f"Unrecognised auth_method: {auth_method}")
    return result


def _download_port_files(
    ml_client: MLClient,
    run_id: str,
    port_name: str,
    target_directory: Path,
    auth_method: str,
) -> None:
    mlflow_client = MlflowClient()
    port_info = _get_output_port_info(mlflow_client, run_id, port_name)

    wasbs_tuple = AzureBlobArtifactRepository.parse_wasbs_uri(port_info["Uri"])
    storage_account_name = wasbs_tuple[1]
    if len(wasbs_tuple) == 4:
        account_dns_suffix = wasbs_tuple[3]
    else:
        account_dns_suffix = "blob.core.windows.net"

    account_url = "https://{account}.{suffix}".format(
        account=storage_account_name, suffix=account_dns_suffix
    )

    sa_cred = _get_credential(ml_client, storage_account_name, auth_method)
    bsc = BlobServiceClient(account_url=account_url, credential=sa_cred)
    abar = AzureBlobArtifactRepository(port_info["Uri"], client=bsc)

    # Download everything
    abar.download_artifacts("", target_directory)


def download_rai_insights(
    ml_client: MLClient, rai_insight_id: str, path: str, auth_method: str = "arm_direct"
) -> None:
    """Download an RAIInsight dashboard from AzureML

    This is a workaround, pending implementation of the required functionality in SDKv2.
    Authentication to Azure is the largest pain point, due to how permissions on the storage
    account are set up.

    If the current user (specifically, their `DefaultAzureCredential()`) has the
    "Storage Blob Data Contributor" role on the storage account containing the dashboard output
    (this will normally be the default workspace store), then the `arm_direct` method can
    be used. Howevever, even if a user created the storage account, they will not have
    this role by default. It can be added manually in the portal, or if the user
    is the owner of the storage account, they the `fetch_key` can be used instead.
    This will fetch one of the storage account keys and use it for authentication

    param MLCient ml_client: Instance of MLClient to use for communicating with AzureML
    param str rai_insight_id: The id of the dashboard to be downloaded (will be the run id of the Gather component)
    param str path: Path to download the dashboard (must not exist)
    param str auth_method: See above. `arm_direct` or `fetch_key`
    """
    v1_ws = _get_v1_workspace_client(ml_client)

    mlflow.set_tracking_uri(v1_ws.get_mlflow_tracking_uri())

    output_directory = Path(path)
    output_directory.mkdir(parents=True, exist_ok=False)

    _download_port_files(
        ml_client,
        rai_insight_id,
        OutputPortNames.RAI_INSIGHTS_GATHER_RAIINSIGHTS_PORT,
        output_directory,
        auth_method,
    )

    # Ensure empty directories are present
    tool_dirs = ["causal", "counterfactual", "error_analysis", "explainer"]
    for t in tool_dirs:
        os.makedirs(Path(path) / t, exist_ok=True)


def download_rai_insights_ux(
    ml_client: MLClient, rai_insight_id: str, path: str
) -> None:
    v1_ws = _get_v1_workspace_client(ml_client)

    mlflow.set_tracking_uri(v1_ws.get_mlflow_tracking_uri())

    mlflow_client = MlflowClient()

    output_directory = Path(path)
    output_directory.mkdir(parents=True, exist_ok=False)

    _download_port_files(
        mlflow_client,
        rai_insight_id,
        OutputPortNames.RAI_INSIGHTS_GATHER_RAIINSIGHTS_UX_PORT,
        output_directory,
        ml_client._credential,
    )
