# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing import List, Optional, Tuple

from azureml.core import Run
from azure.ml import MLClient

from ._constants import PropertyKeyValues
from ._utilities import _get_v1_workspace_client


def list_rai_insights(
    ml_client: MLClient, experiment_name: str, model_id: Optional[str] = None
) -> List[str]:
    """List RAI Insights available from an experiment.

    This is a workaround, pending implementation of the required functionality in SDKv2.

    For a given experiment and (optional) model id, list the available RAI Insights
    which ahve been computed using the DPv2 components. The insights can
    then be downloaded using download_rai_insights.

    param MLCient ml_client: Instance of MLClient to use for communicating with AzureML
    param str experiment_name: Name of the experiment to search under
    param str model_id: Optional id in AzureML of the desired model
    """
    # Return the Run ids for runs having RAI insights

    filter_properties = {
        PropertyKeyValues.RAI_INSIGHTS_TYPE_KEY: PropertyKeyValues.RAI_INSIGHTS_TYPE_GATHER
    }
    if model_id is not None:
        filter_properties[PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY] = model_id

    # Have to use V1 client for now
    v1_workspace = _get_v1_workspace_client(ml_client)
    v1_experiment = v1_workspace.experiments[experiment_name]

    all_runs = Run.list(
        v1_experiment, properties=filter_properties, include_children=True
    )

    return [r.id for r in all_runs]
