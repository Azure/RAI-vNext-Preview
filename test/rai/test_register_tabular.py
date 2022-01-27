# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import pathlib
import time

from azure.ml import MLClient
from azure.ml.entities import JobInput
from azure.ml.entities import ComponentJob, PipelineJob

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestRegisterTabularDataset:
    def test_smoke_registration(
        self, ml_client: MLClient, component_config, registered_adult_model_id: str
    ):
        version_string = component_config["version"]

        # Pipeline globals
        pipeline_inputs = {
            "my_parquet_file": JobInput(dataset=f"Adult_Train_PQ:{version_string}"),
        }

        # The job to convert the dataset to Tabular
        reg_tabular_job_inputs = {
            "dataset_input_path": "${{inputs.my_parquet_file}}",
            "dataset_base_name": "registered_tabular",
        }
        reg_tabular_job = ComponentJob(
            component=f"RegisterTabularDataset:{version_string}",
            inputs=reg_tabular_job_inputs,
        )

        # Define the pipeline
        experiment_name = f"Register_Tabular_{version_string}"
        insights_pipeline_job = PipelineJob(
            experiment_name=experiment_name,
            description="Test registering tabular dataset",
            jobs={
                "reg-job": reg_tabular_job,
            },
            inputs=pipeline_inputs,
            outputs=None,
            compute="cpucluster",
        )

        # Send it
        insights_pipeline_job = submit_and_wait(ml_client, insights_pipeline_job)
        assert insights_pipeline_job is not None
