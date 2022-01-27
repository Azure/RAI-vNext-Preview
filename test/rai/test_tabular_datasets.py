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
        conversion_pipeline_job = PipelineJob(
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
        conversion_pipeline_job = submit_and_wait(ml_client, conversion_pipeline_job)
        assert conversion_pipeline_job is not None

    def test_use_tabular_dataset(
        self, ml_client: MLClient, component_config, registered_adult_model_id: str
    ):
        version_string = component_config["version"]
        epoch_secs = int(time.time())

        # Pipeline globals
        pipeline_inputs = {
            "my_parquet_file": JobInput(dataset=f"Adult_Train_PQ:{version_string}"),
        }

        # The job to convert the training dataset to Tabular
        train_tabular_base = "train_tabular_adult"
        reg_tabular_job_inputs = {
            "dataset_input_path": "${{inputs.my_parquet_file}}",
            "dataset_base_name": "train_tabular_base",
            "dataset_name_suffix": epoch_secs,
        }
        reg_tabular_job = ComponentJob(
            component=f"RegisterTabularDataset:{version_string}",
            inputs=reg_tabular_job_inputs,
        )

        # Define the pipeline
        experiment_name = f"Register_Tabular_for_Use_{version_string}"
        conversion_pipeline_job = PipelineJob(
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
        conversion_pipeline_job = submit_and_wait(ml_client, conversion_pipeline_job)
        assert conversion_pipeline_job is not None

        # ----

        # Now we want to consume the dataset in one of our pipelines

        # Pipeline globals
        pipeline_inputs = {
            "target_column_name": "income",
            "my_test_data": JobInput(dataset=f"Adult_Test_PQ:{version_string}"),
        }

        # The job to fetch the model
        fetch_job_inputs = {"model_id": registered_adult_model_id}
        fetch_job_outputs = {"model_info_output_path": None}
        fetch_job = ComponentJob(
            component=f"FetchRegisteredModel:{version_string}",
            inputs=fetch_job_inputs,
            outputs=fetch_job_outputs,
        )

        # The job to convert the tabular dataset to a file
        to_parquet_inputs = {
            "tabular_dataset_name": f"{train_tabular_base}_{epoch_secs}"
        }
        to_parquet_outputs = {"dataset_output_path": None}
        to_parquet_job = ComponentJob(
            component=f"TabularToParquet:{version_string}",
            inputs=to_parquet_inputs,
            outputs=to_parquet_outputs,
        )

        # Top level RAI Insights component
        create_rai_inputs = {
            "title": "Run built from Python",
            "task_type": "classification",
            "model_info_path": "${{jobs.fetch-model-job.outputs.model_info_output_path}}",
            "train_dataset": "${{jobs.to-parquet-job.dataset_output_path}}",
            "test_dataset": "${{inputs.my_test_data}}",
            "target_column_name": "${{inputs.target_column_name}}",
            "categorical_column_names": '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
        }
        create_rai_outputs = {"rai_insights_dashboard": None}
        create_rai_job = ComponentJob(
            component=f"RAIInsightsConstructor:{version_string}",
            inputs=create_rai_inputs,
            outputs=create_rai_outputs,
        )

        # Setup explanation
        explain_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "comment": "For miniSDK testing",
        }
        explain_outputs = {"explanation": None}
        explain_job = ComponentJob(
            component=f"RAIInsightsExplanation:{version_string}",
            inputs=explain_inputs,
            outputs=explain_outputs,
        )

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.explain-rai-job.outputs.explanation}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = ComponentJob(
            component=f"RAIInsightsGather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        experiment_name = f"Use_Tabular_Dataset_{version_string}"
        insights_pipeline_job = PipelineJob(
            experiment_name=experiment_name,
            description="Simple test for mini SDK",
            jobs={
                "fetch-model-job": fetch_job,
                "to-parquet-job": to_parquet_job,
                "create-rai-job": create_rai_job,
                "explain-rai-job": explain_job,
                "gather-job": gather_job,
            },
            inputs=pipeline_inputs,
            outputs=None,
            compute="cpucluster",
        )

        # Send it
        insights_pipeline_job = submit_and_wait(ml_client, insights_pipeline_job)
        assert insights_pipeline_job is not None
