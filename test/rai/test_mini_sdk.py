# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import pathlib
import pytest
import tempfile

from responsibleai import RAIInsights

from azure.ml import MLClient
from azure.ml.entities import JobInput
from azure.ml.entities import ComponentJob, PipelineJob

from azure_ml_rai import (
    download_rai_insights,
    download_rai_insights_ux,
    list_rai_insights,
)

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestMiniSDK:
    @pytest.mark.skip(reason="Auth issues in builds")
    def test_sdk_smoke(
        self, ml_client: MLClient, component_config, registered_adult_model_id: str
    ):
        version_string = component_config["version"]

        # Pipeline globals
        pipeline_inputs = {
            "target_column_name": "income",
            "my_training_data": JobInput(dataset=f"Adult_Train_PQ:{version_string}"),
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

        # Top level RAI Insights component
        create_rai_inputs = {
            "title": "Run built from Python",
            "task_type": "classification",
            "model_info_path": "${{jobs.fetch-model-job.outputs.model_info_output_path}}",
            "train_dataset": "${{inputs.my_training_data}}",
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
        experiment_name = f"MiniSDK_{version_string}"
        insights_pipeline_job = PipelineJob(
            experiment_name=experiment_name,
            description="Simple test for mini SDK",
            jobs={
                "fetch-model-job": fetch_job,
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

        available_insights = list_rai_insights(
            ml_client, experiment_name, registered_adult_model_id
        )
        assert len(available_insights) == 1

        with tempfile.TemporaryDirectory() as td:
            insight_name = "my_insight"
            target_dir = pathlib.Path(td) / insight_name

            download_rai_insights(ml_client, available_insights[0], str(target_dir))

            rai_i = RAIInsights.load(target_dir)
            assert rai_i is not None

        with tempfile.TemporaryDirectory() as td2:
            insight_name = "my_insight_ux"
            target_dir = pathlib.Path(td2) / insight_name

            download_rai_insights_ux(ml_client, available_insights[0], str(target_dir))
