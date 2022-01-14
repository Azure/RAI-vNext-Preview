# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

from azure.ml import MLClient
from azure.ml.entities import JobInput
from azure.ml.entities import ComponentJob, PipelineJob

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestCounterfactuallComponent:
    def test_classification_all_args(
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

        # Setup counterfactual
        counterfactual_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "total_CFs": "10",
            "method": "kdtree",
            "desired_class": "opposite",
            "permitted_range": '{"Capital Gain": [0, 2000], "Hours per Week": [0, 10]}',
            "features_to_vary": '["Capital Gain", "Hours per Week", "Age", "Country", "Sex"]',
            "feature_importance": "True",
        }
        counterfactual_outputs = {"counterfactual": None}
        counterfactual_job = ComponentJob(
            component=f"RAIInsightsCounterfactual:{version_string}",
            inputs=counterfactual_inputs,
            outputs=counterfactual_outputs,
        )

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.counterfactual-rai-job.outputs.counterfactual}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = ComponentJob(
            component=f"RAIInsightsGather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"Counterfactual_Classification_All_Args_{version_string}",
            description="Expected failure due to multiple tool instances",
            jobs={
                "fetch-model-job": fetch_job,
                "create-rai-job": create_rai_job,
                "counterfactual-rai-job": counterfactual_job,
                "gather-job": gather_job,
            },
            inputs=pipeline_inputs,
            outputs=None,
            compute="cpucluster",
        )

        # Send it
        insights_pipeline_job = submit_and_wait(ml_client, insights_pipeline_job)
        assert insights_pipeline_job is not None
