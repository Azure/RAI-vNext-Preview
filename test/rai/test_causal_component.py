# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

from azure.ml import MLClient
from azure.ml.entities import JobInput
from azure.ml.entities import CommandComponent, PipelineJob

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestCausalComponent:
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
        fetch_job = CommandComponent(
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
        create_rai_job = CommandComponent(
            component=f"RAIInsightsConstructor:{version_string}",
            inputs=create_rai_inputs,
            outputs=create_rai_outputs,
        )

        causal_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "treatment_features": '["Age", "Sex"]',
            "heterogeneity_features": '["Marital Status"]',
            "nuisance_model": "automl",
            "heterogeneity_model": "forest",
            "alpha": "0.06",
            "upper_bound_on_cat_expansion": "51",
            "treatment_cost": "[0.1, 0.2]",
            "min_tree_leaf_samples": "3",
            "max_tree_depth": "3",
            "skip_cat_limit_checks": "True",
            "categories": "auto",
            "n_jobs": "2",
            "verbose": "0",
            "random_state": "10",
        }
        causal_outputs = {"causal": None}
        causal_job = CommandComponent(
            component=f"RAIInsightsCausal:{version_string}",
            inputs=causal_inputs,
            outputs=causal_outputs,
        )

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.causal-rai-job.outputs.causal}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = CommandComponent(
            component=f"RAIInsightsGather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"Causal_Component_Classification_All_Args_{version_string}",
            description="Test causal component with all arguments",
            jobs={
                "fetch-model-job": fetch_job,
                "create-rai-job": create_rai_job,
                "causal-rai-job": causal_job,
                "gather-job": gather_job,
            },
            inputs=pipeline_inputs,
            outputs=None,
            compute="cpucluster",
        )

        # Send it
        insights_pipeline_job = submit_and_wait(ml_client, insights_pipeline_job)
        assert insights_pipeline_job is not None

    def test_regression_all_args(
        self, ml_client: MLClient, component_config, registered_boston_model_id: str
    ):
        version_string = component_config["version"]

        # Pipeline globals
        pipeline_inputs = {
            "target_column_name": "y",
            "my_training_data": JobInput(dataset=f"Boston_Train_PQ:{version_string}"),
            "my_test_data": JobInput(dataset=f"Boston_Test_PQ:{version_string}"),
        }

        # The job to fetch the model
        fetch_job_inputs = {"model_id": registered_boston_model_id}
        fetch_job_outputs = {"model_info_output_path": None}
        fetch_job = CommandComponent(
            component=f"FetchRegisteredModel:{version_string}",
            inputs=fetch_job_inputs,
            outputs=fetch_job_outputs,
        )

        # Top level RAI Insights component
        create_rai_inputs = {
            "title": "Run built from Python",
            "task_type": "regression",
            "model_info_path": "${{jobs.fetch-model-job.outputs.model_info_output_path}}",
            "train_dataset": "${{inputs.my_training_data}}",
            "test_dataset": "${{inputs.my_test_data}}",
            "target_column_name": "${{inputs.target_column_name}}",
            "categorical_column_names": "[]",
        }
        create_rai_outputs = {"rai_insights_dashboard": None}
        create_rai_job = CommandComponent(
            component=f"RAIInsightsConstructor:{version_string}",
            inputs=create_rai_inputs,
            outputs=create_rai_outputs,
        )

        causal_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "treatment_features": '["NOX", "ZN"]',
            "heterogeneity_features": '["RM", "AGE"]',
            "nuisance_model": "automl",
            "heterogeneity_model": "forest",
            "alpha": "0.06",
            "upper_bound_on_cat_expansion": "51",
            "treatment_cost": "[0.1, 0.2]",
            "min_tree_leaf_samples": "3",
            "max_tree_depth": "3",
            "skip_cat_limit_checks": "True",
            "categories": "auto",
            "n_jobs": "2",
            "verbose": "0",
            "random_state": "10",
        }
        causal_outputs = {"causal": None}
        causal_job = CommandComponent(
            component=f"RAIInsightsCausal:{version_string}",
            inputs=causal_inputs,
            outputs=causal_outputs,
        )

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.causal-rai-job.outputs.causal}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = CommandComponent(
            component=f"RAIInsightsGather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"Causal_Regression_All_Args_{version_string}",
            description="Check regression example with all arguments",
            jobs={
                "fetch-model-job": fetch_job,
                "create-rai-job": create_rai_job,
                "causal-rai-job": causal_job,
                "gather-job": gather_job,
            },
            inputs=pipeline_inputs,
            outputs=None,
            compute="cpucluster",
        )

        # Send it
        insights_pipeline_job = submit_and_wait(ml_client, insights_pipeline_job)
        assert insights_pipeline_job is not None
