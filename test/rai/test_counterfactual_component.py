# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

from azure.ml import MLClient
from azure.ml import dsl
from azure.ml.entities import JobInput
from azure.ml.entities import CommandComponent, PipelineJob

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestCounterfactualComponent:
    def test_classification_all_args(
        self,
        ml_client: MLClient,
        component_config,
        registered_adult_model_id: str,
        rai_components,
    ):
        version_string = component_config["version"]

        @dsl.pipeline(
            compute="cpucluster",
            description="Test Counterfactual component with all arguments",
            experiment_name=f"TestCounterfactualComponent_test_classification_all_args_{version_string}",
        )
        def test_counterfactual_classification(
            target_column_name,
            train_data,
            test_data,
        ):
            fetch_model_job = rai_components.fetch_model(
                model_id=registered_adult_model_id
            )

            construct_job = rai_components.rai_constructor(
                title="Run built from DSL",
                task_type="classification",
                model_info_path=fetch_model_job.outputs.model_info_output_path,
                train_dataset=train_data,
                test_dataset=test_data,
                target_column_name=target_column_name,
                categorical_column_names='["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
                maximum_rows_for_test_dataset=5000,  # Should be default
                classes="[]",  # Should be default
            )

            counterfactual_job = rai_components.rai_counterfactual(
                rai_insights_dashboard=construct_job.outputs.rai_insights_dashboard,
                total_cfs=10,  # Case sensitivity bug!
                method="random",
                desired_class="opposite",
                desired_range="[]",
                permitted_range='{"Capital Gain": [0, 20000], "Hours per week": [0, 20]}',
                features_to_vary='["Capital Gain", "Hours per week", "Age", "Country", "Sex"]',
                feature_importance=True,
            )

            return {}

        adult_train_pq = JobInput(type="mltable", path=f"adult_train_pq:{version_string}")
        adult_test_pq = JobInput(type="mltable", path=f"adult_test_pq:{version_string}")
        rai_pipeline = test_counterfactual_classification(
            target_column_name="income",
            train_data=adult_train_pq,
            test_data=adult_test_pq,
        )

        rai_pipeline_job = submit_and_wait(ml_client, rai_pipeline)
        assert rai_pipeline_job is not None
        """
        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.counterfactual-rai-job.outputs.counterfactual}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = CommandComponent(
            component=f"rai_insights_gather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"Counterfactual_Classification_All_Args_{version_string}",
            description="Test counterfactual component with all arguments",
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
        insights_pipeline_job = submit_and_wait(
            ml_client, insights_pipeline_job)
        assert insights_pipeline_job is not None
        """

    def test_regression_all_args(
        self,
        ml_client: MLClient,
        component_config,
        registered_boston_model_id: str,
        rai_components,
    ):
        version_string = component_config["version"]

        @dsl.pipeline(
            compute="cpucluster",
            description="Test Counterfactual component with all arguments",
            experiment_name=f"TestCounterfactualComponent_test_regression_all_args_{version_string}",
        )
        def test_counterfactual_regression(
            target_column_name,
            train_data,
            test_data,
        ):
            fetch_model_job = rai_components.fetch_model(
                model_id=registered_boston_model_id
            )

            construct_job = rai_components.rai_constructor(
                title="Run built from DSL",
                task_type="regression",
                model_info_path=fetch_model_job.outputs.model_info_output_path,
                train_dataset=train_data,
                test_dataset=test_data,
                target_column_name=target_column_name,
                categorical_column_names="[]",
                maximum_rows_for_test_dataset=5000,  # Should be default
                classes="[]",  # Should be default
            )

            counterfactual_job = rai_components.rai_counterfactual(
                rai_insights_dashboard=construct_job.outputs.rai_insights_dashboard,
                total_cfs=10,  # Case sensitivity bug
                method="kdtree",
                desired_class="opposite",  # Required argument bug...
                desired_range="[20, 100]",
                permitted_range='{"ZN": [0, 10], "AGE": [0, 50], "CRIM": [25, 50], "INDUS": [0, 10]}',
                features_to_vary='["ZN", "AGE", "CRIM", "INDUS"]',
                feature_importance=True,
            )

            return {}

        adult_train_pq = JobInput(type="mltable", path=f"boston_train_pq:{version_string}")
        adult_test_pq = JobInput(type="mltable", path=f"boston_test_pq:{version_string}")
        rai_pipeline = test_counterfactual_regression(
            target_column_name="y",
            train_data=adult_train_pq,
            test_data=adult_test_pq,
        )

        rai_pipeline_job = submit_and_wait(ml_client, rai_pipeline)
        assert rai_pipeline_job is not None
        """
        # Setup counterfactual
        counterfactual_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "total_CFs": "10",
            "method": "kdtree",
            "desired_range": "[20, 100]",
            "permitted_range": '{"ZN": [0, 10], "AGE": [0, 50], "CRIM": [25, 50], "INDUS": [0, 10]}',
            "features_to_vary": '["ZN", "AGE", "CRIM", "INDUS"]',
            "feature_importance": "True",
        }
        counterfactual_outputs = {"counterfactual": None}
        counterfactual_job = CommandComponent(
            component=f"rai_insights_counterfactual:{version_string}",
            inputs=counterfactual_inputs,
            outputs=counterfactual_outputs,
        )

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.counterfactual-rai-job.outputs.counterfactual}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = CommandComponent(
            component=f"rai_insights_gather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"Counterfactual_Regression_All_Args_{version_string}",
            description="Check regression example with all arguments",
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
        """
