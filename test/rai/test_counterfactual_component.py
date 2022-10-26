# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

from azure.ai.ml import MLClient, dsl, Input

from test.constants_for_test import Timeouts
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
            fetch_model_job.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

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
            construct_job.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

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
            counterfactual_job.set_limits(timeout=Timeouts.COUNTERFACTUAL_TIMEOUT)

            gather_job = rai_components.rai_gather(
                constructor=construct_job.outputs.rai_insights_dashboard,
                insight_1=counterfactual_job.outputs.counterfactual,
            )
            gather_job.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

            gather_job.outputs.dashboard.mode = "upload"
            gather_job.outputs.ux_json.mode = "upload"

            return {
                "dashboard": gather_job.outputs.dashboard,
                "ux_json": gather_job.outputs.ux_json,
            }

        adult_train = Input(
            type="mltable", path=f"adult_train:{version_string}", mode="download"
        )
        adult_test = Input(
            type="mltable", path=f"adult_test:{version_string}", mode="download"
        )
        rai_pipeline = test_counterfactual_classification(
            target_column_name="income",
            train_data=adult_train,
            test_data=adult_test,
        )

        rai_pipeline_job = submit_and_wait(ml_client, rai_pipeline)
        assert rai_pipeline_job is not None

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
            fetch_model_job.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

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
            construct_job.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

            counterfactual_job = rai_components.rai_counterfactual(
                rai_insights_dashboard=construct_job.outputs.rai_insights_dashboard,
                total_cfs=10,  # Case sensitivity bug
                desired_range="[20, 100]",
                permitted_range='{"ZN": [0, 10], "AGE": [0, 50], "CRIM": [25, 50], "INDUS": [0, 10]}',
                features_to_vary='["ZN", "AGE", "CRIM", "INDUS"]',
                feature_importance=True,
            )
            counterfactual_job.set_limits(timeout=Timeouts.COUNTERFACTUAL_TIMEOUT)

            gather_job = rai_components.rai_gather(
                constructor=construct_job.outputs.rai_insights_dashboard,
                insight_1=None,
                insight_4=counterfactual_job.outputs.counterfactual,
            )
            gather_job.set_limits(timeout=120)

            gather_job.outputs.dashboard.mode = "upload"
            gather_job.outputs.ux_json.mode = "upload"

            return {
                "dashboard": gather_job.outputs.dashboard,
                "ux_json": gather_job.outputs.ux_json,
            }

        adult_train_pq = Input(
            type="uri_file", path=f"boston_train_pq:{version_string}", mode="download"
        )
        adult_test_pq = Input(
            type="uri_file", path=f"boston_test_pq:{version_string}", mode="download"
        )
        rai_pipeline = test_counterfactual_regression(
            target_column_name="y",
            train_data=adult_train_pq,
            test_data=adult_test_pq,
        )

        rai_pipeline_job = submit_and_wait(ml_client, rai_pipeline)
        assert rai_pipeline_job is not None
