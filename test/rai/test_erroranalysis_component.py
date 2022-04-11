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


class TestErrorAnalysisComponent:
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
            description="Test Error Analysis component with all arguments",
            experiment_name=f"TestErrorAnalysisComponent_test_classification_all_args_{version_string}",
        )
        def test_erroranalysis_classification(
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

            erroranalysis_job = rai_components.rai_erroranalysis(
                rai_insights_dashboard=construct_job.outputs.rai_insights_dashboard,
                max_depth=4,
                num_leaves=25,
                min_child_samples=10,
                filter_features='["Marital Status", "Workclass"]',
            )

            return {}

        adult_train_pq = JobInput(
            path=f"adult_train_pq:{version_string}"
        )
        adult_test_pq = JobInput(path=f"adult_test_pq:{version_string}")
        rai_pipeline = test_erroranalysis_classification(
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
            "insight_1": "${{jobs.erroranalysis-rai-job.outputs.error_analysis}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = CommandComponent(
            component=f"rai_insights_gather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"ErrorAnalysis_Classification_All_Args_{version_string}",
            description="Check error analysis with all arguments",
            jobs={
                "fetch-model-job": fetch_job,
                "create-rai-job": create_rai_job,
                "erroranalysis-rai-job": erroranalysis_job,
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
            description="Test Error Analysis component with all arguments",
            experiment_name=f"TestErrorAnalysisComponent_test_regression_all_args_{version_string}",
        )
        def test_erroranalysis_regression(
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

            erroranalysis_job = rai_components.rai_erroranalysis(
                rai_insights_dashboard=construct_job.outputs.rai_insights_dashboard,
                max_depth=4,
                num_leaves=25,
                min_child_samples=10,
                filter_features="[]",
            )

            return {}

        adult_train_pq = JobInput(
            path=f"boston_train_pq:{version_string}"
        )
        adult_test_pq = JobInput(
            path=f"boston_test_pq:{version_string}"
        )
        rai_pipeline = test_erroranalysis_regression(
            target_column_name="y",
            train_data=adult_train_pq,
            test_data=adult_test_pq,
        )

        rai_pipeline_job = submit_and_wait(ml_client, rai_pipeline)
        assert rai_pipeline_job is not None
        """

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.erroranalysis-rai-job.outputs.error_analysis}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = CommandComponent(
            component=f"rai_insights_gather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"ErrorAnalysis_Regression_All_Args_{version_string}",
            description="Check regression example with all arguments",
            jobs={
                "fetch-model-job": fetch_job,
                "create-rai-job": create_rai_job,
                "erroranalysis-rai-job": erroranalysis_job,
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
