# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import pathlib
import time

from azure.ml import MLClient
from azure.ml import dsl
from azure.ml.entities import JobInput
from azure.ml.entities import ComponentJob, PipelineJob
from sympy import maximum

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestRegisterTabularDataset:
    def test_smoke_registration(
        self, ml_client: MLClient, component_config, registered_adult_model_id: str
    ):
        version_string = component_config["version"]
        epoch_secs = int(time.time())

        register_tabular_component = dsl.load_component(client=ml_client,
            name="RegisterTabularDataset", version=version_string
        )

        @dsl.pipeline(
            compute="cpucluster",
            description="Test of Register Tabular component",
            experiment_name="Smoke_Tabular_Datset_registration",
        )
        def my_pipeline(
            parquet_file,
        ):
            _ = register_tabular_component(
                dataset_input_path=parquet_file, dataset_base_name="registered_tabular", dataset_name_suffix=str(epoch_secs)
            )
            return {}

        adult_train_pq = ml_client.datasets.get(
            name="Adult_Train_PQ", version=version_string
        )
        pipeline = my_pipeline(adult_train_pq)

        conversion_pipeline_job = submit_and_wait(ml_client, pipeline)
        assert conversion_pipeline_job is not None

    def test_use_tabular_dataset(
        self, ml_client: MLClient, component_config, registered_adult_model_id: str
    ):
        version_string = component_config["version"]
        epoch_secs = int(time.time())
        train_tabular_base = "train_tabular_adult"

        register_tabular_component = dsl.load_component(client=ml_client,
            name="RegisterTabularDataset", version=version_string
        )

        @dsl.pipeline(
            compute="cpucluster",
            description="Test of Register Tabular component",
            experiment_name="Tabular_Datset_registration",
        )
        def tabular_registration_pipeline(
            parquet_file,
            base_name
        ):
            _ = register_tabular_component(
                dataset_input_path=parquet_file, dataset_base_name=base_name, dataset_name_suffix=str(epoch_secs)
            )
            return {}

        adult_train_pq = ml_client.datasets.get(
            name="Adult_Train_PQ", version=version_string
        )
        pipeline = tabular_registration_pipeline(adult_train_pq, base_name=train_tabular_base)

        conversion_pipeline_job = submit_and_wait(ml_client, pipeline)
        assert conversion_pipeline_job is not None

        # ----

        # Now we want to consume the dataset in one of our pipelines

        fetch_model_component = dsl.load_component(client=ml_client,
            name="FetchRegisteredModel", version=version_string
        )

        tabular_to_parquet_component = dsl.load_component(client=ml_client,
            name="TabularToParquet", version=version_string
        )

        rai_constructor_component = dsl.load_component(client=ml_client,
            name="RAIInsightsConstructor", version=version_string
        )

        rai_explanation_component = dsl.load_component(client=ml_client,
            name="RAIInsightsExplanation", version=version_string
        )

        rai_gather_component = dsl.load_component(client=ml_client,
            name="RAIInsightsGather", version=version_string
        )

        @dsl.pipeline(
            compute="cpucluster",
            description="Test of Register Tabular component",
            experiment_name=f"Use_Tabular_Dataset_{version_string}",
        )
        def use_tabular_rai(
            target_column_name, train_data_name, test_data,
        ):
            fetch_model_job = fetch_model_component(model_id=registered_adult_model_id)

            to_parquet_job = tabular_to_parquet_component(tabular_dataset_name=train_data_name)

            construct_job = rai_constructor_component(
                title="Run built from DSL",
                task_type="classification",
                model_info_path=fetch_model_job.outputs.model_info_output_path,
                train_dataset=to_parquet_job.outputs.dataset_output_path,
                test_dataset=test_data,
                target_column_name=target_column_name,
                categorical_column_names='["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
                maximum_rows_for_test_dataset=5000,
                classes='[]'
            )

        adult_test_pq = ml_client.datasets.get(
            name="Adult_Test_PQ", version=version_string
        )
        rai_pipeline = use_tabular_rai(target_column_name='income', train_data_name=f"{train_tabular_base}_{epoch_secs}", test_data=adult_test_pq)

        rai_pipeline_job = submit_and_wait(ml_client, rai_pipeline)
        assert rai_pipeline_job is not None

    def some_other_func(self):
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
            "train_dataset": "${{jobs.to-parquet-job.outputs.dataset_output_path}}",
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
