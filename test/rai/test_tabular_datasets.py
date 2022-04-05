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

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestRegisterTabularDataset:
    def test_smoke_registration(
        self, ml_client: MLClient, component_config, registered_adult_model_id: str
    ):
        version_string = component_config["version"]
        epoch_secs = int(time.time())

        register_tabular_component = dsl.load_component(
            client=ml_client, name="RegisterTabularDataset", version=version_string
        )

        @dsl.pipeline(
            compute="cpucluster",
            description="Test of Register Tabular component",
            experiment_name="Smoke_Tabular_Datset_registration",
        )
        def my_pipeline(train_parquet, test_parquet):
            _ = register_tabular_component(
                dataset_input_path=train_parquet,
                dataset_base_name="tabular_train_adult",
                dataset_name_suffix=str(epoch_secs),
            )
            _ = register_tabular_component(
                dataset_input_path=test_parquet,
                dataset_base_name="tabular_test_adult",
                dataset_name_suffix=str(epoch_secs),
            )
            return {}

        adult_train_pq = ml_client.datasets.get(
            name="Adult_Train_PQ", version=version_string
        )
        adult_test_pq = ml_client.datasets.get(
            name="Adult_Test_PQ", version=version_string
        )
        pipeline = my_pipeline(adult_train_pq, adult_test_pq)

        conversion_pipeline_job = submit_and_wait(ml_client, pipeline)
        assert conversion_pipeline_job is not None

    def test_use_tabular_dataset(
        self, ml_client: MLClient, component_config, registered_adult_model_id: str
    ):
        version_string = component_config["version"]
        epoch_secs = int(time.time())
        train_tabular_base = "train_tabular_adult"

        register_tabular_component = dsl.load_component(
            client=ml_client, name="RegisterTabularDataset", version=version_string
        )

        @dsl.pipeline(
            compute="cpucluster",
            description="Test of Register Tabular component",
            experiment_name="Tabular_Datset_registration",
        )
        def tabular_registration_pipeline(parquet_file, base_name):
            _ = register_tabular_component(
                dataset_input_path=parquet_file,
                dataset_base_name=base_name,
                dataset_name_suffix=str(epoch_secs),
            )
            return {}

        adult_train_pq = ml_client.datasets.get(
            name="Adult_Train_PQ", version=version_string
        )
        pipeline = tabular_registration_pipeline(
            adult_train_pq, base_name=train_tabular_base
        )

        conversion_pipeline_job = submit_and_wait(ml_client, pipeline)
        assert conversion_pipeline_job is not None

        # ----

        # Now we want to consume the dataset in one of our pipelines

        fetch_model_component = dsl.load_component(
            client=ml_client, name="FetchRegisteredModel", version=version_string
        )

        tabular_to_parquet_component = dsl.load_component(
            client=ml_client, name="TabularToParquet", version=version_string
        )

        rai_constructor_component = dsl.load_component(
            client=ml_client, name="RAIInsightsConstructor", version=version_string
        )

        rai_explanation_component = dsl.load_component(
            client=ml_client, name="RAIInsightsExplanation", version=version_string
        )

        rai_gather_component = dsl.load_component(
            client=ml_client, name="RAIInsightsGather", version=version_string
        )

        @dsl.pipeline(
            compute="cpucluster",
            description="Test of Register Tabular component",
            experiment_name=f"Use_Tabular_Dataset_{version_string}",
        )
        def use_tabular_rai(
            target_column_name,
            train_data_name,
            test_data,
        ):
            fetch_model_job = fetch_model_component(model_id=registered_adult_model_id)

            to_parquet_job = tabular_to_parquet_component(
                tabular_dataset_name=train_data_name
            )

            construct_job = rai_constructor_component(
                title="Run built from DSL",
                task_type="classification",
                model_info_path=fetch_model_job.outputs.model_info_output_path,
                train_dataset=to_parquet_job.outputs.dataset_output_path,
                test_dataset=test_data,
                target_column_name=target_column_name,
                categorical_column_names='["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
                maximum_rows_for_test_dataset=5000,
                classes="[]",
            )

            rai_explanation_job = rai_explanation_component(
                rai_insights_dashboard=construct_job.outputs.rai_insights_dashboard,
                comment="Something, something",
            )

            rai_gather_job = rai_gather_component(
                constructor=construct_job.outputs.rai_insights_dashboard,
                insight_1=rai_explanation_job.outputs.explanation,
                insight_2=None,
                insight_3=None,
                insight_4=None,
            )
            rai_gather_job.outputs.dashboard.mode = "upload"
            rai_gather_job.outputs.ux_json.mode = "upload"

            return {
                "dashboard": rai_gather_job.outputs.dashboard,
                "ux_json": rai_gather_job.outputs.ux_json,
            }

        adult_test_pq = JobInput(path=f"Adult_Test_PQ:{version_string}", mode="download")
        rai_pipeline = use_tabular_rai(
            target_column_name="income",
            train_data_name=f"{train_tabular_base}_{epoch_secs}",
            test_data=adult_test_pq,
        )

        rai_pipeline_job = submit_and_wait(ml_client, rai_pipeline)
        assert rai_pipeline_job is not None
