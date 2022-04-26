# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import pathlib
import tempfile
import uuid

from azure.ml import MLClient, dsl, Input, Output
from azure.ml.entities import load_component
from azure.ml.entities import Job
from responsibleai import RAIInsights

from test.utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def process_file(input_file, output_file, replacements):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            for f, r in replacements.items():
                line = line.replace(f, r)
            outfile.write(line)


class TestRAISmoke:
    def test_classification_pipeline_from_yaml(self, ml_client, component_config):
        current_dir = pathlib.Path(__file__).parent.absolute()
        pipeline_file = current_dir / "pipeline_adult_analyse.yaml"
        pipeline_processed_file = "pipeline_adult_analyse.processed.yaml"

        replacements = {"VERSION_REPLACEMENT_STRING": str(component_config["version"])}
        process_file(pipeline_file, pipeline_processed_file, replacements)

        pipeline_job = Job.load(path=pipeline_processed_file)

        submit_and_wait(ml_client, pipeline_job)

    def test_boston_pipeline_from_yaml(self, ml_client, component_config):
        current_dir = pathlib.Path(__file__).parent.absolute()
        pipeline_file = current_dir / "pipeline_boston_analyse.yaml"
        pipeline_processed_file = "pipeline_boston_analyse.processed.yaml"

        replacements = {"VERSION_REPLACEMENT_STRING": str(component_config["version"])}
        process_file(pipeline_file, pipeline_processed_file, replacements)

        pipeline_job = Job.load(path=pipeline_processed_file)

        submit_and_wait(ml_client, pipeline_job)

    def test_classification_pipeline_from_python(
        self, ml_client: MLClient, component_config
    ):
        # This is for the Adult dataset
        version_string = component_config["version"]

        train_log_reg_component = load_component(
            client=ml_client,
            name="train_logistic_regression_for_rai",
            version=version_string,
        )

        register_model_component = load_component(
            client=ml_client, name="register_model", version=version_string
        )

        rai_constructor_component = load_component(
            client=ml_client, name="rai_insights_constructor", version=version_string
        )

        rai_explanation_component = load_component(
            client=ml_client, name="rai_insights_explanation", version=version_string
        )

        rai_causal_component = load_component(
            client=ml_client, name="rai_insights_causal", version=version_string
        )

        rai_counterfactual_component = load_component(
            client=ml_client, name="rai_insights_counterfactual", version=version_string
        )

        rai_erroranalysis_component = load_component(
            client=ml_client, name="rai_insights_erroranalysis", version=version_string
        )

        rai_gather_component = load_component(
            client=ml_client, name="rai_insights_gather", version=version_string
        )

        @dsl.pipeline(
            compute="cpucluster",
            description="Submission of classification pipeline from Python",
            experiment_name=f"test_classification_pipeline_from_python_{version_string}",
        )
        def rai_classification_pipeline(
            target_column_name,
            train_data,
            test_data,
        ):
            train_job = train_log_reg_component(
                target_column_name=target_column_name, training_data=train_data
            )

            register_job = register_model_component(
                model_input_path=train_job.outputs.model_output,
                model_base_name="test_classification_pipeline_from_python",
                model_name_suffix=-1,  # Should be default
            )

            create_rai_job = rai_constructor_component(
                title="Run built from Python",
                task_type="classification",
                model_info_path=register_job.outputs.model_info_output_path,
                train_dataset=train_data,
                test_dataset=test_data,
                target_column_name=target_column_name,
                categorical_column_names='["Race", "Sex", "Workclass", "Marital Status", "Occupation", "Country"]',
                maximum_rows_for_test_dataset=5000,  # Should be default
                classes="[]",  # Should be default
            )

            explain_job = rai_explanation_component(
                comment="Insert text here",
                rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            )

            causal_job = rai_causal_component(
                rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
                treatment_features='["Age", "Sex"]',
                heterogeneity_features='["Marital Status"]',
                nuisance_model="linear",  # Should be default
                heterogeneity_model="linear",  # Should be default
                alpha=0.05,  # Should be default
                upper_bound_on_cat_expansion=50,  # Should be default
                treatment_cost="0",  # Should be default
                min_tree_leaf_samples=2,  # Should be default
                max_tree_depth=2,  # Should be default
                skip_cat_limit_checks=False,  # Should be default
                categories="auto",  # Should be default
                n_jobs=1,  # Should be default
                verbose=1,  # Should be default
                random_state="None",  # Should be default
            )

            counterfactual_job = rai_counterfactual_component(
                rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
                total_cfs=10,  # Bug filed - should be total_CFs
                desired_class="opposite",
                method="random",  # Should be default
                desired_range="[]",  # Should be default
                permitted_range="{}",  # Should be default
                features_to_vary="all",  # Should be default
                feature_importance=True,  # Should be default
            )

            erroranalysis_job = rai_erroranalysis_component(
                rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
                filter_features='["Race", "Sex"]',
                max_depth=3,  # Should be default
                num_leaves=31,  # Should be default
                min_child_samples=20,  # Should be default
            )

            rai_gather_job = rai_gather_component(
                constructor=create_rai_job.outputs.rai_insights_dashboard,
                insight_1=explain_job.outputs.explanation,
                insight_2=causal_job.outputs.causal,
                insight_3=counterfactual_job.outputs.counterfactual,
                insight_4=erroranalysis_job.outputs.error_analysis,
            )

            rai_gather_job.outputs.dashboard.mode = "upload"
            rai_gather_job.outputs.ux_json.mode = "upload"

            return {
                "dashboard": rai_gather_job.outputs.dashboard,
                "ux_json": rai_gather_job.outputs.ux_json,
            }

        pipeline_job = rai_classification_pipeline(
            target_column_name="income",
            train_data=Input(
                type="uri_file",
                path=f"adult_train_pq:{version_string}",
                mode="download",
            ),
            test_data=Input(
                type="uri_file", path=f"adult_test_pq:{version_string}", mode="download"
            ),
        )

        # Workaround to enable the download
        rand_path = str(uuid.uuid4())
        pipeline_job.outputs.dashboard = Output(
            path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
            mode="upload",
            type="uri_folder",
        )
        pipeline_job.outputs.ux_json = Output(
            path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/ux_json/",
            mode="upload",
            type="uri_folder",
        )

        # Send it
        pipeline_job = submit_and_wait(ml_client, pipeline_job)
        assert pipeline_job is not None

        # Try some downloads
        with tempfile.TemporaryDirectory() as dashboard_path:
            ml_client.jobs.download(
                pipeline_job.name, download_path=dashboard_path, output_name="dashboard"
            )
            expected_path = pathlib.Path(dashboard_path) / 'named-outputs' / 'dashboard'
            rai_i = RAIInsights.load(expected_path)
            assert rai_i is not None

    def test_fetch_registered_model_component(
        self, ml_client, component_config, registered_adult_model_id
    ):
        version_string = component_config["version"]

        fetch_model_component = load_component(
            client=ml_client, name="fetch_registered_model", version=version_string
        )

        rai_constructor_component = load_component(
            client=ml_client, name="rai_insights_constructor", version=version_string
        )

        # Pipeline skips on analysis; relies on the constructor component verifying the model works
        @dsl.pipeline(
            compute="cpucluster",
            description="Test of Fetch Model component",
            experiment_name=f"test_fetch_registered_model_component_{version_string}",
        )
        def fetch_analyse_registered_model(model_id, train_data, test_data):
            fetch_model_job = fetch_model_component(model_id=model_id)

            _ = rai_constructor_component(
                title="Run built from DSL",
                task_type="classification",
                model_info_path=fetch_model_job.outputs.model_info_output_path,
                train_dataset=train_data,
                test_dataset=test_data,
                target_column_name="income",
                categorical_column_names='["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
                maximum_rows_for_test_dataset=5000,
                classes="[]",  # Should be default value
            )

        insights_pipeline_job = fetch_analyse_registered_model(
            model_id=registered_adult_model_id,
            train_data=Input(
                type="uri_file",
                path=f"adult_train_pq:{version_string}",
                mode="download",
            ),
            test_data=Input(
                type="uri_file", path=f"adult_test_pq:{version_string}", mode="download"
            ),
        )

        # Send it
        insights_pipeline_job = submit_and_wait(ml_client, insights_pipeline_job)
        assert insights_pipeline_job is not None
