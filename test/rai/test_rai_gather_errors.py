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


class TestRAIGatherErrors:
    def test_tool_component_mismatch(
        self, ml_client: MLClient, component_config, rai_components
    ):
        # Checks that components from different constructors can't be mixed
        # This is for the Adult dataset
        version_string = component_config["version"]

        @dsl.pipeline(
            compute="cpucluster",
            description="Checks components from different constructors can't be used",
            experiment_name=f"TestRAIGatherErrors_test_tool_component_mismatch_{version_string}",
        )
        def test_constructor_mismatch(
            target_column_name,
            train_data,
            test_data,
        ):
            train_job = rai_components.train_adult(
                target_column_name=target_column_name, training_data=train_data
            )

            # Register twice (nondeterministically)
            model_base_name = "TestRAIGatherErrors_test_tool_component_mismatch"
            reg1_job = rai_components.register_model(
                model_input_path=train_job.outputs.model_output,
                model_base_name=model_base_name,
            )
            reg2_job = rai_components.register_model(
                model_input_path=train_job.outputs.model_output,
                model_base_name=model_base_name,
            )

            # Two RAI constructors
            construct1_job = rai_components.rai_constructor(
                title="Run built from DSL",
                task_type="classification",
                model_info_path=reg1_job.outputs.model_info_output_path,
                train_dataset=train_data,
                test_dataset=test_data,
                target_column_name=target_column_name,
                categorical_column_names='["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
                maximum_rows_for_test_dataset=5000,  # Should be default
                classes="[]",  # Should be default
            )

            construct2_job = rai_components.rai_constructor(
                title="Run built from DSL",
                task_type="classification",
                model_info_path=reg2_job.outputs.model_info_output_path,
                train_dataset=train_data,
                test_dataset=test_data,
                target_column_name=target_column_name,
                categorical_column_names='["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
                maximum_rows_for_test_dataset=5000,  # Should be default
                classes="[]",  # Should be default
            )

            # Setup causal for constructor 1
            causal1_job = rai_components.rai_causal(
                rai_insights_dashboard=construct1_job.outputs.rai_insights_dashboard,
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

            # Setup counterfactual for constructor 2
            counterfactual2_job = rai_components.counterfactual(
                rai_insights_dashboard=construct2_job.outputs.rai_insights_dashboard,
                total_cfs=10,  # Bug filed - should be total_CFs
                desired_class="opposite",
                method="random",  # Should be default
                desired_range="[]",  # Should be default
                permitted_range="{}",  # Should be default
                features_to_vary="all",  # Should be default
                feature_importance=True,  # Should be default
            )

            # Now a single gather components
            gather_job = rai_components.gather(
                constructor=construct1_job.outputs.rai_insights_dashboard,
                insight_1=causal1_job.outputs.causal,
                insight_2=counterfactual2_job.outputs.counterfactual,
            )

            gather_job.outputs.dashboard.mode = "upload"
            gather_job.outputs.ux_json.mode = "upload"

            return {
                "dashboard": gather_job.outputs.dashboard,
                "ux_json": gather_job.outputs.ux_json,
            }

        # Assemble into a pipeline
        pipeline_job = test_constructor_mismatch(
            target_column_name="income",
            train_data=JobInput(path=f"Adult_Train_PQ:{version_string}"),
            test_data=JobInput(path=f"Adult_Test_PQ:{version_string}"),
        )

        # Send it
        pipeline_job = submit_and_wait(ml_client, pipeline_job, "Failed")
        # Want to do more here, but there isn't anything useful in the returned job
        # Problem is, the job might have failed for some other reason
        assert pipeline_job is not None

    def test_multiple_tool_instances(
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

        # Setup two causal analyses
        causal_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "treatment_features": '["Age", "Sex"]',
            "heterogeneity_features": '["Marital Status"]',
        }
        causal_outputs = {"causal": None}
        causal_job_01 = CommandComponent(
            component=f"RAIInsightsCausal:{version_string}",
            inputs=causal_inputs,
            outputs=causal_outputs,
        )

        causal_inputs["treatment_cost"] = "[0.01, 0.02]"
        causal_job_02 = CommandComponent(
            component=f"RAIInsightsCausal:{version_string}",
            inputs=causal_inputs,
            outputs=causal_outputs,
        )

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
            "insight_1": "${{jobs.causal-rai-job-01.outputs.causal}}",
            "insight_2": "${{jobs.causal-rai-job-02.outputs.causal}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = CommandComponent(
            component=f"RAIInsightsGather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Pipeline to construct the RAI Insights
        insights_pipeline_job = PipelineJob(
            experiment_name=f"XFAIL_multiple_tool_instances_{version_string}",
            description="Expected failure due to multiple tool instances",
            jobs={
                "fetch-model-job": fetch_job,
                "create-rai-job": create_rai_job,
                "causal-rai-job-01": causal_job_01,
                "causal-rai-job-02": causal_job_02,
                "gather-job": gather_job,
            },
            inputs=pipeline_inputs,
            outputs=None,
            compute="cpucluster",
        )

        # Send it
        insights_pipeline_job = submit_and_wait(
            ml_client, insights_pipeline_job, "Failed"
        )
        assert insights_pipeline_job is not None
        # Unfortunately we can't check anything else right now
