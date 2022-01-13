# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

from azure.ml import MLClient
from azure.ml.entities import JobInput
from azure.ml.entities import ComponentJob, PipelineJob

from utilities_for_test import submit_and_wait

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class TestRAIGatherErrors:
    def test_tool_component_mismatch(self, ml_client: MLClient, component_config):
        # Checks that components from different constructors can't be mixed
        # This is for the Adult dataset
        version_string = component_config["version"]

        # Configure the global pipeline inputs:
        pipeline_inputs = {
            "target_column_name": "income",
            "my_training_data": JobInput(dataset=f"Adult_Train_PQ:{version_string}"),
            "my_test_data": JobInput(dataset=f"Adult_Test_PQ:{version_string}"),
        }

        # Specify the training job
        train_job_inputs = {
            "target_column_name": "${{inputs.target_column_name}}",
            "training_data": "${{inputs.my_training_data}}",
        }
        train_job_outputs = {"model_output": None}
        train_job = ComponentJob(
            component=f"TrainLogisticRegressionForRAI:{version_string}",
            inputs=train_job_inputs,
            outputs=train_job_outputs,
        )

        # The model registration job
        register_job_inputs = {
            "model_input_path": "${{jobs.train-model-job.outputs.model_output}}",
            "model_base_name": "notebook_registered_logreg",
        }
        register_job_outputs = {"model_info_output_path": None}
        # Register twice (the component is non-deterministic so we can be
        # sure output won't be reused)
        register_job_1 = ComponentJob(
            component=f"RegisterModel:{version_string}",
            inputs=register_job_inputs,
            outputs=register_job_outputs,
        )
        register_job_2 = ComponentJob(
            component=f"RegisterModel:{version_string}",
            inputs=register_job_inputs,
            outputs=register_job_outputs,
        )

        # Top level RAI Insights component
        create_rai_inputs = {
            "title": "Run built from Python",
            "task_type": "classification",
            "model_info_path": "${{jobs.register-model-job-1.outputs.model_info_output_path}}",
            "train_dataset": "${{inputs.my_training_data}}",
            "test_dataset": "${{inputs.my_test_data}}",
            "target_column_name": "${{inputs.target_column_name}}",
            "categorical_column_names": '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
        }
        create_rai_outputs = {"rai_insights_dashboard": None}

        # Have TWO dashboard constructors
        create_rai_1 = ComponentJob(
            component=f"RAIInsightsConstructor:{version_string}",
            inputs=create_rai_inputs,
            outputs=create_rai_outputs,
        )
        create_rai_inputs[
            "model_info_path"
        ] = "${{jobs.register-model-job-2.outputs.model_info_output_path}}"
        create_rai_2 = ComponentJob(
            component=f"RAIInsightsConstructor:{version_string}",
            inputs=create_rai_inputs,
            outputs=create_rai_outputs,
        )

        # Setup causal on constructor 1
        causal_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job-1.outputs.rai_insights_dashboard}}",
            "treatment_features": '["Age", "Sex"]',
            "heterogeneity_features": '["Marital Status"]',
        }
        causal_outputs = {"causal": None}
        causal_job = ComponentJob(
            component=f"RAIInsightsCausal:{version_string}",
            inputs=causal_inputs,
            outputs=causal_outputs,
        )

        # Setup counterfactual on constructor 2
        counterfactual_inputs = {
            "rai_insights_dashboard": "${{jobs.create-rai-job-2.outputs.rai_insights_dashboard}}",
            "total_CFs": "10",
            "desired_class": "opposite",
        }
        counterfactual_outputs = {"counterfactual": None}
        counterfactual_job = ComponentJob(
            component=f"RAIInsightsCounterfactual:{version_string}",
            inputs=counterfactual_inputs,
            outputs=counterfactual_outputs,
        )

        # Configure the gather component
        gather_inputs = {
            "constructor": "${{jobs.create-rai-job-1.outputs.rai_insights_dashboard}}",
            "insight_2": "${{jobs.causal-rai-job.outputs.causal}}",
            "insight_3": "${{jobs.counterfactual-rai-job.outputs.counterfactual}}",
        }
        gather_outputs = {"dashboard": None, "ux_json": None}
        gather_job = ComponentJob(
            component=f"RAIInsightsGather:{version_string}",
            inputs=gather_inputs,
            outputs=gather_outputs,
        )

        # Assemble into a pipeline
        pipeline_job = PipelineJob(
            experiment_name=f"XFAIL_tool_component_mismatch_{version_string}",
            description="Python submitted Adult",
            jobs={
                "train-model-job": train_job,
                "register-model-job-1": register_job_1,
                "register-model-job-2": register_job_2,
                "create-rai-job-1": create_rai_1,
                "create-rai-job-2": create_rai_2,
                "causal-rai-job": causal_job,
                "counterfactual-rai-job": counterfactual_job,
                "gather-job": gather_job,
            },
            inputs=pipeline_inputs,
            outputs=train_job_outputs,
            compute="cpucluster",
        )

        # Send it
        pipeline_job = submit_and_wait(ml_client, pipeline_job, "Failed")
        # Want to do more here, but there isn't anything useful in the returned job
        # Problem is, the job might have failed for some other reason
        assert pipeline_job is not None
