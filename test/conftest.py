# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import pytest
import time

from azure.identity import DefaultAzureCredential

from azure.ml import MLClient
from azure.ml.entities import JobInput
from azure.ml.entities import CommandComponent, PipelineJob

from test.utilities_for_test import submit_and_wait


@pytest.fixture(scope="session")
def component_config():
    config_file = "component_config.json"

    with open(config_file, "r") as cf:
        result = json.load(cf)

    return result


@pytest.fixture(scope="session")
def workspace_config():
    ws_config_file = "config.json"

    with open(ws_config_file) as cf:
        result = json.load(cf)

    return result


@pytest.fixture(scope="session")
def ml_client(workspace_config):
    client = MLClient(
        # For local testing, may need exclude_shared_token_cache_credential=True
        credential=DefaultAzureCredential(),
        subscription_id=workspace_config["subscription_id"],
        resource_group_name=workspace_config["resource_group"],
        workspace_name=workspace_config["workspace_name"],
        logging_enable=True,
    )

    return client


@pytest.fixture(scope="session")
def registered_adult_model_id(ml_client, component_config):
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_adult"

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
    train_job = CommandComponent(
        component=f"TrainLogisticRegressionForRAI:{version_string}",
        inputs=train_job_inputs,
        outputs=train_job_outputs,
    )

    # The model registration job
    register_job_inputs = {
        "model_input_path": "${{jobs.train-model-job.outputs.model_output}}",
        "model_base_name": model_name,
        "model_name_suffix": model_name_suffix,
    }
    register_job_outputs = {"model_info_output_path": None}
    register_job = CommandComponent(
        component=f"RegisterModel:{version_string}",
        inputs=register_job_inputs,
        outputs=register_job_outputs,
    )
    # Assemble into a pipeline
    register_pipeline = PipelineJob(
        experiment_name=f"Register_Adult_Model_Fixture_{version_string}",
        description="Python submitted Adult model registration",
        jobs={
            "train-model-job": train_job,
            "register-model-job": register_job,
        },
        inputs=pipeline_inputs,
        outputs=register_job_outputs,
        compute="cpucluster",
    )

    # Send it
    register_pipeline_job = submit_and_wait(ml_client, register_pipeline)
    assert register_pipeline_job is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    return expected_model_id


@pytest.fixture(scope="session")
def registered_boston_model_id(ml_client, component_config):
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_boston"

    # Configure the global pipeline inputs:
    pipeline_inputs = {
        "target_column_name": "y",
        "my_training_data": JobInput(dataset=f"Boston_Train_PQ:{version_string}"),
        "my_test_data": JobInput(dataset=f"Boston_Test_PQ:{version_string}"),
    }

    # Specify the training job
    train_job_inputs = {
        "target_column_name": "${{parent.inputs.target_column_name}}",
        "training_data": "${{parent.inputs.my_training_data}}",
        "categorical_features": "[]",
        "continuous_features": '["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE","DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]',
    }
    train_job_outputs = {"model_output": None}
    train_job = CommandComponent(
        component=f"TrainBostonForRAI:{version_string}",
        inputs=train_job_inputs,
        outputs=train_job_outputs,
    )

    # The model registration job
    register_job_inputs = {
        "model_input_path": "${{parent.jobs.train-model-job.outputs.model_output}}",
        "model_base_name": model_name,
        "model_name_suffix": model_name_suffix,
    }
    register_job_outputs = {"model_info_output_path": None}
    register_job = CommandComponent(
        component=f"RegisterModel:{version_string}",
        inputs=register_job_inputs,
        outputs=register_job_outputs,
    )
    # Assemble into a pipeline
    register_pipeline = PipelineJob(
        experiment_name=f"Register_Boston_Model_Fixture_{version_string}",
        description="Python submitted Boston model registration",
        jobs={
            "train-model-job": train_job,
            "register-model-job": register_job,
        },
        inputs=pipeline_inputs,
        outputs=register_job_outputs,
        compute="azureml:cpucluster",
    )

    # Send it
    register_pipeline_job = submit_and_wait(ml_client, register_pipeline)
    assert register_pipeline_job is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    return expected_model_id
