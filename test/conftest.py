# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import pytest
import time

from azure.identity import DefaultAzureCredential

from azure.ml import MLClient
from azure.ml import dsl
from azure.ml.entities import JobInput, load_component
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

    train_component = load_component(
        client=ml_client, name="TrainLogisticRegressionForRAI", version=version_string
    )
    register_component = load_component(
        client=ml_client, name="RegisterModel", version=version_string
    )
    adult_train_pq = JobInput(path=f"Adult_Train_PQ:{version_string}")

    @dsl.pipeline(
        compute="cpucluster",
        description="Register Common Model for Adult",
        experiment_name="Fixture_Common_Adult_Model",
    )
    def my_training_pipeline(target_column_name, training_data):
        trained_model = train_component(
            target_column_name=target_column_name, training_data=training_data
        )

        _ = register_component(
            model_input_path=trained_model.outputs.model_output,
            model_base_name=model_name,
            model_name_suffix=model_name_suffix,
        )

        return {}

    training_pipeline = my_training_pipeline("income", adult_train_pq)

    training_pipeline_job = submit_and_wait(ml_client, training_pipeline)
    assert training_pipeline_job is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    return expected_model_id


@pytest.fixture(scope="session")
def registered_boston_model_id(ml_client, component_config):
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_boston"

    train_component = load_component(
        client=ml_client, name="TrainBostonForRAI", version=version_string
    )
    register_component = load_component(
        client=ml_client, name="RegisterModel", version=version_string
    )
    boston_train_pq = JobInput(path=f"Boston_Train_PQ:{version_string}")

    @dsl.pipeline(
        compute="cpucluster",
        description="Register Common Model for Boston",
        experiment_name="Fixture_Common_Boston_Model",
    )
    def my_training_pipeline(target_column_name, training_data):
        trained_model = train_component(
            target_column_name=target_column_name,
            training_data=training_data,
            categorical_features="[]",
            continuous_features='["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE","DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]',
        )

        _ = register_component(
            model_input_path=trained_model.outputs.model_output,
            model_base_name=model_name,
            model_name_suffix=model_name_suffix,
        )

        return {}

    training_pipeline = my_training_pipeline("y", boston_train_pq)

    training_pipeline_job = submit_and_wait(ml_client, training_pipeline)
    assert training_pipeline_job is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    return expected_model_id
