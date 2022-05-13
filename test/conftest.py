# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging
import pytest
import time

from azure.identity import DefaultAzureCredential

from azure.ai.ml import MLClient, dsl, Input
from azure.ai.ml.entities import load_component
from azure.ai.ml.entities import CommandComponent, PipelineJob

from test.utilities_for_test import submit_and_wait
from test.constants_for_test import Timeouts


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class Components:
    def __init__(self, ml_client: MLClient, version_string: str):
        _logger.info("Initialising Components class")
        self.fetch_model = load_component(
            client=ml_client, name="fetch_registered_model", version=version_string
        )

        self.tabular_to_parquet = load_component(
            client=ml_client, name="convert_tabular_to_parquet", version=version_string
        )

        self.rai_constructor = load_component(
            client=ml_client, name="rai_insights_constructor", version=version_string
        )

        self.rai_explanation = load_component(
            client=ml_client, name="rai_insights_explanation", version=version_string
        )

        self.rai_gather = load_component(
            client=ml_client, name="rai_insights_gather", version=version_string
        )

        self.rai_causal = load_component(
            client=ml_client, name="rai_insights_causal", version=version_string
        )

        self.rai_counterfactual = load_component(
            client=ml_client, name="rai_insights_counterfactual", version=version_string
        )

        self.rai_erroranalysis = load_component(
            client=ml_client, name="rai_insights_erroranalysis", version=version_string
        )

        self.train_adult = load_component(
            client=ml_client,
            name="train_logistic_regression_for_rai",
            version=version_string,
        )

        self.register_model = load_component(
            client=ml_client, name="register_model", version=version_string
        )
        _logger.info("Components class initialised")


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
    _logger.info("Creating MLClient")
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
def rai_components(ml_client, component_config):
    version_string = component_config["version"]

    return Components(ml_client, version_string)


@pytest.fixture(scope="session")
def registered_adult_model_id(ml_client, component_config):
    _logger.info("Registering common adult model")
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_adult"

    train_component = load_component(
        client=ml_client,
        name="train_logistic_regression_for_rai",
        version=version_string,
    )
    register_component = load_component(
        client=ml_client, name="register_model", version=version_string
    )
    adult_train_pq = Input(
        type="uri_file", path=f"adult_train_pq:{version_string}", mode="download"
    )

    @dsl.pipeline(
        compute="cpucluster",
        description="Register Common Model for Adult",
        experiment_name="Fixture_Common_Adult_Model",
    )
    def my_training_pipeline(target_column_name, training_data):
        trained_model = train_component(
            target_column_name=target_column_name, training_data=training_data
        )
        trained_model.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

        register_job = register_component(
            model_input_path=trained_model.outputs.model_output,
            model_base_name=model_name,
            model_name_suffix=model_name_suffix,
        )
        register_job.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

        return {}

    training_pipeline = my_training_pipeline("income", adult_train_pq)
    _logger.info("Simple Adult model pipeline constructed")

    training_pipeline_job = submit_and_wait(ml_client, training_pipeline)
    assert training_pipeline_job is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    _logger.info(f"Registered simple Adult model as {expected_model_id}")
    return expected_model_id


@pytest.fixture(scope="session")
def registered_boston_model_id(ml_client, component_config):
    _logger.info("Registering simplified Boston model")
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_boston"

    train_component = load_component(
        client=ml_client, name="train_boston_for_rai", version=version_string
    )
    register_component = load_component(
        client=ml_client, name="register_model", version=version_string
    )
    boston_train_pq = Input(
        type="uri_file", path=f"boston_train_pq:{version_string}", mode="download"
    )

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
        trained_model.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

        register_job = register_component(
            model_input_path=trained_model.outputs.model_output,
            model_base_name=model_name,
            model_name_suffix=model_name_suffix,
        )
        register_job.set_limits(timeout=Timeouts.DEFAULT_TIMEOUT)

        return {}

    training_pipeline = my_training_pipeline("y", boston_train_pq)
    _logger.info("Simple Boston model pipeline constructed")

    training_pipeline_job = submit_and_wait(ml_client, training_pipeline)
    assert training_pipeline_job is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    _logger.info(f"Registered simple Boston model as {expected_model_id}")
    return expected_model_id
