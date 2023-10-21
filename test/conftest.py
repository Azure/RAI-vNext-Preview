# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import time
from test.constants_for_test import Timeouts
from test.utilities_for_test import submit_and_wait

import pytest
from azure.ai.ml import automl, Input, MLClient, dsl, Output
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


class Components:
    def __init__(self, ml_client: MLClient, version_string: str):
        self.fetch_model = ml_client.components.get(
            name="fetch_registered_model", version=version_string
        )

        self.tabular_to_parquet = ml_client.components.get(
            name="convert_tabular_to_parquet", version=version_string
        )

        self.rai_constructor = ml_client.components.get(
            name="rai_insights_constructor", version=version_string
        )

        self.rai_explanation = ml_client.components.get(
            name="rai_insights_explanation", version=version_string
        )

        self.rai_gather = ml_client.components.get(
            name="rai_insights_gather", version=version_string
        )

        self.rai_causal = ml_client.components.get(
            name="rai_insights_causal", version=version_string
        )

        self.rai_counterfactual = ml_client.components.get(
            name="rai_insights_counterfactual", version=version_string
        )

        self.rai_erroranalysis = ml_client.components.get(
            name="rai_insights_erroranalysis", version=version_string
        )

        self.train_adult = ml_client.components.get(
            name="train_logistic_regression_for_rai",
            version=version_string,
        )

        self.register_model = ml_client.components.get(
            name="register_model", version=version_string
        )


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
def rai_components(ml_client, component_config):
    version_string = component_config["version"]

    return Components(ml_client, version_string)


@pytest.fixture(scope="session")
def registered_adult_model_id(ml_client, component_config):
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_adult"

    train_component = ml_client.components.get(
        name="train_logistic_regression_for_rai",
        version=version_string,
    )
    register_component = ml_client.components.get(
        name="register_model", version=version_string
    )
    adult_train = Input(
        type="mltable", path=f"adult_train:{version_string}", mode="download"
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

    training_pipeline = my_training_pipeline("income", adult_train)

    training_pipeline_job = submit_and_wait(ml_client, training_pipeline)
    assert training_pipeline_job is not None

    # download best model and then register from local?

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    return expected_model_id


@pytest.fixture(scope="session")
def registered_boston_model_id(ml_client, component_config):
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_boston"

    train_component = ml_client.components.get(
        name="train_boston_for_rai", version=version_string
    )
    register_component = ml_client.components.get(
        name="register_model", version=version_string
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
            continuous_features='["CRIM", "ZN", "INDUS", "CHAS", "NOX", ' +
                                '"RM", "AGE","DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]',
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

    training_pipeline_job = submit_and_wait(ml_client, training_pipeline)
    assert training_pipeline_job is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"
    return expected_model_id


@pytest.fixture(scope="session")
def registered_electro_model_id(ml_client, component_config):
    version_string = component_config["version"]

    model_name_suffix = int(time.time())
    model_name = "common_fetch_model_electro"

    electro_train = Input(
        type="mltable", path=f"electro_train:{version_string}", mode="download"
    )

    experiment_name = f"Fixture_Common_Electro_Model_{version_string}"
    job_name = f"Train_Electro_Forecasting_Model_{model_name_suffix}"
    training_job = automl.forecasting(
        training_data=electro_train,
        target_column_name="usage",
        outputs={"best_model": Output(type=AssetTypes.CUSTOM_MODEL)},
        primary_metric="normalized_root_mean_squared_error",
        n_cross_validations='auto',
        experiment_name=experiment_name,
        name=job_name,
    )
    training_job.set_forecast_settings(
        forecast_horizon=24,
        time_column_name="datetime",
        time_series_id_column_names=["group_id", "customer_id"],
    )
    training_job.set_training(
        enable_dnn_training=False,
        enable_model_explainability=True,
        allowed_training_algorithms=["LightGBM"],
    )
    training_job.set_limits(
        timeout_minutes=30,
        max_trials=5,
        max_concurrent_trials=3,
        max_nodes=3,
    )

    training_job.compute = "cpucluster"
    assert submit_and_wait(ml_client, training_job) is not None

    expected_model_id = f"{model_name}_{model_name_suffix}:1"

    ml_client.jobs.download(
        name=job_name,
        download_path="test-outputs/",
        all=True
    )
    model = Model(
        path = "./test-outputs/named-outputs/best_model",
        type=AssetTypes.MLFLOW_MODEL,
        name=model_name + "_" + str(model_name_suffix),
    )
    ml_client.models.create_or_update(model)

    return expected_model_id