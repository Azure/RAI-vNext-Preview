# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging
import os
import subprocess
import tempfile
import time

import requests

import pandas as pd

from azureml.core import Workspace

from rai_component_utilities import (
    print_dir_tree,
    load_dataset,
    fetch_model_id,
    load_mlflow_model,
    download_model_to_dir,
)

from model_wrapper import ModelWrapper

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def check_for_ready_string(log_output: str):
    ready_strings = [
        "Serving on http://127",
        "Booting worker with pid",
        "Listening at: http://127.0.0.1:5000",
    ]

    result = False
    for rs in ready_strings:
        if rs in log_output:
            result = True
    return result


class DeployedModelLoader:
    def __init__(self, workspace: Workspace, model_id: str):
        self._sub_id = workspace.subscription_id
        self._resource_group = workspace.resource_group
        self._workspace_name = workspace._workspace_name
        self._model_id = model_id
        self._unwrapped_model_dir = tempfile.mkdtemp()

    def __enter__(self):
        self._server = None

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self._server is not None:
            _logger.info("Sending SIGTERM to server process")
            self._server.terminate()
            time.sleep(5)
            _logger.info("Sending SIGKILL to server process")
            self._server.kill()
            _logger.info("Process killed")
        else:
            _logger.info("No server found")

    def _call_model_and_extract(self, input_df: pd.DataFrame, target: str):
        payload = input_df.to_json(orient="split")
        _logger.info("Payload: {0}".format(payload))
        headers = {"Content-Type": "application/json"}
        r = requests.post(
            "http://127.0.0.1:5000/invocations",
            headers=headers,
            data=payload,
            timeout=100,
        )
        # _logger.info("Call to model completed: {0}".format(r.text))
        decoded = json.loads(r.text)
        _logger.info(f"Decoded response: {decoded}")
        return decoded[target]

    def load(self, path: str):
        _logger.info(f"Ignoring supplied path: {path}")
        _logger.info("Creating workspace object")
        workspace = Workspace(
            self._sub_id,
            self._resource_group,
            self._workspace_name,
        )

        _logger.info("Downloading mlflow model from AzureML")
        download_model_to_dir(workspace, self._model_id, self._unwrapped_model_dir)
        model_name = self._model_id.split(":")[0]

        _logger.info("Trying to create wrapped model")
        self._target_model_dir = ModelWrapper.wrap_mlflow_model(
            os.path.join(self._unwrapped_model_dir, model_name)
        )

        _logger.info("Starting mlflow process")
        launch_args = [
            "mlflow",
            "models",
            "serve",
            "--model-uri",
            self._target_model_dir,
        ]
        self._server = subprocess.Popen(
            args=launch_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        for line in self._server.stdout:
            _logger.info(line.strip())
            if check_for_ready_string(line):
                break
            if self._server.returncode is not None:
                # Server has crashed
                raise RuntimeError("MLFlow server has crashed")
        _logger.info("Server coming up.... pausing")
        time.sleep(10)

        _logger.info("MLFlow model deployed")
        return self

    def save(self, model, path):
        _logger.info("Not saving model - already in AzureML")
        pass

    def predict(self, input_df: pd.DataFrame):
        return self._call_model_and_extract(input_df, "pred")

    def predict_proba(self, input_df: pd.DataFrame):
        return self._call_model_and_extract(input_df, "pred_proba")

    def score(self, input_df: pd.DataFrame):
        return self._call_model_and_extract(input_df, "score")
