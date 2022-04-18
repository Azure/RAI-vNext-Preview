# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from wsgiref.simple_server import server_version

import requests

import numpy as np
import pandas as pd

import mlflow

from azureml.core import Model, Run, Workspace

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
        self._server_pid = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._shutdown_server()

    def __del__(self):
        self._shutdown_server()

    def _shutdown_server(self):
        if self._server_pid is not None:
            _logger.info(f"Sending SIGTERM to server process {self._server_pid}")
            os.kill(self._server_pid, signal.SIGTERM)
            time.sleep(5)
            _logger.info("Sending SIGKILL to server process (assuming Unix)")
            os.kill(self._server_pid, signal.SIGKILL)
            _logger.info("Process killed")
            self._server_pid = None
        else:
            _logger.info("No server found")

    def _convert_to_json(self, input) -> str:
        result = ""
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.Series):
            result = input.to_json(orient="split")
        elif isinstance(input, np.ndarray):
            result = json.dumps(input.tolist())
        else:
            # See what we get
            result = json.dumps(input)

        return result


    def _call_model_and_extract(self, input_data, target: str):
        payload = self._convert_to_json(input_data)
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

        # Convert to numpy
        result = np.asarray(decoded[target])

        return result

    def load(self, path: str):
        _logger.info(f"Ignoring supplied path: {path}")
        _logger.info("Creating workspace object")
        workspace = Run.get_context().experiment.workspace
        # workspace = Workspace(
        #    subscription_id=self._sub_id,
        #    resource_group=self._resource_group,
        #    workspace_name=self._workspace_name,
        # )

        _logger.info("Downloading mlflow model from AzureML")
        self._download_model_to_dir(
            workspace, self._model_id, self._unwrapped_model_dir
        )
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
        server_process = subprocess.Popen(
            args=launch_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        for line in server_process.stdout:
            _logger.info(line.strip())
            if check_for_ready_string(line):
                break
            if server_process.returncode is not None:
                # Server has crashed
                raise RuntimeError("MLFlow server has crashed")
        _logger.info("Server coming up.... pausing")
        time.sleep(10)
        self._server_pid = server_process.pid

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

    def _download_model_to_dir(
        self, workspace: Workspace, model_id: str, target_path: str
    ) -> None:
        mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
        model = Model(workspace, id=model_id)
        model.download(target_dir=target_path, exist_ok=True)
