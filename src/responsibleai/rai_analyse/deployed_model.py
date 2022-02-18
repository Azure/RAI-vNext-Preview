# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


import logging
import subprocess
import time

import requests

import pandas as pd

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class DeployedModel:
    def __init__(self, model_dir: str):
        self._target_model_dir = model_dir

    def __enter__(self):
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
            target_string = "Booting worker with pid" # "Listening at: http://127.0.0.1:5000"
            if target_string in line:
                break
            if self._server.returncode is not None:
                # Server has crashed
                raise RuntimeError("MLFlow server has crashed")

        _logger.info("MLFlow model deployed")
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        _logger.info("Sending SIGTERM to server process")
        self._server.terminate()
        time.sleep(5)
        _logger.info("Sending SIGKILL to server process")
        self._server.kill()
        _logger.info("Process killed")

    def predict(self, input_df: pd.DataFrame):
        payload = input_df.to_json(orient="split")
        _logger.info("Payload: {0}".format(payload))
        headers = {"Content-Type": "application/json"}
        r = requests.post(
            "http://127.0.0.1:5000/invocations",
            headers=headers,
            data=payload,
            timeout=100,
        )
        _logger.info("Call to model completed: {0}".format(r.text))
        return r.text