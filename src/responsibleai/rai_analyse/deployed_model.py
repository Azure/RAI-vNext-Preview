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
            _logger.info(line)
            if "Listening at: http://127.0.0.1:5000" in line:
                break
            if self._server.returncode is not None:
                # Server has crashed
                raise RuntimeError("MLFlow server has crashed")

        _logger.info("MLFlow model deployed")

    def __exit__(self, exception_type, exception_value, exception_traceback):
        _logger.info("Killing server process")
        self._server.kill()
        _logger.info("Process killed")
        _logger.info("Remaining output")
        for line in self._server.stdout:
            _logger.info(line)
        _logger.info("End of process output")

    def predict(self, input_df: pd.DataFrame):
        payload = input_df.to_json(orient="split")
        headers = {"Content-Type": "application/json"}
        r = requests.post(
            "http://127.0.0.1:5000/invocations",
            headers=headers,
            data=payload,
            timeout=100,
        )
        return r.text