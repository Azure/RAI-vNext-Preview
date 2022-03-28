# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging
import subprocess
import time

import requests

import pandas as pd

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
            if check_for_ready_string(line):
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

    def _call_model_and_extract(self, input_df: pd.DataFrame, target: str):
        payload = input_df.to_json(orient="split")
        # _logger.info("Payload: {0}".format(payload))
        headers = {"Content-Type": "application/json"}
        r = requests.post(
            "http://127.0.0.1:5000/invocations",
            headers=headers,
            data=payload,
            timeout=100,
        )
        # _logger.info("Call to model completed: {0}".format(r.text))
        decoded = json.loads(r.text)
        # _logger.info(f"Decoded response: {decoded}")
        return decoded[target]

    def predict(self, input_df: pd.DataFrame):
        return self._call_model_and_extract(input_df, "pred")

    def predict_proba(self, input_df: pd.DataFrame):
        return self._call_model_and_extract(input_df, "pred_proba")
