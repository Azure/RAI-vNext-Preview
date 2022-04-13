# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import os
from pathlib import Path
from typing import List
import uuid

import mlflow

import pandas as pd
import numpy as np

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

def ensure_list(input) -> List:
    if isinstance(input, list):
        _logger.info("input was list")
        return input
    else:
        _logger.info(f"Converting {type(input)} to list")
        return list(input)


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, target_mlflow_dir: str):
        assert os.path.isabs(target_mlflow_dir)
        self._target_mlflow_path = target_mlflow_dir
        self._model = None # Lazy load
        _logger.info("Created ModelWrapper for path: {0}".format(target_mlflow_dir))

    def _load_model(self):
        if self._model is None:
            _logger.info(f"Loading wrapped model with mlflow: {self._target_mlflow_path}")
            self._model = mlflow.sklearn.load_model(self._target_mlflow_path)

    def predict(self, context, X):
        self._load_model()
        _logger.info("Calling predict and predict_proba")
        preds = self._call_model('predict', X)
        pred_probas = self._call_model('predict_probas', X)
        result = {
            "pred": ensure_list(preds),
            "pred_proba": ensure_list(pred_probas),
        }
        return result

    def _call_model(self, method_name: str, X):
        self._load_model()
        if hasattr(self._model, method_name):
            method = getattr(self._model, method_name)
            return method(X)
        else:
            return []

    @staticmethod
    def wrap_mlflow_model(target_mlflow_dir: str):
        _logger.info("target_mlflow_dir: {0}".format(target_mlflow_dir))
        my_dir = str(Path('.').resolve() / str(uuid.uuid4()))  # Won't be cleaned up
        _logger.info("Target directory: {0}".format(my_dir))

        #mlflow_dirname = os.listdir(target_mlflow_dir)[0]
        #wrapped_dirname = os.path.join(my_dir, mlflow_dirname)
        mlflow_dirname = Path(target_mlflow_dir).resolve()
        conda_file = str(mlflow_dirname / 'conda.yaml')

        wrapped_model = ModelWrapper(str(mlflow_dirname))

        _logger.info("Invoking mlflow.pyfunc.save_model")
        mlflow.pyfunc.save_model(path=my_dir, python_model=wrapped_model, conda_env=conda_file)

        return my_dir
