# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import shutil
import os
import pathlib
import tempfile

import cloudpickle
import mlflow

from rai_component_utilities import print_dir_tree

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class ModelWrapper:
    def __init__(self, target_mlflow_path: str):
        self._target_mlflow_path = target_mlflow_path
        _logger.info("Created ModelWrapper for path: {0}".format(target_mlflow_path))

    def _load_model(self):
        if self._model is None:
            _logger.info("Loading wrapped model with mlflow")
            self._model = mlflow.sklearn.load_model(self._target_mlflow_path)

    def predict(self, X):
        _logger.info("Calling predict and predict_proba")
        preds = self._model.predict(X)
        pred_probas = self._model.predict_proba(X)
        result = {
            "pred": preds.to_json(orient="split"),
            "pred_proba": pred_probas.to_json(orient="split"),
        }
        return result

    @staticmethod
    def wrap_mlflow_model(target_mlflow_dir: str):
        _logger.info("target_mlflow_dir: {0}".format(target_mlflow_dir))
        my_dir = tempfile.mkdtemp()  # Won't be cleaned up
        _logger.info("Target directory: {0}".format(my_dir))
        shutil.copytree(target_mlflow_dir, my_dir, dirs_exist_ok=True)

        mlflow_dirname = os.listdir(target_mlflow_dir)[0]
        wrapped_dirname = os.path.join(my_dir, mlflow_dirname)

        # Want to cuckoo the pickle file
        target_pickle_file = os.path.join(wrapped_dirname, "model.pkl")
        os.remove(target_pickle_file)

        # Nest the actual MLFlow model inside this directory
        _logger.info("Nesting actual model")
        shutil.copytree(
            src=os.path.join(target_mlflow_dir, mlflow_dirname),
            dst=os.path.join(wrapped_dirname, 'internal_model'),
        )

        wrapped_model = ModelWrapper('internal_model')

        _logger.info("Pickling wrapped model")
        with open(target_pickle_file, mode="wb") as pkl_file:
            cloudpickle.dump(wrapped_model, pkl_file)

        print("####----####----####")
        print_dir_tree(wrapped_dirname)
        print("####----####----####")

        return wrapped_dirname
