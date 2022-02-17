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

    def _load_model(self):
        if self._model is None:
            self._model = mlflow.sklearn.load_model()


    @staticmethod
    def wrap_mlflow_model(target_mlflow_dir: str):
        my_dir = tempfile.mkdtemp() # Won't be cleaned up
        _logger.info("Target directory: {0}".format(my_dir))
        shutil.copytree(target_mlflow_dir, my_dir, dirs_exist_ok=True)

        target_mlflow_path = pathlib.Path(target_mlflow_dir)
        mlflow_dirname = target_mlflow_path.parts[-1]
        wrapped_dirname = os.path.join(my_dir, mlflow_dirname)
        print("---###---###---")
        print_dir_tree(wrapped_dirname)
        print("---###---###---")

        target_pickle_file = os.path.join(wrapped_dirname, 'model.pkl')
        os.remove(target_pickle_file)

        wrapped_model = ModelWrapper(os.path.abspath(target_mlflow_dir))

        cloudpickle.dump(wrapped_model, target_pickle_file)

        return wrapped_dirname
        

