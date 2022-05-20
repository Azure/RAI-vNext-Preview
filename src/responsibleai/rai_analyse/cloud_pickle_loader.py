# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

from pathlib import Path

import cloudpickle


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class CloudPickleLoader:
    SAVE_FILE = "model.pkl"

    def __init__(self):
        pass

    def load(self, path: str):
        _logger.info(f"Loading from: {path}")

        target_path = Path(path) / CloudPickleLoader.SAVE_FILE

        with open(target_path, "rb") as f:
            model = cloudpickle.load(f)

        return model

    def save(self, model, path):
        _logger.info(f"Saving to: {path}")

        target_path = Path(path) / CloudPickleLoader.SAVE_FILE

        with open(target_path, "wb") as f:
            model = cloudpickle.dump(model, f)
