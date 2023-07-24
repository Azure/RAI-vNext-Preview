import json
import os

import pytest
from raiutils.exceptions import UserConfigValidationException

from src.responsibleai.rai_analyse.constants import DashboardInfo
from src.responsibleai.rai_analyse.rai_component_utilities import \
    fetch_model_id


@pytest.fixture
def temp_dir(tmpdir):
    # Create a temporary directory for testing
    return str(tmpdir)


class TestFetchModelId:
    @pytest.mark.parametrize(
        "model_info", [
            ({"id": "abc123"}),  # Valid input case
            ({"id": 123}),  # Valid input case with integer model ID
            ({"id": None})  # Valid input case with None model ID
        ]
    )
    def test_fetch_model_id(self, temp_dir, model_info):
        model_info_path = os.path.join(temp_dir, DashboardInfo.MODEL_INFO_FILENAME)
        with open(model_info_path, "w") as json_file:
            json.dump(model_info, json_file)
        assert fetch_model_id(temp_dir) == model_info["id"]

    @pytest.mark.parametrize("model_info", [({"otherKeys": "abc"})])
    def test_fetch_model_id_invalid_input(self, temp_dir, model_info):
        model_info_path = os.path.join(temp_dir, DashboardInfo.MODEL_INFO_FILENAME)
        with open(model_info_path, "w") as json_file:
            json.dump(model_info, json_file)
        with pytest.raises(
            UserConfigValidationException,
                match=f"Invalid input, expecting key {DashboardInfo.MODEL_ID_KEY} to exist in the input json"):
            fetch_model_id(temp_dir)
            
    @pytest.mark.parametrize("model_info_")
    def test_fetch_model_id_invalid_model_path(self, temp_dir):
        model_info_path = os.path.join(temp_dir, DashboardInfo.MODEL_INFO_FILENAME)
        with pytest.raises(
            UserConfigValidationException,
                match=f"Failed to open {model_info_path}. Please ensure the model path is correct."):
            fetch_model_id("model_info_")
