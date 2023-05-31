import json
import os

import pytest
from raiutils.exceptions import UserConfigValidationException
from src.responsibleai.rai_analyse.constants import DashboardInfo
from src.responsibleai.rai_analyse.rai_component_utilities import fetch_model_id


class TestFetchModelId:

    @pytest.mark.skip(reason="Skipping this test for now due to import issues, will enable once fixed")
    @pytest.mark.parametrize("model_info_path, model_info, expected_model_id", [("", {"id": "abc"}, "abc")])
    def test_fetch_model_id(self, model_info_path, model_info, expected_model_id):
        model_info_path = os.path.join(model_info_path, DashboardInfo.MODEL_INFO_FILENAME)
        with open(model_info_path, "w") as json_file:
            json.dump(model_info, json_file)
        assert fetch_model_id(model_info_path) == expected_model_id

    @pytest.mark.skip(reason="Skipping this test for now due to import issues, will enable once fixed")
    @pytest.mark.parametrize("model_info_path, model_info", [("", {"otherKeys": "abc"})])
    def test_fetch_model_id_invalid_input(self, model_info_path, model_info):
        model_info_path = os.path.join(model_info_path, DashboardInfo.MODEL_INFO_FILENAME)
        with open(model_info_path, "w") as json_file:
            json.dump(model_info, json_file)
        with pytest.raises(
            UserConfigValidationException,
                match=f"Invalid input, expecting key {DashboardInfo.MODEL_ID_KEY} to exist in the input json"):
            fetch_model_id(model_info_path)
