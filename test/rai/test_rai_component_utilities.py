import hashlib
import json
import os
import re

import pytest
from raiutils.exceptions import UserConfigValidationException

from src.responsibleai.rai_analyse.constants import DashboardInfo
from src.responsibleai.rai_analyse.rai_component_utilities import (
    InstallCondaEnv, _get_mlflow_model_server_command,
    _get_model_conda_env_info, _get_model_conda_env_name, fetch_model_id,
    load_mlflow_model)


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

    def test_fetch_model_id_invalid_model_path(self):
        model_info_path = os.path.join("invalid_model_path", DashboardInfo.MODEL_INFO_FILENAME)
        with pytest.raises(
            UserConfigValidationException,
                match=re.escape(f'Failed to open {model_info_path}. Please ensure the model path is correct.')):
            fetch_model_id("invalid_model_path")


def test_install_automl_conda_env_uses_legacy_pip_resolver(monkeypatch):
    captured = {}
    tracking_variables = (
        "MLFLOW_EXPERIMENT_ID",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_ID",
        "MLFLOW_TRACKING_URI",
    )
    for variable in tracking_variables:
        monkeypatch.setenv(variable, "azureml-value")

    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.shutil.copytree",
        lambda source, destination: None,
    )
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.os.walk",
        lambda path: [],
    )
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities."
        "_get_model_conda_env_info",
        lambda path: ("mlflow-test", True),
    )

    def check_output(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return b"success"

    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.subprocess.check_output",
        check_output,
    )

    installer = InstallCondaEnv(
        use_separate_conda_env=True,
        conda_file=None,
        model_path="model",
    )
    installer.install()

    assert captured["command"][:3] == ["mlflow", "models", "prepare-env"]
    assert installer.local_model_path == captured["command"][4]
    assert installer.conda_env_name == "mlflow-test"
    assert captured["env"]["PIP_USE_DEPRECATED"] == "legacy-resolver"
    assert captured["env"]["PATH"] == os.environ["PATH"]
    assert all(
        variable not in captured["env"] for variable in tracking_variables)


def test_install_generic_conda_env_uses_default_pip_resolver(monkeypatch):
    captured = {}
    monkeypatch.setenv("PIP_USE_DEPRECATED", "legacy-resolver")
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.shutil.copytree",
        lambda source, destination: None,
    )
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.os.walk",
        lambda path: [],
    )
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities."
        "_get_model_conda_env_info",
        lambda path: ("mlflow-test", False),
    )

    def check_output(command, **kwargs):
        captured.update(kwargs)
        return b"success"

    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.subprocess.check_output",
        check_output,
    )

    InstallCondaEnv(
        use_separate_conda_env=True,
        conda_file=None,
        model_path="model",
    ).install()

    assert "PIP_USE_DEPRECATED" not in captured["env"]


def test_install_conda_env_preserves_non_forecasting_environment(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.shutil.copyfile",
        lambda source, destination: None,
    )

    def check_output(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return b"success"

    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.subprocess.check_output",
        check_output,
    )

    InstallCondaEnv(
        use_separate_conda_env=False,
        conda_file="conda.yaml",
        model_path="model",
    ).install()

    assert captured["command"][:3] == ["conda", "env", "update"]
    assert "env" not in captured


def test_get_model_conda_env_name_uses_declared_file(tmp_path):
    conda_file = tmp_path / "forecast-env.yaml"
    conda_contents = "dependencies:\n- python=3.10\n"
    conda_file.write_text(conda_contents)
    (tmp_path / "MLmodel").write_text(
        "flavors:\n"
        "  python_function:\n"
        "    env:\n"
        "      conda: forecast-env.yaml\n"
    )

    expected_hash = hashlib.sha1(
        conda_contents.encode("utf-8"), usedforsecurity=False).hexdigest()
    assert _get_model_conda_env_name(
        str(tmp_path)) == f"mlflow-{expected_hash}"


@pytest.mark.parametrize(
    "model_marker",
    [
        "metadata:\n  azureml.engine: automl\n",
        "  automl-forecasting: {}\n",
    ],
)
def test_get_model_conda_env_info_detects_automl(tmp_path, model_marker):
    (tmp_path / "conda.yaml").write_text(
        "dependencies:\n- python=3.10\n")
    flavors = (
        "flavors:\n"
        "  python_function:\n"
        "    env:\n"
        "      conda: conda.yaml\n"
    )
    if model_marker.startswith("metadata"):
        model_contents = model_marker + flavors
    else:
        model_contents = flavors + model_marker
    (tmp_path / "MLmodel").write_text(model_contents)

    _, is_automl_model = _get_model_conda_env_info(str(tmp_path))

    assert is_automl_model


def test_get_model_conda_env_info_rejects_unmarked_model(tmp_path):
    (tmp_path / "conda.yaml").write_text(
        "dependencies:\n- python=3.10\n")
    (tmp_path / "MLmodel").write_text(
        "flavors:\n"
        "  python_function:\n"
        "    env:\n"
        "      conda: conda.yaml\n"
    )

    _, is_automl_model = _get_model_conda_env_info(str(tmp_path))

    assert not is_automl_model


def test_get_mlflow_model_server_command(monkeypatch):
    conda_executable = "/opt/miniconda/bin/conda"
    monkeypatch.setattr(
        "mlflow.utils.conda.get_conda_bin_executable",
        lambda executable: conda_executable,
    )
    expected_command = [
        conda_executable,
        "run",
        "--no-capture-output",
        "--name",
        "mlflow-test",
        "mlflow",
        "models",
        "serve",
        "-m",
        "local-model",
        "--env-manager",
        "local",
        "-p",
        "5432",
    ]
    assert _get_mlflow_model_server_command(
        "local-model", "mlflow-test") == expected_command

    fallback_command = _get_mlflow_model_server_command("registered-model")
    assert fallback_command[:5] == [
        "mlflow", "models", "serve", "-m", "registered-model"]
    assert fallback_command[5:7] == ["--env-manager", "conda"]


def test_load_mlflow_model_serves_localized_model(monkeypatch, tmp_path):
    captured = {}
    log_directory = tmp_path / "logs" / "azureml"
    log_directory.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities."
        "mlflow.tracking.MlflowClient",
        lambda: object(),
    )

    def install(installer):
        installer.local_model_path = "localized-model"
        installer.conda_env_name = "mlflow-test"

    monkeypatch.setattr(InstallCondaEnv, "install", install)

    def model_server_command(model_uri, conda_env_name=None):
        captured["model_uri"] = model_uri
        captured["conda_env_name"] = conda_env_name
        return ["model-server"]

    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities."
        "_get_mlflow_model_server_command",
        model_server_command,
    )

    class RunningProcess:
        def poll(self):
            return None

    def popen(command, **kwargs):
        kwargs["stdout"].write("Listening at: http")
        kwargs["stdout"].flush()
        return RunningProcess()

    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.subprocess.Popen",
        popen,
    )
    monkeypatch.setattr(
        "src.responsibleai.rai_analyse.rai_component_utilities.requests.post",
        lambda url: type("Response", (), {"status_code": 500})(),
    )

    load_mlflow_model(
        use_model_dependency=True,
        use_separate_conda_env=True,
        model_path="registered-model",
    )

    assert captured == {
        "model_uri": "localized-model",
        "conda_env_name": "mlflow-test",
    }
