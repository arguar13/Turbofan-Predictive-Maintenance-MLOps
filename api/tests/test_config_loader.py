import pytest
from pydantic import ValidationError

from config_loader import load_config


def _valid_config_text(mlflow_uri: str = "http://localhost:5000") -> str:
    return f"""
project:
  name: "predictive_maintenance_turbofan"
  version: "2.0.0"
model:
  window_size: 30
  mlflow_tracking_uri: "{mlflow_uri}"
  model_name: "Turbofan_ConvTransformer"
monitoring:
  f2_weighted_threshold: 0.75
  critical_recall_threshold: 0.75
"""


def test_load_config_resolves_env_var(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(mlflow_uri="${TEST_MLFLOW_URI}"))
    monkeypatch.setenv("TEST_MLFLOW_URI", "http://mlflow-test:5000")

    config = load_config(config_path=str(config_file))

    assert config["model"]["mlflow_tracking_uri"] == "http://mlflow-test:5000"


def test_load_config_default_mlflow_uri_has_a_scheme(tmp_path, monkeypatch):
    """The MLFLOW_TRACKING_URI default must be a URI WITH a scheme.

    MLflow rejects a bare host:port with UnsupportedModelRegistryStoreURIException,
    so `make smoke-test` and `make train-toy` would fail unless the variable
    was exported by hand.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(mlflow_uri="${MLFLOW_TRACKING_URI}"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = load_config(config_path=str(config_file))

    assert config["model"]["mlflow_tracking_uri"] == "http://localhost:5000"
    assert "://" in config["model"]["mlflow_tracking_uri"]


def test_load_config_fails_fast_on_unknown_env_var(tmp_path, monkeypatch):
    """An UNKNOWN variable is reported, not invented.

    A misspelled placeholder in config.yaml must fail right away, not resolve
    silently to an arbitrary default that hides the real cause.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(mlflow_uri="${MISSING_MLFLOW_VAR}"))
    monkeypatch.delenv("MISSING_MLFLOW_VAR", raising=False)

    with pytest.raises(ValueError, match="MISSING_MLFLOW_VAR"):
        load_config(config_path=str(config_file))


def test_load_config_leaves_plain_values_untouched(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text())

    config = load_config(config_path=str(config_file))

    assert config["project"]["name"] == "predictive_maintenance_turbofan"
    assert config["model"]["window_size"] == 30


def test_load_config_fails_fast_on_missing_section(tmp_path):
    """The whole 'model' section is missing."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
project:
  name: "predictive_maintenance_turbofan"
  version: "2.0.0"
monitoring:
  f2_weighted_threshold: 0.75
  critical_recall_threshold: 0.75
"""
    )

    with pytest.raises(ValueError, match="Invalid config.yaml"):
        load_config(config_path=str(config_file))


def test_load_config_fails_fast_on_out_of_range_threshold(tmp_path):
    """An f2_weighted_threshold outside [0, 1] is not a valid threshold."""
    config_file = tmp_path / "config.yaml"
    bad_value = "f2_weighted_threshold: 1.5"
    invalid_text = _valid_config_text().replace("f2_weighted_threshold: 0.75", bad_value)
    config_file.write_text(invalid_text)

    with pytest.raises(ValueError, match="Invalid config.yaml"):
        load_config(config_path=str(config_file))


def test_load_config_fails_fast_on_wrong_type(tmp_path):
    """window_size must be an integer, not a string."""
    config_file = tmp_path / "config.yaml"
    invalid_text = _valid_config_text().replace("window_size: 30", 'window_size: "treinta"')
    config_file.write_text(invalid_text)

    with pytest.raises((ValueError, ValidationError)):
        load_config(config_path=str(config_file))
