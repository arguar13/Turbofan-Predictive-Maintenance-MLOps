import pytest

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


def test_load_config_fails_fast_on_missing_section(tmp_path):
    """The whole 'monitoring' section is missing."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
project:
  name: "predictive_maintenance_turbofan"
  version: "2.0.0"
model:
  window_size: 30
  mlflow_tracking_uri: "http://localhost:5000"
  model_name: "Turbofan_ConvTransformer"
"""
    )

    with pytest.raises(ValueError, match="Invalid config.yaml"):
        load_config(config_path=str(config_file))


def test_load_config_fails_fast_on_unknown_key(tmp_path):
    """An unknown key (e.g. the obsolete model.num_features) is not silently ignored."""
    config_file = tmp_path / "config.yaml"
    invalid_text = _valid_config_text().replace(
        "  window_size: 30\n", "  window_size: 30\n  num_features: 14\n"
    )
    config_file.write_text(invalid_text)

    with pytest.raises(ValueError, match="num_features"):
        load_config(config_path=str(config_file))
