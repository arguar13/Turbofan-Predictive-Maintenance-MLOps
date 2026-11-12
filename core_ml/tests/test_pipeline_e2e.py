"""End-to-end smoke test: raw .txt -> prepare -> train -> MLflow -> Registry.

A synthetic dataset in the exact C-MAPSS format (26 columns per row) and a
temporary SQLite tracking store, so it runs in CI with no DVC, S3 or MLflow
server. Covers contracts, windowing, training, artifact logging and the rule
"only what passes the gate gets registered".
"""

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

import prepare_training_data
import train
from data_paths import prepared_dir


def _write_synthetic_cmapss(data_dir, n_engines: int = 6, cycles: int = 100) -> None:
    # 100 cycles per engine: windows (from cycle 30) cover RUL 70..0, i.e. all
    # three classes (Healthy > 60, Alert, Critical <= 30).
    rng = np.random.default_rng(0)
    lines = []
    for unit in range(1, n_engines + 1):
        for cycle in range(1, cycles + 1):
            settings = rng.normal(0, 0.01, size=3)
            # Sensors that drift towards the end of the engine's life.
            sensors = 500 + rng.normal(0, 1, size=21) + cycle * 0.3
            values = [unit, cycle, *settings, *sensors]
            lines.append(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in values))
    (data_dir / "train_FD001.txt").write_text("\n".join(lines) + "\n")


def _config_with_thresholds(f2: float, critical_recall: float):
    real_load_config = train.load_config

    def _load_config():
        config = real_load_config()
        config["monitoring"]["f2_weighted_threshold"] = f2
        config["monitoring"]["critical_recall_threshold"] = critical_recall
        return config

    return _load_config


@pytest.fixture
def prepared_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_cmapss(data_dir)
    # MLflow with SQLite stores artifacts relative to the cwd: isolate it in tmp.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    # An absolute data_dir works: BASE_DIR / <absolute path> is that absolute path.
    prepare_training_data.generate_training_parquet(data_dir=str(data_dir), max_engines=0)
    return data_dir


def test_prepare_writes_raw_windows_and_feature_names(prepared_data_dir):
    output_dir = prepared_dir(str(prepared_data_dir))
    assert output_dir == prepared_data_dir.parent / "prepared" / "data"
    assert (output_dir / "engine_features.parquet").exists()
    assert (output_dir / "feature_names.json").exists()
    # prepare does NOT fit the scaler: train.py fits it on train only.
    assert not (output_dir / "scaler.joblib").exists()
    # The raw (DVC-tracked) directory stays untouched: writing there would make
    # the data.dvc pointer stale.
    assert sorted(path.name for path in prepared_data_dir.iterdir()) == sorted(
        path.name for path in prepared_data_dir.glob("*.txt")
    )


def test_rejected_model_is_logged_but_never_registered(prepared_data_dir, monkeypatch):
    # Impossible thresholds: the gate must fail.
    monkeypatch.setattr(train, "load_config", _config_with_thresholds(1.0, 1.0))

    run_id = train.train_pipeline(
        data_dir=str(prepared_data_dir), epochs=1, val_split=0.34, enforce_quality_gate=False
    )

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags["quality_gate_passed"] == "False"
    assert "training_data_fingerprint" in run.data.tags
    # The gate is decided on test engines, disjoint from the val ones.
    assert {"best_val_f2_weighted", "test_f2_weighted", "test_critical_recall"} <= set(
        run.data.metrics
    )
    assert int(run.data.params["test_samples"]) > 0
    artifacts = {a.path for a in client.list_artifacts(run_id, "preprocessing")}
    assert "preprocessing/scaler.joblib" in artifacts
    assert {"test_f1_macro", "test_alert_recall", "baseline_test_f1_macro"} <= set(run.data.metrics)
    assert run.data.tags["git_dirty"] in {"true", "false", "unknown"}
    report = mlflow.artifacts.load_dict(f"runs:/{run_id}/evaluation/test_report.json")
    assert len(report["confusion_matrix"]["matrix"]) == 3
    assert client.search_registered_models() == []


def test_rejected_model_fails_the_process_when_gate_is_enforced(prepared_data_dir, monkeypatch):
    monkeypatch.setattr(train, "load_config", _config_with_thresholds(1.0, 1.0))

    with pytest.raises(SystemExit, match="Quality gate failed"):
        train.train_pipeline(data_dir=str(prepared_data_dir), epochs=1, val_split=0.34)


def test_model_that_passes_the_gate_is_registered_as_champion(prepared_data_dir, monkeypatch):
    monkeypatch.setattr(train, "load_config", _config_with_thresholds(0.0, 0.0))

    run_id = train.train_pipeline(data_dir=str(prepared_data_dir), epochs=1, val_split=0.34)

    champion = MlflowClient().get_model_version_by_alias("Turbofan_ConvTransformer", "champion")
    assert champion.run_id == run_id
