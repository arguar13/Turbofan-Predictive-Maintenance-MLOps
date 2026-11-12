"""Endpoint tests against a fake model: no MLflow, no network.

`TestClient(app)` outside a `with` block does not run the lifespan, so the real
champion is never loaded; the model service is swapped through
`app.dependency_overrides`.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

import main
from main import LoadedModel, ModelService

WINDOW_SIZE = 4
FEATURES = ["op_setting_1", "sensor_2", "sensor_3"]
HEADERS = {"X-API-Key": "test-api-key"}


class _AlwaysCritical(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.tensor([[0.0, 1.0, 5.0]]).repeat(batch, 1)


def _fake_loaded_model() -> LoadedModel:
    scaler = StandardScaler().fit(
        pd.DataFrame(np.random.default_rng(0).normal(size=(50, len(FEATURES))), columns=FEATURES)
    )
    return LoadedModel(
        model=_AlwaysCritical(),
        scaler=scaler,
        version="7",
        window_size=WINDOW_SIZE,
        num_features=len(FEATURES),
        feature_names=FEATURES,
    )


def _service(loaded: LoadedModel | None) -> ModelService:
    def loader() -> LoadedModel:
        if loaded is None:
            raise RuntimeError("no champion")
        return loaded

    return ModelService(loader=loader, cooldown_seconds=0)


@pytest.fixture
def client_with_model():
    main.app.dependency_overrides[main.get_model_service] = lambda: _service(_fake_loaded_model())
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.fixture
def client_without_model():
    main.app.dependency_overrides[main.get_model_service] = lambda: _service(None)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _valid_body() -> dict:
    return {"engine_id": "FD001_1", "readings": [[0.1, 0.2, 0.3]] * WINDOW_SIZE}


def test_health_reports_model_shape_and_feature_order(client_with_model):
    response = client_with_model.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "7"
    assert body["window_size"] == WINDOW_SIZE
    assert body["num_features"] == len(FEATURES)
    assert body["feature_names"] == FEATURES


def test_health_is_503_without_model(client_without_model):
    assert client_without_model.get("/health").status_code == 503


def test_predict_requires_api_key(client_with_model):
    assert client_with_model.post("/predict", json=_valid_body()).status_code == 401
    wrong = client_with_model.post("/predict", json=_valid_body(), headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401


def test_predict_returns_class_probabilities_and_version(client_with_model):
    response = client_with_model.post("/predict", json=_valid_body(), headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Critical"
    assert body["model_version"] == "7"
    assert set(body["probabilities"]) == {"Healthy", "Alert", "Critical"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_rejects_shape_that_does_not_match_loaded_model(client_with_model):
    body = {"engine_id": "FD001_1", "readings": [[0.1, 0.2]] * WINDOW_SIZE}

    response = client_with_model.post("/predict", json=body, headers=HEADERS)

    assert response.status_code == 422
    assert "(4, 3)" in response.json()["detail"]


def test_predict_is_503_without_model(client_without_model):
    response = client_without_model.post("/predict", json=_valid_body(), headers=HEADERS)
    assert response.status_code == 503


def test_model_service_retries_after_a_failed_load():
    attempts = {"n": 0}
    loaded = _fake_loaded_model()

    def flaky_loader() -> LoadedModel:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("MLflow todavia no responde")
        return loaded

    service = ModelService(loader=flaky_loader, cooldown_seconds=0)

    assert service.get() is None
    assert service.get() is loaded
    # Once loaded, the loader is never called again.
    assert service.get() is loaded
    assert attempts["n"] == 2


def test_model_service_respects_cooldown_between_attempts():
    attempts = {"n": 0}

    def failing_loader() -> LoadedModel:
        attempts["n"] += 1
        raise RuntimeError("sin champion")

    service = ModelService(loader=failing_loader, cooldown_seconds=3600)

    assert service.get() is None
    assert service.get() is None
    assert attempts["n"] == 1


def test_infer_window_shape_rejects_model_and_scaler_mismatch(monkeypatch):
    class _Shape:
        shape = (-1, 30, 5)

    class _Info:
        class signature:  # noqa: N801 - mimics mlflow's attribute
            class inputs:  # noqa: N801
                inputs = [_Shape()]

    monkeypatch.setattr("mlflow.models.get_model_info", lambda _uri: _Info())
    scaler = StandardScaler().fit(np.zeros((3, 4)))

    with pytest.raises(ValueError, match="inconsistent artifacts"):
        main._infer_window_shape("models:/x/1", scaler, default_window_size=30)
