import hmac
import math
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import joblib
import mlflow
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from mlflow import MlflowClient

from config_loader import load_config
from logging_config import configure_logging
from schemas import HealthResponse, PredictionResponse, SensorWindowRequest

log = configure_logging("api")

# Registry alias this API serves. train.py moves it only to a version that passed
# the quality gate; the API never serves "latest" or a loose file from disk.
CHAMPION_ALIAS = "champion"
# 0=Healthy, 1=Alert, 2=Critical (see core_ml/src/data_processing.py).
CLASSES = ("Healthy", "Alert", "Critical")
# After a failed load the next attempt waits: every attempt hits MLflow, and the
# readinessProbe calls /health every few seconds.
RELOAD_COOLDOWN_SECONDS = float(os.environ.get("MODEL_RELOAD_COOLDOWN_SECONDS", "30"))

config = load_config()
mlflow.set_tracking_uri(config["model"]["mlflow_tracking_uri"])

# The Service is a public LoadBalancer, so /predict requires an API key. It comes
# from envFrom -> secretRef: mlops-secrets (kubernetes/base/secret.yaml), never
# from config.yaml, which is committed. Fail fast: silently starting without it
# would expose an unauthenticated endpoint to the internet.
_API_KEY_HEADER = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_HEADER, auto_error=False)
_raw_api_key = os.environ.get("API_KEY")
if not _raw_api_key:
    raise RuntimeError(
        "API_KEY is not set. The API refuses to start without authentication "
        "(see kubernetes/base/secret.yaml, key API_KEY)."
    )
_API_KEY: str = _raw_api_key


def _is_valid_api_key(provided: str | None, expected: str) -> bool:
    """Constant-time comparison: a plain `==` leaks timing that lets an attacker
    guess the key one character at a time."""
    if provided is None:
        return False
    return hmac.compare_digest(provided, expected)


def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    if not _is_valid_api_key(provided, _API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# ---------------------------------------------------------------------------
# Served model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """Model and scaler from the SAME MLflow run, plus the input shape they expect."""

    model: Any
    scaler: Any
    version: str
    window_size: int
    num_features: int
    feature_names: list[str] | None

    def predict_proba(self, readings: np.ndarray) -> np.ndarray:
        """Raw (window_size, num_features) readings -> probability per class."""
        # Same column names the scaler was fitted with: avoids sklearn's
        # feature-name UserWarning on every request and pins the column order.
        features = (
            pd.DataFrame(readings, columns=self.feature_names)
            if self.feature_names is not None
            else readings
        )
        scaled = np.asarray(self.scaler.transform(features), dtype=np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(scaled[np.newaxis]))
            return torch.softmax(logits, dim=1)[0].numpy()


def _infer_window_shape(model_uri: str, scaler: Any, default_window_size: int) -> tuple[int, int]:
    """Derive (window_size, num_features) from the served artifacts, not config.yaml.

    `num_features` depends on the data (constant sensors are dropped per
    dataset), so its source of truth is the scaler from the model's own run:
    exactly what `scaler.transform()` will accept. The model signature supplies
    window_size and a cross-check: if model and scaler disagree the run is
    broken, and refusing to serve it beats failing on every request.
    """
    num_features = int(scaler.n_features_in_)
    window_size = default_window_size
    try:
        from mlflow.models import get_model_info

        shape = get_model_info(model_uri).signature.inputs.inputs[0].shape
        # (batch, window_size, num_features), with batch = -1
        if len(shape) == 3 and shape[1] > 0 and shape[2] > 0:
            window_size = int(shape[1])
            if int(shape[2]) != num_features:
                raise ValueError(
                    f"The model signature expects {shape[2]} features and the scaler of "
                    f"the same run {num_features}: inconsistent artifacts."
                )
        else:
            log.warning("model_signature_unexpected_shape", shape=list(shape))
    except ValueError:
        raise
    except Exception:
        log.warning("model_signature_unavailable", exc_info=True)
    return window_size, num_features


def load_champion(model_name: str, default_window_size: int) -> LoadedModel:
    """Load the champion model and its scaler from the MLflow Model Registry.

    Both are immutable references to the same run, never loose local files.
    """
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)

    # A concrete VERSION, not the alias: model and scaler come from the same run
    # even if the alias moves between the two calls.
    model_uri = f"models:/{model_name}/{model_version.version}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id, artifact_path="preprocessing/scaler.joblib"
    )
    scaler = joblib.load(scaler_path)
    window_size, num_features = _infer_window_shape(model_uri, scaler, default_window_size)
    feature_names = getattr(scaler, "feature_names_in_", None)
    return LoadedModel(
        model=model,
        scaler=scaler,
        version=str(model_version.version),
        window_size=window_size,
        num_features=num_features,
        feature_names=[str(name) for name in feature_names] if feature_names is not None else None,
    )


class ModelService:
    """Holds the served model and retries loading it while there is none.

    MLflow may not be up yet, or there may be no champion on a fresh deploy.
    Every /health or /predict call retries the load (with a cooldown), so the
    readinessProbe turns green as soon as a champion exists, with no pod restart.

    A NEW champion still needs a rollout restart: a loaded model is never
    swapped in place.
    """

    def __init__(self, loader: Callable[[], LoadedModel], cooldown_seconds: float) -> None:
        self._loader = loader
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None
        self._last_attempt = -math.inf

    def get(self) -> LoadedModel | None:
        if self._current is not None:
            return self._current
        # Non-blocking: if another thread is already loading (MLflow retries can
        # take a while), this request gets a 503 instead of hanging.
        if not self._lock.acquire(blocking=False):
            return None
        try:
            now = time.monotonic()
            if self._current is None and now - self._last_attempt >= self._cooldown_seconds:
                self._last_attempt = now
                try:
                    self._current = self._loader()
                    log.info(
                        "model_loaded",
                        model_version=self._current.version,
                        window_size=self._current.window_size,
                        num_features=self._current.num_features,
                    )
                except Exception:
                    log.exception("model_load_failed", alias=CHAMPION_ALIAS)
            return self._current
        finally:
            self._lock.release()


model_service = ModelService(
    loader=lambda: load_champion(config["model"]["model_name"], config["model"]["window_size"]),
    cooldown_seconds=RELOAD_COOLDOWN_SECONDS,
)


def get_model_service() -> ModelService:
    return model_service


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


def _require_model(service: ModelService) -> LoadedModel:
    loaded = service.get()
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return loaded


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # First load attempt in the background: the server accepts connections (TCP
    # livenessProbe green) even while MLflow is slow to answer.
    threading.Thread(target=model_service.get, name="model-loader", daemon=True).start()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    version=config["project"]["version"],
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
# Deliberately WITHOUT require_api_key: the readinessProbe (kubernetes/base/api.yaml)
# calls it with no custom headers, and it exposes nothing beyond "a model is
# loaded, and this is the input shape it expects".
def health_check(service: ModelServiceDep) -> HealthResponse:
    loaded = _require_model(service)
    return HealthResponse(
        status="ok",
        model_version=loaded.version,
        window_size=loaded.window_size,
        num_features=loaded.num_features,
        feature_names=loaded.feature_names,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(
    request: SensorWindowRequest,
    service: ModelServiceDep,
) -> PredictionResponse:
    loaded = _require_model(service)

    readings = np.asarray(request.readings, dtype=np.float64)
    expected_shape = (loaded.window_size, loaded.num_features)
    if readings.shape != expected_shape:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'readings' must have shape {expected_shape} (window_size x num_features), "
                f"got {readings.shape}. GET /health returns the expected shape and column "
                "order."
            ),
        )

    probabilities = loaded.predict_proba(readings)
    prediction = CLASSES[int(np.argmax(probabilities))]
    log.info(
        "prediction_served",
        engine_id=request.engine_id,
        prediction=prediction,
        model_version=loaded.version,
    )
    return PredictionResponse(
        engine_id=request.engine_id,
        prediction=prediction,
        probabilities={
            name: round(float(p), 6) for name, p in zip(CLASSES, probabilities, strict=True)
        },
        model_version=loaded.version,
    )
