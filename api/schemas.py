"""Pydantic request/response contracts for the inference API.

A request that breaks the contract (empty or ragged matrix, non-finite values,
blank engine_id) is rejected with a 422 before the scaler or the model run.

The EXACT shape (window_size x num_features) is checked in the endpoint against
the model loaded at that moment: the champion can be loaded after startup
(ModelService keeps retrying), so a schema built at import time would be pinned
to a shape that is not known yet.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator


class SensorWindowRequest(BaseModel):
    engine_id: str = Field(..., min_length=1, max_length=64)
    # One row per timestep, features in the order of `feature_names` (GET /health).
    readings: list[list[float]] = Field(..., min_length=1)

    @field_validator("engine_id")
    @classmethod
    def engine_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("engine_id cannot be empty or whitespace.")
        return value

    @model_validator(mode="after")
    def validate_matrix(self) -> SensorWindowRequest:
        n_columns = len(self.readings[0])
        if n_columns == 0:
            raise ValueError("The rows of 'readings' cannot be empty.")
        for row_index, row in enumerate(self.readings):
            if len(row) != n_columns:
                raise ValueError(
                    f"Row {row_index} has {len(row)} values and row 0 has {n_columns}; "
                    "'readings' must be a rectangular matrix."
                )
            if not all(math.isfinite(value) for value in row):
                raise ValueError(
                    f"Row {row_index}: non-finite sensor values (NaN/Inf) are not allowed."
                )
        return self


class PredictionResponse(BaseModel):
    engine_id: str
    prediction: str
    # Per-class probabilities let the caller apply its own threshold (e.g. alert
    # when P(Critical) > 0.3) instead of relying on the argmax alone.
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    window_size: int
    num_features: int
    # Exact column order /predict expects in each row, taken from the scaler
    # logged with the model. None if the scaler did not record it.
    feature_names: list[str] | None
