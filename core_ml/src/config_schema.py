"""Pydantic contract for config.yaml.

A missing key, wrong type or out-of-range value must fail at startup with a
clear error, not hours later as a cryptic KeyError in the middle of a training
run or a request.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    # extra="forbid": an unknown key (a typo, or an obsolete field such as
    # model.num_features) fails at startup instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ModelConfig(_StrictModel):
    window_size: int = Field(gt=0)
    mlflow_tracking_uri: str = Field(min_length=1)
    model_name: str = Field(min_length=1)


class MonitoringConfig(_StrictModel):
    # Quality gate: a new model gets the "champion" alias only if it clears BOTH
    # thresholds; see train.py::_evaluate_quality_gate for why plain accuracy is
    # not enough.
    f2_weighted_threshold: float = Field(ge=0, le=1)
    critical_recall_threshold: float = Field(ge=0, le=1)


class AppConfig(_StrictModel):
    project: ProjectConfig
    model: ModelConfig
    monitoring: MonitoringConfig
