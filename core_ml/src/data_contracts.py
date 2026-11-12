"""Fail-fast data contracts for the predictive-maintenance pipeline.

Each function validates a DataFrame at one point of the pipeline
(data_processing -> prepare_training_data -> train) and raises
`pandera.errors.SchemaError` as soon as the contract breaks, before any compute
(training, MLflow calls) is spent on invalid data.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

__all__ = [
    "SchemaError",
    "SchemaErrors",
    "validate_raw_telemetry",
    "validate_labeled_telemetry",
    "validate_windowed_features",
    "validate_training_batch",
]

_SENSOR_COLUMN_PATTERN = r"^sensor_\d+$"
_OP_SETTING_COLUMN_PATTERN = r"^op_setting_[1-3]$"

# ---------------------------------------------------------------------------
# 1. Combined raw telemetry (output of load_and_combine_data)
# ---------------------------------------------------------------------------
RAW_TELEMETRY_SCHEMA = pa.DataFrameSchema(
    columns={
        "unit_number": pa.Column(int, checks=pa.Check.gt(0), nullable=False),
        "time_in_cycles": pa.Column(int, checks=pa.Check.gt(0), nullable=False),
        _OP_SETTING_COLUMN_PATTERN: pa.Column(
            float, checks=pa.Check(lambda s: s.abs() < 1e6), nullable=False, regex=True
        ),
        # Some C-MAPSS sensors (e.g. sensor_17/sensor_18) only hold integers in the
        # source files, so pandas infers int64; coerce=True accepts both.
        _SENSOR_COLUMN_PATTERN: pa.Column(
            float, checks=pa.Check(lambda s: s.abs() < 1e6), nullable=False, regex=True
        ),
        "dataset_id": pa.Column(str, nullable=False),
        "global_unit": pa.Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)

# ---------------------------------------------------------------------------
# 2. Labelled telemetry (output of build_multiclass_target)
# ---------------------------------------------------------------------------
LABELED_TELEMETRY_SCHEMA = RAW_TELEMETRY_SCHEMA.add_columns(
    {
        "RUL": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
        "failure_type": pa.Column(int, checks=pa.Check.isin([0, 1, 2]), nullable=False),
    }
)

# ---------------------------------------------------------------------------
# 3. Training windows (output of prepare_training_data.py)
# ---------------------------------------------------------------------------
WINDOWED_FEATURE_SCHEMA = pa.DataFrameSchema(
    columns={
        "engine_id": pa.Column(str, checks=pa.Check.str_length(min_value=1), nullable=False),
        "event_timestamp": pa.Column("datetime64[ns]", nullable=False),
        "created_timestamp": pa.Column("datetime64[ns]", nullable=False),
        "windowed_features": pa.Column(object, nullable=False),
        "failure_type": pa.Column(int, checks=pa.Check.isin([0, 1, 2]), nullable=False),
    },
    strict=False,
    coerce=False,
)

# ---------------------------------------------------------------------------
# 4. Batch loaded by train.py (_load_training_data). Deliberately LOOSER than
#    WINDOWED_FEATURE_SCHEMA on event_timestamp's exact dtype: it is not a model
#    input, so any datetime variant is enough.
_TRAINING_BATCH_SCHEMA = pa.DataFrameSchema(
    columns={
        "engine_id": pa.Column(str, checks=pa.Check.str_length(min_value=1), nullable=False),
        # dtype=None on purpose: pandera compares an annotated dtype for EXACT
        # equality. The check below accepts any datetime variant instead.
        "event_timestamp": pa.Column(
            dtype=None,
            checks=pa.Check(
                lambda s: pd.api.types.is_datetime64_any_dtype(s),
                element_wise=False,
                error="event_timestamp must be datetime64 (naive or tz-aware)",
            ),
            nullable=False,
        ),
        "windowed_features": pa.Column(object, nullable=False),
        "failure_type": pa.Column(int, checks=pa.Check.isin([0, 1, 2]), nullable=False),
    },
    strict=False,
    coerce=False,
)


def validate_raw_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if the combined raw telemetry breaks its contract."""
    return RAW_TELEMETRY_SCHEMA.validate(df, lazy=True)


def validate_labeled_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if the labelled telemetry (RUL / failure_type) is invalid."""
    return LABELED_TELEMETRY_SCHEMA.validate(df, lazy=True)


def validate_windowed_features(df: pd.DataFrame, expected_length: int) -> pd.DataFrame:
    """Raise if the flattened windows break their contract.

    `expected_length` is `window_size * num_features`: window_size comes from
    config.yaml and num_features from the data (the sensors that survive
    clean_and_prepare), so it is checked here rather than in the static schema.
    """
    validated = WINDOWED_FEATURE_SCHEMA.validate(df, lazy=True)

    bad_length = validated["windowed_features"].map(len) != expected_length
    if bad_length.any():
        raise SchemaError(
            schema=WINDOWED_FEATURE_SCHEMA,
            data=validated,
            message=(
                f"{int(bad_length.sum())} row(s) have 'windowed_features' of a length "
                f"other than the expected {expected_length} "
                "(window_size * num_features)."
            ),
        )
    return validated


def validate_training_batch(df: pd.DataFrame, expected_length: int) -> pd.DataFrame:
    """Raise if the loaded engine_features.parquet batch cannot be trained on.

    For train.py, after `_load_training_data`. The DataFrame freshly written by
    prepare_training_data.py goes through `validate_windowed_features` instead.
    """
    validated = _TRAINING_BATCH_SCHEMA.validate(df, lazy=True)

    bad_length = validated["windowed_features"].map(len) != expected_length
    if bad_length.any():
        raise SchemaError(
            schema=_TRAINING_BATCH_SCHEMA,
            data=validated,
            message=(
                f"{int(bad_length.sum())} row(s) have 'windowed_features' of a length "
                f"other than the expected {expected_length} "
                "(window_size * num_features)."
            ),
        )
    return validated
