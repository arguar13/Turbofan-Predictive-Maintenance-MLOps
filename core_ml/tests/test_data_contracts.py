import pandas as pd
import pytest

from data_contracts import (
    SchemaError,
    SchemaErrors,
    validate_labeled_telemetry,
    validate_raw_telemetry,
    validate_windowed_features,
)


def _raw_telemetry_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 2],
            "time_in_cycles": [1, 2, 1],
            "op_setting_1": [0.1, 0.2, 0.1],
            "op_setting_2": [0.0, 0.0, 0.0],
            "op_setting_3": [100.0, 100.0, 100.0],
            "sensor_1": [518.67, 518.67, 518.67],
            "sensor_2": [641.82, 642.15, 641.90],
            "dataset_id": ["FD001", "FD001", "FD001"],
            "global_unit": ["FD001_1", "FD001_1", "FD001_2"],
        }
    )


def test_validate_raw_telemetry_accepts_valid_data():
    validated = validate_raw_telemetry(_raw_telemetry_df())
    assert len(validated) == 3


def test_validate_raw_telemetry_coerces_integer_sensor_columns():
    df = _raw_telemetry_df()
    df["sensor_1"] = df["sensor_1"].astype(int)  # like sensor_17/18 in C-MAPSS
    validated = validate_raw_telemetry(df)
    assert validated["sensor_1"].dtype == "float64"


def test_validate_raw_telemetry_rejects_null_sensor_value():
    df = _raw_telemetry_df()
    df.loc[0, "sensor_1"] = None
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_raw_telemetry(df)


def test_validate_raw_telemetry_rejects_non_positive_unit_number():
    df = _raw_telemetry_df()
    df.loc[0, "unit_number"] = 0
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_raw_telemetry(df)


def test_validate_labeled_telemetry_rejects_invalid_failure_type():
    df = _raw_telemetry_df()
    df["RUL"] = [10, 9, 20]
    df["failure_type"] = [0, 0, 9]  # 9 is not a valid class (0, 1, 2)
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_labeled_telemetry(df)


def test_validate_labeled_telemetry_accepts_valid_labels():
    df = _raw_telemetry_df()
    df["RUL"] = [10, 9, 20]
    df["failure_type"] = [0, 1, 2]
    validated = validate_labeled_telemetry(df)
    assert len(validated) == 3


def _windowed_features_df() -> pd.DataFrame:
    now = pd.Timestamp.now()
    return pd.DataFrame(
        {
            "engine_id": ["ENG_001", "ENG_002"],
            "event_timestamp": [now, now],
            "created_timestamp": [now, now],
            "windowed_features": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            "failure_type": [0, 2],
        }
    )


def test_validate_windowed_features_accepts_matching_length():
    validated = validate_windowed_features(_windowed_features_df(), expected_length=4)
    assert len(validated) == 2


def test_validate_windowed_features_rejects_wrong_array_length():
    df = _windowed_features_df()
    df.at[0, "windowed_features"] = df.at[0, "windowed_features"][:-1]
    with pytest.raises(SchemaError):
        validate_windowed_features(df, expected_length=4)


def test_validate_windowed_features_rejects_blank_engine_id():
    df = _windowed_features_df()
    df.loc[0, "engine_id"] = ""
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_windowed_features(df, expected_length=4)
