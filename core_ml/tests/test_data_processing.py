import numpy as np
import pandas as pd
import pytest

from data_processing import (
    build_multiclass_target,
    clean_and_prepare,
    create_sliding_windows,
    fit_window_scaler,
    load_and_combine_data,
    scale_windows,
    select_engines,
)


def _sample_unit_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1],
            "time_in_cycles": [1, 2, 3, 100],
            "global_unit": ["FD001_1", "FD001_1", "FD001_1", "FD001_1"],
            "dataset_id": ["FD001", "FD001", "FD001", "FD001"],
            "op_setting_1": [0.0, 0.0, 0.0, 0.0],
            "op_setting_2": [0.0, 0.0, 0.0, 0.0],
            "op_setting_3": [100.0, 100.0, 100.0, 100.0],
            "sensor_1": [0.1, 0.2, 0.3, 0.4],
            "sensor_2": [1.0, 1.0, 1.0, 1.0],  # constant sensor
        }
    )


def test_build_multiclass_target_labels_by_remaining_useful_life():
    df = _sample_unit_df()

    result = build_multiclass_target(df)

    # RUL = max_cycle (100) - time_in_cycles; RUL > 60 means Healthy
    assert result["RUL"].tolist() == [99, 98, 97, 0]
    assert result["failure_type"].tolist() == [0, 0, 0, 2]


def test_build_multiclass_target_flags_critical_near_end_of_life():
    df = _sample_unit_df().iloc[:2].copy()
    df["time_in_cycles"] = [1, 100]  # RUL 99 and 0 -> Healthy, Critical

    result = build_multiclass_target(df)

    assert result["failure_type"].tolist() == [0, 2]


def test_clean_and_prepare_drops_invariant_sensors_and_helper_columns():
    df = build_multiclass_target(_sample_unit_df())

    cleaned = clean_and_prepare(df)

    assert "sensor_2" not in cleaned.columns  # constant -> dropped
    assert "sensor_1" in cleaned.columns
    assert "RUL" not in cleaned.columns
    assert "dataset_id" not in cleaned.columns


def test_clean_and_prepare_drops_duplicate_rows():
    df = build_multiclass_target(_sample_unit_df())
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_and_prepare(df_with_dupe)

    assert len(cleaned) == len(df)


def _two_unit_df(cycles_per_unit: int = 5) -> pd.DataFrame:
    """Two engines with `cycles_per_unit` rows each: several windows per engine at
    window_size=3, which is what it takes to prove windows never mix engines."""
    rows = []
    for unit_idx, unit in enumerate(["FD001_1", "FD001_2"]):
        for cycle in range(1, cycles_per_unit + 1):
            rows.append(
                {
                    "unit_number": unit_idx + 1,
                    "time_in_cycles": cycle,
                    "global_unit": unit,
                    # A distinct value range per engine, so a window's content
                    # reveals which engine it came from.
                    "sensor_1": float(unit_idx * 1000 + cycle),
                    "failure_type": 0,
                }
            )
    return pd.DataFrame(rows)


def test_create_sliding_windows_tags_every_window_with_its_source_engine():
    df = _two_unit_df(cycles_per_unit=5)

    X, y, groups, feature_names = create_sliding_windows(df, window_size=3)

    # 5 cycles, window of 3 -> 3 windows per engine, 2 engines -> 6 windows.
    assert X.shape == (6, 3, 1)
    assert len(y) == len(groups) == 6
    assert set(groups) == {"FD001_1", "FD001_2"}
    assert feature_names == ["sensor_1"]
    # sensor_1 = unit_idx*1000 + cycle (unscaled): FD001_1 spans 1-5 and FD001_2
    # 1001-1005, so no window may contain values from both ranges.
    for window, group in zip(X, groups, strict=True):
        if group == "FD001_1":
            assert (window < 1000).all()
        else:
            assert (window > 1000).all()


def test_create_sliding_windows_does_not_mutate_input_and_returns_float32():
    df = _two_unit_df(cycles_per_unit=5)
    original = df.copy()

    X, y, _groups, _features = create_sliding_windows(df, window_size=3)

    pd.testing.assert_frame_equal(df, original)
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_create_sliding_windows_orders_cycles_within_each_engine():
    df = _two_unit_df(cycles_per_unit=5).sample(frac=1.0, random_state=0)

    X, _y, groups, _features = create_sliding_windows(df, window_size=3)

    # With cycles in order, sensor_1 (unit_idx*1000 + cycle) strictly increases
    # within every window.
    for window in X:
        assert np.all(np.diff(window[:, 0]) > 0)
    assert len(groups) == 6


def test_create_sliding_windows_skips_engines_shorter_than_the_window():
    df = pd.concat(
        [
            _two_unit_df(cycles_per_unit=5),
            _two_unit_df(cycles_per_unit=2).iloc[:2].assign(global_unit="X"),
        ]
    )

    _X, _y, groups, _features = create_sliding_windows(df, window_size=3)

    assert "X" not in set(groups)


def _multi_dataset_df(engines_per_dataset: int = 10, cycles: int = 3) -> pd.DataFrame:
    rows = []
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        for unit in range(1, engines_per_dataset + 1):
            for cycle in range(1, cycles + 1):
                rows.append(
                    {
                        "unit_number": unit,
                        "time_in_cycles": cycle,
                        "dataset_id": ds,
                        "global_unit": f"{ds}_{unit}",
                    }
                )
    return pd.DataFrame(rows)


def test_select_engines_samples_whole_engines_evenly_across_sub_datasets():
    df = _multi_dataset_df(engines_per_dataset=10, cycles=3)

    selected = select_engines(df, max_engines=8, seed=0)

    per_dataset = selected.groupby("dataset_id")["global_unit"].nunique()
    assert per_dataset.to_dict() == {"FD001": 2, "FD002": 2, "FD003": 2, "FD004": 2}
    # Every selected engine keeps all of its cycles (never loose rows).
    assert (selected.groupby("global_unit").size() == 3).all()


def test_select_engines_is_deterministic_and_optional():
    df = _multi_dataset_df()

    first = select_engines(df, max_engines=8, seed=1)
    second = select_engines(df, max_engines=8, seed=1)

    pd.testing.assert_frame_equal(first, second)
    assert len(select_engines(df, max_engines=0)) == len(df)
    assert len(select_engines(df, max_engines=None)) == len(df)


def test_load_and_combine_data_fails_fast_when_no_files_are_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_FD00X"):
        load_and_combine_data(str(tmp_path))


def test_window_scaler_uses_only_the_windows_it_is_fitted_on():
    train = np.full((4, 3, 2), 10.0, dtype=np.float32)
    train[:, :, 1] = 20.0
    train[0] += 1.0  # some variance
    other = np.full((2, 3, 2), 1000.0, dtype=np.float32)  # val/test far from train

    scaler = fit_window_scaler(train, ["a", "b"])

    # The mean comes from train only: "other" does not leak into it.
    np.testing.assert_allclose(scaler.mean_, train.reshape(-1, 2).mean(axis=0), rtol=1e-6)
    assert list(scaler.feature_names_in_) == ["a", "b"]
    scaled = scale_windows(scaler, other)
    assert scaled.shape == other.shape
    assert scaled.dtype == np.float32
    assert (scaled > 10).all()  # far outside the train distribution, as it should be
