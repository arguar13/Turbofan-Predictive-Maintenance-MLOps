import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_contracts import validate_labeled_telemetry, validate_raw_telemetry

# Columns that identify or label a row but are not model inputs.
NON_FEATURE_COLUMNS = ("unit_number", "time_in_cycles", "global_unit", "failure_type")


def load_and_combine_data(data_dir: str) -> pd.DataFrame:
    """Load and concatenate C-MAPSS FD001-FD004 with a globally unique engine id.

    The result is validated against the raw-data contract before it is returned,
    so invalid telemetry (nulls, non-positive units or cycles) stops here.
    """
    index_names = ["unit_number", "time_in_cycles"]
    setting_names = ["op_setting_1", "op_setting_2", "op_setting_3"]
    sensor_names = [f"sensor_{i}" for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    datasets = ["FD001", "FD002", "FD003", "FD004"]
    train_list = []

    for ds in datasets:
        file_path = os.path.join(data_dir, f"train_{ds}.txt")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
            df["dataset_id"] = ds
            df["global_unit"] = df["dataset_id"] + "_" + df["unit_number"].astype(str)
            train_list.append(df)

    if not train_list:
        # Otherwise pd.concat([]) fails with "No objects to concatenate", which
        # says neither what was missing nor where it was looked for.
        raise FileNotFoundError(
            f"No train_FD00X.txt file found in {data_dir}. "
            "Download the data with `make dvc-pull` (or copy it there)."
        )

    combined = pd.concat(train_list, ignore_index=True)
    return validate_raw_telemetry(combined)


def select_engines(df: pd.DataFrame, max_engines: int | None, seed: int = 42) -> pd.DataFrame:
    """Subsample whole ENGINES (never single rows), stratified by sub-dataset.

    Taking "the first N windows" would bias training: the files are sorted by
    sub-dataset and engine, so the first windows all come from FD001 (one
    operating condition, one fault mode). Here every sub-dataset contributes the
    same quota of randomly chosen engines (fixed seed), and each engine is kept
    whole so its time series stays intact. `max_engines=None` (or <= 0) keeps all.
    """
    if not max_engines or max_engines <= 0:
        return df

    engines = df[["dataset_id", "global_unit"]].drop_duplicates()
    if len(engines) <= max_engines:
        return df

    rng = np.random.default_rng(seed)
    per_dataset = max(1, max_engines // engines["dataset_id"].nunique())
    selected: list[str] = []
    for _, group in engines.groupby("dataset_id", sort=True):
        units = group["global_unit"].to_numpy()
        take = min(per_dataset, len(units))
        selected.extend(rng.choice(units, size=take, replace=False).tolist())

    return df[df["global_unit"].isin(selected)].reset_index(drop=True)


def build_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label every cycle Healthy (RUL > 60), Alert (31-60) or Critical (<= 30).

    Training trajectories run to failure, so RUL = last cycle - current cycle.
    RUL and failure_type are validated before returning, so no label outside
    the contract (e.g. a non-existent class) ever reaches training.
    """
    df = df.copy()
    max_cycles = df.groupby("global_unit")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_in_cycles"]

    df["failure_type"] = 0  # Healthy
    df.loc[df["RUL"] <= 60, "failure_type"] = 1  # Alert
    df.loc[df["RUL"] <= 30, "failure_type"] = 2  # Critical
    return validate_labeled_telemetry(df)


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and constant sensors, plus the columns no longer needed."""
    df = df.drop_duplicates()
    sensor_cols = [col for col in df.columns if "sensor" in col]
    std_dev = df[sensor_cols].std()
    invariant_sensors = std_dev[std_dev < 1e-6].index.tolist()

    df = df.drop(columns=invariant_sensors + ["RUL", "dataset_id"])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns, in the order the scaler expects them."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def create_sliding_windows(
    df: pd.DataFrame, window_size: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Unscaled (samples, window, features) windows, labels, engine ids and feature order.

    `groups` holds the engine (`global_unit`) behind each window. With stride 1
    the windows of one engine overlap heavily, so any consumer must split with
    `groups` (GroupShuffleSplit), never IID over the rows of X/y.

    Scaling does NOT happen here: this stage does not know the engine split, and
    a scaler fitted on every engine would leak val/test statistics into
    training. train.py fits it on the train engines only (`fit_window_scaler`).
    """
    features = feature_columns(df)

    # Windows assume consecutive cycles within each engine; sorting a copy leaves
    # the caller's DataFrame untouched.
    ordered = df.sort_values(["global_unit", "time_in_cycles"], kind="stable")

    values = ordered[features].to_numpy(dtype=np.float32)
    units = ordered["global_unit"].to_numpy()
    labels = ordered["failure_type"].to_numpy()

    # After sorting, each engine's rows are contiguous: the positions where
    # global_unit changes delimit every series.
    boundaries = np.flatnonzero(units[1:] != units[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(units)]))

    X_parts, y_parts, group_parts = [], [], []
    for start, end in zip(starts, ends, strict=True):
        n_windows = (end - start) - window_size + 1
        if n_windows <= 0:
            continue
        # sliding_window_view returns (n_windows, n_features, window_size) as
        # zero-copy views, instead of a Python loop with .iloc per window.
        windows = np.lib.stride_tricks.sliding_window_view(values[start:end], window_size, axis=0)
        X_parts.append(windows.transpose(0, 2, 1))
        # Each window takes the label of its last cycle
        y_parts.append(labels[start + window_size - 1 : end])
        group_parts.append(np.full(n_windows, units[start], dtype=object))

    if not X_parts:
        raise ValueError(
            f"No engine has at least window_size={window_size} cycles: no windows to build."
        )

    X = np.ascontiguousarray(np.concatenate(X_parts), dtype=np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    groups = np.concatenate(group_parts)
    return X, y, groups, features


def fit_window_scaler(X_train: np.ndarray, feature_names: list[str]) -> StandardScaler:
    """Fit the StandardScaler on the train windows ONLY.

    Fitted on a named DataFrame so the scaler keeps `feature_names_in_`, which
    the API publishes as the exact column order /predict expects. Overlapping
    windows count middle cycles a little more often, but the resulting mean and
    std are practically the per-row ones.
    """
    rows = X_train.reshape(-1, X_train.shape[2])
    return StandardScaler().fit(pd.DataFrame(rows, columns=feature_names))


def scale_windows(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply the train-fitted scaler to (n, window, features) windows."""
    flat = pd.DataFrame(X.reshape(-1, X.shape[2]), columns=scaler.feature_names_in_)
    return np.asarray(scaler.transform(flat), dtype=np.float32).reshape(X.shape)
