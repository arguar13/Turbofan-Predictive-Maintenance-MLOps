"""Build engine_features.parquet + feature_names.json from the raw C-MAPSS .txt files.

Output goes to `prepared/<dataset>/` (see data_paths.py). This is the pipeline's
only data-preparation step: it turns the raw files into the windows train.py
consumes. With no online feature serving, a plain Parquet file read with
`pd.read_parquet` needs no extra infrastructure.
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from config_loader import load_config
from data_contracts import validate_windowed_features
from data_paths import prepared_dir, raw_dir
from data_processing import (
    build_multiclass_target,
    clean_and_prepare,
    create_sliding_windows,
    load_and_combine_data,
    select_engines,
)
from logging_config import configure_logging

log = configure_logging("prepare-training-data")

# 0 = every engine (709, ~140k windows: ~400 MB in memory as float32, ~24 MB as
# Parquet). With the gate judged on unseen test engines, a 100-engine sample
# (60 for train) did not reach the Critical recall floor; the full dataset does.
# `--max-engines N` is for quick iterations.
DEFAULT_MAX_ENGINES = 0


def _utc_now_naive() -> datetime:
    """Naive UTC: WINDOWED_FEATURE_SCHEMA (data_contracts.py) expects tz-less datetime64[ns]."""
    return datetime.now(UTC).replace(tzinfo=None)


def generate_training_parquet(
    data_dir: str = "data",
    max_engines: int | None = DEFAULT_MAX_ENGINES,
    seed: int = 42,
) -> Path:
    """Write engine_features.parquet (UNSCALED windows) + feature_names.json.

    `data_dir` is relative to core_ml/ ("data" for the full dataset, "data_toy"
    for the ~1,100-row DVC-tracked sample). Artifacts go to
    `prepared/<data_dir>/`, never next to the raw files, and that directory is
    returned. No scaler here: this stage does not know the engine split, so
    train.py fits it on the train engines and logs it with the model.
    """
    window_size = load_config()["model"]["window_size"]
    data_path = raw_dir(data_dir)
    output_path = prepared_dir(data_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info(
        "preprocessing_started",
        data_path=str(data_path),
        output_path=str(output_path),
        max_engines=max_engines,
    )

    # Each data_processing.py stage validates its own contract. Subsampling is by
    # ENGINE and BEFORE windowing, so the feature set comes from exactly the data
    # that is trained on.
    raw_df = load_and_combine_data(str(data_path))
    raw_df = select_engines(raw_df, max_engines=max_engines, seed=seed)
    target_df = build_multiclass_target(raw_df)
    clean_df = clean_and_prepare(target_df)

    X, y, groups, feature_names = create_sliding_windows(clean_df, window_size=window_size)
    num_features = X.shape[2]
    log.info(
        "windows_created",
        engines=int(raw_df["global_unit"].nunique()),
        windows=len(X),
        num_features=num_features,
        features=feature_names,
    )

    # One row per flattened window (the format of WINDOWED_FEATURE_SCHEMA).
    log.info("flattening_windows")
    records = []

    # event_timestamp/created_timestamp are required by the contract but unused by
    # train.py. event_timestamp is synthetic (one window every 10 s, starting 23 h
    # ago), not the time the readings were taken.
    base_time = _utc_now_naive() - timedelta(hours=23)

    for i in range(len(X)):
        # float32, not a list of Python floats (stored as double): the model's
        # dtype and half the disk.
        flat_features = X[i].ravel()
        records.append(
            {
                # The real engine ("FD001_23"): train.py groups the
                # train/val/test split by this id.
                "engine_id": str(groups[i]),
                "event_timestamp": base_time + timedelta(seconds=i * 10),
                "created_timestamp": _utc_now_naive(),
                "windowed_features": flat_features,
                "failure_type": int(y[i]),
            }
        )

    features_df = pd.DataFrame(records)

    # Validate before writing anything to disk.
    expected_length = window_size * num_features
    features_df = validate_windowed_features(features_df, expected_length=expected_length)

    features_path = output_path / "engine_features.parquet"
    features_df.to_parquet(features_path, index=False)
    log.info("parquet_written", features_path=str(features_path), rows=len(features_df))

    # Column order inside each flattened window: train.py fits a named scaler with
    # it, and the API publishes those names on /health.
    feature_names_path = output_path / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2))
    log.info("feature_names_written", path=str(feature_names_path), num_features=num_features)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory with the raw input .txt files, relative to core_ml/; artifacts "
        "go to prepared/<data-dir>/. Use 'data_toy' for the small DVC-tracked dataset.",
    )
    parser.add_argument(
        "--max-engines",
        type=int,
        default=DEFAULT_MAX_ENGINES,
        help="Maximum number of engines, split evenly across FD001-FD004 (0 = all). "
        "Whole engines are subsampled, never single windows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_training_parquet(data_dir=args.data_dir, max_engines=args.max_engines, seed=args.seed)
