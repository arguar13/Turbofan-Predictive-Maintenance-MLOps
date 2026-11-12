"""Train the predictive-maintenance ConvTransformer with full MLflow lineage.

Every run is tied to the tuple Git commit + DVC data hash + hyperparameters +
MLflow run ID + container image tag, through the run's tags and params (see
`_build_lineage_tags`).

Input: `engine_features.parquet`, built from the raw C-MAPSS .txt files by
`prepare_training_data.py`. The toy dataset (local smoke test, seconds, no GPU)
and the full dataset go through exactly the same path.

The model and the scaler are never loose files: both are artifacts of the same
MLflow run. A version is registered, and given the "champion" alias the API
serves, only if:
  1. it passes the quality gate: weighted F2 AND Critical-class recall, each
     against its own threshold in config.yaml's `monitoring.*` (see
     `_evaluate_quality_gate`). Plain accuracy would be blind to the asymmetric
     cost of calling a Critical engine Healthy or Alert; and
  2. it is not worse than the current champion when both were evaluated on the
     same data and split (see `_should_replace_champion`).
The gate is decided on a TEST set of engines that take part neither in training
nor in checkpoint selection (that uses VAL), so the certifying metrics are not
biased by the selection. A rejected model stays as an auditable run and never
reaches the Registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

# Only called with a fixed argv (no external input), see _get_git_commit_sha.
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from config_loader import load_config
from data_contracts import validate_training_batch
from data_paths import prepared_dir
from data_processing import fit_window_scaler, scale_windows
from logging_config import configure_logging

log = configure_logging("train")

BASE_DIR = Path(__file__).resolve().parent.parent
# Repository root (parent of core_ml/), where the project's .git lives.
REPO_ROOT = BASE_DIR.parent
NUM_CLASSES = 3
CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = "Predictive_Maintenance_ConvTransformer"
# 0=Healthy, 1=Alert, 2=Critical (see data_processing.py::build_multiclass_target).
# A false negative on this class (an engine that is really Critical called
# Healthy/Alert) is the cost the whole quality gate below exists to bound.
CRITICAL_CLASS_INDEX = 2
CLASS_NAMES = ("Healthy", "Alert", "Critical")


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Without it self-attention is invariant to the order of the 30 timesteps,
    and that order is precisely the degradation signal.
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ConvTransformer(nn.Module):
    """Hybrid CNN + self-attention classifier over a window of sensor readings.

    Degradation shows up as short trends spread across correlated sensors,
    which a local Conv1d captures with few parameters; attention then relates
    any two timesteps of the window without an RNN's sequential bottleneck.
    Deliberately small (d_model=64, 2 layers) so the full dataset trains on CPU.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = NUM_CLASSES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_features, out_channels=d_model, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window_size, num_features]
        x = x.transpose(1, 2)  # [batch, num_features, window_size] for Conv1d
        x = self.relu(self.bn(self.conv(x)))
        x = x.transpose(1, 2)  # [batch, window_size, d_model] for the batch_first encoder
        x = self.pos_encoder(x)
        out = self.transformer(x)
        pooled = out.mean(dim=1)  # global average pooling over the timesteps
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Lineage: Git commit + DVC data hash + container image tag
# ---------------------------------------------------------------------------


def _get_git_commit_sha() -> str:
    """CI's variable first (a runner checkout can be a shallow clone), then `git rev-parse`."""
    ci_sha = os.environ.get("CI_COMMIT_SHA") or os.environ.get("GITHUB_SHA")
    if ci_sha:
        return ci_sha
    try:
        # Fixed argv and shell=False: bandit flags subprocess regardless, but there
        # is no injection surface. cwd=REPO_ROOT, not core_ml/: a stray nested .git
        # in core_ml/ would otherwise be resolved and tag the wrong commit.
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _is_git_worktree_dirty() -> str:
    """ "true" if tracked files have uncommitted changes.

    Without this flag, a run trained on locally modified code is attributed to a
    commit that does not contain that code, and the lineage tuple looks
    reproducible when it is not. Untracked files (data, venvs) do not count.
    """
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        return "true" if result.stdout.strip() else "false"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _get_dvc_data_hash(data_dir: str) -> str:
    """md5 recorded in `<data_dir>.dvc` (data.dvc or data_toy.dvc)."""
    dvc_file = BASE_DIR / f"{data_dir}.dvc"
    if not dvc_file.exists():
        return "unknown"
    with open(dvc_file) as file:
        metadata = yaml.safe_load(file) or {}
    outs = metadata.get("outs") or [{}]
    return outs[0].get("md5", "unknown")


def _get_container_image_tag() -> str:
    return os.environ.get("IMAGE_TAG") or os.environ.get("CI_COMMIT_SHA") or "local-dev"


def _build_lineage_tags(data_dir: str) -> dict[str, str]:
    return {
        "git_commit_sha": _get_git_commit_sha(),
        "git_dirty": _is_git_worktree_dirty(),
        "dvc_data_hash": _get_dvc_data_hash(data_dir),
        "dvc_data_dir": data_dir,
        "container_image_tag": _get_container_image_tag(),
    }


def _training_data_fingerprint(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> str:
    """Hash of the exact arrays and split parameters the run trains and tests on.

    `dvc_data_hash` only reflects the committed `.dvc` pointer, not what is on
    disk (a locally modified data/ reports the same hash) nor the subsampling in
    prepare_training_data.py. This fingerprint is what decides whether two runs
    are comparable metric by metric.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    digest.update("|".join(map(str, groups)).encode())
    digest.update(f"val_split={val_split};test_split={test_split};seed={seed}".encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def _load_training_data(data_path: Path) -> pd.DataFrame:
    parquet_path = data_path / "engine_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run first:\n"
            f"  poetry run python src/prepare_training_data.py --data-dir {data_path.name}"
        )
    return pd.read_parquet(parquet_path)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def _group_split_indices(
    indices: np.ndarray, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split `indices` in two without ever separating the windows of one engine."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    first, second = next(splitter.split(indices, groups=groups[indices]))
    return indices[first], indices[second]


def _split_train_val_test(
    groups: np.ndarray, val_split: float, test_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / val / test indices, grouped by engine.

    - train: fits the weights.
    - val: picks the checkpoint and drives early stopping.
    - test: ONLY decides the quality gate and the champion comparison.

    Gating on val would be optimistic: the chosen epoch is, by construction, the
    one that fits val best. Test engines never influence any training decision,
    so the gate measures real generalisation. `val_split` and `test_split` are
    fractions of ENGINES, not of windows.
    """
    if not (0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1):
        raise ValueError(
            f"val_split ({val_split}) and test_split ({test_split}) must lie in (0, 1) "
            "and sum to less than 1."
        )
    all_indices = np.arange(len(groups))
    n_engines = len(np.unique(groups))
    if n_engines < 3:
        # Only the tiny smoke-test dataset gets here: three splits need three engines.
        log.warning("too_few_engines_for_three_way_split", n_engines=n_engines)
        return all_indices, all_indices, all_indices

    rest, test = _group_split_indices(all_indices, groups, test_split, seed)
    # val_split is a fraction of the total, so it is rescaled to what is left after test.
    train, val = _group_split_indices(rest, groups, val_split / (1 - test_split), seed)
    return train, val, test


def _predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.argmax(model(X), dim=1).numpy()


def _evaluate(model: nn.Module, X: torch.Tensor, y: np.ndarray) -> dict[str, float]:
    """Accuracy, weighted F2 and Critical recall of the model on (X, y)."""
    return _gate_metrics(y, _predict(model, X))


def _gate_metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """The metrics that decide the checkpoint and the quality gate."""
    return {
        "accuracy": float((predictions == y).mean()),
        "f2_weighted": float(
            fbeta_score(y, predictions, beta=2, average="weighted", zero_division=0)
        ),
        "critical_recall": float(
            recall_score(
                y, predictions, average=None, labels=np.arange(NUM_CLASSES), zero_division=0
            )[CRITICAL_CLASS_INDEX]
        ),
    }


def _test_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, majority_class: int
) -> tuple[dict[str, float], dict]:
    """Metrics that expose the weak spots, not just the ones the gate uses.

    Weighted metrics are dominated by Healthy (~2/3 of the windows); macro F1 and
    the per-class figures expose the hardest class (Alert). The "always predict
    the train majority class" baseline shows how much the model actually adds.
    Returns (scalar metrics, full report).
    """
    labels = list(range(NUM_CLASSES))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    metrics = {
        "test_f1_macro": report["macro avg"]["f1-score"],
        "test_f1_weighted": report["weighted avg"]["f1-score"],
    }
    for name in CLASS_NAMES:
        for key in ("precision", "recall", "f1-score"):
            metrics[f"test_{name.lower()}_{key.replace('-score', '')}"] = report[name][key]

    baseline = np.full_like(y_true, majority_class)
    baseline_report = classification_report(
        y_true, baseline, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    metrics["baseline_test_accuracy"] = baseline_report["accuracy"]
    metrics["baseline_test_f1_macro"] = baseline_report["macro avg"]["f1-score"]

    artifact = {
        "classification_report": report,
        "confusion_matrix": {
            "labels": list(CLASS_NAMES),
            "rows_are": "true class",
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "baseline_majority_class": CLASS_NAMES[majority_class],
    }
    return metrics, artifact


def _should_replace_champion(
    client: MlflowClient, model_name: str, fingerprint: str, challenger_f2: float
) -> bool:
    """Champion/challenger: never replace a better champion with a worse model.

    Passing the gate is not enough: a retrain can clear the thresholds and still
    be worse than what is serving. Metrics are compared only when both runs share
    the data/split fingerprint (same test engines); otherwise only the gate
    decides, because F2 measured on different engines is not comparable.
    """
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return True  # no champion yet

    if champion.run_id is None:
        return True
    champion_run = client.get_run(champion.run_id)
    if champion_run.data.tags.get("training_data_fingerprint") != fingerprint:
        log.info("champion_not_comparable", champion_version=champion.version)
        return True

    champion_f2 = champion_run.data.metrics.get("test_f2_weighted")
    if champion_f2 is None or challenger_f2 >= champion_f2:
        return True

    log.warning(
        "challenger_worse_than_champion",
        champion_version=champion.version,
        champion_f2_weighted=champion_f2,
        challenger_f2_weighted=challenger_f2,
    )
    return False


def _evaluate_quality_gate(
    f2_weighted: float,
    critical_recall: float,
    f2_threshold: float,
    critical_recall_threshold: float,
) -> bool:
    """Both thresholds must pass, not either one.

    F2 already weighs recall twice as much as precision, but a weighted F2 can
    stay high on the strength of Healthy and Alert (the majority classes) while
    the model systematically misses Critical, the class that matters most. The
    Critical recall floor closes that gap.
    """
    return f2_weighted >= f2_threshold and critical_recall >= critical_recall_threshold


def _checkpoint_score(
    f2_weighted: float,
    critical_recall: float,
    f2_threshold: float,
    critical_recall_threshold: float,
) -> tuple[bool, float]:
    """Sort key for the best checkpoint: (passes the gate, F2).

    Ranking by F2 alone ignores the Critical recall floor the gate also demands.
    Seen on the real data: the best-F2 epoch (0.826) had Critical recall 0.61 and
    failed the gate, while another epoch (F2 0.807, recall 0.86) passed it. With
    this key any passing epoch beats a failing one, and F2 breaks ties. It does
    not force the gate: if no epoch passes, the best-F2 one is restored and the
    gate fails anyway.
    """
    passes = _evaluate_quality_gate(
        f2_weighted, critical_recall, f2_threshold, critical_recall_threshold
    )
    return passes, f2_weighted


def train_pipeline(
    data_dir: str = "data",
    epochs: int = 3,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    val_split: float = 0.2,
    test_split: float = 0.2,
    enforce_quality_gate: bool = True,
    seed: int = 42,
    patience: int = 5,
) -> str:
    # Weight init and DataLoader shuffling are otherwise non-deterministic: the
    # same commit, data and hyperparameters could pass the gate in one run and
    # fail it in the next. Same seed as the engine split below.
    torch.manual_seed(seed)
    config = load_config()
    window_size = config["model"]["window_size"]
    model_name = config["model"]["model_name"]
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["model"]["mlflow_tracking_uri"])
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = prepared_dir(data_dir)

    # Without the feature order the scaler cannot carry column names, and the API
    # could not publish what each row must contain.
    feature_names_path = data_path / "feature_names.json"
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"{feature_names_path} not found. Run prepare_training_data.py "
            f"--data-dir {data_dir} first (it writes the parquet and feature_names.json together)."
        )
    feature_names: list[str] = json.loads(feature_names_path.read_text())

    log.info("loading_training_data", data_path=str(data_path))
    training_data = _load_training_data(data_path)

    if len(training_data) == 0:
        raise ValueError(f"The training dataset in {data_path} is empty.")

    # Validate the batch contract before building tensors (see data_contracts.py).
    sample_array_length = len(training_data["windowed_features"].iloc[0])
    training_data = validate_training_batch(training_data, expected_length=sample_array_length)

    if sample_array_length % window_size != 0:
        raise ValueError(
            f"windowed_features has length {sample_array_length}, not a multiple of "
            f"window_size={window_size}: the parquet was built with another config."
        )
    num_features = sample_array_length // window_size
    if len(feature_names) != num_features:
        raise ValueError(
            f"feature_names.json declares {len(feature_names)} features and the parquet "
            f"{num_features}: run prepare_training_data.py again."
        )
    log.info(
        "features_detected", num_features=num_features, sample_array_length=sample_array_length
    )

    X = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(window_size, num_features)
            for value in training_data["windowed_features"]
        ]
    )
    y = training_data["failure_type"].to_numpy()
    groups = training_data["engine_id"].to_numpy()
    fingerprint = _training_data_fingerprint(X, y, groups, val_split, test_split, seed)

    train_idx, val_idx, test_idx = _split_train_val_test(groups, val_split, test_split, seed)
    # Scaler fitted on train engines only: val/test statistics would leak into the
    # checkpoint choice and the certification.
    scaler = fit_window_scaler(X[train_idx], feature_names)
    X_train, y_train = scale_windows(scaler, X[train_idx]), y[train_idx]
    X_val, y_val = scale_windows(scaler, X[val_idx]), y[val_idx]
    X_test, y_test = scale_windows(scaler, X[test_idx]), y[test_idx]
    log.info(
        "engine_split",
        train_engines=len(np.unique(groups[train_idx])),
        val_engines=len(np.unique(groups[val_idx])),
        test_engines=len(np.unique(groups[test_idx])),
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True
    )

    model = ConvTransformer(num_features=num_features, num_classes=NUM_CLASSES)
    # "balanced" class weights from y_train only: the windows are dominated by
    # Healthy, and an unweighted loss lets the model all but ignore Critical while
    # still reporting high accuracy. Classes absent from y_train (tiny or heavily
    # subsampled data) keep weight 1 instead of making compute_class_weight raise.
    present_classes = np.unique(y_train)
    class_weights = np.ones(NUM_CLASSES)
    class_weights[present_classes] = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=y_train
    )
    if len(present_classes) < NUM_CLASSES:
        log.warning("classes_missing_in_train_split", present=present_classes.tolist())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lineage_tags = _build_lineage_tags(data_dir)
    f2_threshold = config["monitoring"]["f2_weighted_threshold"]
    critical_recall_threshold = config["monitoring"]["critical_recall_threshold"]

    with mlflow.start_run(run_name="ConvTransformer_Training") as run:
        mlflow.set_tags({**lineage_tags, "training_data_fingerprint": fingerprint})
        mlflow.log_params(
            {
                "window_size": window_size,
                "num_features": num_features,
                "epochs": epochs,
                "patience": patience,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "val_split": val_split,
                "test_split": test_split,
                "seed": seed,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
            }
        )

        # The checkpoint is chosen with the gate's own criteria (_checkpoint_score),
        # not by val accuracy or by assuming the last epoch is best: on the full
        # dataset val accuracy collapsed from overfitting late in training (epoch
        # 20: train loss 0.21, accuracy 0.18) while an earlier epoch generalised
        # better. Picking by one metric and gating by another could promote a model
        # that is not the best by the gate's standard.
        # `epochs` is a ceiling: training stops after `patience` epochs without an F2
        # improvement, and the restored checkpoint is always the best one. Patience
        # tracks plain F2, not the checkpoint key: tracking the key would freeze the
        # counter at the first gate-passing epoch and stop before better ones.
        best_score: tuple[bool, float] | None = None
        best_f2_seen = -1.0
        best_f2_weighted = -1.0
        best_val_accuracy = -1.0
        best_critical_recall = -1.0
        best_state_dict: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            val_metrics = _evaluate(model, X_val_tensor, y_val)
            epoch_val_accuracy = val_metrics["accuracy"]
            epoch_f2_weighted = val_metrics["f2_weighted"]
            epoch_critical_recall = val_metrics["critical_recall"]
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("val_f2_weighted", epoch_f2_weighted, step=epoch)
            mlflow.log_metric("val_critical_recall", epoch_critical_recall, step=epoch)

            log.info(
                "epoch_completed",
                epoch=epoch + 1,
                total_epochs=epochs,
                train_loss=avg_loss,
                val_accuracy=epoch_val_accuracy,
                val_f2_weighted=epoch_f2_weighted,
                val_critical_recall=epoch_critical_recall,
            )

            epoch_score = _checkpoint_score(
                epoch_f2_weighted, epoch_critical_recall, f2_threshold, critical_recall_threshold
            )
            if best_score is None or epoch_score > best_score:
                best_score = epoch_score
                best_f2_weighted = epoch_f2_weighted
                best_val_accuracy = epoch_val_accuracy
                best_critical_recall = epoch_critical_recall
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch_f2_weighted > best_f2_seen:
                best_f2_seen = epoch_f2_weighted
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    log.info(
                        "early_stopping",
                        epoch=epoch + 1,
                        patience=patience,
                        best_f2_seen=best_f2_seen,
                        restored_checkpoint_f2=best_f2_weighted,
                    )
                    break

        # Set by the first epoch whenever epochs >= 1. An explicit error rather
        # than an assert, which optimised bytecode strips.
        if best_state_dict is None:
            raise RuntimeError("epochs must be >= 1: no training epoch was completed.")
        model.load_state_dict(best_state_dict)
        # Val metrics are reported for diagnosis; the gate uses ONLY test (unseen engines).
        test_predictions = _predict(model, X_test_tensor)
        test_metrics = _gate_metrics(y_test, test_predictions)
        test_accuracy = test_metrics["accuracy"]
        f2_weighted = test_metrics["f2_weighted"]
        critical_recall = test_metrics["critical_recall"]
        mlflow.log_metrics(
            {
                "best_val_accuracy": best_val_accuracy,
                "best_val_f2_weighted": best_f2_weighted,
                "best_val_critical_recall": best_critical_recall,
                "test_accuracy": test_accuracy,
                "test_f2_weighted": f2_weighted,
                "test_critical_recall": critical_recall,
            }
        )
        majority_class = int(np.bincount(y_train, minlength=NUM_CLASSES).argmax())
        diagnostics, test_report = _test_diagnostics(y_test, test_predictions, majority_class)
        mlflow.log_metrics(diagnostics)
        mlflow.log_dict(test_report, "evaluation/test_report.json")

        log.info(
            "evaluation_completed",
            val_f2_weighted=best_f2_weighted,
            val_critical_recall=best_critical_recall,
            test_accuracy=test_accuracy,
            test_f2_weighted=f2_weighted,
            test_critical_recall=critical_recall,
            test_f1_macro=diagnostics["test_f1_macro"],
            test_alert_recall=diagnostics["test_alert_recall"],
            f2_threshold=f2_threshold,
            critical_recall_threshold=critical_recall_threshold,
        )

        example_input = X_train_tensor[:1]
        signature = mlflow.models.infer_signature(
            example_input.numpy(), model(example_input).detach().numpy()
        )
        # Logged WITHOUT registered_model_name: registering is a decision taken
        # after the gate, not a side effect of logging.
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input,
            signature=signature,
            # MLflow 3 defaults to torch.export ("pt2"), which loads as a
            # GraphModule without .eval()/.train() and breaks the serving API.
            # "pickle" keeps a regular nn.Module.
            serialization_format="pickle",
        )

        # The scaler fitted on train travels with the model, as an artifact of the
        # same run: it is what the API downloads to scale incoming windows.
        with tempfile.TemporaryDirectory() as tmp_dir:
            scaler_file = Path(tmp_dir) / "scaler.joblib"
            joblib.dump(scaler, scaler_file)
            mlflow.log_artifact(str(scaler_file), artifact_path="preprocessing")

        quality_gate_passed = _evaluate_quality_gate(
            f2_weighted, critical_recall, f2_threshold, critical_recall_threshold
        )
        mlflow.set_tag("quality_gate_passed", str(quality_gate_passed))
        mlflow.set_tag("quality_gate_f2_threshold", str(f2_threshold))
        mlflow.set_tag("quality_gate_critical_recall_threshold", str(critical_recall_threshold))

        run_id = run.info.run_id
        log.info("lineage_tuple", mlflow_run_id=run_id, **lineage_tags)

        client = MlflowClient()
        if quality_gate_passed and _should_replace_champion(
            client, model_name, fingerprint, f2_weighted
        ):
            registered = mlflow.register_model(model_info.model_uri, model_name)
            client.set_registered_model_alias(
                name=model_name, alias=CHAMPION_ALIAS, version=str(registered.version)
            )
            mlflow.set_tag("promoted_version", str(registered.version))
            log.info(
                "quality_gate_passed",
                test_accuracy=test_accuracy,
                test_f2_weighted=f2_weighted,
                critical_recall=critical_recall,
                promoted_version=registered.version,
                alias=CHAMPION_ALIAS,
            )
        elif quality_gate_passed:
            mlflow.set_tag("promoted_version", "none: the current champion is better")
        else:
            log.warning(
                "quality_gate_failed",
                test_accuracy=test_accuracy,
                test_f2_weighted=f2_weighted,
                critical_recall=critical_recall,
                f2_threshold=f2_threshold,
                critical_recall_threshold=critical_recall_threshold,
            )

    if enforce_quality_gate and not quality_gate_passed:
        raise SystemExit(
            f"Quality gate failed (test): f2_weighted={f2_weighted:.4f} "
            f"(threshold={f2_threshold}), critical_recall={critical_recall:.4f} "
            f"(threshold={critical_recall_threshold}). "
            "Stopping here: review the data and hyperparameters."
        )

    return run_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping: stop after N consecutive epochs without a val_f2_weighted "
        "improvement (the best checkpoint is always restored, never the last).",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of engines for validation (checkpoint selection).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of engines for test: they only decide the quality gate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-enforce-quality-gate",
        dest="enforce_quality_gate",
        action="store_false",
        help="Do not fail the process when the quality gate fails (for smoke tests on the "
        "toy dataset, which cannot certify model quality).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        enforce_quality_gate=args.enforce_quality_gate,
        seed=args.seed,
    )
