from types import SimpleNamespace

import numpy as np
import pytest
from mlflow.exceptions import MlflowException

from train import (
    _checkpoint_score,
    _split_train_val_test,
    _evaluate_quality_gate,
    _should_replace_champion,
    _training_data_fingerprint,
)


def _fake_windows(
    engine_window_counts: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic X/y/groups with `n` windows per engine, each window filled with a
    constant per-engine value."""
    X, y, groups = [], [], []
    for engine_index, (engine, n) in enumerate(engine_window_counts.items()):
        for i in range(n):
            X.append(np.full((3, 2), fill_value=float(engine_index), dtype=float))
            y.append(i % 3)
            groups.append(engine)
    return np.array(X), np.array(y), np.array(groups)


def test_quality_gate_requires_both_thresholds_to_pass():
    # Good overall F2, poor Critical recall: F2 alone would pass, and this is
    # exactly the model critical_recall_threshold exists to reject.
    assert not _evaluate_quality_gate(
        f2_weighted=0.90, critical_recall=0.40, f2_threshold=0.75, critical_recall_threshold=0.75
    )


def test_quality_gate_rejects_good_critical_recall_with_bad_f2():
    assert not _evaluate_quality_gate(
        f2_weighted=0.50, critical_recall=0.95, f2_threshold=0.75, critical_recall_threshold=0.75
    )


def test_quality_gate_passes_when_both_thresholds_are_met():
    assert _evaluate_quality_gate(
        f2_weighted=0.80, critical_recall=0.80, f2_threshold=0.75, critical_recall_threshold=0.75
    )


def test_training_data_fingerprint_is_deterministic_and_split_sensitive():
    X, y, groups = _fake_windows({"ENG_A": 4, "ENG_B": 4})

    def fp(X_, val_split=0.2, test_split=0.2, seed=42):
        return _training_data_fingerprint(X_, y, groups, val_split, test_split, seed)

    base = fp(X)

    assert base == fp(X)
    assert base != fp(X, val_split=0.3)
    assert base != fp(X, test_split=0.3)
    assert base != fp(X, seed=7)
    assert base != fp(X * 2)


class _FakeClient:
    """The bare minimum of MlflowClient that _should_replace_champion uses."""

    def __init__(self, fingerprint: str | None = None, f2: float | None = None) -> None:
        self._has_champion = fingerprint is not None
        self._fingerprint = fingerprint
        self._f2 = f2

    def get_model_version_by_alias(self, name, alias):
        if not self._has_champion:
            raise MlflowException(f"Registered model alias {alias} not found.")
        return SimpleNamespace(version="3", run_id="run-champion")

    def get_run(self, run_id):
        metrics = {} if self._f2 is None else {"test_f2_weighted": self._f2}
        return SimpleNamespace(
            data=SimpleNamespace(
                tags={"training_data_fingerprint": self._fingerprint}, metrics=metrics
            )
        )


def test_champion_is_replaced_when_there_is_none_yet():
    assert _should_replace_champion(_FakeClient(), "m", "fp", challenger_f2=0.1)


def test_champion_is_kept_when_challenger_is_worse_on_the_same_data():
    client = _FakeClient(fingerprint="fp", f2=0.90)
    assert not _should_replace_champion(client, "m", "fp", challenger_f2=0.80)


def test_champion_is_replaced_when_challenger_is_better_on_the_same_data():
    client = _FakeClient(fingerprint="fp", f2=0.80)
    assert _should_replace_champion(client, "m", "fp", challenger_f2=0.85)


def test_gate_alone_decides_when_runs_are_not_comparable():
    # Same "worse" F2, but a different data fingerprint: not comparable.
    client = _FakeClient(fingerprint="other-data", f2=0.99)
    assert _should_replace_champion(client, "m", "fp", challenger_f2=0.50)


def test_checkpoint_that_passes_the_gate_beats_one_with_higher_f2_that_fails():
    # Seen on real data: the higher F2 came with Critical recall below the floor.
    thresholds = {"f2_threshold": 0.75, "critical_recall_threshold": 0.75}
    fails_gate = _checkpoint_score(0.826, 0.61, **thresholds)
    passes_gate = _checkpoint_score(0.807, 0.86, **thresholds)

    assert passes_gate > fails_gate


def test_checkpoint_score_falls_back_to_f2_when_nothing_passes_the_gate():
    thresholds = {"f2_threshold": 0.99, "critical_recall_threshold": 0.99}
    assert _checkpoint_score(0.80, 0.5, **thresholds) > _checkpoint_score(0.70, 0.9, **thresholds)


def test_three_way_split_keeps_every_engine_in_exactly_one_partition():
    groups = np.repeat([f"ENG_{i}" for i in range(20)], 10)

    train, val, test = _split_train_val_test(groups, val_split=0.2, test_split=0.2, seed=42)

    train_e, val_e, test_e = (set(groups[idx]) for idx in (train, val, test))
    assert train_e.isdisjoint(val_e)
    assert train_e.isdisjoint(test_e)
    assert val_e.isdisjoint(test_e)
    assert len(train_e | val_e | test_e) == 20
    # 20% of the engines for test and 20% for val (of the total).
    assert len(test_e) == 4
    assert len(val_e) == 4


def test_three_way_split_falls_back_below_three_engines():
    groups = np.repeat(["ENG_A", "ENG_B"], 5)

    train, val, test = _split_train_val_test(groups, val_split=0.2, test_split=0.2, seed=42)

    assert len(train) == len(val) == len(test) == len(groups)


def test_three_way_split_rejects_fractions_that_leave_no_train():
    groups = np.repeat([f"ENG_{i}" for i in range(10)], 3)

    with pytest.raises(ValueError, match="sum to less than 1"):
        _split_train_val_test(groups, val_split=0.5, test_split=0.5, seed=42)
