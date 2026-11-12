import pytest
from pydantic import ValidationError

from schemas import SensorWindowRequest


def test_valid_window_is_accepted():
    instance = SensorWindowRequest(
        engine_id="ENG_001", readings=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    assert instance.engine_id == "ENG_001"
    assert len(instance.readings) == 3


def test_rejects_ragged_matrix():
    with pytest.raises(ValidationError, match="rectangular"):
        SensorWindowRequest(engine_id="ENG_001", readings=[[1.0, 2.0], [3.0], [5.0, 6.0]])


def test_rejects_empty_rows():
    with pytest.raises(ValidationError, match="cannot be empty"):
        SensorWindowRequest(engine_id="ENG_001", readings=[[], []])


def test_rejects_nan_values():
    with pytest.raises(ValidationError, match="non-finite"):
        SensorWindowRequest(engine_id="ENG_001", readings=[[1.0, float("nan")], [3.0, 4.0]])


def test_rejects_infinite_values():
    with pytest.raises(ValidationError, match="non-finite"):
        SensorWindowRequest(engine_id="ENG_001", readings=[[1.0, float("inf")], [3.0, 4.0]])


def test_rejects_blank_engine_id():
    with pytest.raises(ValidationError):
        SensorWindowRequest(engine_id="   ", readings=[[1.0, 2.0]])


def test_rejects_empty_readings():
    with pytest.raises(ValidationError):
        SensorWindowRequest(engine_id="ENG_001", readings=[])
