"""Where each dataset's raw and derived files live.

The raw C-MAPSS .txt files are DVC-tracked (`data.dvc`, `data_toy.dvc`), so their
directory must contain exactly what the pointer describes. Derived files (the
Parquet windows and the column order) go to a sibling, git-ignored
`prepared/<dataset>/`; written next to the raw files, they would invalidate the
DVC pointer every time someone runs the preparation.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def raw_dir(data_dir: str) -> Path:
    """Directory with the raw .txt files; `data_dir` is relative to core_ml/ (or absolute)."""
    return BASE_DIR / data_dir


def prepared_dir(data_dir: str) -> Path:
    """Directory with the derived artifacts of dataset `data_dir`."""
    raw = raw_dir(data_dir)
    return raw.parent / "prepared" / raw.name
