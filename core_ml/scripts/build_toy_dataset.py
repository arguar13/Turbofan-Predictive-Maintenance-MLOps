"""Build the DVC-tracked "toy" dataset (1,116 rows).

Takes a FIXED, deterministic subset of train_FD001.txt (engines `TOY_UNITS`) and
copies the raw lines verbatim, without re-parsing or reformatting, so the sample
is byte-for-byte real production data.

Usage:
    poetry -C core_ml run python scripts/build_toy_dataset.py

Then version the result:
    dvc add data_toy
    git add data_toy.dvc
    dvc push
"""

from pathlib import Path

TOY_UNITS = {1, 2, 3, 4, 5}
SOURCE_DATASET = "FD001"

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILE = BASE_DIR / "data" / f"train_{SOURCE_DATASET}.txt"
TARGET_DIR = BASE_DIR / "data_toy"
TARGET_FILE = TARGET_DIR / f"train_{SOURCE_DATASET}.txt"


def build_toy_dataset() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {SOURCE_FILE}. "
            "Run 'dvc pull' in core_ml/ to download the raw data."
        )

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    kept_lines = []
    with open(SOURCE_FILE, encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            unit_number = int(stripped.split()[0])
            if unit_number in TOY_UNITS:
                kept_lines.append(line)

    if not kept_lines:
        raise RuntimeError("The unit filter kept no rows; check TOY_UNITS.")

    with open(TARGET_FILE, "w", encoding="utf-8") as target:
        target.writelines(kept_lines)

    print(
        f"Toy dataset written to {TARGET_FILE}: {len(kept_lines)} rows "
        f"(units {sorted(TOY_UNITS)} of {SOURCE_DATASET})."
    )


if __name__ == "__main__":
    build_toy_dataset()
