"""Small file helpers used by the pipeline stage CLIs."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def read_json(path: Path) -> Any:
    """Read a JSON document.

    Args:
        path: File to read.

    Returns:
        The decoded document.
    """
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, document: Any) -> None:
    """Write a JSON document with stable key order, creating parent directories.

    Args:
        path: Destination file.
        document: JSON-serialisable object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_table(directory: Path, name: str) -> "pd.DataFrame":
    """Read a pipeline table stored as ``<name>.parquet`` or a ``<name>/`` Parquet directory.

    Args:
        directory: Stage output directory.
        name: Table name, e.g. ``"commits"``.

    Returns:
        The table.
    """
    import pyarrow.parquet as pq

    directory = Path(directory)
    single = directory / f"{name}.parquet"
    # Ray Data writes pandas metadata that would turn timestamps into Arrow-backed
    # columns; ignoring it gives plain numpy dtypes, which the feature code expects.
    table = pq.read_table(single if single.exists() else directory / name)
    return table.to_pandas(ignore_metadata=True)
