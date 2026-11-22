"""Steps shared by the CSV and Excel loaders.

Both loaders first read a file as a grid of text with no header. The helpers
here find the header row, build unique display names, trim empty rows and
columns, and enforce the row and column limits. Typing happens later
(``normalize.py``), so date and number rules live in one place.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from analyst.core.errors import IngestionError

HEADER_SCAN_ROWS = 30
_NUMBER_LIKE = re.compile(r"^[\s$€£¥+\-(]*[\d.,]+\s*%?\)?\s*$")


@dataclass(frozen=True)
class RawTable:
    """A file read as text: one string column per source column, missing cells as ``<NA>``.

    Attributes:
        dataframe: Text columns named by the user's original headers (made unique).
        source_name: The uploaded file name.
        sheet_name: Worksheet for Excel files, otherwise ``None``.
        encoding: Detected text encoding for CSV files, otherwise ``None``.
        delimiter: Detected or chosen delimiter for CSV files, otherwise ``None``.
        header_row: Zero-based row index of the header in the file, or ``None`` when the
            file has no header and names were generated.
        skipped_rows: Rows above the header that were dropped (titles, notes).
    """

    dataframe: pd.DataFrame
    source_name: str
    sheet_name: str | None
    encoding: str | None
    delimiter: str | None
    header_row: int | None
    skipped_rows: int


def _is_blank(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
        or (isinstance(value, str) and not value.strip())
    )


def _looks_numeric(value: str) -> bool:
    return bool(_NUMBER_LIKE.match(value))


def detect_header_row(grid: pd.DataFrame, scan_rows: int = HEADER_SCAN_ROWS) -> int | None:
    """Return the index of the first row that looks like a header, or ``None``.

    A header row fills at least half of the table's columns and is mostly text
    rather than numbers. Rows above it (titles, notes, blank lines) are treated
    as preamble. If the first such well-filled row is mostly numbers, the file
    has no header and ``None`` is returned.
    """
    width = grid.shape[1]
    if width == 0:
        return None
    needed = max(1, math.ceil(width / 2))
    for idx in range(min(scan_rows, len(grid))):
        cells = [str(v).strip() for v in grid.iloc[idx].tolist() if not _is_blank(v)]
        if len(cells) < needed:
            continue
        text_cells = [c for c in cells if not _looks_numeric(c)]
        return idx if len(text_cells) / len(cells) >= 0.6 else None
    return None


def make_unique_names(raw_names: list[object]) -> list[str]:
    """Clean header cells into unique display names.

    Blank headers become ``Column N`` (1-based position); repeats get `` (2)``,
    `` (3)`` and so on. Surrounding whitespace and line breaks are removed.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for position, raw in enumerate(raw_names, start=1):
        name = "" if _is_blank(raw) else " ".join(str(raw).split())
        if not name:
            name = f"Column {position}"
        key = name.casefold()
        if key in seen:
            seen[key] += 1
            candidate = f"{name} ({seen[key]})"
            while candidate.casefold() in seen:
                seen[key] += 1
                candidate = f"{name} ({seen[key]})"
            name = candidate
            key = name.casefold()
        seen[key] = 1
        names.append(name)
    return names


def apply_header(grid: pd.DataFrame, header_row: int | None) -> tuple[pd.DataFrame, int]:
    """Split a headerless text grid into named columns and data rows.

    Returns the table and the number of preamble rows dropped above the header.
    """
    if header_row is None:
        table = grid.copy()
        table.columns = make_unique_names([None] * grid.shape[1])
        return table.reset_index(drop=True), 0
    if header_row < 0 or header_row >= len(grid):
        raise IngestionError(
            "no_header",
            f"header_row {header_row} outside 0..{len(grid) - 1}",
            user_message="The chosen header row is outside the file.",
        )
    table = grid.iloc[header_row + 1 :].copy()
    table.columns = make_unique_names(grid.iloc[header_row].tolist())
    return table.reset_index(drop=True), header_row


def drop_empty(table: pd.DataFrame) -> pd.DataFrame:
    """Drop rows and generated-name columns that contain no values at all."""
    blank = pd.DataFrame(
        {
            name: (table[name].isna() | table[name].str.strip().eq("").fillna(False)).astype(bool)
            for name in table.columns
        },
        index=table.index,
    )
    table = table.loc[~blank.all(axis=1)]
    empty_cols = [
        name
        for name in table.columns
        if blank.loc[table.index, name].all() and re.fullmatch(r"Column \d+", str(name))
    ]
    return table.drop(columns=empty_cols).reset_index(drop=True)


def check_limits(table: pd.DataFrame, *, max_rows: int, max_columns: int) -> None:
    """Raise ``IngestionError`` if the table is empty or exceeds the limits."""
    if table.shape[1] > max_columns:
        raise IngestionError(
            "too_many_columns",
            f"{table.shape[1]} columns > {max_columns}",
            user_message=f"This file has {table.shape[1]:,} columns; the limit is {max_columns:,}.",
        )
    if len(table) > max_rows:
        raise IngestionError(
            "too_many_rows",
            f"{len(table)} rows > {max_rows}",
            user_message=f"This file has more than {max_rows:,} rows, which is the limit.",
        )
    if len(table) == 0 or table.shape[1] == 0:
        raise IngestionError("empty", "no data rows after the header")
