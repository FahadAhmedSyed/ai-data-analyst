"""Read an uploaded Excel workbook into a text table (T1.04, ADR 0008).

Workbooks are opened in memory with python-calamine, the Rust reader behind
pandas' ``calamine`` engine, which handles both ``.xlsx`` and legacy ``.xls``.
Cell values are converted to text in a fixed, unambiguous format (ISO dates,
shortest round-trip numbers) so typing in ``normalize.py`` treats CSV and Excel
input the same way.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import islice

import pandas as pd
from python_calamine import CalamineSheet, CalamineWorkbook

from analyst.config.settings import Settings, get_settings
from analyst.core.errors import IngestionError
from analyst.core.logging import get_logger
from analyst.ingestion.tabular import (
    RawTable,
    apply_header,
    check_limits,
    detect_header_row,
    drop_empty,
)

log = get_logger(__name__)

ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# An .xlsx is a zip of XML; real workbooks compress about 10:1 (spike d).
MAX_UNCOMPRESSED_RATIO = 10
MAX_ZIP_ENTRIES = 10_000
HEADER_ALLOWANCE = 32


@dataclass(frozen=True)
class SheetInfo:
    """A worksheet as offered to the user for selection."""

    name: str
    visible: bool
    rows: int
    columns: int


def _check_container(data: bytes, settings: Settings) -> None:
    """Size, signature and zip-bomb checks, done before any parsing."""
    if len(data) > settings.max_upload_bytes:
        raise IngestionError(
            "too_large",
            f"{len(data)} bytes",
            user_message=f"This file is larger than the {settings.max_upload_mb} MB upload limit.",
        )
    if not data:
        raise IngestionError("empty", "zero-byte workbook")
    if data.startswith(OLE_MAGIC):
        return  # legacy .xls: no zip container to inspect
    if not data.startswith(ZIP_MAGIC):
        raise IngestionError(
            "corrupt",
            "no xlsx or xls signature",
            user_message="This doesn't look like an Excel workbook. If it's a CSV file, "
            "rename it to .csv and upload it again.",
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise IngestionError("corrupt", f"bad zip: {exc}") from exc
    uncompressed = sum(entry.file_size for entry in entries)
    if len(entries) > MAX_ZIP_ENTRIES or uncompressed > MAX_UNCOMPRESSED_RATIO * (
        settings.max_upload_bytes
    ):
        raise IngestionError(
            "too_large",
            f"{len(entries)} entries, {uncompressed} bytes uncompressed",
            user_message="This workbook expands to far more data than the upload limit allows.",
        )


def _open(data: bytes, settings: Settings) -> CalamineWorkbook:
    _check_container(data, settings)
    try:
        return CalamineWorkbook.from_filelike(io.BytesIO(data))
    except Exception as exc:  # calamine raises several unrelated error types
        raise IngestionError("corrupt", f"{type(exc).__name__}: {exc}") from exc


def list_sheets(data: bytes, *, settings: Settings | None = None) -> list[SheetInfo]:
    """List the workbook's sheets with their visibility and used size."""
    settings = settings or get_settings()
    workbook = _open(data, settings)
    infos = []
    for meta in workbook.sheets_metadata:
        sheet = workbook.get_sheet_by_name(meta.name)
        infos.append(
            SheetInfo(
                name=meta.name,
                visible="Hidden" not in str(meta.visible),
                rows=sheet.height,
                columns=sheet.width,
            )
        )
    return infos


def cell_to_text(value: object) -> str | None:
    """Convert one cell value to text, or ``None`` for an empty cell.

    Dates become ISO ``YYYY-MM-DD``; date-times ``YYYY-MM-DD HH:MM:SS``; whole
    numbers lose the ``.0`` that Excel's floating-point storage adds; other
    numbers use Python's shortest round-trip form; booleans become ``true`` or
    ``false``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat(timespec="seconds")
    if isinstance(value, timedelta):
        return str(value)
    return str(value)


def _choose_sheet(workbook: CalamineWorkbook, sheet: str | None) -> CalamineSheet:
    names = workbook.sheet_names
    if not names:
        raise IngestionError("empty", "workbook has no sheets")
    if sheet is None:
        visible = [m.name for m in workbook.sheets_metadata if "Hidden" not in str(m.visible)]
        sheet = (visible or names)[0]
    if sheet not in names:
        raise IngestionError(
            "corrupt",
            f"sheet {sheet!r} not in workbook",
            user_message=f"The workbook has no sheet named '{sheet}'.",
        )
    return workbook.get_sheet_by_name(sheet)


def load_excel(
    data: bytes,
    source_name: str,
    *,
    sheet: str | None = None,
    header_row: int | None = None,
    detect_header: bool = True,
    settings: Settings | None = None,
) -> RawTable:
    """Read one worksheet into a ``RawTable``.

    Args:
        data: The uploaded workbook's bytes (``.xlsx`` or ``.xls``).
        source_name: The file name, kept for display.
        sheet: Worksheet name; defaults to the first visible sheet.
        header_row: Override the detected header row (zero-based).
        detect_header: When ``header_row`` is not given, detect it (``True``) or
            treat the sheet as having no header (``False``).
        settings: Limits to apply; defaults to the process settings.

    Raises:
        IngestionError: For oversized, corrupt, empty or over-limit workbooks,
            and for unknown sheet names.
    """
    settings = settings or get_settings()
    workbook = _open(data, settings)
    worksheet = _choose_sheet(workbook, sheet)

    limit = settings.max_rows + HEADER_ALLOWANCE
    rows = [[cell_to_text(v) for v in row] for row in islice(worksheet.iter_rows(), limit)]
    if not rows or not any(any(v is not None for v in row) for row in rows):
        raise IngestionError("empty", f"sheet {worksheet.name!r} has no values")
    width = max(len(r) for r in rows)
    grid = pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype="string")

    if header_row is None and detect_header:
        header_row = detect_header_row(grid)
    table, skipped = apply_header(grid, header_row)
    table = drop_empty(table)
    check_limits(table, max_rows=settings.max_rows, max_columns=settings.max_columns)

    log.info(
        "excel_loaded",
        extra={
            "bytes": len(data),
            "format": "xls" if data.startswith(OLE_MAGIC) else "xlsx",
            "sheets": len(workbook.sheet_names),
            "header_row": header_row,
            "rows": len(table),
            "columns": table.shape[1],
        },
    )
    return RawTable(
        dataframe=table,
        source_name=source_name,
        sheet_name=worksheet.name,
        encoding=None,
        delimiter=None,
        header_row=header_row,
        skipped_rows=skipped,
    )
