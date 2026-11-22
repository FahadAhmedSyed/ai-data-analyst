"""Read an uploaded CSV file into a text table (T1.03).

The file is decoded and parsed entirely in memory; nothing is written to disk.
Every cell is kept as text so that typing (``normalize.py``) controls how dates
and numbers are interpreted.
"""

from __future__ import annotations

import codecs
import csv
import io
from collections import Counter

import pandas as pd
from charset_normalizer import from_bytes

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

CANDIDATE_DELIMITERS = (",", ";", "\t", "|")
SNIFF_BYTES = 64 * 1024
SNIFF_LINES = 50
_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls
# Rows read beyond the row limit so a file just over the limit is detected,
# allowing for a header and preamble rows.
HEADER_ALLOWANCE = 32


def detect_encoding(data: bytes) -> str:
    """Return the text encoding of ``data``.

    Order: byte-order mark, then strict UTF-8, then Windows-1252, then
    charset-normalizer on a sample. Windows-1252 is tried before statistical
    detection because spreadsheet exports from Windows in New Zealand and other
    Western regions use it, and detectors often mistake short Western text for
    Central European or Baltic code pages. Raises ``IngestionError("encoding")``
    if nothing fits.
    """
    for bom, name in _BOMS:
        if data.startswith(bom):
            return name
    for candidate in ("utf-8", "cp1252"):
        try:
            data.decode(candidate)
        except UnicodeDecodeError:
            continue
        return candidate
    best = from_bytes(data[:SNIFF_BYTES]).best()
    if best is None or not best.encoding:
        raise IngestionError("encoding", "charset-normalizer found no match")
    return best.encoding


def _sample_lines(text: str) -> str:
    """First complete lines of the text, for delimiter detection."""
    sample = text[:SNIFF_BYTES]
    if len(text) > SNIFF_BYTES and "\n" in sample:
        sample = sample[: sample.rindex("\n")]
    return "\n".join(sample.splitlines()[:SNIFF_LINES])


def detect_delimiter(text: str) -> str:
    """Pick the delimiter that splits the first lines into the most consistent field counts.

    Each candidate is tried with a real CSV reader, so delimiters inside quoted
    fields are ignored. Ties favour the order comma, semicolon, tab, pipe. A
    file with a single column gets a comma.
    """
    sample = _sample_lines(text)
    best = ","
    best_score = (0.0, 0)
    for delim in CANDIDATE_DELIMITERS:
        try:
            counts = [len(row) for row in csv.reader(io.StringIO(sample), delimiter=delim) if row]
        except csv.Error:
            continue
        if not counts:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        if mode < 2:
            continue
        score = (freq / len(counts), mode)
        if score > best_score:
            best, best_score = delim, score
    return best


def load_csv(
    data: bytes,
    source_name: str,
    *,
    delimiter: str | None = None,
    header_row: int | None = None,
    detect_header: bool = True,
    settings: Settings | None = None,
) -> RawTable:
    """Read CSV bytes into a ``RawTable``.

    Args:
        data: The uploaded file's bytes.
        source_name: The file name, kept for display.
        delimiter: Override the detected delimiter.
        header_row: Override the detected header row (zero-based).
        detect_header: When ``header_row`` is not given, detect it (``True``) or
            treat the file as having no header (``False``).
        settings: Limits to apply; defaults to the process settings.

    Raises:
        IngestionError: For oversized, empty, binary, undecodable or malformed files,
            and for files over the row or column limits.
    """
    settings = settings or get_settings()
    if len(data) > settings.max_upload_bytes:
        raise IngestionError(
            "too_large",
            f"{len(data)} bytes",
            user_message=f"This file is larger than the {settings.max_upload_mb} MB upload limit.",
        )
    if not data.strip():
        raise IngestionError("empty", "file is empty or whitespace")
    if data.startswith(_ZIP_MAGIC) or data.startswith(_OLE_MAGIC):
        raise IngestionError(
            "corrupt",
            "binary spreadsheet with a text extension",
            user_message="This looks like an Excel file saved with a CSV name. "
            "Rename it to .xlsx or .xls and upload it again.",
        )

    encoding = detect_encoding(data)
    try:
        text = data.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise IngestionError("encoding", f"{encoding}: {exc}") from exc
    if "\x00" in text:
        raise IngestionError("corrupt", "NUL characters in decoded text")

    sep = delimiter or detect_delimiter(text)
    try:
        grid = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            header=None,
            dtype="string",
            keep_default_na=False,
            na_values=[""],
            skip_blank_lines=True,
            nrows=settings.max_rows + HEADER_ALLOWANCE,
            engine="c",
        )
    except pd.errors.EmptyDataError as exc:
        raise IngestionError("empty", str(exc)) from exc
    except pd.errors.ParserError as exc:
        raise IngestionError(
            "corrupt",
            str(exc),
            user_message="Some rows have more fields than the first rows. Check the delimiter "
            "or for unquoted commas inside values.",
        ) from exc

    if header_row is None and detect_header:
        header_row = detect_header_row(grid)
    table, skipped = apply_header(grid, header_row)
    table = drop_empty(table)
    check_limits(table, max_rows=settings.max_rows, max_columns=settings.max_columns)

    log.info(
        "csv_loaded",
        extra={
            "bytes": len(data),
            "encoding": encoding,
            "delimiter": sep,
            "header_row": header_row,
            "rows": len(table),
            "columns": table.shape[1],
        },
    )
    return RawTable(
        dataframe=table,
        source_name=source_name,
        sheet_name=None,
        encoding=encoding,
        delimiter=sep,
        header_row=header_row,
        skipped_rows=skipped,
    )
