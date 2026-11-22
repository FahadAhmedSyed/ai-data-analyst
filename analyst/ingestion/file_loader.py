"""Entry point for uploads: choose the loader by file extension."""

from __future__ import annotations

from pathlib import PurePath

from analyst.config.settings import Settings
from analyst.core.errors import IngestionError
from analyst.ingestion.csv_loader import load_csv
from analyst.ingestion.excel_loader import load_excel
from analyst.ingestion.tabular import RawTable

CSV_EXTENSIONS = frozenset({".csv", ".txt"})
EXCEL_EXTENSIONS = frozenset({".xlsx", ".xls"})
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS


def file_kind(filename: str) -> str:
    """Return ``"csv"`` or ``"excel"`` for a supported file name, else raise."""
    suffix = PurePath(filename).suffix.lower()
    if suffix in CSV_EXTENSIONS:
        return "csv"
    if suffix in EXCEL_EXTENSIONS:
        return "excel"
    raise IngestionError("unsupported_type", f"extension {suffix!r}")


def load_file(
    data: bytes,
    filename: str,
    *,
    sheet: str | None = None,
    delimiter: str | None = None,
    header_row: int | None = None,
    detect_header: bool = True,
    settings: Settings | None = None,
) -> RawTable:
    """Load an uploaded file as a ``RawTable``, dispatching on its extension.

    ``sheet`` applies to Excel files and ``delimiter`` to CSV files; the other
    is ignored.
    """
    if file_kind(filename) == "csv":
        return load_csv(
            data,
            filename,
            delimiter=delimiter,
            header_row=header_row,
            detect_header=detect_header,
            settings=settings,
        )
    return load_excel(
        data,
        filename,
        sheet=sheet,
        header_row=header_row,
        detect_header=detect_header,
        settings=settings,
    )
