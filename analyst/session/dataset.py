"""Loading an uploaded file into a session: read, type, store, profile (architecture section 5).

``load_dataset`` is the one place that knows the order of those steps, so the interface only
chooses a file and how to read it. The uploaded bytes are parsed and then let go: nothing
here keeps them (privacy.md, section 2), and nothing is written to disk.

Reading and typing happen before the session's database is touched, so a file that cannot
be read leaves the dataset already loaded as it was. Once the database starts loading the
old table is gone, and a failure after that leaves the session without a dataset; the
caller (``AnalystSession.load``) forgets everything that depended on the old one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from analyst.config.settings import Settings, get_settings
from analyst.core.logging import get_logger
from analyst.core.models import DatasetProfile, QualityIssue
from analyst.database.session_db import SessionDatabase
from analyst.ingestion.file_loader import file_kind, load_file
from analyst.ingestion.normalize import normalize
from analyst.ingestion.profiler import profile_database, quality_summary
from analyst.session.display import preview_frame

log = get_logger(__name__)

HeaderMode = Literal["auto", "row", "none"]


@dataclass(frozen=True)
class ReadOptions:
    """How the user asked for the file to be read (FR-1.2, FR-1.4). The defaults detect it all.

    Attributes:
        sheet: Worksheet of an Excel file; ``None`` for the first visible one.
        delimiter: Separator of a CSV file; ``None`` to detect it.
        header: ``auto`` finds the header row, ``row`` uses ``header_row``, ``none`` says the
            file has no header and names are made up.
        header_row: The header's row number as the user counts it, from 1. Used with ``row``.
    """

    sheet: str | None = None
    delimiter: str | None = None
    header: HeaderMode = "auto"
    header_row: int = 1

    def header_arguments(self) -> tuple[int | None, bool]:
        """``header_row`` and ``detect_header`` for ``load_file``, whose rows count from zero."""
        if self.header == "row":
            return max(self.header_row, 1) - 1, True
        return None, self.header == "auto"


_DELIMITER_NAMES = {",": "commas", ";": "semicolons", "\t": "tabs", "|": "pipes"}


@dataclass(frozen=True)
class ReadInfo:
    """What reading found, for the interface to show beside the preview."""

    kind: str  # "csv" or "excel"
    encoding: str | None
    delimiter: str | None
    header_row: int | None  # as the user counts: from 1; None when the file has no header
    skipped_rows: int
    sheet: str | None = None

    def describe(self) -> str:
        """One sentence on how the file was read, so a wrong guess is easy to spot (FR-1.4)."""
        if self.kind == "csv":
            separator = _DELIMITER_NAMES.get(self.delimiter or "", "a custom separator")
            encoding = f" ({self.encoding})" if self.encoding else ""
            first = f"Read as a text file with columns separated by {separator}{encoding}"
        else:
            first = f"Read from the sheet {self.sheet}" if self.sheet else "Read from the sheet"
        if self.header_row is None:
            second = "no header row was used, so columns are named Column 1, Column 2 and so on"
        else:
            second = f"the header is on row {self.header_row}"
            if self.skipped_rows:
                one = self.skipped_rows == 1
                second += (
                    f", and {self.skipped_rows} {'row' if one else 'rows'} above it "
                    f"{'was' if one else 'were'} left out"
                )
        return f"{first}; {second}."


@dataclass(frozen=True)
class LoadedDataset:
    """Everything the interface shows about a loaded file."""

    profile: DatasetProfile
    issues: tuple[QualityIssue, ...]
    read: ReadInfo
    preview: pd.DataFrame


def load_dataset(
    db: SessionDatabase,
    data: bytes,
    filename: str,
    options: ReadOptions | None = None,
    *,
    settings: Settings | None = None,
) -> LoadedDataset:
    """Read ``data`` and load it into ``db`` (replacing what was there), then profile it.

    Raises:
        IngestionError: The file is unsupported, empty, damaged, too big or too long. This
            is raised before the database changes, except ``too_large`` from the load itself.
    """
    settings = settings or get_settings()
    options = options or ReadOptions()
    started = time.perf_counter()
    kind = file_kind(filename)
    header_row, detect_header = options.header_arguments()
    raw = load_file(
        data,
        filename,
        sheet=options.sheet,
        delimiter=options.delimiter,
        header_row=header_row,
        detect_header=detect_header,
        settings=settings,
    )
    typed = normalize(raw, settings=settings)
    try:
        db.load(typed)
        profile = profile_database(db)
        issues = quality_summary(profile, typed.notes)
        preview = preview_frame(db, profile)
    except BaseException:
        db.close()  # never leave a table that was not profiled
        raise
    log.info(
        "dataset_loaded",
        extra={
            "kind": kind,
            "rows": profile.row_count,
            "columns": profile.column_count,
            "issues": len(issues),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return LoadedDataset(
        profile=profile,
        issues=issues,
        read=ReadInfo(
            kind=kind,
            encoding=raw.encoding,
            delimiter=raw.delimiter,
            header_row=None if raw.header_row is None else raw.header_row + 1,
            skipped_rows=raw.skipped_rows,
            sheet=raw.sheet_name,
        ),
        preview=preview,
    )
