"""Regenerate simple.xls. Run once with: uv run --with xlwt python tests/fixtures/excel/make_xls.py

xlwt is not a project dependency; the generated file is committed.
"""

from datetime import date
from pathlib import Path

import xlwt

book = xlwt.Workbook()
sheet = book.add_sheet("Readings")
date_style = xlwt.easyxf(num_format_str="DD/MM/YYYY")
sheet.write(0, 0, "Site")
sheet.write(0, 1, "Date")
sheet.write(0, 2, "Rainfall (mm)")
rows = [("Site A", date(2024, 4, 3), 1.5), ("Site B", date(2024, 4, 4), 12.0)]
for i, (site, day, rain) in enumerate(rows, start=1):
    sheet.write(i, 0, site)
    sheet.write(i, 1, day, date_style)
    sheet.write(i, 2, rain)
book.add_sheet("Notes").write(0, 0, "Collected by the council")
book.save(str(Path(__file__).with_name("simple.xls")))
