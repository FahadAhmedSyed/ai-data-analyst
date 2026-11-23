"""An independent check of the gold answers: 45 of them recomputed with pandas.

The gold SQL is run by DuckDB on data read by the application's ingestion. Here the same
generated files are read again by plain pandas (no application code) and the answers worked
out by hand in pandas. If the SQL, the ingestion or the scoring had a slip, the two would
disagree. A pandas answer is judged with the same ``compare_results`` the benchmark uses, so
the check also shows that a correct answer in another shape (other column names, extra
columns, integer for float) is accepted.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from eval.generate_datasets import generate
from eval.scoring import compare_results


def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(generate(name)), **kw)


def _sheet(name: str, sheet: str, skip: int) -> pd.DataFrame:
    data = io.BytesIO(generate(name))
    return pd.read_excel(data, sheet_name=sheet, skiprows=skip, engine="openpyxl")


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    sales = _csv("sales", encoding="utf-8-sig", dtype=str)
    sales["order_date"] = pd.to_datetime(sales["Order Date"], format="%d/%m/%Y")
    for column in ("Unit Price ($)", "Revenue ($)"):
        sales[column] = sales[column].str.replace(r"[$,]", "", regex=True).astype(float)
    sales = sales.rename(
        columns={"Region": "region", "Product": "product", "Category": "category",
                 "Quantity": "quantity", "Revenue ($)": "revenue", "Customer ID": "customer",
                 "Order ID": "order_id"}
    )  # fmt: skip
    sales["quantity"] = sales["quantity"].astype(int)
    sales["order_id"] = sales["order_id"].astype(int)

    survey = _csv("survey", sep=";", encoding="cp1252")
    survey["at"] = pd.to_datetime(survey["Submitted At"], format="%d/%m/%Y %H:%M")
    survey = survey.rename(
        columns={"Age group": "age", "Channel": "channel", "Satisfaction": "sat",
                 "Likelihood to recommend": "nps", "Comment": "comment"}
    )  # fmt: skip

    rain = _sheet("rainfall", "Daily", 2).rename(
        columns={"Date": "date", "Site": "site", "Rainfall (mm)": "mm",
                 "Max Temp (°C)": "high", "Min Temp (°C)": "low"}
    )  # fmt: skip
    rain["date"] = pd.to_datetime(rain["date"])

    water = _csv("water_quality").rename(columns={"Site": "site", "Sample ID": "sample"})

    finance = _csv("finance").rename(
        columns={"Department": "dept", "Budget": "budget", "Actual": "actual"}
    )
    finance["month"] = pd.to_datetime(finance["Month"], format="%Y-%m")

    incidents = _sheet("incidents", "Incidents", 1).rename(
        columns={"Reported On": "reported", "Resolved On": "resolved", "Site": "site",
                 "Severity": "severity", "Cost": "cost", "Incident ID": "id"}
    )  # fmt: skip
    incidents["reported"] = pd.to_datetime(incidents["reported"])
    incidents["resolved"] = pd.to_datetime(incidents["resolved"])
    return {
        "sales": sales, "survey": survey, "rainfall": rain, "water_quality": water,
        "finance": finance, "incidents": incidents,
    }  # fmt: skip


def one(value) -> pd.DataFrame:
    return pd.DataFrame({"answer": [value]})


def counts(series: pd.Series, **kw) -> pd.DataFrame:
    return series.value_counts(**kw).rename_axis("key").reset_index()


def _days_to_resolve(incidents: pd.DataFrame) -> pd.DataFrame:
    return incidents.assign(days=(incidents.resolved - incidents.reported).dt.days)


# Each answer is worked out from the raw files, without the application and without SQL.
ANSWERS = {
    "S01": lambda f: one(f["sales"].revenue.sum()),
    "S02": lambda f: one(len(f["sales"])),
    "S03": lambda f: one(f["rainfall"].mm.max()),
    "S04": lambda f: one(f["survey"].sat.mean()),
    "S05": lambda f: one(len(f["incidents"])),
    "S06": lambda f: one(f["water_quality"]["sample"].nunique()),
    "S07": lambda f: one(f["sales"].revenue.mean()),
    "S08": lambda f: one(f["rainfall"].site.nunique()),
    "S09": lambda f: one(f["survey"].nps.median()),
    "S10": lambda f: one(f["incidents"].cost.sum()),
    "F01": lambda f: one(
        f["sales"].query("category == 'Electronics' and region == 'North'").revenue.sum()
    ),
    "F02": lambda f: one(len(f["rainfall"].query("site == 'Site C' and mm > 25"))),
    "F04": lambda f: one(len(f["incidents"].query("severity == 'High' and resolved.isna()"))),
    "F05": lambda f: f["sales"][
        (f["sales"].revenue > 5000)
        & (f["sales"].order_date >= "2024-03-01")
        & (f["sales"].order_date < "2024-04-01")
    ][["order_id", "order_date", "revenue"]],
    "F06": lambda f: one((f["sales"].customer == "C007").sum()),
    "F07": lambda f: one((f["survey"].sat <= 2).sum()),
    "F08": lambda f: f["rainfall"].query("site == 'Site A' and mm > 40")[["date", "mm"]],
    "F09": lambda f: one((f["incidents"].cost > 10000).sum()),
    "G01": lambda f: f["sales"].groupby("region", dropna=False).revenue.sum().reset_index()
    .sort_values("revenue", ascending=False),
    "G02": lambda f: f["sales"].groupby("product").quantity.sum().nlargest(5).reset_index(),
    "G04": lambda f: f["rainfall"][f["rainfall"].date.dt.year == 2023]
    .groupby("site").mm.sum().reset_index(),
    "G06": lambda f: f["sales"].groupby("customer").revenue.sum().nlargest(3).reset_index(),
    "G09": lambda f: f["water_quality"].groupby("site")["sample"].nunique().reset_index(),
    "D01": lambda f: f["rainfall"][f["rainfall"].date.dt.month.isin([12, 1, 2])]
    .groupby("site").mm.mean().reset_index(),
    "D03": lambda f: one(
        f["sales"][f["sales"].order_date > f["sales"].order_date.max() - pd.DateOffset(months=12)]
        .revenue.sum()
    ),
    "D05": lambda f: one((f["incidents"].resolved - f["incidents"].reported).dt.days.mean()),
    "D06": lambda f: f["rainfall"].query("site == 'Site B'").groupby(f["rainfall"].date.dt.year)
    .mm.sum().nlargest(1).reset_index(),
    "D07": lambda f: f["sales"].query("order_date.dt.year == 2024")
    .groupby(f["sales"].order_date.dt.quarter).revenue.sum().reset_index(),
    "D08": lambda f: one(
        f["rainfall"].query("date >= '2023-06-01' and date < '2023-09-01'").mm.sum()
    ),
    "D10": lambda f: one(len(f["incidents"].query("reported >= '2024-07-01'"))),
    "D11": lambda f: f["finance"].query("month.dt.year == 2024").groupby("month").actual.sum()
    .nlargest(1).reset_index(),
    "N01": lambda f: f["rainfall"].groupby("site").mm.apply(lambda s: s.isna().sum()).reset_index(),
    "N02": lambda f: one(f["finance"].query("month.dt.year == 2024").actual.sum()),
    "N03": lambda f: one(100 * f["survey"].comment.notna().mean()),
    "N04": lambda f: one(f["finance"].duplicated().sum()),
    "N06": lambda f: f["rainfall"].query("mm == 0").groupby("site").size().reset_index(),
    "M01": lambda f: f["sales"].groupby("category").revenue
    .agg(["size", "sum", "mean"]).reset_index(),
    "M05": lambda f: _days_to_resolve(f["incidents"]).groupby("severity")
    .agg(n=("id", "size"), cost=("cost", "sum"), days=("days", "max")).reset_index(),
    "W04": lambda f: f["sales"].groupby("product").revenue.sum().sort_values(ascending=False)
    .reset_index().assign(rank=lambda d: np.arange(1, len(d) + 1)),
    "W06": lambda f: f["finance"].groupby("dept").actual.sum().sort_values(ascending=False)
    .reset_index().assign(share=lambda d: 100 * d.actual / d.actual.sum()),
    "T03": lambda f: f["incidents"].query("severity == 'High'").groupby("site").size()
    .nlargest(1).reset_index(),
    "I01": lambda f: counts(f["survey"].channel).head(1),
    "I03": lambda f: counts(f["survey"].comment.dropna()),
    "E01": lambda f: one(None),
    "E02": lambda f: one(len(f["survey"].query("at.dt.year == 2023"))),
}  # fmt: skip


def test_the_answers_cover_a_good_share_of_the_cases(shipped_cases) -> None:
    ids = {c.id for c in shipped_cases}
    assert set(ANSWERS) <= ids
    assert len(ANSWERS) >= 45


@pytest.mark.parametrize("case_id", sorted(ANSWERS))
def test_the_gold_answer_agrees_with_pandas(case_id: str, frames, shipped_cases, pool) -> None:
    case = next(c for c in shipped_cases if c.id == case_id)
    expected = pool.get(case.dataset).gold.expected(case.gold_sql)
    candidate = ANSWERS[case_id](frames)
    verdict = compare_results(candidate, expected, case.compare)
    assert verdict.matched, f"{case_id}: {verdict.reason}\n{candidate}\n{expected.frame}"
