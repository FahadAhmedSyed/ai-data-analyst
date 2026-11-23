"""Rules the interface code keeps, checked from its source (security.md, architecture section 3).

The page is drawn by Streamlit, which tests exercise in ``tests/e2e``. These checks look at
what no run of the page can show: that a whole class of mistake is not in the code at all.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pandas as pd
import pytest

import analyst
import analyst.ui
from analyst.database.session_db import SessionDatabase
from analyst.session.state import AnalystSession, ChatEntry

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(analyst.__file__).parent
UI = PACKAGE / "ui"


def sources(*roots: Path) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        found += [root] if root.is_file() else sorted(root.rglob("*.py"))
    return found


def imported(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def code_without_comments_and_docstrings(path: Path) -> str:
    """The source with docstrings blanked, so a rule that is *explained* in one does not trip."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            first = body[0] if body else None
            is_text = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            if is_text and isinstance(first.value.value, str):  # type: ignore[union-attr]
                first.value = ast.Constant(value="")  # type: ignore[union-attr]
    return ast.unparse(tree)


# -- no markup, no shared caches -------------------------------------------------------------------


@pytest.mark.parametrize("path", sources(PACKAGE, ROOT / "app.py"), ids=lambda p: p.name)
def test_no_page_code_allows_html_in_text(path: Path) -> None:
    assert "unsafe_allow_html" not in code_without_comments_and_docstrings(path)
    assert "unsafe_allow_javascript" not in code_without_comments_and_docstrings(path)


@pytest.mark.parametrize("path", sources(UI), ids=lambda p: p.name)
def test_no_page_code_uses_streamlits_shared_caches(path: Path) -> None:
    """``st.cache_data`` and ``st.cache_resource`` are shared by every browser session (T8)."""
    code = code_without_comments_and_docstrings(path)
    assert not re.search(r"\bcache_(data|resource)\b|\bst\.cache\b|\bexperimental_memo\b", code)
    assert "lru_cache" not in code and "functools.cache" not in code


@pytest.mark.parametrize("path", sources(UI), ids=lambda p: p.name)
def test_no_page_code_writes_user_data_to_disk(path: Path) -> None:
    code = code_without_comments_and_docstrings(path)
    assert not re.search(
        r"\bopen\(|write_bytes|write_text|\bto_csv\(|tempfile|NamedTemporary", code
    )


# -- layers ----------------------------------------------------------------------------------------


def test_streamlit_is_only_imported_by_the_interface() -> None:
    outside = [
        path.relative_to(ROOT).as_posix()
        for path in sources(PACKAGE)
        if UI not in path.parents and any(m.split(".")[0] == "streamlit" for m in imported(path))
    ]
    assert outside == []


@pytest.mark.parametrize("path", sources(UI), ids=lambda p: p.name)
def test_the_interface_does_not_reach_around_the_session(path: Path) -> None:
    """It never touches the database, the SQL checks or the model's parts: the session does.

    What it may take from those layers is a type or a constant (``LLMClient``, the question
    length), never the thing that runs SQL or calls the model.
    """
    banned_libraries = {"duckdb", "sqlglot", "openai"}
    banned_modules = (
        "analyst.database",
        "analyst.sql.validator",
        "analyst.pipeline.query_loop",
        "analyst.pipeline.generator",
        "analyst.pipeline.explainer",
    )
    for name in imported(path):
        assert name.split(".")[0] not in banned_libraries, name
        assert not name.startswith(banned_modules), name


# -- nothing of a user's lives in a module ---------------------------------------------------------


def interface_modules() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(analyst.ui.__path__, "analyst.ui.")]


def test_the_interface_has_the_modules_the_rules_above_cover() -> None:
    assert {
        "analyst.ui.app",
        "analyst.ui.chat",
        "analyst.ui.dataset_view",
        "analyst.ui.gate",
        "analyst.ui.notices",
        "analyst.ui.session_store",
        "analyst.ui.sidebar",
    } <= set(interface_modules())


@pytest.mark.parametrize("name", interface_modules())
def test_no_interface_module_holds_a_session_a_table_or_bytes_at_module_level(name: str) -> None:
    """Module globals are shared by every browser session; only constants may live there."""
    module = importlib.import_module(name)
    holding = [
        key
        for key, value in vars(module).items()
        if not key.startswith("__")
        and isinstance(
            value,
            AnalystSession | SessionDatabase | ChatEntry | pd.DataFrame | bytes | bytearray | list,
        )
        and not (isinstance(value, list) and key == "__all__")
    ]
    assert holding == []
