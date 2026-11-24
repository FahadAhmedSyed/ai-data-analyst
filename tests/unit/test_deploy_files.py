"""The deployment files (T1.20): what Railway builds from, and that no secret goes into it."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
RAILWAY = json.loads((ROOT / "railway.json").read_text(encoding="utf-8"))


def instructions() -> list[str]:
    """The Dockerfile's instructions, comments dropped and continuation lines joined."""
    text = re.sub(r"\\\n\s*(#.*\n\s*)*", " ", DOCKERFILE)
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def test_the_image_uses_the_python_the_project_requires() -> None:
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    (base,) = [ln for ln in instructions() if ln.startswith("FROM ")]
    assert requires == ">=3.12,<3.13"
    assert re.fullmatch(r"FROM python:3\.12(\.\d+)?-slim\S*", base)


def test_the_image_installs_the_checked_requirements_file() -> None:
    lines = instructions()
    assert "COPY requirements.txt ./" in lines
    assert "RUN pip install -r requirements.txt" in lines
    assert lines.index("COPY requirements.txt ./") < lines.index("COPY analyst ./analyst")


def test_the_image_copies_only_what_the_app_needs() -> None:
    copies = [ln for ln in instructions() if ln.startswith("COPY ")]
    assert set(copies) == {
        "COPY requirements.txt ./",
        "COPY app.py ./",
        "COPY .streamlit ./.streamlit",
        "COPY analyst ./analyst",
    }
    for line in copies:
        assert (ROOT / line.split()[1]).exists(), line


def test_the_app_runs_as_a_user_that_is_not_root() -> None:
    lines = instructions()
    assert lines.index("USER app") > lines.index("RUN pip install -r requirements.txt")
    assert "USER root" not in lines


def test_a_container_closes_the_app_when_no_password_is_set() -> None:
    assert any("REQUIRE_GATE=true" in ln for ln in instructions() if ln.startswith("ENV "))


def test_no_secret_is_set_in_the_image() -> None:
    text = "\n".join(instructions())
    for name in ("OPENAI_API_KEY", "APP_PASSWORD", "LLM_PRICES"):
        assert name not in text
    names = re.findall(
        r"\b([A-Z][A-Z0-9_]*)=", " ".join(ln for ln in instructions() if ln[:3] in {"ENV", "ARG"})
    )
    assert names  # the pattern found the variables, so the check below looks at something
    for name in names:
        assert name == "REQUIRE_GATE" or not re.search(r"KEY|SECRET|TOKEN|PASSWORD", name), name


def test_the_start_command_uses_the_port_railway_gives_and_listens_on_all_addresses() -> None:
    (cmd,) = [ln for ln in instructions() if ln.startswith("CMD ")]
    command = json.loads(cmd.removeprefix("CMD "))
    assert command[:2] == ["sh", "-c"]
    run = command[2]
    assert run.startswith("exec streamlit run app.py")  # exec: signals reach Streamlit
    assert "--server.port=${PORT:-8501}" in run
    assert "--server.address=0.0.0.0" in run
    assert "--server.headless=true" in run
    assert "--reload" not in run and "developmentMode" not in run


def test_railway_builds_this_dockerfile_and_checks_streamlits_health_route() -> None:
    assert RAILWAY["build"] == {"builder": "DOCKERFILE", "dockerfilePath": "Dockerfile"}
    deploy = RAILWAY["deploy"]
    assert deploy["healthcheckPath"] == "/_stcore/health"
    assert 0 < deploy["healthcheckTimeout"] <= 300
    assert deploy["restartPolicyType"] == "ON_FAILURE"
    assert 0 < deploy["restartPolicyMaxRetries"] <= 10
    assert "startCommand" not in deploy  # the image's CMD is the single place for it


def test_the_health_route_is_the_one_streamlit_serves() -> None:
    from streamlit.web.server.starlette.starlette_routes import ROUTE_HEALTH

    assert "/" + ROUTE_HEALTH.lstrip("/") == RAILWAY["deploy"]["healthcheckPath"]


@pytest.mark.parametrize(
    "pattern", [".env", ".env.*", ".git", ".venv", "eval/data", "uploads", "*.duckdb", "*.db"]
)
def test_secrets_and_data_are_kept_out_of_the_build(pattern: str) -> None:
    assert pattern in DOCKERIGNORE


def test_the_example_settings_file_is_the_one_env_file_that_is_not_ignored() -> None:
    assert "!.env.example" in DOCKERIGNORE
    assert DOCKERIGNORE.index("!.env.example") > DOCKERIGNORE.index(".env.*")


def test_what_the_image_copies_is_not_excluded_from_the_build() -> None:
    for needed in ("requirements.txt", "app.py", ".streamlit", "analyst"):
        assert needed not in DOCKERIGNORE


def test_streamlits_config_is_safe_for_a_public_address() -> None:
    server = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())["server"]
    assert server["enableXsrfProtection"] is True
    assert server["headless"] is True
    assert server.get("enableCORS", True) is True
    assert server.get("runOnSave", False) is False
    assert server.get("developmentMode", False) is False
