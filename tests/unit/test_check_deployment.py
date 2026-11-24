"""The live-site checker (T1.20): run against a small local server, never the internet."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from scripts import check_deployment as cd

PAGE = "<!doctype html><html><head><title>Streamlit</title></head><body></body></html>"


def routes(**overrides: tuple[int, str]) -> dict[str, tuple[int, str]]:
    """A healthy copy: the routes it serves and what each returns."""
    healthy = {"/": (200, PAGE), cd.HEALTH_PATH: (200, "ok")}
    return {**healthy, **overrides}


class Handler(BaseHTTPRequestHandler):
    table: ClassVar[dict[str, tuple[int, str]]] = {}

    def do_GET(self) -> None:
        status, body = self.table.get(self.path, (200, PAGE))  # like Streamlit: any path
        if status in (301, 302):
            self.send_response(status)
            self.send_header("Location", body)
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *_args: object) -> None:
        """Keep the test output clean."""


@pytest.fixture
def serve() -> Iterator:
    servers: list[HTTPServer] = []

    def start(table: dict[str, tuple[int, str]]) -> str:
        handler = type("H", (Handler,), {"table": table})
        server = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def by_name(results: list[cd.Result]) -> dict[str, cd.Result]:
    return {r.name: r for r in results}


# -- the address ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://app.up.railway.app", "https://app.up.railway.app"),
        ("https://app.up.railway.app/", "https://app.up.railway.app"),
        ("  https://app.up.railway.app/some/path?x=1  ", "https://app.up.railway.app"),
        ("http://localhost:8501", "http://localhost:8501"),
    ],
)
def test_the_address_is_reduced_to_scheme_and_host(given: str, expected: str) -> None:
    assert cd.normalise(given) == expected


@pytest.mark.parametrize(
    "given", ["", "app.up.railway.app", "ftp://x.example", "https://", "file:///etc"]
)
def test_an_address_that_is_not_a_website_is_refused(given: str) -> None:
    with pytest.raises(ValueError, match="give the address"):
        cd.normalise(given)


def test_a_bad_address_exits_with_two_and_says_why(capsys: pytest.CaptureFixture) -> None:
    assert cd.main(["not-an-address"]) == 2
    assert "give the address" in capsys.readouterr().err


def test_fetch_refuses_other_schemes_without_opening_anything() -> None:
    reply = cd.fetch("file:///etc/passwd")
    assert reply.status == 0
    assert reply.body == ""


# -- the checks against a real (local) server -----------------------------------------------------


def test_a_healthy_copy_on_this_computer_passes_without_https(serve) -> None:
    results = cd.run_checks(serve(routes()))
    assert all(r.ok for r in results), [r for r in results if not r.ok]
    assert "plain HTTP is sent to HTTPS" not in by_name(results)  # only for an https address


def test_a_public_address_over_plain_http_fails_the_https_check() -> None:
    results = by_name(cd.run_checks("http://app.example", get=fake({})))
    assert not results["served over HTTPS"].ok
    assert "plain HTTP is sent to HTTPS" not in results


@pytest.mark.parametrize("host", ["localhost:8501", "127.0.0.1:8501", "[::1]:8501"])
def test_every_loopback_spelling_counts_as_this_computer(host: str) -> None:
    assert cd._is_local(f"http://{host}")
    assert cd._uses_https(f"http://{host}").ok


def test_a_health_check_that_does_not_say_ok_fails(serve) -> None:
    results = by_name(cd.run_checks(serve(routes(**{cd.HEALTH_PATH: (200, "starting")}))))
    assert not results["health check answers ok"].ok


def test_a_server_error_on_the_health_check_fails_and_says_the_status(serve) -> None:
    results = by_name(cd.run_checks(serve(routes(**{cd.HEALTH_PATH: (500, "")}))))
    assert results["health check answers ok"].detail == "status 500"


def test_a_page_that_does_not_load_fails(serve) -> None:
    results = by_name(cd.run_checks(serve(routes(**{"/": (502, "")}))))
    assert not results["the page loads"].ok


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/.env", "OPENAI_API_KEY=abc"),
        ("/.env", "APP_PASSWORD=hunter2"),
        ("/.git/config", "[core]\n\trepositoryformatversion = 0"),
        ("/.streamlit/secrets.toml", "key = '" + "sk-" + "x" * 30 + "'"),
    ],
)
def test_a_served_secret_file_fails_and_is_named(serve, path: str, body: str) -> None:
    result = by_name(cd.run_checks(serve(routes(**{path: (200, body)}))))
    leak = result["no secret or source file is served"]
    assert not leak.ok
    assert path in leak.detail


def test_the_apps_own_page_returned_for_a_secret_path_is_not_a_leak(serve) -> None:
    # Streamlit answers every unknown path with its page; that is not the file.
    assert by_name(cd.run_checks(serve(routes())))["no secret or source file is served"].ok


def test_nothing_answering_fails_every_check_that_needs_an_answer() -> None:
    results = by_name(
        cd.run_checks("http://127.0.0.1:9", get=lambda _url: cd.Response(0, error="X"))
    )
    assert not results["health check answers ok"].ok
    assert not results["the page loads"].ok
    assert results["health check answers ok"].detail == "X"


# -- the redirect check (needs an https address, so the fetcher is replaced) ----------------------


def fake(table: dict[str, cd.Response]) -> cd.Fetch:
    return lambda url: table.get(url, cd.Response(200, PAGE))


def https_table(**overrides: cd.Response) -> dict[str, cd.Response]:
    base = "https://app.example"
    return {
        base + cd.HEALTH_PATH: cd.Response(200, "ok"),
        base + "/": cd.Response(200, PAGE),
        "http://app.example/": cd.Response(301, location="https://app.example/"),
        **overrides,
    }


def test_a_healthy_https_copy_passes_every_check() -> None:
    results = cd.run_checks("https://app.example", get=fake(https_table()))
    assert all(r.ok for r in results), [r for r in results if not r.ok]
    assert len(results) == 5


def test_plain_http_that_serves_the_page_instead_of_redirecting_fails() -> None:
    table = https_table(**{"http://app.example/": cd.Response(200, PAGE)})
    result = by_name(cd.run_checks("https://app.example", get=fake(table)))
    assert not result["plain HTTP is sent to HTTPS"].ok


def test_a_redirect_to_another_plain_http_address_fails() -> None:
    table = https_table(**{"http://app.example/": cd.Response(301, location="http://other/")})
    result = by_name(cd.run_checks("https://app.example", get=fake(table)))
    assert not result["plain HTTP is sent to HTTPS"].ok


def test_nothing_listening_on_plain_http_is_fine() -> None:
    table = https_table(**{"http://app.example/": cd.Response(0, error="URLError")})
    result = by_name(cd.run_checks("https://app.example", get=fake(table)))
    assert result["plain HTTP is sent to HTTPS"].ok


def test_a_redirect_is_reported_not_followed(serve) -> None:
    base = serve({"/": (302, "https://elsewhere.example/")})
    reply = cd.fetch(base + "/")
    assert reply.status == 302
    assert reply.location == "https://elsewhere.example/"


# -- the report and the exit status ---------------------------------------------------------------


def test_the_report_names_each_check_and_reminds_about_the_form() -> None:
    text = cd.report([cd.Result("a", True), cd.Result("b", False, "status 500")])
    assert "PASS  a" in text
    assert "FAIL  b (status 500)" in text
    assert "1 check(s) failed" in text
    assert "private window" in text


def test_the_report_says_when_everything_passed() -> None:
    assert "All checks passed." in cd.report([cd.Result("a", True)])


def test_main_exits_one_when_a_check_fails_and_zero_when_none_does(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(cd, "run_checks", lambda _base: [cd.Result("a", False)])
    assert cd.main(["https://app.example"]) == 1
    monkeypatch.setattr(cd, "run_checks", lambda _base: [cd.Result("a", True)])
    assert cd.main(["https://app.example"]) == 0
    assert "All checks passed." in capsys.readouterr().out
