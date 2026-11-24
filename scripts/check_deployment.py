"""Check a running copy of the app from outside (T1.20; see docs/deployment.md).

    python scripts/check_deployment.py https://your-app.up.railway.app

Standard library only, so it runs anywhere without installing the app. It makes plain HTTP
requests and checks what can be checked that way: the address answers over HTTPS, the health
check says ok, the page loads, plain HTTP is sent to HTTPS, and a few files that must never be
served are not. It sends no password and no data, and it cannot see inside the page: Streamlit
draws it over a WebSocket. Open the address in a private window as well, and make sure the
password form comes first.

Exit status: 0 when every check passed, 1 when one failed, 2 for a bad address.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

HEALTH_PATH = "/_stcore/health"
TIMEOUT_S = 15
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# Streamlit answers any path with its page, so a file is judged by what the body contains.
SECRET_PATHS = ("/.env", "/.git/config", "/.streamlit/secrets.toml")
SECRET_MARKERS = re.compile(r"OPENAI_API_KEY|APP_PASSWORD|repositoryformatversion|\bsk-[\w-]{20,}")
MAX_BODY_BYTES = 200_000


@dataclass(frozen=True)
class Response:
    """What came back from one request. ``status`` is 0 when nothing answered."""

    status: int
    body: str = ""
    location: str = ""
    error: str = ""


Fetch = Callable[[str], Response]


@dataclass(frozen=True)
class Result:
    """One line of the report."""

    name: str
    ok: bool
    detail: str = ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report a redirect instead of following it, so the check can see where it points."""

    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


def fetch(url: str) -> Response:
    """GET ``url`` without following redirects. Never raises."""
    if urlsplit(url).scheme not in ("http", "https"):
        return Response(0, error="not an http or https address")
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": "deployment-check/1"})  # noqa: S310
    try:
        with opener.open(request, timeout=TIMEOUT_S) as reply:
            body = reply.read(MAX_BODY_BYTES).decode("utf-8", "replace")
            return Response(reply.status, body, reply.headers.get("Location", ""))
    except urllib.error.HTTPError as err:
        return Response(err.code, "", err.headers.get("Location", "") if err.headers else "")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        return Response(0, error=type(err).__name__)


def normalise(address: str) -> str:
    """The base address without a trailing slash. Raises ``ValueError`` if it is not a site."""
    parts = urlsplit(address.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        msg = "give the address as https://host (or http://host to check a copy on this computer)"
        raise ValueError(msg)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _health(base: str, get: Fetch) -> Result:
    reply = get(base + HEALTH_PATH)
    if reply.status == 200 and reply.body.strip() == "ok":
        return Result("health check answers ok", True)
    return Result("health check answers ok", False, reply.error or f"status {reply.status}")


def _page(base: str, get: Fetch) -> Result:
    reply = get(base + "/")
    if reply.status == 200 and "<html" in reply.body.lower():
        return Result("the page loads", True)
    return Result("the page loads", False, reply.error or f"status {reply.status}")


def _is_local(base: str) -> bool:
    return urlsplit(base).hostname in LOCAL_HOSTS


def _uses_https(base: str) -> Result:
    name = "served over HTTPS"
    if base.startswith("https://"):
        return Result(name, True)
    if _is_local(base):
        return Result(name, True, "a copy on this computer, so HTTPS is not expected")
    return Result(name, False, "the address starts with http://")


def _http_redirects(base: str, get: Fetch) -> Result:
    name = "plain HTTP is sent to HTTPS"
    reply = get("http://" + base.removeprefix("https://") + "/")
    if reply.status in (301, 302, 307, 308) and reply.location.startswith("https://"):
        return Result(name, True)
    if reply.status == 0:
        return Result(name, True, "nothing answers on plain HTTP")
    return Result(name, False, f"status {reply.status}, not a redirect to https://")


def _secrets_not_served(base: str, get: Fetch) -> Result:
    name = "no secret or source file is served"
    leaks = []
    for path in SECRET_PATHS:
        reply = get(base + path)
        if reply.status == 200 and SECRET_MARKERS.search(reply.body):
            leaks.append(path)
    if leaks:
        return Result(name, False, "served: " + ", ".join(leaks))
    return Result(name, True)


def run_checks(base: str, get: Fetch = fetch) -> list[Result]:
    """Run every check against ``base`` (from ``normalise``)."""
    results = [_uses_https(base), _health(base, get), _page(base, get)]
    if base.startswith("https://"):
        results.append(_http_redirects(base, get))
    results.append(_secrets_not_served(base, get))
    return results


def report(results: list[Result]) -> str:
    """The lines to print."""
    lines = [
        f"{'PASS' if r.ok else 'FAIL'}  {r.name}" + (f" ({r.detail})" if r.detail else "")
        for r in results
    ]
    failed = sum(not r.ok for r in results)
    lines.append("")
    lines.append(
        "All checks passed." if not failed else f"{failed} check(s) failed. See docs/deployment.md."
    )
    lines.append(
        "Still to do by hand: open the address in a private window and confirm the password "
        "form comes first."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Check a running copy of the app.")
    parser.add_argument("address", help="for example https://your-app.up.railway.app")
    args = parser.parse_args(argv)
    try:
        base = normalise(args.address)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    results = run_checks(base)
    print(report(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
