"""HTTP helpers shared by the per-dataset download scripts.

Everything here goes through urllib with an explicit SSL context so the
scripts work on Python builds that ship without a usable CA bundle (the
python.org macOS installers, notably) as long as `certifi` is importable.
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "pictsure-testing/1.0"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


SSL_CONTEXT = _ssl_context()


def resolve_hf_token() -> str | None:
    """HF token from the environment or the huggingface-cli cache, if any.

    Every dataset and model used here is public; a token only helps to avoid
    anonymous rate limits.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        return cached.read_text().strip()
    return None


def _open(url: str, token: str | None, timeout: int):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=timeout)


def fetch_bytes(url: str, token: str | None = None, timeout: int = 120, retries: int = 3) -> bytes:
    """GET `url`, retrying on transient 5xx / connection errors."""
    for attempt in range(retries):
        try:
            with _open(url, token, timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_json(url: str, token: str | None = None, timeout: int = 120, retries: int = 3):
    return json.loads(fetch_bytes(url, token=token, timeout=timeout, retries=retries))


def download(url: str, dest: Path, token: str | None = None, quiet: bool = False) -> bool:
    """Download `url` to `dest` unless it is already there. True if fetched."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_bytes(url, token=token)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    if not quiet:
        print(f"  fetched {dest.parent.name}/{dest.name}")
    return True


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")
