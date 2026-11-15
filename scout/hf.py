"""Cliente mínimo da API do Hugging Face. ÚNICO módulo do projeto que toca a rede.

Só urllib. Timeout em toda requisição, UA genérico, sem token salvo (HF_TOKEN só
se estiver no ambiente). Nunca devolve dado parcial em silêncio: toda falha vira HFError.
Nunca importado por solver, estimate, collect nem parsers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scout import __version__

BASE = "https://huggingface.co"
TIMEOUT_S = 20
MAX_BYTES = 32 * 1024 * 1024
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
DEFAULT_CACHE = Path("~/.cache/localai-scout/hf").expanduser()


class HFError(Exception):
    """reason em {network, gated, not_found, rate_limited, bad_json, offline_cache_miss, bad_repo}."""

    def __init__(self, url: str, status: int | None, reason: str, retry_after: str | None = None):
        super().__init__(f"{reason}: {url}" + (f" (HTTP {status})" if status else ""))
        self.url, self.status, self.reason, self.retry_after = url, status, reason, retry_after


def model_card_url(repo: str) -> str:
    return f"{BASE}/{_check_repo(repo)}"


def _check_repo(repo: str) -> str:
    if not isinstance(repo, str) or not REPO_RE.match(repo) or ".." in repo:
        raise HFError(f"{BASE}/{repo!r}", None, "bad_repo")
    return repo


def _cache_file(cache: Path | None, url: str) -> Path:
    return (cache or DEFAULT_CACHE) / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")


def _read_cache(cache: Path | None, url: str) -> dict | None:
    try:
        doc = json.loads(_cache_file(cache, url).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and "status" in doc else None


def _write_cache(cache: Path | None, url: str, status: int, data: Any) -> None:
    path = _cache_file(cache, url)
    doc = {"url": url, "status": status, "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "data": data}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    except OSError:
        pass  # cache é otimização; falhar em gravar não invalida a resposta


def _fetch(url: str, cache: Path | None, offline: bool, allow_404: bool) -> Any:
    """Devolve o JSON; None só para 404 permitido. Offline lê apenas o cache."""
    if offline:
        hit = _read_cache(cache, url)
        if hit is None:
            raise HFError(url, None, "offline_cache_miss")
        if hit["status"] == 404:
            if allow_404:
                return None
            raise HFError(url, 404, "not_found")
        return hit["data"]
    headers = {"User-Agent": f"localai-scout/{__version__}", "Accept": "application/json"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as resp:
            raw = resp.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            if allow_404:
                _write_cache(cache, url, 404, None)
                return None
            raise HFError(url, 404, "not_found") from e
        if e.code in (401, 403):
            raise HFError(url, e.code, "gated") from e
        if e.code == 429:
            ra = e.headers.get("Retry-After") if e.headers else None
            raise HFError(url, 429, "rate_limited", retry_after=ra) from e
        raise HFError(url, e.code, "network") from e
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:  # ValueError: URL malformada
        raise HFError(url, None, "network") from e
    if len(raw) > MAX_BYTES:
        raise HFError(url, None, "bad_json")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as e:
        raise HFError(url, None, "bad_json") from e
    _write_cache(cache, url, 200, data)
    return data


def get_model_info(repo: str, *, blobs: bool = False, cache: Path | None = None, offline: bool = False) -> dict:
    url = f"{BASE}/api/models/{_check_repo(repo)}" + ("?blobs=true" if blobs else "")
    data = _fetch(url, cache, offline, allow_404=False)
    if not isinstance(data, dict):
        raise HFError(url, None, "bad_json")
    return data


def get_config(repo: str, *, cache: Path | None = None, offline: bool = False) -> dict | None:
    url = f"{BASE}/{_check_repo(repo)}/raw/main/config.json"
    data = _fetch(url, cache, offline, allow_404=True)
    if data is not None and not isinstance(data, dict):
        raise HFError(url, None, "bad_json")
    return data
