"""hf.py sem rede: urlopen substituído por monkeypatch."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from scout import hf


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def net(monkeypatch):
    calls: list = []
    state = {"handler": lambda req, timeout: Resp(b"{}")}

    def fake(req, timeout=None):
        calls.append((req, timeout))
        return state["handler"](req, timeout)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return calls, state


def http_error(code, headers=None):
    import email.message
    m = email.message.Message()
    for k, v in (headers or {}).items():
        m[k] = v
    return urllib.error.HTTPError("u", code, "x", m, None)


def raiser(exc):
    def h(req, timeout):
        raise exc
    return h


def test_model_info_url_headers_timeout(net, tmp_path):
    calls, state = net
    state["handler"] = lambda r, t: Resp(json.dumps({"id": "a/b"}).encode())
    assert hf.get_model_info("a/b", cache=tmp_path) == {"id": "a/b"}
    hf.get_model_info("a/b", blobs=True, cache=tmp_path)
    assert calls[0][0].full_url == "https://huggingface.co/api/models/a/b"
    assert calls[1][0].full_url.endswith("/api/models/a/b?blobs=true")
    assert calls[0][1] == 20
    h = {k.lower(): v for k, v in calls[0][0].header_items()}
    assert h["user-agent"].startswith("localai-scout/") and "authorization" not in h


def test_hf_token_only_from_env(net, monkeypatch, tmp_path):
    calls, _ = net
    monkeypatch.setenv("HF_TOKEN", "tok")
    hf.get_model_info("a/b", cache=tmp_path)
    assert calls[0][0].get_header("Authorization") == "Bearer tok"


def test_config_404_is_none_and_cached(net, tmp_path):
    calls, state = net
    state["handler"] = raiser(http_error(404))
    assert hf.get_config("a/b", cache=tmp_path) is None
    assert calls[0][0].full_url == "https://huggingface.co/a/b/raw/main/config.json"
    assert hf.get_config("a/b", cache=tmp_path, offline=True) is None
    assert len(calls) == 1


def test_model_info_404_raises(net, tmp_path):
    net[1]["handler"] = raiser(http_error(404))
    with pytest.raises(hf.HFError) as e:
        hf.get_model_info("a/b", cache=tmp_path)
    assert e.value.reason == "not_found" and e.value.status == 404


@pytest.mark.parametrize("exc,reason", [
    (http_error(401), "gated"), (http_error(403), "gated"), (http_error(429), "rate_limited"),
    (http_error(500), "network"), (urllib.error.URLError("dns"), "network"),
    (TimeoutError(), "network"), (ConnectionResetError(), "network")])
def test_error_reasons(net, tmp_path, exc, reason):
    net[1]["handler"] = raiser(exc)
    with pytest.raises(hf.HFError) as e:
        hf.get_model_info("a/b", cache=tmp_path)
    assert e.value.reason == reason


def test_rate_limit_keeps_retry_after(net, tmp_path):
    net[1]["handler"] = raiser(http_error(429, {"Retry-After": "30"}))
    with pytest.raises(hf.HFError) as e:
        hf.get_model_info("a/b", cache=tmp_path)
    assert e.value.retry_after == "30"


def test_bad_json_and_wrong_type(net, tmp_path):
    net[1]["handler"] = lambda r, t: Resp(b"<html>")
    with pytest.raises(hf.HFError) as e:
        hf.get_model_info("a/b", cache=tmp_path)
    assert e.value.reason == "bad_json"
    net[1]["handler"] = lambda r, t: Resp(b"[1,2]")
    with pytest.raises(hf.HFError) as e:
        hf.get_config("a/b", cache=tmp_path)
    assert e.value.reason == "bad_json"


def test_offline_uses_cache_and_never_network(net, tmp_path):
    calls, state = net
    state["handler"] = lambda r, t: Resp(b'{"x": 1}')
    hf.get_model_info("a/b", cache=tmp_path)
    n = len(calls)
    assert hf.get_model_info("a/b", cache=tmp_path, offline=True) == {"x": 1}
    with pytest.raises(hf.HFError) as e:
        hf.get_model_info("a/b", blobs=True, cache=tmp_path, offline=True)
    assert e.value.reason == "offline_cache_miss"
    assert len(calls) == n
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and "fetched_at" in json.loads(files[0].read_text())


@pytest.mark.parametrize("repo", ["", "sembarra", "a/b/c", "../x/y", "-a/b", "a/..", "a b/c", "a/b?x=1"])
def test_bad_repo_rejected_before_network(net, tmp_path, repo):
    with pytest.raises(hf.HFError) as e:
        hf.get_model_info(repo, cache=tmp_path)
    assert e.value.reason == "bad_repo" and not net[0]


def test_model_card_url():
    assert hf.model_card_url("Qwen/Qwen3-4B") == "https://huggingface.co/Qwen/Qwen3-4B"
    with pytest.raises(hf.HFError):
        hf.model_card_url("x")


def test_unwritable_cache_does_not_break(net, tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    assert hf.get_model_info("a/b", cache=blocker / "sub") == {}
