"""scout serve: servidor real em thread, porta efêmera, catálogo de teste, coleta falsa."""
from __future__ import annotations

import http.client
import json
import threading

import pytest

from conftest import CATALOG_PATH, SYSTEM_PATHS, assert_valid, load
from scout.server import ScoutServer

TOKEN = "t" * 43
PAGES = "https://jkfa.github.io"
DEV = "http://localhost:5173"
SYSTEM = load(SYSTEM_PATHS["cpu8"])
INTENT = {"intent": {"tasks": ["chat"], "ctx": 4096}}


@pytest.fixture(scope="module")
def srv():
    calls = []
    s = ScoutServer(0, TOKEN, CATALOG_PATH, origins=frozenset({PAGES, DEV}),
                    collect_fn=lambda: calls.append(1) or SYSTEM)
    s.calls = calls
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield s
    s.shutdown()
    s.server_close()


def req(srv, method, path, body=None, headers=None, token=TOKEN, raw=None, host="default"):
    h = {} if headers is None else dict(headers)
    if token:
        h.setdefault("Authorization", f"Bearer {token}")
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    c.putheader("Host", f"127.0.0.1:{srv.port}" if host == "default" else host)
    for k, v in h.items():
        c.putheader(k, v)
    if data is not None:
        c.putheader("Content-Length", str(len(data)))
    c.endheaders(data)
    r = c.getresponse()
    payload = r.read()
    c.close()
    doc = json.loads(payload) if payload else None
    return r.status, doc, r


def assert_error(doc, code):
    assert_valid("error.schema.json", doc)
    assert doc["error"]["code"] == code
    assert TOKEN not in json.dumps(doc) and "Traceback" not in json.dumps(doc)


def test_bind_is_loopback(srv):
    assert srv.server_address[0] == "127.0.0.1"


def test_health_open(srv):
    st, doc, _ = req(srv, "GET", "/health", token=None)
    assert st == 200 and doc["ok"] is True and "version" in doc


PROTECTED = [("GET", "/system", None), ("GET", "/catalog", None), ("POST", "/plan", INTENT),
             ("POST", "/bench", {"model_id": "test/tiny-1b"})]


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_requires_token(srv, method, path, body):
    for tok in (None, "errado", TOKEN[:-1]):
        st, doc, _ = req(srv, method, path, body, token=tok)
        assert st == 401
        assert_error(doc, "unauthorized")
    st, _, _ = req(srv, method, path, body, token=None, headers={"Authorization": TOKEN})
    assert st == 401  # sem esquema Bearer
    st, _, _ = req(srv, method, path, body)
    assert st == 200


def test_unknown_route_and_method(srv):
    assert req(srv, "GET", "/nada", token=None)[0] == 401
    st, doc, _ = req(srv, "GET", "/nada")
    assert st == 404
    assert_error(doc, "not_found")
    st, doc, _ = req(srv, "POST", "/system", {})
    assert st == 405
    assert_error(doc, "method_not_allowed")


def test_system_uses_fake_collect_and_refresh(srv):
    st, doc, _ = req(srv, "GET", "/system")
    assert st == 200 and doc == SYSTEM
    n = len(srv.calls)
    req(srv, "GET", "/system")
    assert len(srv.calls) == n  # cache
    req(srv, "GET", "/system?refresh=1")
    assert len(srv.calls) == n + 1


def test_catalog_summary(srv):
    st, doc, _ = req(srv, "GET", "/catalog")
    assert st == 200 and doc["kind"] == "catalog_summary"
    assert [m["id"] for m in doc["models"]] == sorted(m["id"] for m in load(CATALOG_PATH)["models"])
    assert set(doc["models"][0]) == {"id", "source_url", "tasks", "license", "commercial_ok", "complete"}


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:5174", "null"])
def test_forbidden_origin_before_work(srv, origin):
    n = len(srv.calls)
    for method, path in (("GET", "/system"), ("GET", "/health"), ("OPTIONS", "/system")):
        st, doc, r = req(srv, method, path, headers={"Origin": origin})
        assert st == 403
        assert_error(doc, "forbidden_origin")
        assert r.getheader("Access-Control-Allow-Origin") is None
    assert len(srv.calls) == n
    st, _, _ = req(srv, "GET", "/system", token=None, headers={"Origin": origin})
    assert st == 403  # origin antes do token


def test_allowed_origin_gets_exact_cors(srv):
    st, _, r = req(srv, "GET", "/health", headers={"Origin": PAGES})
    assert st == 200
    assert r.getheader("Access-Control-Allow-Origin") == PAGES
    assert "Origin" in r.getheader("Vary")
    st, _, r = req(srv, "GET", "/system", token=None, headers={"Origin": DEV})
    assert st == 401 and r.getheader("Access-Control-Allow-Origin") == DEV  # erro legível pelo navegador


@pytest.mark.parametrize("host", ["evil.example", "evil.example:%PORT%", "127.0.0.1:1", "localhost", "10.0.0.1:%PORT%"])
def test_forged_host_refused(srv, host):
    st, doc, _ = req(srv, "GET", "/health", host=host.replace("%PORT%", str(srv.port)))
    assert st == 403
    assert_error(doc, "forbidden_host")
    st, doc, _ = req(srv, "GET", "/system", host=host.replace("%PORT%", str(srv.port)))
    assert st == 403


def test_localhost_host_ok(srv):
    assert req(srv, "GET", "/health", host=f"localhost:{srv.port}")[0] == 200


def test_preflight_pna(srv):
    st, _, r = req(srv, "OPTIONS", "/system", token=None, headers={
        "Origin": DEV, "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Private-Network": "true"})
    assert st == 204
    assert r.getheader("Access-Control-Allow-Origin") == DEV
    assert r.getheader("Access-Control-Allow-Private-Network") == "true"
    assert r.getheader("Access-Control-Allow-Headers") == "Authorization, Content-Type"
    assert "POST" in r.getheader("Access-Control-Allow-Methods")
    assert "Origin" in r.getheader("Vary")
    st, _, r = req(srv, "OPTIONS", "/plan", token=None, headers={"Origin": PAGES})
    assert r.getheader("Access-Control-Allow-Private-Network") is None
    assert r.getheader("Access-Control-Allow-Origin") != "*"


def test_plan_valid_and_deterministic(srv):
    body = {**INTENT, "system": SYSTEM}
    st, a, _ = req(srv, "POST", "/plan", body)
    assert st == 200
    assert_valid("feasibility.schema.json", a)
    st, b, _ = req(srv, "POST", "/plan", body)
    assert a == b
    # sem system: usa a coleta ao vivo (falsa) e dá o mesmo plano
    st, c, _ = req(srv, "POST", "/plan", INTENT)
    assert st == 200 and c == a
    assert c["inputs"]["catalog"]["sha256"] == __import__("hashlib").sha256(CATALOG_PATH.read_bytes()).hexdigest()


def test_plan_default_tasks(srv):
    st, doc, _ = req(srv, "POST", "/plan", {"intent": {}})
    assert st == 200 and doc["inputs"]["intent"]["ctx"] == 8192


@pytest.mark.parametrize("body", [
    {}, {"intent": []}, {"intent": {"ctx": 0}}, {"intent": {"ctx": "8"}}, {"intent": {"ctx": True}},
    {"intent": {"tasks": ["nope"]}}, {"intent": {"tasks": []}}, {"intent": {"kv_quant": "x9"}},
    {"intent": {"commercial_use": "yes"}}, {"intent": {"zzz": 1}},
    {"intent": {}, "system": {"kind": "system"}}, {"intent": {}, "system": 5},
    {"intent": {}, "system": {"kind": "system", "schema_version": 1}},
])
def test_plan_bad_input(srv, body):
    st, doc, _ = req(srv, "POST", "/plan", body)
    assert st == 400
    assert_error(doc, "schema_invalid")


def test_invalid_json(srv):
    st, doc, _ = req(srv, "POST", "/plan", raw=b"{nope", headers={"Content-Type": "application/json"})
    assert st == 400
    assert_error(doc, "invalid_json")
    st, doc, _ = req(srv, "POST", "/plan", raw=b"[1]", headers={"Content-Type": "application/json"})
    assert st == 400
    assert_error(doc, "schema_invalid")


def test_content_type_required(srv):
    for h in ({}, {"Content-Type": "text/plain"}):
        st, doc, _ = req(srv, "POST", "/plan", raw=b"{}", headers=h)
        assert st == 415
        assert_error(doc, "unsupported_media_type")
    st, _, _ = req(srv, "POST", "/plan", raw=b'{"intent":{}}',
                   headers={"Content-Type": "application/json; charset=utf-8"})
    assert st == 200


def test_body_too_large_not_read(srv):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    c.putrequest("POST", "/plan", skip_accept_encoding=True)
    c.putheader("Authorization", f"Bearer {TOKEN}")
    c.putheader("Content-Type", "application/json")
    c.putheader("Content-Length", str((1 << 20) + 1))
    c.endheaders()  # nenhum byte do body enviado: o servidor decide pelo cabeçalho
    r = c.getresponse()
    assert r.status == 413
    assert_error(json.loads(r.read()), "payload_too_large")
    c.close()


def test_missing_content_length_411(srv):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    c.putrequest("POST", "/plan", skip_accept_encoding=True)
    c.putheader("Authorization", f"Bearer {TOKEN}")
    c.putheader("Content-Type", "application/json")
    c.endheaders()
    r = c.getresponse()
    assert r.status == 411
    assert_error(json.loads(r.read()), "bad_request")
    c.close()


def test_bench_never_runs(srv):
    st, doc, _ = req(srv, "POST", "/bench", {"model_id": "test/tiny-1b"})
    assert st == 200 and doc["ran"] is False and "test/tiny-1b" in doc["command"]
    assert "measured" not in doc
    st, doc, _ = req(srv, "POST", "/bench", {"model_id": "test/tiny-1b", "run": True})
    assert st == 501
    assert_error(doc, "bench_disabled")
    st, doc, _ = req(srv, "POST", "/bench", {"model_id": "x; rm -rf /"})
    assert st == 404
    assert_error(doc, "not_found")
    assert req(srv, "POST", "/bench", {})[0] == 400


def test_internal_error_has_no_traceback(srv, monkeypatch):
    def boom():
        raise RuntimeError("segredo-interno")

    monkeypatch.setattr(srv, "_collect_fn", boom)
    st, doc, _ = req(srv, "GET", "/system?refresh=1")
    assert st == 500
    assert_error(doc, "internal")
    assert "segredo-interno" not in json.dumps(doc)


def test_token_not_logged(srv, capfd):
    req(srv, "GET", "/catalog")
    assert TOKEN not in capfd.readouterr().err
