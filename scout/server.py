"""Backend LOCAL (`scout serve`). Bind somente em 127.0.0.1.

Segurança (ver ARCHITECTURE.md e DECISIONS.md A1-A15): Host validado (DNS rebinding),
Origin em allowlist, token Bearer em tudo menos /health, sem persistência, sem rede.
O token nunca é gravado em disco nem em log.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from scout import __version__
from scout.cli import _parse_tasks, _read_json, load_tuning, run_plan
from scout.solver import TASKS, Intent

BIND_HOST = "127.0.0.1"  # fixo: não existe opção de host
DEFAULT_PORT = 8765
DEFAULT_ORIGINS = ("https://jkfa.github.io", "http://localhost:5173")
PAGES_URL = "https://jkfa.github.io/localai-scout/"
DEV_URL = "http://localhost:5173/"
MAX_BODY = 1 << 20  # 1 MiB (A7)

STATUS = {
    "bad_request": 400, "invalid_json": 400, "schema_invalid": 400, "unauthorized": 401,
    "forbidden_origin": 403, "forbidden_host": 403, "not_found": 404, "method_not_allowed": 405,
    "payload_too_large": 413, "unsupported_media_type": 415, "bench_disabled": 501, "internal": 500,
}
ROUTES = {"/health": "GET", "/system": "GET", "/plan": "POST", "/catalog": "GET", "/bench": "POST"}


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status or STATUS[code]


def allowed_origins() -> frozenset[str]:
    raw = os.environ.get("SCOUT_ALLOWED_ORIGIN")
    if raw is None:
        return frozenset(DEFAULT_ORIGINS)
    return frozenset(o.strip() for o in raw.split(",") if o.strip())


class ScoutServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port: int, token: str, catalog_path: Path,
                 origins: frozenset[str] | None = None,
                 collect_fn: Callable[[], dict] | None = None,
                 tuning_path: Path | None = None) -> None:
        self.token = token
        self.origins = allowed_origins() if origins is None else origins
        self.catalog, self.catalog_raw = _read_json(catalog_path, "catalog")
        self.tuning = load_tuning(tuning_path)
        if collect_fn is None:
            from scout.collect import collect

            collect_fn = collect
        self._collect_fn = collect_fn
        self._system: dict | None = None
        self._lock = threading.Lock()
        super().__init__((BIND_HOST, port), Handler)
        self.port = self.server_address[1]
        self.allowed_hosts = frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}"})

    def system(self, refresh: bool = False) -> dict:
        with self._lock:
            if self._system is None or refresh:
                self._system = self._collect_fn()
            return self._system


class Handler(BaseHTTPRequestHandler):
    server: ScoutServer
    server_version = "scout"
    sys_version = ""

    # --- saída -------------------------------------------------------------
    def _send(self, status: int, body: Any | None, origin: str | None, extra: dict[str, str] | None = None) -> None:
        data = b"" if body is None else json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(status)
        if body is not None:
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Vary", "Origin")
        if origin is not None and origin in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if data and self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, e: ApiError, origin: str | None) -> None:
        self.close_connection = True  # corpo possivelmente não lido
        self._send(e.status, {"error": {"code": e.code, "message": e.message}}, origin)

    # --- validações --------------------------------------------------------
    def _check_host_origin(self) -> str | None:
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in self.server.allowed_hosts:
            raise ApiError("forbidden_host", "Host não permitido")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.origins:
            raise ApiError("forbidden_origin", "Origin não permitido")
        return origin

    def _check_token(self) -> None:
        auth = self.headers.get("Authorization") or ""
        scheme, _, value = auth.partition(" ")
        given = value.strip().encode("utf-8") if scheme.lower() == "bearer" else b""
        # Ausente, malformado e errado dão a mesma resposta.
        if not hmac.compare_digest(given, self.server.token.encode("utf-8")) or not given:
            raise ApiError("unauthorized", "Token ausente ou inválido")

    def _read_body(self) -> Any:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ApiError("unsupported_media_type", "Content-Type deve ser application/json")
        if self.headers.get("Transfer-Encoding"):
            raise ApiError("bad_request", "Content-Length obrigatório", 411)
        cl = self.headers.get("Content-Length")
        if cl is None:
            raise ApiError("bad_request", "Content-Length obrigatório", 411)
        if not cl.isdigit():
            raise ApiError("bad_request", "Content-Length inválido")
        n = int(cl)
        if n > MAX_BODY:
            raise ApiError("payload_too_large", f"Body maior que {MAX_BODY} bytes")
        raw = self.rfile.read(n)
        try:
            doc = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError("invalid_json", "Body não é JSON válido") from None
        if not isinstance(doc, dict):
            raise ApiError("schema_invalid", "Body deve ser um objeto JSON")
        return doc

    # --- despacho ----------------------------------------------------------
    def _handle(self) -> None:
        origin: str | None = self.headers.get("Origin")
        try:
            origin = self._check_host_origin()
            if self.command == "OPTIONS":
                return self._preflight(origin)
            parts = urlsplit(self.path)
            path = parts.path
            if path != "/health":
                self._check_token()
            if path not in ROUTES:
                raise ApiError("not_found", "Rota inexistente")
            if self.command != ROUTES[path]:
                raise ApiError("method_not_allowed", f"Use {ROUTES[path]} em {path}")
            body = self._read_body() if self.command == "POST" else None
            if path == "/health":
                out: Any = {"ok": True, "version": __version__}
            elif path == "/system":
                out = self.server.system(refresh="refresh=1" in parts.query.split("&"))
            elif path == "/catalog":
                out = self._catalog()
            elif path == "/plan":
                out = self._plan(body or {})
            else:
                out = self._bench(body or {})
            self._send(200, out, origin)
        except ApiError as e:
            self._error(e, origin)
        except Exception:  # nunca vaza traceback
            self._error(ApiError("internal", "Erro interno"), origin)

    def _preflight(self, origin: str | None) -> None:
        extra = {
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Max-Age": "600",
        }
        if (self.headers.get("Access-Control-Request-Private-Network") or "").lower() == "true":
            extra["Access-Control-Allow-Private-Network"] = "true"
        self._send(204, None, origin, extra)

    do_GET = do_POST = do_OPTIONS = do_HEAD = do_PUT = do_DELETE = do_PATCH = _handle

    # --- endpoints ---------------------------------------------------------
    def _catalog(self) -> dict:
        models = [
            {"id": m["id"], "source_url": m.get("source_url"), "tasks": m.get("tasks"),
             "license": m.get("license"), "commercial_ok": m.get("commercial_ok"),
             "complete": bool(m.get("complete"))}
            for m in sorted(self.server.catalog.get("models", []), key=lambda x: x["id"])
        ]
        return {"schema_version": 1, "kind": "catalog_summary", "models": models}

    def _plan(self, body: dict) -> dict:
        srv = self.server
        raw_intent = body.get("intent")
        if not isinstance(raw_intent, dict):
            raise ApiError("schema_invalid", "intent ausente ou não é objeto")
        extra = set(raw_intent) - {"tasks", "ctx", "commercial_use", "batch", "kv_quant"}
        if extra:
            raise ApiError("schema_invalid", "campos desconhecidos em intent: " + ", ".join(sorted(map(str, extra))))
        tasks = raw_intent.get("tasks")
        if tasks is None:
            tasks = _parse_tasks(None, srv.catalog)
        elif (not isinstance(tasks, list) or not tasks
              or any(not isinstance(t, str) or t not in TASKS for t in tasks)):
            raise ApiError("schema_invalid", f"intent.tasks deve ser lista não vazia de {', '.join(TASKS)}")
        ctx = raw_intent.get("ctx", 8192)
        batch = raw_intent.get("batch", 1)
        for name, v in (("ctx", ctx), ("batch", batch)):
            if type(v) is not int or v < 1 or v > 1 << 31:
                raise ApiError("schema_invalid", f"intent.{name} deve ser inteiro >= 1")
        commercial = raw_intent.get("commercial_use", False)
        if type(commercial) is not bool:
            raise ApiError("schema_invalid", "intent.commercial_use deve ser booleano")
        kv = raw_intent.get("kv_quant", "f16")
        if not isinstance(kv, str):
            raise ApiError("schema_invalid", "intent.kv_quant deve ser string")
        intent = Intent(tasks=tuple(tasks), ctx=ctx, commercial_use=commercial, batch=batch, kv_quant=kv)

        client_system = "system" in body
        system = body["system"] if client_system else srv.system()
        if not isinstance(system, dict) or system.get("kind") != "system" or system.get("schema_version") != 1:
            raise ApiError("schema_invalid", "system deve ser um system.json (kind=system, schema_version=1)")
        try:
            return run_plan(system, srv.catalog, srv.catalog_raw, srv.tuning, intent)
        except ValueError as e:
            raise ApiError("schema_invalid", str(e)[:400] or "intent inválido") from None
        except (KeyError, TypeError, AttributeError, IndexError):
            if client_system:  # system do cliente malformado
                raise ApiError("schema_invalid", "system não corresponde ao schema esperado") from None
            raise

    def _bench(self, body: dict) -> dict:
        model_id = body.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ApiError("schema_invalid", "model_id ausente")
        if body.get("run") is True:
            raise ApiError("bench_disabled", "Execução de bench está desabilitada; rode pelo terminal")
        if model_id not in {m["id"] for m in self.server.catalog.get("models", [])}:
            raise ApiError("not_found", "model_id fora do catálogo")
        return {"command": f"scout bench --run {model_id}", "ran": False,
                "missing": ["model_file_path", "runtime"]}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Só stderr; nunca cabeçalhos, token nem corpo. Query descartada.
        sys.stderr.write(f"scout: {self.command} {urlsplit(self.path).path} -> {args[1] if len(args) > 1 else '-'}\n")


def serve(port: int, catalog_path: Path, tuning_path: Path | None = None) -> int:
    token = secrets.token_urlsafe(32)
    httpd = ScoutServer(port, token, catalog_path, tuning_path=tuning_path)  # OSError se a porta estiver ocupada
    sys.stdout.write(
        f"scout serve em http://127.0.0.1:{httpd.port} (Ctrl+C encerra)\n\n"
        f"Token (muda a cada start): {token}\n"
        f"Pareamento (Pages): {PAGES_URL}#token={token}&port={httpd.port}\n"
        f"Pareamento (dev):   {DEV_URL}#token={token}&port={httpd.port}\n")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
