// Cliente do backend local (scout serve). O token vive só nesta variável de módulo:
// nunca em localStorage/sessionStorage/cookie, nunca em URL.

let token = null;
let port = 8765;

export class ApiError extends Error {
  constructor(kind, message, { status = null, code = null } = {}) {
    super(message);
    this.kind = kind; // unreachable | timeout | unauthorized | forbidden | server | bad_response
    this.status = status;
    this.code = code;
  }
}

export const hasToken = () => token !== null;
export const getPort = () => port;

export function setToken(t) {
  token = typeof t === "string" && t.trim() ? t.trim() : null;
}

export function setPort(p) {
  const n = Number(p);
  if (Number.isInteger(n) && n >= 1 && n <= 65535) port = n;
}

// Lê #token=...&port=... e apaga o fragment da URL na hora.
export function consumeFragment() {
  const raw = location.hash.startsWith("#") ? location.hash.slice(1) : "";
  if (!raw) return false;
  const p = new URLSearchParams(raw);
  const t = p.get("token");
  const found = t !== null || p.has("port");
  if (t) setToken(t);
  if (p.get("port")) setPort(p.get("port"));
  if (found) history.replaceState(null, "", location.pathname + location.search);
  return Boolean(t);
}

const base = () => `http://127.0.0.1:${port}`;

const UNREACHABLE =
  "Não consegui falar com o backend em 127.0.0.1. Ou o servidor está fora do ar (rode `scout serve`), " +
  "o navegador bloqueou o acesso à rede local (Private Network Access / permissão de rede local; o Safari bloqueia isso), ou a origem desta página não está permitida no servidor (nesse caso o erro 403 fica ilegível para o navegador). " +
  "O modo ARQUIVO não depende de servidor: envie o feasibility.json.";

async function request(method, path, body, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const headers = {};
  if (path !== "/health" && token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let res;
  try {
    res = await fetch(base() + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal: ctrl.signal, cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer" });
  } catch (e) {
    if (e && e.name === "AbortError") throw new ApiError("timeout", `O backend não respondeu em ${Math.round(timeoutMs / 1000)} s. Ele pode estar coletando o hardware; tente de novo. Se persistir, use o modo ARQUIVO.`);
    throw new ApiError("unreachable", UNREACHABLE);
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try { data = await res.json(); } catch { /* corpo não-JSON */ }
  if (!res.ok) {
    const code = data && data.error && typeof data.error.code === "string" ? data.error.code : null;
    const msg = data && data.error && typeof data.error.message === "string" ? data.error.message : `HTTP ${res.status}`;
    if (res.status === 401) throw new ApiError("unauthorized", "Token recusado (401). O token muda a cada `scout serve`: abra de novo o link impresso no terminal ou cole o token atual.", { status: 401, code });
    if (res.status === 403) throw new ApiError("forbidden", `Acesso negado (403): ${msg}. A origem desta página não está na lista permitida do servidor (veja SCOUT_ALLOWED_ORIGIN), ou o Host foi recusado. Use o modo ARQUIVO se não puder mudar isso.`, { status: 403, code });
    throw new ApiError("server", `Erro do backend (${res.status}${code ? ", " + code : ""}): ${msg}`, { status: res.status, code });
  }
  if (data === null || typeof data !== "object") throw new ApiError("bad_response", "O backend respondeu algo que não é JSON.", { status: res.status });
  return data;
}

export const health = () => request("GET", "/health", undefined, 4000);
export const getSystem = (refresh = false) => request("GET", "/system" + (refresh ? "?refresh=1" : ""), undefined, 60000);
// `tasks` ausente = todas as do catálogo (como a CLI).
export const postPlan = (intent) => request("POST", "/plan", { intent }, 60000);
