"""scout bench: mede decode/prefill reais e calibra banda efetiva + overhead por pool.

Princípio (schemas/bench_spec.md): t_por_token = bytes_por_token / banda_efetiva + overhead_fixo.
Nunca altera eta_gpu/eta_cpu nem data/tuning.json (a gravação é de scout.calstore). A única rede aqui é a API LOCAL do Ollama (127.0.0.1:11434).
O ajuste e o parsing são funções puras; I/O passa por `_exec` e `_ollama_request`.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scout import __version__
from scout import collect as C
from scout import estimate as E
from scout.calibration import normalize_power
from scout.solver import TASKS, Intent, solve

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "bench_prompt.txt"
OLLAMA_URL = "http://127.0.0.1:11434"
GEN_TOKENS = 128
PREFILL_TOKENS = 512  # llama-bench -p
SEED = 42
TOL = 0.05
MAX_ROUNDS = 6
OVERHEAD_MIN_SHARE = 0.05
MIB = 1024 * 1024
MIN_GEN_TOKENS = 16


class BenchError(Exception):
    """Falha de medição com motivo legível; nunca vira dado parcial."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Target:
    model_id: str
    quant: str
    file_bytes: int
    weights_bytes: int
    n_layers: int
    n_gpu_layers: int
    ctx: int
    threads: int | None
    pool: str
    bytes_per_token: int


@dataclass(frozen=True)
class Round:
    decode_tps: float
    prefill_tps: float | None
    prompt_tokens: int | None


# ------------------------------------------------------------ alvo (o que o plan recomendou)


def build_target(system: dict, catalog: dict, model_id: str, ctx: int, tuning: dict) -> Target:
    model = next((m for m in catalog.get("models", []) if m.get("id") == model_id), None)
    if model is None:
        raise BenchError("model_not_in_catalog", f"{model_id}: não está no catálogo")
    tasks = tuple(t for t in TASKS if t in (model.get("tasks") or [])) or TASKS
    result = solve(system, {**catalog, "models": [model]}, Intent(tasks=tasks, ctx=ctx), tuning)
    cand = next((c for c in result["candidates"] if c["model"]["id"] == model_id), None)
    if cand is None:
        rej = next((r for r in result["rejected"] if r["model"]["id"] == model_id), None)
        why = f"{rej['reason_code']}: {rej['message']}" if rej else "modelo excluído do plano (incompleto ou sem a tarefa)"
        raise BenchError("not_a_candidate", f"{model_id}: o plan não recomenda este modelo ({why})")
    if any(n.get("code") == "decode_not_applicable" for n in cand.get("notes", [])):
        raise BenchError("decode_not_applicable", f"{model_id}: modelo de embeddings não faz decode; nada a medir")
    if cand["placement"] == "hybrid":
        raise BenchError("hybrid_unsupported", f"{model_id}: placement híbrido; o tempo por token não é atribuível a um só pool")
    quant = next((q for q in model["quants"] if q["name"] == cand["quant"]["name"]
                  and q["format"] == cand["quant"]["format"]), None)
    if not quant or not isinstance(quant.get("file_bytes"), int):
        raise BenchError("no_file_bytes", f"{model_id} {cand['quant']['name']}: catálogo sem file_bytes; não há como conferir a quant")
    weights = cand["weights_bytes"]
    n_layers = model["n_layers"]
    layer_read = E.bytes_per_token(weights, model["params_active"], model["params_total"]) // n_layers
    n_gpu = cand["n_gpu_layers"]
    return Target(model_id, cand["quant"]["name"], quant["file_bytes"], weights, n_layers, n_gpu, ctx,
                  cand["n_threads"], cand["by_pool"][0]["pool"], layer_read * n_layers)


# ------------------------------------------------------------ energia


def decide_power(bat_present: bool | None, on_battery: bool | None, profile: str | None,
                 profile_capable: bool | None) -> dict:
    """Decisão pura (calibration_store_spec.md §4) sobre o que foi observado. None = não observado.
    Não aplicável exige evidência positiva: sem bateria (bat_present False) ou sem nenhuma fonte de
    perfil (profile_capable False). Qualquer outra falta de leitura é desconhecido (null fora da lista)."""
    na: list[str] = []
    on_ac: bool | None = None
    if bat_present is False:
        na.append("on_ac")
    elif bat_present is True and on_battery is not None:
        on_ac = not on_battery
    prof: str | None = None
    if profile_capable is False:
        na.append("profile")
    elif profile:
        prof = profile
    return {"on_ac": on_ac, "profile": prof, "not_applicable": na}


def _read_power_linux(sysfs: str, which: Callable[[str], str | None]) -> dict:
    probes = C.Probes()
    root = f"{sysfs}/class/power_supply"
    bat_present: bool | None = None
    on_battery: bool | None = None
    try:
        names = sorted(os.listdir(root))
    except OSError:
        names = None  # não deu para listar: desconhecido, não "sem bateria"
    if names is not None:
        bat_present = any(n.startswith("BAT") for n in names)
        if bat_present:
            sup: list[dict[str, str | None]] = []
            for n in names:
                d: dict[str, str | None] = {}
                for k in ("type", "status", "online", "scope"):
                    txt = C.read_text(f"{root}/{n}/{k}", probes, limit=32)
                    d[k] = txt.strip() if txt is not None else None
                sup.append(d)
            on_battery = C.P.parse_power(sup)[1]
    pp_path = f"{sysfs}/firmware/acpi/platform_profile"
    has_file = os.path.exists(pp_path)
    has_daemon = which("powerprofilesctl") is not None
    profile: str | None = None
    if has_file:
        txt = C.read_text(pp_path, probes, limit=64)
        profile = txt.strip() if txt and txt.strip() else None
    if profile is None and has_daemon:
        out = C.run_cmd(["powerprofilesctl", "get"], probes, timeout=3.0)
        profile = out.strip() if out and out.strip() else None
    return decide_power(bat_present, on_battery, profile, has_file or has_daemon)


def read_power(sysfs: str = "/sys", which: Callable[[str], str | None] = shutil.which) -> dict:
    """Estado de energia NO MOMENTO da medição: {on_ac, profile, not_applicable}. Nunca levanta.
    Linux lê /sys; darwin lê só a bateria (pmset); demais plataformas: tudo desconhecido."""
    try:
        system = {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}.get(platform.system(), "other")
        if system == "linux":
            return _read_power_linux(sysfs, which)
        if system == "darwin":
            pw = C.collect_power(system, C.Probes())
            has, ob = pw["has_battery"]["value"], pw["on_battery"]["value"]
            return decide_power(has, ob, None, None)
    except Exception:  # noqa: BLE001 - leitura de energia nunca derruba a CLI
        pass
    return decide_power(None, None, None, None)


# ------------------------------------------------------------ versão do motor


def engine_version(tool: str) -> str | None:
    """ÚNICA fonte da versão do motor (bench grava, plan lê). None em qualquer falha."""
    try:
        if tool == "ollama":
            v = _ollama_request("GET", "/api/version", timeout=2.0).get("version")
            return v.strip() if isinstance(v, str) and v.strip() else None
        if tool == "llama-bench":
            cp = subprocess.run(["llama-bench", "--version"], capture_output=True, text=True, errors="replace",
                                timeout=3.0, check=True, stdin=subprocess.DEVNULL)
            for blob in (cp.stdout, cp.stderr):  # llama.cpp costuma imprimir versão em stderr
                for line in blob.splitlines():
                    if line.strip():
                        return line.strip()
    except Exception:  # noqa: BLE001 - ausente, travado, sem permissão, rede caída: tudo vira None
        return None
    return None


def live_system() -> dict | None:
    """Leitura ao vivo sem privilégio (sem dmidecode), para conferir o --system. None se falhar."""
    try:
        return C.collect()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------ estabilização (pura)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(a, b)


def first_stable_pair(values: list[float | None]) -> int | None:
    """Índice j (>=1) da primeira rodada cujo valor está a até 5% do anterior; None se nunca."""
    for j in range(1, len(values)):
        a, b = values[j - 1], values[j]
        if a is not None and b is not None and _close(a, b):
            return j
    return None


def stabilize(measure: Callable[[int], Round], tol_rounds: int = MAX_ROUNDS) -> dict[str, Any]:
    """1 aquecimento descartado; rodadas até decode E prefill terem um par estável; teto de 6.
    Reporta todas as rodadas; `used_*` marca as duas que entraram na conta. Sem par estável,
    usa as duas últimas e marca stable=false (prefill: valor null)."""
    warm = measure(0)
    rounds: list[Round] = []
    while len(rounds) < tol_rounds:
        rounds.append(measure(len(rounds) + 1))
        dj = first_stable_pair([r.decode_tps for r in rounds])
        pj = first_stable_pair([r.prefill_tps for r in rounds])
        if dj is not None and (pj is not None or all(r.prefill_tps is None for r in rounds)):
            break
    dj = first_stable_pair([r.decode_tps for r in rounds])
    pj = first_stable_pair([r.prefill_tps for r in rounds])
    d_used = (dj - 1, dj) if dj is not None else (max(len(rounds) - 2, 0), len(rounds) - 1)
    has_prefill = any(r.prefill_tps is not None for r in rounds)
    decode = sum(rounds[i].decode_tps for i in set(d_used)) / len(set(d_used))
    prefill = None
    p_used: tuple[int, ...] = ()
    if pj is not None:
        p_used = (pj - 1, pj)
        prefill = sum(rounds[i].prefill_tps for i in p_used) / 2  # type: ignore[misc]
    out_rounds = []
    for i, r in enumerate(rounds):
        out_rounds.append({"round": i + 1, "decode_tps": r.decode_tps, "prefill_tps": r.prefill_tps,
                           "prompt_tokens": r.prompt_tokens,
                           "used_decode": i in d_used, "used_prefill": i in p_used})
    return {
        "warmup": {"decode_tps": warm.decode_tps, "prefill_tps": warm.prefill_tps, "discarded": True},
        "rounds": out_rounds, "decode_tps": decode, "stable": dj is not None,
        "prefill_tps": prefill, "prefill_stable": pj is not None if has_prefill else None,
        "prompt_tokens": rounds[-1].prompt_tokens,
    }


# ------------------------------------------------------------ ajuste (puro)


def fit_calibration(points: list[tuple[int, float]]) -> dict[str, Any]:
    """points = [(bytes_por_token, decode_tps)] de UM pool. -> banda, overhead, forma e os dois valores.
    2p: mínimos quadrados em t = bytes/banda + overhead (exige tamanhos diferentes). Descarta o overhead
    se < 5% do tempo por token do menor modelo ou negativo (ou se a inclinação não for positiva) e refaz
    com 1p: banda = média de bytes*tok/s ponderada pelos bytes. Com 1 modelo: 1p, overhead 0."""
    if not points or any(b <= 0 or t <= 0 for b, t in points):
        raise ValueError("pontos inválidos")
    pts = sorted(points)
    bw1 = sum(b * (b * t) for b, t in pts) / sum(b for b, _ in pts)
    one = {"bandwidth_bytes_per_s": int(round(bw1)), "overhead_s_per_token": 0.0}
    two = None
    reason = "um único modelo" if len(pts) < 2 else None
    if len(pts) >= 2 and len({b for b, _ in pts}) >= 2:
        xs, ys = [float(b) for b, _ in pts], [1.0 / t for _, t in pts]
        n = len(pts)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        overhead = my - slope * mx
        two = {"bandwidth_bytes_per_s": int(round(1.0 / slope)) if slope > 0 else None, "overhead_s_per_token": overhead}
        t_min = ys[0]
        if slope <= 0:
            reason = "inclinação não positiva"
        elif overhead < 0:
            reason = "overhead negativo"
        elif overhead < OVERHEAD_MIN_SHARE * t_min:
            reason = "overhead abaixo de 5% do tempo por token do menor modelo"
        else:
            return {**two, "form": "2p", "two_param": two, "one_param": one, "discard_reason": None}
    elif len(pts) >= 2:
        reason = "modelos do mesmo tamanho"
    return {**one, "form": "1p", "two_param": two, "one_param": one, "discard_reason": reason}


def build_calibration(results: list[dict], date: str) -> dict[str, dict]:
    """measured -> {pool: calibration}. Um ajuste por pool."""
    by_pool: dict[str, list[dict]] = {}
    for r in results:
        by_pool.setdefault(r["pool"], []).append(r)
    out: dict[str, dict] = {}
    for pool, rs in sorted(by_pool.items()):
        fit = fit_calibration([(r["bytes_per_token"], r["decode_tps"]) for r in rs])
        powers = {json.dumps(r["power"], sort_keys=True) for r in rs}
        power = normalize_power(rs[0]["power"]) if len(powers) == 1 else normalize_power(None)
        versions = {r.get("tool_version") for r in rs}
        prefills = [r["prefill_tps"] for r in rs]
        if all(p is not None for p in prefills):
            prefill, reason = sum(prefills) / len(prefills), None
        else:
            prefill, reason = None, "prefill não estabilizou (ou não medido) em algum modelo"
        out[pool] = {
            "bandwidth_bytes_per_s": fit["bandwidth_bytes_per_s"], "overhead_s_per_token": fit["overhead_s_per_token"],
            "form": fit["form"], "stable": all(r["stable"] for r in rs), "power": power,
            "tool": "+".join(sorted({r["tool"] for r in rs})),
            "tool_version": next(iter(versions)) if len(versions) == 1 else None,
            "models": [{"id": r["model_id"], "architecture": r["architecture"], "bytes": r["bytes_per_token"]} for r in rs],
            "date": date, "prefill_tokens_per_s": prefill, "prefill_reason": reason,
            "fit": {"two_param": fit["two_param"], "one_param": fit["one_param"], "discard_reason": fit["discard_reason"]},
        }
    return out


# ------------------------------------------------------------ ferramentas (I/O isolado)


def _exec(cmd: list[str], timeout: float) -> str:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
                            check=True, stdin=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise BenchError("tool_missing", f"{cmd[0]} não encontrado") from e
    except PermissionError as e:
        raise BenchError("tool_failed", f"sem permissão para executar {cmd[0]}") from e
    except subprocess.TimeoutExpired as e:
        raise BenchError("tool_timeout", f"{cmd[0]} excedeu {timeout:g}s") from e
    except subprocess.CalledProcessError as e:
        raise BenchError("tool_failed", f"{cmd[0]} saiu com código {e.returncode}") from e
    return cp.stdout


def llama_bench_cmd(gguf: str, n_gpu_layers: int, threads: int | None) -> list[str]:
    cmd = ["llama-bench", "-m", gguf, "-p", str(PREFILL_TOKENS), "-n", str(GEN_TOKENS), "-ngl", str(n_gpu_layers)]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def parse_llama_bench(text: str) -> tuple[float, float | None, str | None]:
    """-o json -> (decode tok/s, prefill tok/s, versão). Falha alto se não há linha de geração."""
    i = text.find("[")
    try:
        rows = json.loads(text[i:text.rfind("]") + 1]) if i >= 0 else None
    except ValueError:
        rows = None
    if not isinstance(rows, list):
        raise BenchError("bad_output", "saída do llama-bench não é JSON")
    dec = pre = None
    ver = None
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("avg_ts"), (int, float)):
            continue
        ver = ver or (f"{r.get('build_number')} ({r.get('build_commit')})" if r.get("build_commit") else None)
        if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            dec = float(r["avg_ts"])
        elif r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            pre = float(r["avg_ts"])
    if not dec or dec <= 0:
        raise BenchError("bad_output", "llama-bench não devolveu velocidade de geração")
    return dec, pre, ver


def llama_bench_round(gguf: str, target: Target) -> tuple[Round, str | None]:
    cmd = llama_bench_cmd(gguf, target.n_gpu_layers, target.threads) + ["-r", "1", "-o", "json"]
    dec, pre, ver = parse_llama_bench(_exec(cmd, 900.0))
    return Round(dec, pre, PREFILL_TOKENS), ver


def _ollama_request(method: str, path: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    """Só 127.0.0.1:11434."""
    req = urllib.request.Request(OLLAMA_URL + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise BenchError("ollama_unreachable", f"API local do Ollama indisponível ({path})") from e
    if not isinstance(doc, dict):
        raise BenchError("bad_output", f"resposta inesperada do Ollama em {path}")
    return doc


def ollama_check_quant(name: str, target: Target) -> dict:
    """Confere o blob com o file_bytes do catálogo (±1 MiB). Devolve {architecture}."""
    tags = _ollama_request("GET", "/api/tags", timeout=10.0)
    entry = next((m for m in tags.get("models", []) if name in (m.get("name"), m.get("model"))), None)
    if entry is None:
        raise BenchError("ollama_model_missing", f"modelo {name!r} não está instalado no Ollama")
    size = entry.get("size")
    if not isinstance(size, int) or abs(size - target.file_bytes) > MIB:
        raise BenchError("quant_mismatch",
                         f"{name}: blob de {size} bytes não casa com {target.quant} do catálogo ({target.file_bytes} bytes, ±1 MiB)")
    arch = None
    try:
        info = _ollama_request("POST", "/api/show", {"model": name}, timeout=30.0).get("model_info") or {}
        arch = info.get("general.architecture")
    except BenchError:
        pass
    return {"architecture": arch}


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def ollama_round(name: str, target: Target, prompt: str, n: int) -> Round:
    body = {"model": name, "prompt": f"[rodada {n}] {prompt}", "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0, "seed": SEED, "num_predict": GEN_TOKENS, "num_ctx": target.ctx,
                        "num_gpu": target.n_gpu_layers, **({"num_thread": target.threads} if target.threads else {})}}
    r = _ollama_request("POST", "/api/generate", body)
    try:
        ec, ed = int(r["eval_count"]), int(r["eval_duration"])
        pc, pd = int(r["prompt_eval_count"]), int(r["prompt_eval_duration"])
    except (KeyError, TypeError, ValueError) as e:
        raise BenchError("bad_output", "resposta do Ollama sem contadores de avaliação") from e
    if ec < MIN_GEN_TOKENS or ed <= 0:
        raise BenchError("short_generation", f"geração curta demais ({ec} tokens) para medir decode")
    return Round(ec / ed * 1e9, pc / pd * 1e9 if pd > 0 and pc > 0 else None, pc)


# ------------------------------------------------------------ orquestração


def choose_tool(gguf: str | None, ollama_model: str | None) -> tuple[str | None, list[str]]:
    """-> (ferramenta | None, o que falta). llama-bench exige binário no PATH e GGUF em disco."""
    missing: list[str] = []
    if gguf:
        if shutil.which("llama-bench") and os.path.isfile(gguf):
            return "llama-bench", []
        if not shutil.which("llama-bench"):
            missing.append("llama-bench não está no PATH")
        if not os.path.isfile(gguf):
            missing.append("arquivo GGUF não existe em disco")
    if ollama_model:
        return "ollama", []
    missing.append("sem --ollama-model para usar a API local do Ollama")
    return None, missing


def measure_model(target: Target, tool: str, gguf: str | None, ollama_model: str | None,
                  power: dict, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    arch: str | None = None
    if tool == "ollama":
        assert ollama_model
        arch = ollama_check_quant(ollama_model, target)["architecture"]
        prompt = load_prompt()
        fn: Callable[[int], Round] = lambda n: ollama_round(ollama_model, target, prompt, n)
    else:
        assert gguf

        def fn(n: int) -> Round:
            return llama_bench_round(gguf, target)[0]
    res = stabilize(fn)
    version = engine_version(tool)
    bw = int(round(target.bytes_per_token * res["decode_tps"]))
    return {
        "model_id": target.model_id, "tool": tool, "tool_version": version,
        "ollama_model": ollama_model if tool == "ollama" else None, "architecture": arch,
        "quant": target.quant, "n_gpu_layers": target.n_gpu_layers, "ctx": target.ctx, "threads": target.threads,
        "pool": target.pool, "gen_tokens": GEN_TOKENS, "prompt_tokens": res["prompt_tokens"],
        "warmup": res["warmup"], "rounds": res["rounds"],
        "decode_tps": res["decode_tps"], "stable": res["stable"],
        "prefill_tps": res["prefill_tps"], "prefill_stable": res["prefill_stable"],
        "bytes_per_token": target.bytes_per_token, "effective_bandwidth_bytes_per_s": bw,
        "power": power, "on_battery_warning": power["on_ac"] is False,
        "measured_at": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
