"""Calibração local da máquina: identidade, comparação e escolha da entrada.

Puro: sem arquivo, sem rede, sem relógio (schemas/calibration_store_spec.md
§2–§5). Quem lê o arquivo, a energia e a versão do motor é a CLI
(scout/calstore.py, scout/bench.py); aqui só se compara o que já chegou.

Três resultados por comparação: "equal" | "different" | "unknown".
Efeitos de uma linha `match`: "none" | "discard" | "cap_medium" | "downgrade".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from scout.estimate import fmt_gb, fmt_gbps

CALIBRATION_SCHEMA_VERSION = 2  # 2: bytes por token inclui o KV lido (banda de significado novo)

ORIGINS: tuple[str, ...] = ("flag", "env", "default", "disabled")
POWER_ATTRS: tuple[str, ...] = ("on_ac", "profile")

# (campo, camada, escopo), na ordem de calibration.schema.json#/$defs/match_field.
MATCH_FIELDS: tuple[tuple[str, int | str, str], ...] = (
    ("platform", 1, "all"),
    ("os.arch", 1, "all"),
    ("os.kernel", "software", "all"),
    ("cpu.model", 1, "all"),
    ("cpu.cores_physical", 1, "all"),
    ("memory.ram_total_bytes", 1, "all"),
    ("gpu.name", 1, "gpu"),
    ("gpu.backend", 1, "gpu"),
    ("gpu.vram_total_bytes", 1, "gpu"),
    ("memory.speed_mts", 2, "cpu"),
    ("memory.channels", 2, "cpu"),
)
# Camada 1 (identidade de HARDWARE, `machine` do arquivo) e o contexto da
# MEDIÇÃO (`measured_with` de cada entrada: kernel e camada 2), rev. 4.
MACHINE_FIELDS: tuple[str, ...] = tuple(f for f, layer, _ in MATCH_FIELDS if layer == 1)
MEASURED_WITH_FIELDS: tuple[str, ...] = tuple(f for f, layer, _ in MATCH_FIELDS if layer != 1)
LAYER2_FIELDS: tuple[str, ...] = tuple(f for f, layer, _ in MATCH_FIELDS if layer == 2)
TOLERANT_FIELDS = frozenset({"memory.ram_total_bytes", "gpu.vram_total_bytes"})  # ±5% do maior

# Escopo -> pools a que a linha se aplica. Camada 2 só no pool cpu (§2): no
# `unified` a dimensão MT/s × canais não existe à parte (não aplicável).
SCOPE_POOLS: dict[str, tuple[str, ...]] = {"all": ("gpu", "cpu", "unified"), "gpu": ("gpu",), "cpu": ("cpu",)}


@dataclass(frozen=True)
class CalibrationInput:
    """O que a CLI entrega ao solver. origin: flag|env|default|disabled.
    doc: o calibration.json já lido e validado, ou None (ausente/desligado)."""

    origin: str
    doc: dict | None = None


# --------------------------------------------------------------------------
# Identidade (§2)


def _value(node: Any) -> Any:
    return node.get("value") if isinstance(node, dict) else None


def pick_gpu(gpus_with_vram: list[dict]) -> dict:
    """Maior VRAM total; empate pelo menor índice. Mesma regra do pool gpu do solver."""
    return sorted(gpus_with_vram, key=lambda g: (-_value(g["vram_total_bytes"]), g["index"]))[0]


def pool_gpu(system: dict) -> dict | None:
    """A GPU que vira o pool gpu do solver, ou None se não há pool gpu
    (memória unificada, sem GPU, sem VRAM total ou sem VRAM livre)."""
    if _value(system.get("memory", {}).get("unified")) is True:
        return None
    gpus = _value(system.get("gpus"))
    if not gpus:
        return None
    with_vram = [g for g in gpus if _value(g.get("vram_total_bytes")) is not None]
    if not with_vram:
        return None
    chosen = pick_gpu(with_vram)
    if _value(chosen.get("vram_free_bytes")) is None:
        return None
    return chosen


def machine_identity(system: dict) -> dict[str, Any]:
    """Camada 1 (calibration.schema.json#/$defs/machine): o `value` de cada
    campo do system.json. Nada de hostname, serial, usuário, caminho."""
    cpu = system.get("cpu", {})
    gpu = pool_gpu(system) or {}
    platform = system.get("platform")
    return {
        "platform": platform if isinstance(platform, str) else None,
        "os.arch": _value(system.get("os", {}).get("arch")),
        "cpu.model": _value(cpu.get("model")),
        "cpu.cores_physical": _value(cpu.get("cores_physical")),
        "memory.ram_total_bytes": _value(system.get("memory", {}).get("ram_total_bytes")),
        "gpu.name": _value(gpu.get("name")),
        "gpu.backend": _value(gpu.get("backend")),
        "gpu.vram_total_bytes": _value(gpu.get("vram_total_bytes")),
    }


def measured_with(system: dict) -> dict[str, Any]:
    """Contexto da medição ($defs/measured_with): kernel e camada 2, gravados em
    cada entrada pelo bench e nunca reescritos por benches posteriores."""
    mem = system.get("memory", {})
    return {
        "os.kernel": _value(system.get("os", {}).get("kernel")),
        "memory.speed_mts": _value(mem.get("speed_mts")),
        "memory.channels": _value(mem.get("channels")),
    }


def present_pools(system: dict) -> tuple[str, ...]:
    """Pools que o solver monta para este system (mesma regra de _build_pools)."""
    if _value(system.get("memory", {}).get("unified")) is True:
        return ("unified",)
    return (("gpu",) if pool_gpu(system) is not None else ()) + ("cpu",)


_KERNEL_RE = re.compile(r"^\d+\.\d+")


def kernel_key(s: str) -> str:
    """"6.99.0-1-exemplo" -> "6.99"; sem o padrão major.minor, a string inteira."""
    m = _KERNEL_RE.match(s)
    return m.group(0) if m else s


def within_5pct(a: int, b: int) -> bool:
    """|a − b| ≤ 5% de max(a, b), em inteiros (sem float)."""
    return 20 * abs(a - b) <= max(a, b)


def compare_field(field: str, stored: Any, current: Any) -> str:
    """equal | different | unknown (null em algum lado)."""
    if stored is None or current is None:
        return "unknown"
    if field == "os.kernel":
        return "equal" if kernel_key(str(stored)) == kernel_key(str(current)) else "different"
    if field in TOLERANT_FIELDS and isinstance(stored, int) and isinstance(current, int) \
            and not isinstance(stored, bool) and not isinstance(current, bool):
        return "equal" if within_5pct(stored, current) else "different"
    return "equal" if stored == current else "different"


def field_effect(layer: int | str, result: str) -> str:
    """Camada 1: diferente ou null -> discard. Camada 2: diferente -> discard,
    desconhecido -> cap_medium. Software (kernel): diferente -> downgrade,
    desconhecido -> cap_medium."""
    if result == "equal":
        return "none"
    if layer == 1:
        return "discard"
    if layer == 2:
        return "discard" if result == "different" else "cap_medium"
    return "downgrade" if result == "different" else "cap_medium"


def match_lines(stored: dict, current: dict, file_pools: set[str] | frozenset[str]) -> list[dict[str, Any]]:
    """Uma linha por campo da camada 1. O efeito é o que a linha causa nos
    pools do seu escopo que têm entradas no arquivo; sem nenhum, `none`."""
    lines: list[dict[str, Any]] = []
    for field, layer, scope in MATCH_FIELDS:
        if layer != 1:
            continue
        s, c = stored.get(field), current.get(field)
        result = compare_field(field, s, c)
        applies = any(p in file_pools for p in SCOPE_POOLS[scope])
        lines.append({
            "field": field, "layer": layer, "scope": scope, "stored": s, "current": c,
            "result": result, "effect": field_effect(layer, result) if applies else "none",
        })
    return lines


def entry_match_lines(stored_mw: Any, current_mw: dict, pool_id: str) -> list[dict[str, Any]]:
    """Linhas de kernel (software) e camada 2 de UMA entrada (o measured_with
    dela) contra o system atual. Camada 2 só tem efeito no pool cpu."""
    stored_mw = stored_mw if isinstance(stored_mw, dict) else {}
    lines: list[dict[str, Any]] = []
    for field, layer, scope in MATCH_FIELDS:
        if layer == 1:
            continue
        s, c = stored_mw.get(field), current_mw.get(field)
        result = compare_field(field, s, c)
        lines.append({
            "field": field, "layer": layer, "scope": scope, "stored": s, "current": c, "result": result,
            "effect": field_effect(layer, result) if pool_id in SCOPE_POOLS[scope] else "none",
        })
    return lines


def entry_eligible(entry: dict, current_mw: dict, pool_id: str) -> bool:
    """Inelegível: camada 2 do measured_with conhecida dos dois lados e
    diferente (só no pool cpu, §4 rev. 4)."""
    return not any(ln["effect"] == "discard" for ln in entry_match_lines(entry.get("measured_with"), current_mw, pool_id))


def _applies_to(line: dict, pool_id: str) -> bool:
    return pool_id in SCOPE_POOLS[line["scope"]]


# --------------------------------------------------------------------------
# Energia (§4)


def normalize_power(power: dict | None) -> dict[str, Any]:
    """Formato $defs/power. Sem leitura nenhuma: tudo desconhecido (nunca
    "não aplicável", que exige evidência positiva)."""
    p = power if isinstance(power, dict) else {}
    na = [a for a in POWER_ATTRS if a in (p.get("not_applicable") or [])]
    return {
        "on_ac": None if "on_ac" in na else p.get("on_ac"),
        "profile": None if "profile" in na else p.get("profile"),
        "not_applicable": na,
    }


def _attr_state(power: dict, attr: str) -> str:
    """known | na | unknown."""
    if attr in power["not_applicable"]:
        return "na"
    return "unknown" if power.get(attr) is None else "known"


def compare_power_attr(stored: dict, current: dict, attr: str) -> str:
    """Os dois não aplicáveis -> equal; os dois conhecidos -> pelos valores;
    conhecido × não aplicável -> different; desconhecido em algum lado -> unknown."""
    stored, current = normalize_power(stored), normalize_power(current)
    s, c = _attr_state(stored, attr), _attr_state(current, attr)
    if s == "unknown" or c == "unknown":
        return "unknown"
    if s == "na" and c == "na":
        return "equal"
    if s == "known" and c == "known":
        return "equal" if stored[attr] == current[attr] else "different"
    return "different"


def describe_power(power: dict | None) -> str:
    """"na tomada, perfil performance", "na bateria, perfil balanced",
    "sem bateria", "perfil não lido"…"""
    p = normalize_power(power)
    parts: list[str] = []
    st = _attr_state(p, "on_ac")
    if st == "na":
        parts.append("sem bateria")
    elif st == "known":
        parts.append("na tomada" if p["on_ac"] else "na bateria")
    else:
        parts.append("tomada/bateria não lida")
    st = _attr_state(p, "profile")
    if st == "known":
        parts.append(f"perfil {p['profile']}")
    elif st == "unknown":
        parts.append("perfil não lido")
    return ", ".join(parts)


def _date_key(d: Any) -> int:
    try:
        return int(str(d).replace("-", ""))
    except ValueError:
        return 0


def _slowest(entries: list[tuple[int, dict]]) -> tuple[int, dict]:
    """Menor banda; empate: data mais recente; depois a ordem do arquivo."""
    return min(entries, key=lambda ie: (ie[1]["bandwidth_bytes_per_s"], -_date_key(ie[1].get("date")), ie[0]))


def choose_entry(entries: list[dict], current: dict | None) -> tuple[dict, str]:
    """(entrada, power_match) com power_match em same | other_state | unknown (§4)."""
    if not entries:
        raise ValueError("pool de calibração sem entradas")
    cur = normalize_power(current)
    indexed = list(enumerate(entries))

    def results(e: dict) -> list[str]:
        return [compare_power_attr(e.get("power"), cur, a) for a in POWER_ATTRS]

    same = [ie for ie in indexed if all(r == "equal" for r in results(ie[1]))]
    if same:
        return _slowest(same)[1], "same"
    if all(_attr_state(cur, a) != "unknown" for a in POWER_ATTRS):
        return _slowest(indexed)[1], "other_state"
    # Estado atual não provado: menor banda entre as compatíveis com a parte
    # conhecida (nenhum atributo "different"); todas, se nenhuma for.
    compatible = [ie for ie in indexed if "different" not in results(ie[1])]
    return _slowest(compatible or indexed)[1], "unknown"


# --------------------------------------------------------------------------
# Avaliação completa (§2–§5)


@dataclass(frozen=True)
class PoolDecision:
    entry: dict | None          # entrada aplicada; None se não vale (o pool usa eta)
    reasons: frozenset[str]     # motivos de confiança herdados pelos candidatos
    # Entradas cujo prefill vale para o TTFT: só a escolhida com power_match
    # same; senão todas as elegíveis, para o TTFT usar o MENOR prefill (§4).
    prefill_entries: tuple[dict, ...] = ()


def _w(code: str, severity: str, message: str, field: str | None = None) -> dict[str, Any]:
    w: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if field is not None:
        w["field"] = field
    return w


def fmt_field_value(field: str, v: Any) -> str:
    """Valor de campo de identidade para mensagens. Bytes só aparecem via fmt_gb (regra 4)."""
    if v is None:
        return "null"
    if field.endswith("_bytes") and isinstance(v, int) and not isinstance(v, bool):
        return fmt_gb(v)
    return f"'{v}'" if isinstance(v, str) else str(v)


def _memory_unverified_text(platform: Any, current_missing: bool) -> str:
    if platform != "linux":
        return ("O coletor ainda não lê MT/s e canais da RAM nesta plataforma; a calibração do "
                "pool cpu fica em confiança média.")
    if current_missing:
        return ("MT/s e canais da RAM não foram lidos neste system.json; sem essa evidência a "
                "calibração do pool cpu fica em confiança média. Rode uma vez `sudo scout collect` "
                "(ou `scout collect --dmidecode <arquivo de sudo dmidecode -t memory>`) e refaça o "
                "plan para liberar confiança alta.")
    return ("A calibração foi gravada sem MT/s e canais da RAM; gere o system.json com o dmidecode "
            "legível (`sudo scout collect` ou `scout collect --dmidecode <arquivo de sudo dmidecode "
            "-t memory>`) e rode `scout bench --run` de novo para liberar confiança alta.")


def _validate_entry(pool_id: str, e: Any) -> None:
    bw = e.get("bandwidth_bytes_per_s") if isinstance(e, dict) else None
    ovh = e.get("overhead_s_per_token", 0) if isinstance(e, dict) else None
    if (isinstance(bw, bool) or not isinstance(bw, int) or bw < 1
            or isinstance(ovh, bool) or not isinstance(ovh, (int, float)) or ovh < 0):
        raise ValueError(f"calibração do pool {pool_id} inválida: banda inteira > 0 e overhead >= 0")


def _eligible_split(entries: list[dict], current_mw: dict, pool_id: str) -> list[dict]:
    return [e for e in entries if entry_eligible(e, current_mw, pool_id)]


def evaluate(calibration: CalibrationInput | None, system: dict, power: dict | None,
             engines: dict[str, str | None] | None, present: tuple[str, ...]
             ) -> tuple[dict[str, Any], dict[str, PoolDecision], list[dict[str, Any]], dict | None]:
    """Devolve (eco inputs.calibration, decisão por pool do arquivo, avisos,
    inputs.power ou None)."""
    base = {"calibration_schema_version": CALIBRATION_SCHEMA_VERSION, "store": None, "match": [], "pools": {}}
    if calibration is None or calibration.origin == "disabled":
        return {"origin": "disabled", "status": "disabled", **base}, {}, [], None
    if calibration.origin not in ORIGINS:
        raise ValueError(f"origem de calibração inválida: {calibration.origin!r}")
    doc = calibration.doc
    if doc is None:
        return {"origin": calibration.origin, "status": "absent", **base}, {}, [], None
    if not isinstance(doc, dict) or not isinstance(doc.get("pools"), dict) or not doc["pools"]:
        raise ValueError("calibration.json sem pools")

    engines = engines or {}
    cur_power = normalize_power(power)
    stored_machine = doc.get("machine") if isinstance(doc.get("machine"), dict) else {}
    current_machine = machine_identity(system)
    current_mw = measured_with(system)
    file_pools = sorted(doc["pools"])
    match = match_lines(stored_machine, current_machine, frozenset(file_pools))
    outdated = doc.get("calibration_schema_version") != CALIBRATION_SCHEMA_VERSION

    warnings: list[dict[str, Any]] = []
    if outdated:
        warnings.append(_w(
            "calibration_schema_outdated", "warn",
            f"A calibração foi gravada com calibration_schema_version "
            f"{doc.get('calibration_schema_version')}; este scout espera {CALIBRATION_SCHEMA_VERSION}. "
            "Calibração descartada; rode `scout bench --run` para recalibrar.",
            "inputs.calibration.calibration_schema_version"))
    else:
        for ln in match:
            if ln["effect"] != "discard":
                continue
            pools = ", ".join(p for p in file_pools if _applies_to(ln, p))
            if ln["result"] == "unknown":
                code, what = "calibration_machine_unknown", "não foi lido"
            else:
                code, what = "calibration_machine_mismatch", "difere"
            warnings.append(_w(
                code, "warn",
                f"{ln['field']} {what} (calibração: {fmt_field_value(ln['field'], ln['stored'])}; agora: "
                f"{fmt_field_value(ln['field'], ln['current'])}): calibração do pool {pools} descartada, tokens/s volta "
                "ao eta. Rode `scout bench --run` nesta máquina para recalibrar.",
                ln["field"]))

    echo_pools: dict[str, Any] = {}
    decisions: dict[str, PoolDecision] = {}
    applied_any = False
    mem_warning: dict[str, Any] | None = None
    for pid in file_pools:
        entries = doc["pools"][pid]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"calibração do pool {pid}: lista de entradas vazia")
        for e in entries:
            _validate_entry(pid, e)
        # Camada 2 entrada por entrada, ANTES da energia (§4 rev. 4).
        eligible = _eligible_split(entries, current_mw, pid)
        entry, pmatch = choose_entry(eligible or entries, cur_power)
        pool_match = entry_match_lines(entry.get("measured_with"), current_mw, pid) if eligible else []
        tool = entry.get("tool")
        stored_v, current_v = entry.get("tool_version"), engines.get(tool)
        eng = "unknown" if stored_v is None or current_v is None else (
            "equal" if stored_v == current_v else "different")

        discard: list[str] = []
        if outdated:
            discard.append("calibration_schema_outdated")
        for ln in match:
            if ln["effect"] == "discard" and _applies_to(ln, pid):
                discard.append("calibration_machine_unknown" if ln["result"] == "unknown"
                               else "calibration_machine_mismatch")
        if not eligible:
            discard.append("calibration_machine_mismatch")
            if not outdated:
                diffs = list(dict.fromkeys(
                    f"{ln['field']}: calibração {fmt_field_value(ln['field'], ln['stored'])}, agora {fmt_field_value(ln['field'], ln['current'])}"
                    for e in entries for ln in entry_match_lines(e.get("measured_with"), current_mw, pid)
                    if ln["effect"] == "discard"))
                first = next(ln["field"] for ln in entry_match_lines(entries[0].get("measured_with"), current_mw, pid)
                             if ln["effect"] == "discard")
                warnings.append(_w(
                    "calibration_machine_mismatch", "warn",
                    f"Nenhuma medição do pool {pid} foi feita com a RAM atual ({'; '.join(diffs)}): "
                    "calibração do pool descartada, tokens/s volta ao eta. Rode `scout bench --run` "
                    "nesta máquina para recalibrar.", first))
        reasons: list[str] = []
        if pid not in present:
            status = "pool_absent"
            reasons = ["calibration_pool_absent"] + discard
        elif discard:
            status = "discarded"
            reasons = discard
        else:
            status = "applied"
            applied_any = True
            if pmatch == "other_state":
                reasons.append("calibration_power_mismatch")
                warnings.append(_w(
                    "calibration_power_mismatch", "warn",
                    f"Calibrado {describe_power(entry.get('power'))}; você está "
                    f"{describe_power(cur_power)}. Usando a medição mais lenta desta máquina "
                    f"({fmt_gbps(entry['bandwidth_bytes_per_s'])} efetivos) para não prometer "
                    "velocidade. Rode `scout bench --run` neste estado para uma estimativa exata.",
                    f"inputs.calibration.pools.{pid}.power_match"))
            elif pmatch == "unknown":
                reasons.append("calibration_power_unknown")
                unk = [("tomada/bateria desconhecida" if a == "on_ac" else "perfil desconhecido")
                       for a in POWER_ATTRS if _attr_state(cur_power, a) == "unknown"]
                warnings.append(_w(
                    "calibration_power_unknown", "info",
                    f"Estado de energia atual não lido ({', '.join(unk)}); usando a "
                    "medição mais lenta compatível desta máquina "
                    f"({fmt_gbps(entry['bandwidth_bytes_per_s'])} efetivos).",
                    f"inputs.calibration.pools.{pid}.power_match"))
            if eng == "different":
                reasons.append("calibration_engine_changed")
                warnings.append(_w(
                    "calibration_engine_changed", "warn",
                    f"A calibração do pool {pid} foi medida com {tool} {stored_v}; agora está "
                    f"{current_v}. Rode `scout bench --run` para recalibrar.",
                    f"inputs.calibration.pools.{pid}.engine"))
            elif eng == "unknown":
                reasons.append("calibration_engine_unknown")
                if current_v is None:
                    msg = (f"Versão atual de {tool} não lida (motor parado ou ausente); "
                           "inicie-o e refaça o plan.")
                else:
                    msg = (f"A calibração do pool {pid} foi gravada sem a versão de {tool}; "
                           "rode `scout bench --run` para recalibrar.")
                warnings.append(_w("calibration_engine_unknown", "info", msg,
                                   f"inputs.calibration.pools.{pid}.engine"))
            # Kernel e camada 2 DA ENTRADA ESCOLHIDA (§4 rev. 4).
            kl = next(ln for ln in pool_match if ln["field"] == "os.kernel")
            if kl["result"] == "different":
                reasons.append("calibration_kernel_changed")
                warnings.append(_w(
                    "calibration_kernel_changed", "warn",
                    f"A calibração foi medida com o kernel {kernel_key(str(kl['stored']))}; agora é "
                    f"{kernel_key(str(kl['current']))}. Rode `scout bench --run` para recalibrar.",
                    "os.kernel"))
            elif kl["result"] == "unknown":
                reasons.append("calibration_kernel_unknown")
                side = "neste system.json" if kl["current"] is None else "na medição usada"
                warnings.append(_w(
                    "calibration_kernel_unknown", "info",
                    f"Versão do kernel não registrada {side}; a calibração fica em confiança média. "
                    "Rode `scout bench --run` para recalibrar.", "os.kernel"))
            mem = [ln for ln in pool_match if ln["effect"] == "cap_medium" and ln["layer"] == 2]
            if mem:
                reasons.append("calibration_memory_unverified")
                if mem_warning is None:
                    cur_missing = any(ln["current"] is None for ln in mem)
                    mem_warning = _w("calibration_memory_unverified", "info",
                                     _memory_unverified_text(current_machine["platform"], cur_missing),
                                     mem[0]["field"])

        if status != "applied":
            # Pool descartado ou ausente: kernel e camada 2 não modulam nada (revisão, rodada 2).
            pool_match = [{**ln, "effect": "none"} for ln in pool_match]
        echo_pools[pid] = {
            "status": status,
            "reasons": list(dict.fromkeys(reasons)),
            "power_match": pmatch,
            "available_power": [normalize_power(e.get("power")) for e in entries],
            "engine": {"tool": tool, "stored_version": stored_v, "current_version": current_v, "result": eng},
            "calibration": entry,
            "match": pool_match,
        }
        if status == "applied":
            prefill_entries = (entry,) if pmatch == "same" else tuple(eligible)
            decisions[pid] = PoolDecision(entry, frozenset(reasons), prefill_entries)
        else:
            decisions[pid] = PoolDecision(None, frozenset())
    if mem_warning is not None:
        warnings.append(mem_warning)

    # Mensagens idênticas (ex.: motor parado em dois pools) saem uma vez só.
    seen: set[tuple[str, str]] = set()
    uniq = []
    for w in warnings:
        if (w["code"], w["message"]) not in seen:
            seen.add((w["code"], w["message"]))
            uniq.append(w)

    echo = {
        "origin": calibration.origin,
        "status": "used" if applied_any else "discarded",
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "store": {
            "scout_version": doc.get("scout_version"),
            "calibration_schema_version": doc.get("calibration_schema_version"),
            "machine": stored_machine,
        },
        "match": match,
        "pools": echo_pools,
    }
    return echo, decisions, uniq, cur_power


def chosen_tools(doc: dict, system: dict, power: dict | None) -> set[str]:
    """Ferramentas das entradas que o plan aplicaria (pools descartados ou
    ausentes não contam). A CLI lê a versão só destas (§3, Rede)."""
    _, decisions, _, _ = evaluate(CalibrationInput("env", doc), system, power, None, present_pools(system))
    return {d.entry["tool"] for d in decisions.values() if d.entry is not None and isinstance(d.entry.get("tool"), str)}
