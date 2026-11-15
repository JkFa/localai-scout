"""Arquivo local da calibração: caminho, leitura, validação, merge e escrita.

Só stdlib e sem rede (schemas/calibration_store_spec.md §1 e §6). A comparação
de "igual" vem de scout.calibration, para o bench e o plan concordarem.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from scout.calibration import (CALIBRATION_SCHEMA_VERSION, fmt_field_value, MACHINE_FIELDS, MATCH_FIELDS, MEASURED_WITH_FIELDS,
                               POWER_ATTRS, SCOPE_POOLS, CalibrationInput, compare_field, compare_power_attr,
                               entry_eligible, normalize_power)

FILE_NAME = "calibration.json"
APP_DIR = "localai-scout"
FILE_SCHEMA_VERSION = 1
POOLS = ("gpu", "cpu", "unified")
MACHINE_KEYS = MACHINE_FIELDS


class CalibrationError(Exception):
    """Arquivo ausente (origem flag), malformado ou de formato não suportado."""


# ------------------------------------------------------------------ caminho (puro)


def resolve_calibration_path(flag: str | None, env: Mapping[str, str], platform: str,
                             home: Path | None) -> tuple[Path | None, str]:
    """flag > SCOUT_CALIBRATION > padrão do SO. Sem home no padrão: (None, "default")."""
    if flag:
        return Path(flag), "flag"
    if env.get("SCOUT_CALIBRATION"):
        return Path(env["SCOUT_CALIBRATION"]), "env"
    if platform == "darwin":
        return (home / "Library" / "Application Support" / APP_DIR / FILE_NAME if home else None), "default"
    if platform in ("win32", "cygwin") or platform.startswith("windows"):
        base = env.get("LOCALAPPDATA")
        if base:
            return Path(base) / APP_DIR / FILE_NAME, "default"
        return (home / "AppData" / "Local" / APP_DIR / FILE_NAME if home else None), "default"
    xdg = env.get("XDG_STATE_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / APP_DIR / FILE_NAME, "default"
    return (home / ".local" / "state" / APP_DIR / FILE_NAME if home else None), "default"


def find_home(env: Mapping[str, str]) -> Path | None:
    """HOME/USERPROFILE do ambiente; senão o do SO; senão None (nunca levanta)."""
    for key in ("HOME", "USERPROFILE"):
        if env.get(key):
            return Path(env[key])
    try:
        return Path.home()
    except (RuntimeError, KeyError, OSError):
        return None


# ------------------------------------------------------------------ validação (stdlib)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_doc(doc: Any) -> list[str]:
    """Checagem equivalente (no que o código consome) a calibration.schema.json. Lista de erros."""
    if not isinstance(doc, dict):
        return ["raiz não é objeto"]
    errs: list[str] = []
    if doc.get("schema_version") != FILE_SCHEMA_VERSION or isinstance(doc.get("schema_version"), bool):
        errs.append(f"schema_version {doc.get('schema_version')!r} não suportado (esperado {FILE_SCHEMA_VERSION})")
        return errs
    if doc.get("kind") != "calibration":
        errs.append("kind deve ser 'calibration'")
    if not isinstance(doc.get("scout_version"), str):
        errs.append("scout_version ausente")
    if not _is_int(doc.get("calibration_schema_version")):
        errs.append("calibration_schema_version deve ser inteiro")
    machine = doc.get("machine")
    if not isinstance(machine, dict):
        errs.append("machine ausente")
    else:
        miss = [k for k in MACHINE_KEYS if k not in machine]
        extra = sorted(set(machine) - set(MACHINE_KEYS))
        if miss or extra:
            errs.append(f"machine com chaves erradas (faltam: {', '.join(miss) or '-'}; "
                        f"sobram: {', '.join(extra) or '-'})")
    pools = doc.get("pools")
    if not isinstance(pools, dict) or not pools:
        errs.append("pools ausente ou vazio")
        return errs
    for pid, entries in pools.items():
        if pid not in POOLS:
            errs.append(f"pool desconhecido: {pid!r}")
            continue
        if not isinstance(entries, list) or not entries:
            errs.append(f"pools.{pid}: lista vazia ou inválida")
            continue
        for i, e in enumerate(entries):
            errs += [f"pools.{pid}[{i}]: {m}" for m in _entry_errors(e)]
    return errs


def _entry_errors(e: Any) -> list[str]:
    if not isinstance(e, dict):
        return ["não é objeto"]
    errs: list[str] = []
    bw, ovh = e.get("bandwidth_bytes_per_s"), e.get("overhead_s_per_token")
    if not _is_int(bw) or bw < 1:
        errs.append("bandwidth_bytes_per_s deve ser inteiro >= 1")
    if not _is_num(ovh) or ovh < 0:
        errs.append("overhead_s_per_token deve ser número >= 0")
    if e.get("form") not in ("1p", "2p"):
        errs.append("form deve ser 1p ou 2p")
    if not isinstance(e.get("stable"), bool):
        errs.append("stable deve ser booleano")
    p = e.get("power")
    if not isinstance(p, dict) or set(p) != {"on_ac", "profile", "not_applicable"} \
            or not isinstance(p.get("not_applicable"), list) \
            or any(a not in POWER_ATTRS for a in p["not_applicable"]):
        errs.append("power deve ter on_ac, profile e not_applicable")
    if e.get("tool") not in ("ollama", "llama-bench"):
        errs.append("tool deve ser ollama ou llama-bench")
    if "tool_version" not in e or not (e["tool_version"] is None or (isinstance(e["tool_version"], str) and e["tool_version"])):
        errs.append("tool_version deve ser string ou null")
    mw = e.get("measured_with")
    if not isinstance(mw, dict) or set(mw) != set(MEASURED_WITH_FIELDS):
        errs.append("measured_with deve ter os.kernel, memory.speed_mts e memory.channels")
    models = e.get("models")
    if not isinstance(models, list) or not models:
        errs.append("models vazio")
    if not isinstance(e.get("date"), str) or len(e["date"]) != 10:
        errs.append("date deve ser YYYY-MM-DD")
    pf = e.get("prefill_tokens_per_s")
    if "prefill_tokens_per_s" not in e or not (pf is None or (_is_num(pf) and pf > 0)):
        errs.append("prefill_tokens_per_s deve ser número > 0 ou null")
    return errs


# ------------------------------------------------------------------ leitura


def read_calibration(path: Path | None, origin: str) -> CalibrationInput:
    """Lê e valida. Ausente: erro com origem `flag`, doc None nas demais. Malformado: erro alto.
    Com origem `disabled` ou sem caminho, não lê nada."""
    if origin == "disabled" or path is None:
        return CalibrationInput(origin, None)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if origin == "flag":
            raise CalibrationError(f"arquivo de calibração {path} não existe (--calibration)") from None
        return CalibrationInput(origin, None)
    except OSError as e:
        raise CalibrationError(f"não foi possível ler a calibração {path}: {e.strerror or e}") from e
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as e:
        raise CalibrationError(f"{path} não é JSON válido: {e}. Apague-o e rode `scout bench --run`.") from e
    errs = validate_doc(doc)
    if errs:
        raise CalibrationError(f"{path} inválido: {'; '.join(errs[:4])}. Apague-o e rode `scout bench --run`.")
    return CalibrationInput(origin, doc)


def tools_in(doc: dict | None) -> list[str]:
    """Ferramentas distintas das entradas, em ordem estável."""
    if not doc:
        return []
    return sorted({e["tool"] for es in doc["pools"].values() for e in es})


# ------------------------------------------------------------------ recusas antes de medir (§6)

LAYER1_ALL = tuple(f for f, layer, scope in MATCH_FIELDS if layer == 1 and scope == "all")
LAYER1_GPU = tuple(f for f, layer, scope in MATCH_FIELDS if layer == 1 and scope == "gpu")


def identity_refusal(identity: dict, live_identity: dict | None, pools: set[str]) -> tuple[str, str] | None:
    """(reason, mensagem) se o bench deve recusar antes de medir; None se pode.
    `identity` é a do --system; `live_identity` a de uma leitura ao vivo sem privilégio."""
    needed = list(LAYER1_ALL) + (list(LAYER1_GPU) if "gpu" in pools else [])
    nulls = [f for f in needed if identity.get(f) is None]
    if nulls:
        return ("identity_incomplete",
                f"o system.json não tem {', '.join(nulls)}; sem a identidade de hardware a calibração não "
                "pode ser comparada depois. Rode `scout collect` de novo.")
    if live_identity is None:
        return ("system_not_this_machine", "não foi possível ler esta máquina ao vivo para conferir o system.json")
    bad = [f for f in needed if compare_field(f, identity.get(f), live_identity.get(f)) != "equal"]
    if bad:
        return ("system_not_this_machine",
                f"o system.json não descreve esta máquina ({', '.join(bad)} difere da leitura ao vivo); "
                "rode `scout collect` aqui e use esse arquivo.")
    return None


def mixed_tools(plans: list[tuple[str, str | None]]) -> str | None:
    """plans = [(pool, tool|None)]. Devolve o pool que mistura ferramentas, ou None."""
    seen: dict[str, set[str]] = {}
    for pool, tool in plans:
        if tool is not None:
            seen.setdefault(pool, set()).add(tool)
    return next((p for p, ts in sorted(seen.items()) if len(ts) > 1), None)


# ------------------------------------------------------------------ merge (§6)


def same_power_key(a: dict | None, b: dict | None) -> bool:
    """Mesma chave de energia: os dois atributos iguais (§4) ou estados idênticos (inclusive desconhecidos)."""
    if all(compare_power_attr(a, b, x) == "equal" for x in POWER_ATTRS):
        return True
    return normalize_power(a) == normalize_power(b)


def identity_drop_reason(stored: dict, current: dict, pool: str) -> str | None:
    """Motivo para descartar as entradas antigas do pool pela camada 1 (`machine` do arquivo contra o
    --system atual): diferente ou null cai. Camada 2 e kernel são por entrada (`entry_drop_reason`)."""
    for field, layer, scope in MATCH_FIELDS:
        if layer != 1 or pool not in SCOPE_POOLS[scope]:
            continue
        r = compare_field(field, stored.get(field), current.get(field))
        if r != "equal":
            return f"{field} {'difere' if r == 'different' else 'não lido'} " \
                   f"(gravado: {fmt_field_value(field, stored.get(field))}; " \
                   f"agora: {fmt_field_value(field, current.get(field))})"
    return None


def entry_drop_reason(entry: dict, current_mw: dict, pool: str) -> str | None:
    """Camada 2 do `measured_with` DELA conhecida e diferente do --system atual. Desconhecida e kernel não derrubam."""
    if entry_eligible(entry, current_mw, pool):
        return None
    mw = entry.get("measured_with") or {}
    diffs = [f"{f}: gravado {mw.get(f)!r}, agora {current_mw.get(f)!r}" for f in MEASURED_WITH_FIELDS
             if f != "os.kernel" and compare_field(f, mw.get(f), current_mw.get(f)) == "different"]
    return "; ".join(diffs) or "camada 2 difere"


def merge_calibration(old: dict | None, machine: dict, new_pools: dict[str, dict], scout_version: str,
                      measured: dict | None = None) -> tuple[dict, list[str]]:
    """-> (documento novo, mensagens do que caiu e por quê). Não toca em `old`.
    `machine` e `measured` vêm do --system atual (camada 1; kernel e camada 2). Só a entrada NOVA recebe
    `measured_with`; o das antigas nunca é reescrito."""
    notes: list[str] = []
    measured = dict(measured) if measured is not None else {f: None for f in MEASURED_WITH_FIELDS}
    pools: dict[str, list[dict]] = {}
    if old is not None:
        if old.get("calibration_schema_version") != CALIBRATION_SCHEMA_VERSION:
            notes.append(f"calibração antiga descartada: calibration_schema_version "
                         f"{old.get('calibration_schema_version')} != {CALIBRATION_SCHEMA_VERSION}")
        else:
            for pid in sorted(old["pools"]):
                why = identity_drop_reason(old.get("machine") or {}, machine, pid)
                if why:
                    notes.append(f"pool {pid} antigo descartado ({len(old['pools'][pid])} entrada(s)): {why}")
                    continue
                kept = []
                for e in old["pools"][pid]:
                    ewhy = entry_drop_reason(e, measured, pid)
                    if ewhy:
                        notes.append(f"pool {pid}: entrada antiga ({'; '.join(describe_state(e))}) descartada: {ewhy}")
                    else:
                        kept.append(dict(e))
                if kept:
                    pools[pid] = kept
    for pid, entry in sorted(new_pools.items()):
        entry = {**entry, "measured_with": dict(measured)}
        entries = pools.setdefault(pid, [])
        idx = next((i for i, e in enumerate(entries) if same_power_key(e.get("power"), entry.get("power"))), None)
        if idx is None:
            entries.append(entry)
        else:
            entries[idx] = entry
            notes.append(f"pool {pid}: entrada do mesmo estado de energia substituída")
    doc = {"schema_version": FILE_SCHEMA_VERSION, "kind": "calibration", "scout_version": scout_version,
           "calibration_schema_version": CALIBRATION_SCHEMA_VERSION, "machine": dict(machine), "pools": pools}
    return doc, notes


def describe_state(e: dict) -> list[str]:
    from scout.calibration import describe_power

    return [describe_power(e.get("power"))]


# ------------------------------------------------------------------ escrita (§6)


def serialize(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_calibration_file(path: Path, doc: dict) -> None:
    """Atômica (tmp no mesmo diretório, fsync, os.replace) e determinística. Valida antes de gravar.
    Qualquer falha remove o temporário e deixa o arquivo antigo intacto."""
    errs = validate_doc(doc)
    if errs:
        raise CalibrationError(f"calibração a gravar é inválida: {'; '.join(errs[:3])}")
    text = serialize(doc)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".calibration-", suffix=".tmp")
    except OSError as e:
        raise CalibrationError(f"não foi possível gravar em {path.parent}: {e.strerror or e}") from e
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise CalibrationError(f"não foi possível gravar {path}: {e.strerror or e}") from e
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
