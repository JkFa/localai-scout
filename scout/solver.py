"""Solver do M1: system + catálogo + intenção -> feasibility.json.

Puro: não lê arquivo, não toca a rede, não lê relógio. Toda aritmética passa
pelas funções de scout.estimate. Saída determinística: ordenação total, sem
iteração em set, sem timestamp.

REGRA NEGATIVA — swap: o solver NUNCA soma swap ao orçamento nem sugere
aumentá-lo. Swap evita o crash por falta de memória e não resolve lentidão:
pesos paginados para disco derrubam a geração para ~0,1 token/s. As
alavancas corretas, nesta ordem: KV em q8_0, menos contexto, bpw menor dentro
do piso, modelo menor. O aviso `swap_not_a_fix` sai sempre no relatório.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from scout import __version__
from scout import estimate as E

TASKS: tuple[str, ...] = ("chat", "code", "embeddings", "vlm", "stt", "diffusion")
PLACEMENTS: tuple[str, ...] = ("gpu_full", "hybrid", "cpu_only")
FULL_GPU_FORMATS: tuple[str, ...] = ("EXL2", "AWQ", "GPTQ")
# Preferência de qualidade (adendo de m1_interface.md, regra 3).
QUALITY_PREF_MAX_PARAMS = 7_000_000_000
QUALITY_PREF_LOW_BPW = 5.5
QUALITY_PREF_HIGH_BPW = 6.5
PROVENANCE_RANK: dict[str, int] = {"catalog": 0, "measured": 0, "calculated": 1, "estimated": 2}

SWAP_MESSAGE = (
    "Aumentar swap não faz modelo maior rodar: pesos paginados para disco derrubam a "
    "geração para ~0,1 token/s. Alavancas, nesta ordem: KV em q8_0, menos contexto, "
    "bpw menor dentro do piso, modelo menor."
)


@dataclass(frozen=True)
class Intent:
    tasks: tuple[str, ...]
    ctx: int = 8192
    commercial_use: bool = False
    batch: int = 1
    kv_quant: str = "f16"


@dataclass(frozen=True)
class Pool:
    id: str
    gpu_index: int | None
    total: int
    reserved: int
    usable: int
    bw: int | None
    bw_estimated: bool
    eta: float
    low_confidence: bool
    provenance: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gpu_index": self.gpu_index,
            "total_bytes": self.total,
            "reserved_bytes": self.reserved,
            "usable_bytes": self.usable,
            "bandwidth_bytes_per_s": self.bw,
            "eta": self.eta,
            "provenance": dict(sorted(self.provenance.items())),
        }


@dataclass(frozen=True)
class Attempt:
    """Uma tentativa (quant, placement) já calculada."""

    placement: str
    n_gpu: int
    by_pool: tuple[dict[str, Any], ...]
    total: int
    tier: str
    tps: float | None
    pools_used: tuple[Pool, ...]
    missing: int  # bytes para chegar ao tier limite; <= 0 se cabe


# --------------------------------------------------------------------------
# Leitura de campos do system.json (Field = {value, source, confidence})


def _field(node: Any) -> tuple[Any, str, str]:
    if not isinstance(node, dict):
        return None, "none", "low"
    return node.get("value"), str(node.get("source", "none")), str(node.get("confidence", "low"))


def _measured(conf: str) -> str:
    return "measured" if conf in ("high", "medium") else "estimated"


def _bw_estimated(source: str, conf: str) -> bool:
    return source.startswith("estimate:") or conf == "low"


def _worst(*provs: str) -> str:
    return max(provs, key=lambda p: PROVENANCE_RANK[p])


def _warning(code: str, severity: str, message: str, field: str | None = None) -> dict[str, Any]:
    w: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if field is not None:
        w["field"] = field
    return w


def _build_pools(system: dict, tuning: dict) -> tuple[list[Pool], list[dict[str, Any]]]:
    """Pools de memória a partir do system. Levanta ValueError sem ram_total."""
    warnings: list[dict[str, Any]] = []
    eta_gpu = float(tuning["eta_gpu"])
    eta_cpu = float(tuning["eta_cpu"])
    mem = system.get("memory", {})

    ram_total, _, ram_conf = _field(mem.get("ram_total_bytes"))
    if ram_total is None:
        raise ValueError("memory.ram_total_bytes ausente: sem RAM total não há orçamento (A23).")
    ram_prov = _measured(ram_conf)

    bw_cpu, bw_cpu_src, bw_cpu_conf = _field(mem.get("bandwidth_bytes_per_s"))
    bw_cpu_est = bw_cpu is not None and _bw_estimated(bw_cpu_src, bw_cpu_conf)
    if bw_cpu is None:
        warnings.append(_warning(
            "hardware_field_missing", "warn",
            "Banda da RAM não detectada: tokens/s sai indefinido onde a CPU participa.",
            "memory.bandwidth_bytes_per_s"))
    elif bw_cpu_est:
        warnings.append(_warning(
            "bandwidth_estimated", "info",
            "Banda da RAM estimada (fonte: " + bw_cpu_src + ").",
            "memory.bandwidth_bytes_per_s"))

    unified, _, _ = _field(mem.get("unified"))
    if unified is None:
        warnings.append(_warning(
            "hardware_field_missing", "info",
            "Memória unificada não determinada; tratada como pools separados.",
            "memory.unified"))

    if unified is True:
        usable = E.unified_budget(ram_total)
        bw_prov = "estimated" if bw_cpu is None or bw_cpu_est else "measured"
        prov = {
            "total_bytes": ram_prov,
            "reserved_bytes": _worst("calculated", ram_prov),
            "usable_bytes": _worst("calculated", ram_prov),
            "eta": "estimated",
        }
        if bw_cpu is not None:
            prov["bandwidth_bytes_per_s"] = bw_prov
        warnings.append(_warning(
            "unified_memory_single_pool", "info",
            "Memória unificada: um único pool, orçamento de 50% da RAM total.",
            "memory.unified"))
        pool = Pool("unified", None, ram_total, ram_total - usable, usable, bw_cpu, bw_cpu_est,
                    eta_gpu, ram_conf == "low", prov)
        return [pool], warnings

    pools: list[Pool] = []
    gpu_pool = _build_gpu_pool(system, eta_gpu, warnings)
    if gpu_pool is not None:
        pools.append(gpu_pool)

    reserved = E.RAM_OS_RESERVE_BYTES + E.RAM_APP_RESERVE_BYTES
    cpu_prov = {
        "total_bytes": ram_prov,
        "reserved_bytes": "calculated",
        "usable_bytes": _worst("calculated", ram_prov),
        "eta": "estimated",
    }
    if bw_cpu is not None:
        cpu_prov["bandwidth_bytes_per_s"] = "estimated" if bw_cpu_est else "measured"
    pools.append(Pool("cpu", None, ram_total, reserved, E.ram_usable(ram_total), bw_cpu,
                      bw_cpu_est, eta_cpu, ram_conf == "low", cpu_prov))
    return pools, warnings


def _build_gpu_pool(system: dict, eta_gpu: float, warnings: list[dict[str, Any]]) -> Pool | None:
    gpus, _, _ = _field(system.get("gpus"))
    if gpus is None:
        warnings.append(_warning(
            "hardware_field_missing", "warn",
            "Detecção de GPU falhou; plano considera só a CPU.", "gpus"))
        return None
    if not gpus:
        return None

    with_vram = [g for g in gpus if _field(g.get("vram_total_bytes"))[0] is not None]
    if not with_vram:
        warnings.append(_warning(
            "hardware_field_missing", "warn",
            "VRAM total da GPU não detectada; GPU fora do plano.",
            f"gpus.value[{min(g['index'] for g in gpus)}].vram_total_bytes"))
        return None
    chosen = sorted(with_vram, key=lambda g: (-_field(g["vram_total_bytes"])[0], g["index"]))[0]
    idx = chosen["index"]
    if len(gpus) > 1:
        warnings.append(_warning(
            "multi_gpu_ignored", "info",
            f"Mais de uma GPU detectada; o plano usa só a GPU de índice {idx} (maior VRAM).",
            "gpus"))

    vram_total, _, vt_conf = _field(chosen["vram_total_bytes"])
    vram_free, _, vf_conf = _field(chosen.get("vram_free_bytes"))
    if vram_free is None:
        warnings.append(_warning(
            "hardware_field_missing", "warn",
            "VRAM livre da GPU não detectada; sem ela a reserva do display é desconhecida e a GPU fica fora do plano.",
            f"gpus.value[{idx}].vram_free_bytes"))
        return None

    display = E.vram_display(vram_total, vram_free)
    reserved = display + E.GPU_RESERVE_BYTES
    usable = E.vram_usable(vram_total, vram_free)
    if vram_total - reserved < 0:
        warnings.append(_warning(
            "vram_usable_clamped", "warn",
            "VRAM utilizável negativa pela fórmula (display + reserva excedem o total); truncada em 0.",
            f"gpus.value[{idx}].vram_free_bytes"))

    bw, bw_src, bw_conf = _field(chosen.get("bandwidth_bytes_per_s"))
    bw_est = bw is not None and _bw_estimated(bw_src, bw_conf)
    if bw is None:
        warnings.append(_warning(
            "hardware_field_missing", "info",
            "Banda da GPU não detectada: tokens/s sai indefinido onde a GPU participa.",
            f"gpus.value[{idx}].bandwidth_bytes_per_s"))
    elif bw_est:
        warnings.append(_warning(
            "bandwidth_estimated", "info",
            "Banda da GPU estimada (fonte: " + bw_src + ").",
            f"gpus.value[{idx}].bandwidth_bytes_per_s"))

    vt_prov, vf_prov = _measured(vt_conf), _measured(vf_conf)
    prov = {
        "total_bytes": vt_prov,
        "reserved_bytes": _worst("calculated", vt_prov, vf_prov),
        "usable_bytes": _worst("calculated", vt_prov, vf_prov),
        "eta": "estimated",
    }
    if bw is not None:
        prov["bandwidth_bytes_per_s"] = "estimated" if bw_est else "measured"
    low = vt_conf == "low" or vf_conf == "low"
    return Pool("gpu", idx, vram_total, reserved, usable, bw, bw_est, eta_gpu, low, prov)


def _system_warnings(system: dict) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    on_batt, _, _ = _field(system.get("power", {}).get("on_battery"))
    if on_batt is True:
        warnings.append(_warning(
            "on_battery", "warn",
            "Máquina na bateria: espere throttling e tokens/s abaixo da estimativa.",
            "power.on_battery"))
    disks, _, _ = _field(system.get("disks"))
    for i, d in enumerate(disks or []):
        kind, _, _ = _field(d.get("kind"))
        if kind == "hdd":
            warnings.append(_warning(
                "hdd_storage", "warn",
                f"Disco rotativo em {d.get('mount')}: carregar o modelo será lento.",
                f"disks.value[{i}].kind"))
            break
    cores_perf, _, _ = _field(system.get("cpu", {}).get("cores_performance"))
    if cores_perf is None:
        warnings.append(_warning(
            "hardware_field_missing", "info",
            "Núcleos de performance não detectados: n_threads fica indefinido.",
            "cpu.cores_performance"))
    return warnings


# --------------------------------------------------------------------------
# Cálculo por modelo


def _model_ref(m: dict) -> dict[str, Any]:
    return {k: m.get(k) for k in (
        "id", "source_url", "license", "license_source", "commercial_ok", "tasks",
        "params_total", "params_active", "n_layers")}


def _quant_ref(q: dict) -> dict[str, Any]:
    return {"format": q["format"], "name": q["name"], "bpw": q.get("bpw"), "file_bytes": q.get("file_bytes")}


def _usage(pool: Pool, weights: int, kv: int, buf: int) -> dict[str, Any]:
    total = weights + kv + buf
    return {
        "pool": pool.id,
        "weights_bytes": weights,
        "kv_bytes": kv,
        "buffer_bytes": buf,
        "total_bytes": total,
        "budget_bytes": pool.usable,
        "tier": E.tier(total, pool.usable),
    }


def _worst_tier(usages: tuple[dict[str, Any], ...]) -> str:
    return max((u["tier"] for u in usages), key=E.TIERS.index)


def _missing(usages: tuple[dict[str, Any], ...]) -> int:
    return max(u["total_bytes"] - E.limit_cap(u["budget_bytes"]) for u in usages)


def _attempts(
    fmt: str,
    weights: int,
    kv: int,
    buf: int,
    n_layers: int,
    layer_bytes_read: int,
    pools: dict[str, Pool],
) -> list[Attempt]:
    """Todas as tentativas calculáveis de um (modelo, quant), em ordem de PLACEMENTS."""
    out: list[Attempt] = []
    gpu = pools.get("gpu") or pools.get("unified")
    cpu = pools.get("cpu")
    bytes_camada = weights // n_layers

    def tps(n_gpu: int) -> float | None:
        bw_gpu = gpu.bw if gpu is not None else None
        eta_gpu = gpu.eta if gpu is not None else 1.0
        bw_cpu = cpu.bw if cpu is not None else None
        eta_cpu = cpu.eta if cpu is not None else 1.0
        return E.tokens_per_s(layer_bytes_read, n_layers, n_gpu, bw_gpu, bw_cpu, eta_gpu, eta_cpu)

    def make(placement: str, n_gpu: int, usages: tuple[dict[str, Any], ...], used: tuple[Pool, ...]) -> Attempt:
        return Attempt(placement, n_gpu, usages, weights + kv + buf, _worst_tier(usages),
                       tps(n_gpu), used, _missing(usages))

    # gpu_full (ou o pool unificado inteiro)
    if gpu is not None:
        out.append(make("gpu_full", n_layers, (_usage(gpu, weights, kv, buf),), (gpu,)))

    if fmt != "GGUF":
        return out  # EXL2/AWQ/GPTQ: só residência total na GPU

    # hybrid: KV inteiro e buffer no lado GPU (m1_interface.md)
    if gpu is not None and gpu.id == "gpu" and cpu is not None:
        n_gpu = E.n_gpu_layers(weights, n_layers, gpu.usable, kv, buf)
        if 1 <= n_gpu < n_layers:
            w_gpu = bytes_camada * n_gpu
            usages = (_usage(gpu, w_gpu, kv, buf), _usage(cpu, weights - w_gpu, 0, 0))
            out.append(make("hybrid", n_gpu, usages, (gpu, cpu)))

    if cpu is not None:
        out.append(make("cpu_only", 0, (_usage(cpu, weights, kv, buf),), (cpu,)))
    return out


def _choose(viable: list[Attempt]) -> Attempt:
    """Maior tokens/s entre os placements viáveis.

    Se algum tokens/s é desconhecido (banda null), a comparação não é
    definida; cai para a ordem gpu_full > hybrid > cpu_only em vez de
    preferir a CPU só porque a banda da GPU não foi medida.
    """
    if any(a.tps is None for a in viable):
        return viable[0]  # viable já está em ordem de PLACEMENTS
    return sorted(viable, key=lambda a: (-a.tps, E.TIERS.index(a.tier), PLACEMENTS.index(a.placement)))[0]


def _round_tps(x: float | None) -> float | None:
    if x is None:
        return None
    r = round(x, 2)
    return r if r > 0 else x


def _quant_order(q: dict) -> tuple:
    bpw = q.get("bpw")
    return (bpw is None, -(bpw or 0.0), q["format"], q["name"])


def _evaluate_model(m: dict, intent: Intent, pools: dict[str, Pool], tuning_default: bool,
                    n_threads: int | None) -> tuple[dict | None, dict | None]:
    """Devolve (candidato, None) ou (None, rejeitado)."""
    ref = _model_ref(m)

    def reject(code: str, message: str, closest: Attempt | None = None, cq: dict | None = None,
               skipped: list | None = None) -> tuple[None, dict]:
        return None, {
            "model": ref,
            "reason_code": code,
            "message": message,
            "closest": None if closest is None else {
                "quant": _quant_ref(cq),
                "placement": closest.placement,
                "total_bytes": closest.total,
                "by_pool": list(closest.by_pool),
            },
            "missing_bytes": None if closest is None else closest.missing,
            "skipped_quants": skipped or [],
        }

    # Embeddings (DECISIONS.md): ctx acima do máximo não rejeita; o texto é
    # truncado/fatiado em ctx_max. Só quando a única tarefa avaliada é embeddings.
    evaluated = [t for t in (m.get("tasks") or []) if t in intent.tasks]
    embed_only = evaluated == ["embeddings"]
    ctx = min(intent.ctx, m["ctx_max"]) if embed_only else intent.ctx
    if ctx > m["ctx_max"]:
        return reject("ctx_exceeds_max",
                      f"Contexto pedido ({intent.ctx} tokens) acima do máximo do modelo ({m['ctx_max']} tokens).")
    if m["kv_model"] == "mla":
        return reject("kv_model_unsupported",
                      "Atenção latente (MLA): a fórmula de KV superestima; modelo não avaliado nesta versão.")

    params_total: int = m["params_total"]
    params_active: int = m["params_active"]
    n_layers: int = m["n_layers"]
    floor_bpw = E.min_bpw(params_total)
    kv = E.kv_bytes(n_layers, m["n_kv_heads"], m["head_dim"], ctx, intent.kv_quant, intent.batch)
    buf = E.buffer_bytes(ctx)

    skipped: list[dict[str, Any]] = []
    closest: tuple[Attempt, dict] | None = None
    for q in sorted(m["quants"], key=_quant_order):
        sq = {"format": q["format"], "name": q["name"]}
        bpw = q.get("bpw")
        if bpw is None:
            skipped.append({**sq, "reason_code": "bpw_unknown"})
            continue
        # Piso de qualidade: filtro duro ANTES de qualquer cálculo.
        if bpw < floor_bpw:
            skipped.append({**sq, "reason_code": "below_quality_floor"})
            continue
        # EXL2/AWQ/GPTQ exigem GPU dedicada (pool "gpu"). Em memória unificada
        # (Apple Silicon) não contam como residentes na GPU: rejeitados (DECISIONS.md).
        if q["format"] in FULL_GPU_FORMATS and "gpu" not in pools:
            skipped.append({**sq, "reason_code": "no_gpu_for_format"})
            continue

        weights, fallback = E.weights_bytes(params_total, bpw, q.get("file_bytes"))
        # MoE: memória usa params_total; leitura por token usa a fração ativa (A20).
        layer_read = E.bytes_per_token(weights, params_active, params_total) // n_layers
        attempts = _attempts(q["format"], weights, kv, buf, n_layers, layer_read, pools)
        viable = [a for a in attempts if a.tier != "inviavel"]
        if viable:
            cand = _candidate(ref, m, q, _choose(viable), weights, kv, buf, fallback, skipped,
                              intent, tuning_default, n_threads)
            if embed_only:
                cand["ctx_effective"] = ctx
            return cand, None
        for a in attempts:  # empate: fica a primeira (maior bpw, ordem de PLACEMENTS)
            if closest is None or a.missing < closest[0].missing:
                closest = (a, q)
        reason = "format_requires_full_gpu" if q["format"] in FULL_GPU_FORMATS else "does_not_fit"
        skipped.append({**sq, "reason_code": reason})

    if closest is not None:
        a, q = closest
        if q["format"] in FULL_GPU_FORMATS:
            return reject("format_requires_full_gpu",
                          f"{q['format']} exige residência total na VRAM e não cabe na GPU.",
                          a, q, skipped)
        return reject("does_not_fit",
                      "Nenhuma quant dentro do piso de qualidade cabe na memória disponível.",
                      a, q, skipped)
    codes = [s["reason_code"] for s in skipped]
    if codes and all(c == "below_quality_floor" for c in codes):
        return reject("below_quality_floor",
                      f"Todas as quants estão abaixo do piso de qualidade ({floor_bpw} bpw) para o tamanho do modelo.",
                      skipped=skipped)
    if codes and all(c in ("no_gpu_for_format", "below_quality_floor") for c in codes) \
            and "no_gpu_for_format" in codes:
        return reject("no_gpu_for_format",
                      "As quants disponíveis exigem GPU dedicada (CUDA/ROCm) e não há uma utilizável nesta plataforma.", skipped=skipped)
    return reject("no_usable_quant", "Nenhuma quant utilizável (bpw desconhecido ou ausente).",
                  skipped=skipped)


def _candidate(ref: dict, m: dict, q: dict, a: Attempt, weights: int, kv: int, buf: int,
               fallback: bool, skipped: list, intent: Intent, tuning_default: bool,
               n_threads: int | None) -> dict[str, Any]:
    reasons: set[str] = {"ttft_unspecified"}
    if fallback:
        reasons.add("weights_fallback")
        if q.get("bpw_source") == "table":
            reasons.add("bpw_from_table")
    if tuning_default:
        reasons.add("eta_default")
    if a.tps is None:
        reasons.add("bandwidth_unknown")
    # Banda só conta se o pool tem camadas neste placement.
    bw_pools = [p for p in a.pools_used
                if (p.id == "cpu" and a.n_gpu < m["n_layers"]) or (p.id != "cpu" and a.n_gpu > 0)]
    if any(p.bw_estimated for p in bw_pools):
        reasons.add("bandwidth_estimated")
    low_hw = any(p.low_confidence for p in a.pools_used)
    if low_hw:
        reasons.add("hardware_low_confidence")

    if fallback or low_hw:
        confidence = "low"
    elif reasons & {"eta_default", "bandwidth_estimated", "bandwidth_unknown"}:
        confidence = "medium"
    else:
        confidence = "high"

    w_prov = "estimated" if fallback else "catalog"
    hw_prov = "estimated" if low_hw else "calculated"
    blocked = intent.commercial_use and m.get("commercial_ok") is not True
    blocked_reason = None
    if blocked:
        blocked_reason = "license_unknown" if m.get("commercial_ok") is None else "non_commercial"

    return {
        "model": ref,
        "quant": _quant_ref(q),
        "placement": a.placement,
        "n_gpu_layers": a.n_gpu,
        "n_threads": n_threads,
        "weights_bytes": weights,
        "kv_bytes": kv,
        "buffer_bytes": buf,
        "total_bytes": a.total,
        "by_pool": list(a.by_pool),
        "tier": a.tier,
        "tokens_per_s_est": _round_tps(a.tps),
        "ttft_s_est": None,  # A17: sem fórmula de prefill até haver medição
        "confidence": confidence,
        "confidence_reasons": sorted(reasons),
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "primary_for": [],
        "skipped_quants": skipped,
        "provenance": {
            "weights_bytes": w_prov,
            "kv_bytes": "calculated",
            "buffer_bytes": "calculated",
            "total_bytes": _worst(w_prov, "calculated"),
            "n_gpu_layers": hw_prov if a.placement == "hybrid" else "calculated",
            "tokens_per_s_est": "estimated",
        },
    }


def _quality_alternative(chosen: dict, pool: list[dict]) -> dict | None:
    """Regra 3 do adendo de m1_interface.md (PROMPT.md M1: abaixo de 7e9, modelo
    MENOR em bpw ALTO vence o maior em bpw destrutivo). Devolve o substituto ou None."""
    if not (chosen["model"]["params_total"] < QUALITY_PREF_MAX_PARAMS
            and chosen["quant"]["bpw"] < QUALITY_PREF_LOW_BPW):
        return None
    chosen_tier = E.TIERS.index(chosen["tier"])
    for c in pool:  # pool já está na ordem final
        if (c is not chosen
                and c["model"]["params_total"] < chosen["model"]["params_total"]
                and c["quant"]["bpw"] >= QUALITY_PREF_HIGH_BPW
                and E.TIERS.index(c["tier"]) <= chosen_tier):
            return c
    return None


def _assign_primary(candidates: list[dict], tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    """Preenche primary_for (regras 2 e 3 do adendo). Não reordena candidatos."""
    warnings: list[dict[str, Any]] = []
    for task in tasks:
        pool = [c for c in candidates if not c["blocked"] and task in (c["model"]["tasks"] or [])]
        if not pool:
            continue
        chosen = pool[0]
        alt = _quality_alternative(chosen, pool)
        if alt is not None:
            i = candidates.index(chosen)
            warnings.append(_warning(
                "quality_preference", "info",
                f"Principal de {task}: {alt['model']['id']} ({alt['quant']['name']}) no lugar de "
                f"{chosen['model']['id']} ({chosen['quant']['name']}): abaixo de 7B, modelo menor em "
                f"bpw alto é preferível a modelo maior em bpw destrutivo.",
                f"candidates[{i}]"))
            chosen = alt
        chosen["primary_for"].append(task)
    return warnings


def _sort_key(c: dict) -> tuple:
    tps = c["tokens_per_s_est"]
    return (E.TIERS.index(c["tier"]), tps is None, -(tps or 0.0), c["model"]["id"],
            c["quant"]["format"], c["quant"]["name"])


def _catalog_sha256(catalog: dict) -> str:
    """sha256 da serialização canônica. A CLI sobrescreve com o hash dos bytes do arquivo."""
    canon = json.dumps(catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------


def solve(system: dict, catalog: dict, intent: Intent, tuning: dict) -> dict:
    """Plano de viabilidade (feasibility.json). Sem arquivo, sem rede, sem relógio."""
    if intent.kv_quant not in E.KV_ELEM_BITS:
        raise ValueError(f"kv_quant inválido: {intent.kv_quant!r}")
    if intent.ctx < 1 or intent.batch < 1:
        raise ValueError("ctx e batch precisam ser >= 1")
    tasks = tuple(t for t in TASKS if t in intent.tasks)
    if not tasks:
        raise ValueError("intent.tasks vazio ou sem tarefa conhecida")

    pool_list, warnings = _build_pools(system, tuning)
    warnings = [_warning("swap_not_a_fix", "info", SWAP_MESSAGE)] + warnings + _system_warnings(system)
    pools = {p.id: p for p in pool_list}
    tuning_default = tuning.get("source", "default") == "default"
    n_threads, _, _ = _field(system.get("cpu", {}).get("cores_performance"))

    models = list(catalog.get("models", []))
    candidates: list[dict] = []
    rejected: list[dict] = []
    n_incomplete = n_mismatch = 0
    for m in sorted(models, key=lambda x: x["id"]):
        if not m.get("complete"):
            n_incomplete += 1
            continue
        if not any(t in tasks for t in (m.get("tasks") or [])):
            n_mismatch += 1
            continue
        cand, rej = _evaluate_model(m, intent, pools, tuning_default, n_threads)
        if cand is not None:
            candidates.append(cand)
        else:
            rejected.append(rej)

    candidates.sort(key=_sort_key)
    warnings += _assign_primary(candidates, tasks)

    verified = sorted(m["verified_at"] for m in models if m.get("verified_at"))
    out: dict[str, Any] = {
        "schema_version": 1,
        "kind": "feasibility",
        "scout_version": __version__,
        "inputs": {
            "intent": {
                "tasks": list(tasks),
                "ctx": intent.ctx,
                "commercial_use": intent.commercial_use,
                "batch": intent.batch,
                "kv_quant": intent.kv_quant,
            },
            "system": system,
            "catalog": {
                "sha256": _catalog_sha256(catalog),
                "model_count": len(models),
                "oldest_verified_at": verified[0] if verified else None,
            },
            "tuning": {
                "eta_gpu": tuning["eta_gpu"],
                "eta_cpu": tuning["eta_cpu"],
                "source": tuning.get("source", "default"),
            },
        },
        "pools": [p.to_json() for p in pool_list],
        "candidates": candidates,
        "rejected": rejected,
        "excluded": {"incomplete": n_incomplete, "task_mismatch": n_mismatch},
        "warnings": warnings,
    }
    if (system.get("fixture") or {}).get("synthetic") or (catalog.get("fixture") or {}).get("synthetic"):
        out["fixture"] = {"synthetic": True, "description": "Plano derivado de entradas sintéticas (fixture)."}
    return out
