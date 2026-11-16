"""QA da Tarefa 5, a partir do enunciado (schemas/calibration_store_spec.md), sem olhar a implementação.

Duas áreas que a suíte do `nucleo` não cobria:
  * escolha da GPU do pool gpu com várias GPUs, e a identidade gravada bater com a GPU que o solver usa;
  * os três textos de `calibration_memory_unverified`, vistos pelo plan (não pela função privada).

A identidade da calibração é montada À MÃO a partir do system.json (cópia do `value` de cada campo, §2),
nunca por `machine_identity`: assim um erro na função de identidade não se esconde atrás de si mesmo.
"""
from __future__ import annotations

import copy

import pytest

from conftest import assert_valid
from scout import calibration as C
from scout.calibration import CalibrationInput
from scout.solver import Intent, solve

AC = {"on_ac": True, "profile": "performance", "not_applicable": []}
ENG = {"ollama": "9.9.9"}
GB = 1_000_000_000


def node(v):
    return {"value": v, "source": "fixture", "confidence": "low" if v is None else "high"}


def gpu(index, name, total, free="auto", backend="cuda"):
    free = int(total * 0.9) if free == "auto" and total is not None else (None if free == "auto" else free)
    return {"index": index, "vendor": node("nvidia"), "name": node(name), "vram_total_bytes": node(total),
            "vram_free_bytes": node(free), "bandwidth_bytes_per_s": node(300 * GB),
            "driver_version": node("0"), "compute_capability": node("8.6"), "backend": node(backend),
            "bf16": node(None), "fp8": node(None), "flash_attention": node(None)}


def with_gpus(systems, gpus):
    s = copy.deepcopy(systems["rtx3060"])
    s["gpus"] = {"value": gpus, "source": "fixture", "confidence": "high"}
    return s


def v(n):
    return n["value"]


def ident(system, chosen=None):
    """Identidade de §2 copiada à mão, com kernel e camada 2 (cal_doc os separa). `chosen`: a GPU do pool gpu."""
    return {
        "platform": system["platform"],
        "os.arch": v(system["os"]["arch"]), "os.kernel": v(system["os"]["kernel"]),
        "cpu.model": v(system["cpu"]["model"]), "cpu.cores_physical": v(system["cpu"]["cores_physical"]),
        "memory.ram_total_bytes": v(system["memory"]["ram_total_bytes"]),
        "gpu.name": v(chosen["name"]) if chosen else None,
        "gpu.backend": v(chosen["backend"]) if chosen else None,
        "gpu.vram_total_bytes": v(chosen["vram_total_bytes"]) if chosen else None,
        "memory.speed_mts": v(system["memory"]["speed_mts"]),
        "memory.channels": v(system["memory"]["channels"]),
    }


def entry(power=AC, bw=20 * GB):
    return {"bandwidth_bytes_per_s": bw, "overhead_s_per_token": 0.0, "form": "1p", "stable": True,
            "power": copy.deepcopy(power), "tool": "ollama", "tool_version": "9.9.9", "date": "2000-01-01",
            "models": [{"id": "test/dense-7b", "architecture": "testarch", "bytes": 1}],
            "prefill_tokens_per_s": None}


MEASURED_WITH = ("os.kernel", "memory.speed_mts", "memory.channels")  # rev. 4: por entrada, não na máquina


def cal_doc(ident_all, pools):
    """`ident_all` traz os 11 campos copiados à mão; a camada 1 vai para `machine` e kernel e
    camada 2 para o `measured_with` de cada entrada (calibration_store_spec.md §2 rev. 4)."""
    machine = {k: v for k, v in ident_all.items() if k not in MEASURED_WITH}
    mw = {k: ident_all[k] for k in MEASURED_WITH}
    pools = {pid: [dict(e, measured_with=dict(mw)) for e in es] for pid, es in pools.items()}
    return {"schema_version": 1, "kind": "calibration", "scout_version": "0.0.0",
            "calibration_schema_version": C.CALIBRATION_SCHEMA_VERSION, "machine": machine, "pools": pools}


def plan(system, catalog, tuning, doc=None):
    cal = CalibrationInput("env", doc) if doc is not None else CalibrationInput("disabled", None)
    out = solve(system, catalog, Intent(tasks=("chat",)), tuning, AC if doc is not None else None, cal,
                ENG if doc is not None else None)
    assert_valid("feasibility.schema.json", out)
    return out


def gpu_pool(out):
    return next((p for p in out["pools"] if p["id"] == "gpu"), None)


# ------------------------------------------------------------------ várias GPUs


def test_pick_gpu_largest_vram_wins_even_when_not_index_zero(systems, catalog, tuning):
    # 8 GB (idx 0), 24 GB (idx 1), 12 GB (idx 2): o pool gpu é o de maior VRAM total, a idx 1.
    gpus = [gpu(0, "G-8", 8 * GB), gpu(1, "G-24", 24 * GB), gpu(2, "G-12", 12 * GB)]
    s = with_gpus(systems, gpus)
    chosen = C.pool_gpu(s)
    assert chosen["index"] == 1 and v(chosen["name"]) == "G-24"
    pool = gpu_pool(plan(s, catalog, tuning))
    assert pool["gpu_index"] == 1 and pool["total_bytes"] == 24 * GB
    assert C.machine_identity(s)["gpu.name"] == "G-24"
    assert C.machine_identity(s)["gpu.vram_total_bytes"] == 24 * GB


def test_pick_gpu_tie_breaks_on_lowest_index(systems, catalog, tuning):
    # Empate de VRAM (16 GB): vence o menor índice, mesmo que apareça depois na lista.
    gpus = [gpu(3, "G-c", 16 * GB), gpu(1, "G-a", 16 * GB), gpu(2, "G-b", 16 * GB)]
    s = with_gpus(systems, gpus)
    assert v(C.pool_gpu(s)["name"]) == "G-a"
    assert gpu_pool(plan(s, catalog, tuning))["gpu_index"] == 1


def test_pick_gpu_ignores_gpu_without_total_vram(systems, catalog, tuning):
    # A idx 0 não tem VRAM total lida: não pode ser escolhida; a idx 1 vira o pool.
    s = with_gpus(systems, [gpu(0, "G-null", None, free=None), gpu(1, "G-8", 8 * GB)])
    assert v(C.pool_gpu(s)["name"]) == "G-8"
    assert gpu_pool(plan(s, catalog, tuning))["gpu_index"] == 1


def test_pick_gpu_function_on_list():
    gpus = [gpu(0, "a", 4 * GB), gpu(1, "b", 4 * GB), gpu(2, "c", 6 * GB)]
    assert C.pick_gpu(gpus)["index"] == 2
    assert C.pick_gpu(gpus[:2])["index"] == 0
    assert C.pick_gpu(list(reversed(gpus[:2])))["index"] == 0


def test_no_gpu_pool_cases_agree_between_pool_gpu_and_solver(systems, catalog, tuning):
    unified = copy.deepcopy(systems["apple24"])
    none_gpus = with_gpus(systems, [])
    all_null = with_gpus(systems, [gpu(0, "x", None, free=None)])
    for s in (unified, none_gpus, all_null):
        assert C.pool_gpu(s) is None
        assert gpu_pool(plan(s, catalog, tuning)) is None
        assert C.machine_identity(s)["gpu.name"] is None


def test_chosen_gpu_without_free_vram_means_no_gpu_pool_in_both(systems, catalog, tuning):
    # A maior VRAM (idx 1) não tem vram_free: o solver não monta pool gpu, e a identidade não inventa GPU.
    s = with_gpus(systems, [gpu(0, "G-8", 8 * GB), gpu(1, "G-24", 24 * GB, free=None)])
    assert gpu_pool(plan(s, catalog, tuning)) is None
    assert C.pool_gpu(s) is None
    assert C.machine_identity(s)["gpu.name"] is None


@pytest.mark.parametrize("gpus", [
    [gpu(0, "A", 8 * GB, backend="vulkan"), gpu(1, "B", 24 * GB, backend="rocm")],
    [gpu(0, "A", 24 * GB, backend="cuda"), gpu(1, "B", 24 * GB, backend="rocm")],
    [gpu(2, "A", 12 * GB), gpu(0, "B", 12 * GB, backend="rocm"), gpu(1, "C", 6 * GB)],
])
def test_pool_gpu_identity_is_the_gpu_the_solver_uses(systems, catalog, tuning, gpus):
    s = with_gpus(systems, gpus)
    pool = gpu_pool(plan(s, catalog, tuning))
    used = next(g for g in gpus if g["index"] == pool["gpu_index"])  # a GPU que o solver usa
    m = C.machine_identity(s)
    assert (m["gpu.name"], m["gpu.backend"], m["gpu.vram_total_bytes"]) == (
        v(used["name"]), v(used["backend"]), v(used["vram_total_bytes"]))
    assert pool["total_bytes"] == v(used["vram_total_bytes"])


def _gpu_match(out):
    return {m["field"]: m for m in out["inputs"]["calibration"]["match"] if m["field"].startswith("gpu.")}


def test_calibration_with_identity_of_the_chosen_gpu_applies_and_survives_reordering(systems, catalog, tuning):
    gpus = [gpu(0, "G-8", 8 * GB), gpu(1, "G-24", 24 * GB, backend="rocm"), gpu(2, "G-12", 12 * GB)]
    s = with_gpus(systems, gpus)
    d = cal_doc(ident(s, gpus[1]), {"gpu": [entry()]})  # à mão: a de 24 GB
    for order in (gpus, list(reversed(gpus))):
        s2 = with_gpus(systems, order)
        out = plan(s2, catalog, tuning, d)
        assert out["inputs"]["calibration"]["pools"]["gpu"]["status"] == "applied"
        lines = _gpu_match(out)
        assert {k: m["result"] for k, m in lines.items()} == {
            "gpu.name": "equal", "gpu.backend": "equal", "gpu.vram_total_bytes": "equal"}


def test_calibration_with_identity_of_another_gpu_is_discarded(systems, catalog, tuning):
    gpus = [gpu(0, "G-8", 8 * GB), gpu(1, "G-24", 24 * GB)]
    s = with_gpus(systems, gpus)
    d = cal_doc(ident(s, gpus[0]), {"gpu": [entry()]})  # identidade da GPU errada (a de índice 0)
    out = plan(s, catalog, tuning, d)
    pool = out["inputs"]["calibration"]["pools"]["gpu"]
    assert pool["status"] == "discarded" and "calibration_machine_mismatch" in pool["reasons"]
    lines = _gpu_match(out)
    assert lines["gpu.name"]["result"] == "different" and lines["gpu.vram_total_bytes"]["result"] == "different"
    assert lines["gpu.name"]["stored"] == "G-8" and lines["gpu.name"]["current"] == "G-24"


# ------------------------------------------------------------------ calibration_memory_unverified


def _mem(system, mts, ch):
    s = copy.deepcopy(system)
    s["memory"]["speed_mts"] = node(mts)
    s["memory"]["channels"] = node(ch)
    return s


def _unverified(out):
    return [w for w in out["warnings"] if w["code"] == "calibration_memory_unverified"]


def _cpu_pool_plan(system, machine, catalog, tuning):
    out = plan(system, catalog, tuning, cal_doc(machine, {"cpu": [entry()]}))
    c = next(c for c in out["candidates"] if c["model"]["id"] == "test/dense-7b")
    assert out["inputs"]["calibration"]["pools"]["cpu"]["status"] == "applied"
    return out, c


def test_text_missing_in_current_system_linux(systems, catalog, tuning):
    s = _mem(systems["ddr5_64"], None, None)
    stored = ident(systems["ddr5_64"])  # gravada com MT/s e canais
    out, c = _cpu_pool_plan(s, stored, catalog, tuning)
    (w,) = _unverified(out)  # no máximo um
    assert w["severity"] == "info"
    assert "não foram lidos neste system.json" in w["message"]
    assert "sudo scout collect" in w["message"] and "--dmidecode" in w["message"]
    assert "confiança alta" in w["message"]
    assert c["confidence"] == "medium" and "calibration_memory_unverified" in c["confidence_reasons"]


def test_text_missing_in_stored_calibration_linux(systems, catalog, tuning):
    s = systems["ddr5_64"]
    stored = ident(s)
    stored["memory.speed_mts"] = None
    stored["memory.channels"] = None
    out, c = _cpu_pool_plan(s, stored, catalog, tuning)
    (w,) = _unverified(out)
    assert w["severity"] == "info"
    assert "A calibração foi gravada sem MT/s e canais da RAM" in w["message"]
    assert "scout bench --run" in w["message"] and "dmidecode" in w["message"]
    assert "não foram lidos neste system.json" not in w["message"]
    assert c["confidence"] == "medium"


@pytest.mark.parametrize("platform", ["darwin", "windows"])
@pytest.mark.parametrize("missing_side", ["system", "calibration"])
def test_text_other_platform_does_not_promise_a_path_that_does_not_exist(systems, catalog, tuning, platform,
                                                                         missing_side):
    # Intel Mac / Windows: o atributo existe, falta a leitura. Mesmo texto nos dois lados em falta.
    base = copy.deepcopy(systems["ddr5_64"])
    base["platform"] = platform
    if missing_side == "system":
        s, stored = _mem(base, None, None), ident(base)
    else:
        s, stored = base, ident(base)
        stored["memory.speed_mts"] = stored["memory.channels"] = None
    out, c = _cpu_pool_plan(s, stored, catalog, tuning)
    (w,) = _unverified(out)
    assert w["severity"] == "info"
    assert "ainda não lê MT/s e canais da RAM nesta plataforma" in w["message"]
    for forbidden in ("sudo", "dmidecode", "scout bench --run", "scout collect"):
        assert forbidden not in w["message"]
    assert c["confidence"] == "medium" and "calibration_memory_unverified" in c["confidence_reasons"]


def test_one_known_one_unknown_field_still_yields_a_single_warning(systems, catalog, tuning):
    # só speed_mts desconhecido, canais conhecidos: continua um único aviso
    s = _mem(systems["ddr5_64"], None, 2)
    out, _ = _cpu_pool_plan(s, ident(systems["ddr5_64"]), catalog, tuning)
    assert len(_unverified(out)) == 1


def test_known_equal_on_both_sides_has_no_warning_and_reaches_high(systems, catalog, tuning):
    s = systems["ddr5_64"]
    out, c = _cpu_pool_plan(s, ident(s), catalog, tuning)
    assert _unverified(out) == []
    assert c["confidence"] == "high" and "calibration_memory_unverified" not in c["confidence_reasons"]


def test_unified_pool_with_unknown_memory_is_not_applicable_so_no_cap(systems, catalog, tuning):
    # Apple Silicon: MT/s x canais não existe à parte. Nem aviso nem teto, e `high` sem exceção escrita à mão.
    s = copy.deepcopy(systems["apple24"])
    s["memory"]["speed_mts"] = node(None)
    s["memory"]["channels"] = node(None)
    out = plan(s, catalog, tuning, cal_doc(ident(s), {"unified": [entry()]}))
    assert out["inputs"]["calibration"]["pools"]["unified"]["status"] == "applied"
    assert _unverified(out) == []
    c = next(c for c in out["candidates"] if c["model"]["id"] == "test/dense-7b")
    assert c["confidence"] == "high"
