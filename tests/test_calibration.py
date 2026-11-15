"""Calibração local no plan (schemas/calibration_store_spec.md §2–§5).

Toda calibração aqui é sintética, montada em memória no formato de
schemas/examples/calibration_example.json, com a identidade da fixture de
sistema usada. Nenhum arquivo real de dados é lido.
"""
from __future__ import annotations

import copy

import pytest

from conftest import schema_validator
from scout import calibration as C
from scout import cli
from scout import estimate as E
from scout.calibration import CalibrationInput
from scout.solver import Intent, solve

AC = {"on_ac": True, "profile": "performance", "not_applicable": []}
BATT = {"on_ac": False, "profile": "balanced", "not_applicable": []}
NA = {"on_ac": None, "profile": None, "not_applicable": ["on_ac", "profile"]}
ENG = {"ollama": "9.9.9"}


def entry(models=("test/dense-7b",), stable=True, power=AC, bw=20_000_000_000, ovh=0.002, prefill=None,
          date="2000-01-01", version="9.9.9", mw=None):
    """mw: measured_with explícito; None = o do system do plan (doc() preenche)."""
    return {**({"measured_with": dict(mw)} if mw is not None else {}),"bandwidth_bytes_per_s": bw, "overhead_s_per_token": ovh, "form": "2p" if ovh else "1p",
            "stable": stable, "power": copy.deepcopy(power), "tool": "ollama", "tool_version": version,
            "date": date, "models": [{"id": m, "architecture": "testarch", "bytes": 1} for m in models],
            "prefill_tokens_per_s": prefill}


def doc(system: dict, pools: dict, **fields) -> dict:
    """Campos de camada 1 vão para `machine`; kernel e camada 2 para o
    measured_with de cada entrada que não trouxe o seu."""
    m = C.machine_identity(system)
    m.update({k: v for k, v in fields.items() if k in C.MACHINE_FIELDS})
    mw = dict(C.measured_with(system), **{k: v for k, v in fields.items() if k in C.MEASURED_WITH_FIELDS})
    assert set(fields) <= set(C.MACHINE_FIELDS) | set(C.MEASURED_WITH_FIELDS)
    pools = {pid: [e if "measured_with" in e else dict(e, measured_with=dict(mw)) for e in es]
             for pid, es in pools.items()}
    return {"schema_version": 1, "kind": "calibration", "scout_version": "0.0.0",
            "calibration_schema_version": C.CALIBRATION_SCHEMA_VERSION, "machine": m, "pools": pools}


def _c(out: dict, mid: str) -> dict:
    return next(c for c in out["candidates"] if c["model"]["id"] == mid)


def _model(catalog: dict, mid: str) -> dict:
    return next(m for m in catalog["models"] if m["id"] == mid)


def _valid(out: dict) -> None:
    errs = list(schema_validator("feasibility.schema.json").iter_errors(out))
    assert errs == [], [f"{list(e.path)}: {e.message[:200]}" for e in errs[:3]]


def _codes(out: dict) -> list[str]:
    return [w["code"] for w in out["warnings"]]


def _plan(systems, catalog, tuning, pools=None, power=AC, engines=ENG, sysname="ddr5_64", d=None,
          tasks=("chat",), **intent):
    system = systems[sysname]
    d = d if d is not None else doc(system, pools)
    out = solve(system, catalog, Intent(tasks=tasks, **intent), tuning, power,
                CalibrationInput("env", d), engines)
    _valid(out)
    return out


# ---------------------------------------------------------------- consumo (migrado)


def test_calibrated_cpu_uses_bytes_over_bandwidth_plus_overhead(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": [entry()]})
    c = _c(out, "test/dense-7b")
    assert c["placement"] == "cpu_only"
    m = _model(catalog, "test/dense-7b")
    layer = E.bytes_per_token(c["weights_bytes"], m["params_active"], m["params_total"]) // m["n_layers"]
    expected = 1 / (layer * m["n_layers"] / 20_000_000_000 + 0.002)
    assert c["tokens_per_s_est"] == pytest.approx(expected, abs=0.006)
    assert c["tokens_per_s_basis"] == "calibration"
    assert c["provenance"]["tokens_per_s_est"] == "calculated"
    assert "eta_default" not in c["confidence_reasons"]
    assert c["confidence"] == "high"
    cal = out["inputs"]["calibration"]
    assert cal["status"] == "used" and cal["origin"] == "env"
    assert cal["pools"]["cpu"]["status"] == "applied" and cal["pools"]["cpu"]["power_match"] == "same"
    assert cal["pools"]["cpu"]["calibration"]["bandwidth_bytes_per_s"] == 20_000_000_000
    assert all(ln["effect"] == "none" for ln in cal["match"])
    assert out["inputs"]["power"] == AC
    assert "calibration" not in out["inputs"]["tuning"]


def test_no_calibration_disabled_or_absent_uses_eta(systems, catalog, tuning):
    base = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",)), tuning)
    absent = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",)), tuning, AC,
                   CalibrationInput("default", None), ENG)
    _valid(base)
    _valid(absent)
    assert base["inputs"]["calibration"]["status"] == "disabled"
    assert absent["inputs"]["calibration"]["status"] == "absent"
    assert "power" not in base["inputs"] and "power" not in absent["inputs"]
    assert base["candidates"] == absent["candidates"]
    c = _c(base, "test/dense-7b")
    assert c["confidence"] == "medium" and "eta_default" in c["confidence_reasons"]
    assert c["tokens_per_s_basis"] == "eta"


def test_tuning_with_calibration_key_fails(systems, catalog, tuning):
    with pytest.raises(ValueError, match="calibration"):
        solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",)), dict(tuning, calibration={}))


def test_unstable_calibration_caps_medium(systems, catalog, tuning):
    c = _c(_plan(systems, catalog, tuning, {"cpu": [entry(stable=False)]}), "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_unstable" in c["confidence_reasons"]


def test_other_architecture_caps_medium(systems, catalog, tuning):
    c = _c(_plan(systems, catalog, tuning, {"cpu": [entry(models=("test/other-4b",))]}), "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_other_arch" in c["confidence_reasons"]
    assert c["tokens_per_s_basis"] == "calibration"


def test_hybrid_with_one_calibrated_pool_is_mixed(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": [entry(models=("test/moe-30b-a3b",))]},
                sysname="rtx3060", tasks=("chat", "code"))
    hyb = [c for c in out["candidates"] if c["placement"] == "hybrid"]
    assert hyb, "fixture rtx3060 precisa de um híbrido"
    for c in hyb:
        assert c["tokens_per_s_basis"] == "mixed"
        assert c["provenance"]["tokens_per_s_est"] == "estimated"
        assert "eta_default" in c["confidence_reasons"] and c["confidence"] != "high"


def test_ttft_only_with_prompt_and_own_stable_prefill(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": [entry(prefill=40.0)]}, prompt_tokens=532)
    c = _c(out, "test/dense-7b")
    assert c["ttft_s_est"] == pytest.approx(532 / 40.0, abs=0.006)
    assert "ttft_unspecified" not in c["confidence_reasons"]
    assert _c(_plan(systems, catalog, tuning, {"cpu": [entry(prefill=40.0)]}), "test/dense-7b")["ttft_s_est"] is None
    other = _plan(systems, catalog, tuning, {"cpu": [entry(models=("test/other-4b",), prefill=40.0)]},
                  prompt_tokens=532)
    assert _c(other, "test/dense-7b")["ttft_s_est"] is None


def test_slow_prefill_cites_local_measurement(systems, catalog, tuning):
    ref = {"tokens_per_s": 11.7, "measured_on": "REFERÊNCIA", "tool": "t", "date": "2026-11-15"}
    out = _plan(systems, catalog, dict(tuning, prefill_reference=ref), {"cpu": [entry(prefill=30.0)]})
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "slow_prefill_cpu")
    assert "nesta máquina" in msg and "~30 tokens/s" in msg and "1800 tokens" in msg
    assert "REFERÊNCIA" not in msg


@pytest.mark.parametrize("bad", [{"bandwidth_bytes_per_s": 0}, {"bandwidth_bytes_per_s": 1.5e10},
                                 {"overhead_s_per_token": -1}])
def test_invalid_entry_fails_loudly(systems, catalog, tuning, bad):
    d = doc(systems["cpu8"], {"cpu": [dict(entry(), **bad)]})
    with pytest.raises(ValueError):
        solve(systems["cpu8"], catalog, Intent(tasks=("chat",)), tuning, AC, CalibrationInput("env", d), ENG)


def test_embeddings_ignore_calibration(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": [entry(models=("test/embed-110m",))]}, sysname="cpu8",
                tasks=("embeddings",))
    c = _c(out, "test/embed-110m")
    assert c["tokens_per_s_est"] is None and "tokens_per_s_basis" not in c


def test_architecture_normalized_match(systems, catalog, tuning):
    from scout.solver import norm_arch

    assert norm_arch("qwen3_moe") == norm_arch("qwen3moe") == norm_arch("Qwen3-MoE")
    assert norm_arch(None) is None and norm_arch("") is None
    cat = copy.deepcopy(catalog)
    _model(cat, "test/dense-7b")["architecture"] = "test_arch"
    e = entry(models=("test/other-4b",))
    e["models"][0]["architecture"] = "TestArch"
    c = _c(_plan(systems, cat, tuning, {"cpu": [e]}), "test/dense-7b")
    assert "calibration_other_arch" not in c["confidence_reasons"] and c["confidence"] == "high"
    e["models"][0]["architecture"] = None
    assert "calibration_other_arch" in _c(_plan(systems, cat, tuning, {"cpu": [e]}), "test/dense-7b")["confidence_reasons"]


def test_run_plan_reads_power_only_with_calibration_file(monkeypatch, systems, catalog, tuning):
    import scout.bench as bench

    calls = []
    monkeypatch.setattr(bench, "read_power", lambda: calls.append(1) or dict(AC))
    intent = Intent(tasks=("chat",))
    out = cli.run_plan(systems["ddr5_64"], catalog, b"x", tuning, intent)
    assert calls == [] and out["inputs"]["calibration"]["status"] == "disabled"
    cli.run_plan(systems["ddr5_64"], catalog, b"x", tuning, intent, calibration=CalibrationInput("env", None))
    assert calls == []
    d = doc(systems["ddr5_64"], {"cpu": [entry(), entry(power=BATT, bw=10_000_000_000)]})
    first = cli.run_plan(systems["ddr5_64"], catalog, b"x", tuning, intent, calibration=CalibrationInput("env", d),
                         engines=ENG)
    again = cli.run_plan(systems["ddr5_64"], catalog, b"x", tuning, intent, calibration=CalibrationInput("env", d),
                         engines=ENG)
    assert first == again and len(calls) == 2
    md = cli.render_markdown(first)
    assert "Calibração local: usada (origem: env); energia agora: na tomada, perfil performance." in md
    assert "estados gravados: na tomada, perfil performance; na bateria, perfil balanced." in md


# ---------------------------------------------------------------- identidade (§2)


def test_layer1_difference_discards_and_falls_back_to_eta(systems, catalog, tuning):
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"cpu.model": "OUTRA CPU"})
    out = _plan(systems, catalog, tuning, d=d)
    cal = out["inputs"]["calibration"]
    assert cal["status"] == "discarded" and cal["pools"]["cpu"]["status"] == "discarded"
    assert cal["pools"]["cpu"]["reasons"] == ["calibration_machine_mismatch"]
    line = next(ln for ln in cal["match"] if ln["field"] == "cpu.model")
    assert (line["layer"], line["result"], line["effect"]) == (1, "different", "discard")
    w = next(w for w in out["warnings"] if w["code"] == "calibration_machine_mismatch")
    assert "OUTRA CPU" in w["message"] and "FIXTURE CPU 16C/32T" in w["message"] and w["field"] == "cpu.model"
    assert _c(out, "test/dense-7b")["tokens_per_s_basis"] == "eta"
    assert "power" in out["inputs"]


def test_layer1_null_is_unknown_and_discards(systems, catalog, tuning):
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"cpu.cores_physical": None})
    # Arquivo fora do schema (camada 1 null): o eco não valida, mas o solver não pode aplicar.
    out = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",)), tuning, AC, CalibrationInput("env", d), ENG)
    assert out["inputs"]["calibration"]["pools"]["cpu"]["reasons"] == ["calibration_machine_unknown"]
    assert "calibration_machine_unknown" in _codes(out)


@pytest.mark.parametrize("delta,status", [(0.04, "applied"), (0.06, "discarded")])
def test_ram_total_tolerance(systems, catalog, tuning, delta, status):
    ram = systems["ddr5_64"]["memory"]["ram_total_bytes"]["value"]
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"memory.ram_total_bytes": int(ram * (1 - delta))})
    assert _plan(systems, catalog, tuning, d=d)["inputs"]["calibration"]["pools"]["cpu"]["status"] == status


def test_gpu_fields_only_matter_for_gpu_pool(systems, catalog, tuning):
    # cpu pool numa máquina com GPU: trocar a GPU não descarta o cpu
    d = doc(systems["rtx3060"], {"cpu": [entry()]}, **{"gpu.name": "OUTRA GPU"})
    out = _plan(systems, catalog, tuning, d=d, sysname="rtx3060")
    assert out["inputs"]["calibration"]["pools"]["cpu"]["status"] == "applied"
    assert next(ln for ln in out["inputs"]["calibration"]["match"] if ln["field"] == "gpu.name")["effect"] == "none"
    d = doc(systems["rtx3060"], {"gpu": [entry()]}, **{"gpu.name": "OUTRA GPU"})
    out = _plan(systems, catalog, tuning, d=d, sysname="rtx3060")
    assert out["inputs"]["calibration"]["pools"]["gpu"]["status"] == "discarded"


def test_layer2_unknown_caps_medium_with_guidance(systems, catalog, tuning):
    s = copy.deepcopy(systems)
    d = doc(s["ddr5_64"], {"cpu": [entry()]})
    s["ddr5_64"]["memory"]["speed_mts"] = {"value": None, "source": "none", "confidence": "low"}
    out = _plan(s, catalog, tuning, d=d)
    c = _c(out, "test/dense-7b")
    assert out["inputs"]["calibration"]["pools"]["cpu"]["status"] == "applied"
    assert c["confidence"] == "medium" and "calibration_memory_unverified" in c["confidence_reasons"]
    w = [w for w in out["warnings"] if w["code"] == "calibration_memory_unverified"]
    assert len(w) == 1 and w[0]["severity"] == "info" and "dmidecode" in w[0]["message"]
    assert "sudo scout collect" in w[0]["message"]


def test_layer2_different_discards(systems, catalog, tuning):
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"memory.speed_mts": 4800})
    out = _plan(systems, catalog, tuning, d=d)
    assert out["inputs"]["calibration"]["pools"]["cpu"]["reasons"] == ["calibration_machine_mismatch"]
    assert _c(out, "test/dense-7b")["tokens_per_s_basis"] == "eta"


def test_layer2_not_applicable_to_unified_pool(systems, catalog, tuning):
    d = doc(systems["apple24"], {"unified": [entry()]})
    assert d["pools"]["unified"][0]["measured_with"]["memory.channels"] is None and d["machine"]["gpu.name"] is None
    out = _plan(systems, catalog, tuning, d=d, sysname="apple24")
    pc = out["inputs"]["calibration"]["pools"]["unified"]
    assert pc["status"] == "applied" and pc["reasons"] == []
    ch = next(ln for ln in pc["match"] if ln["field"] == "memory.channels")
    assert (ch["result"], ch["effect"]) == ("unknown", "none")  # não aplicável: sem teto
    assert "calibration_memory_unverified" not in _codes(out)


def test_memory_unverified_texts():
    assert "sudo scout collect" in C._memory_unverified_text("linux", True)
    assert "scout bench --run" in C._memory_unverified_text("linux", False)
    other = C._memory_unverified_text("darwin", True)
    assert "nesta plataforma" in other and "sudo" not in other


def test_schema_version_mismatch_discards_all(systems, catalog, tuning):
    d = doc(systems["ddr5_64"], {"cpu": [entry()]})
    d["calibration_schema_version"] = 99
    out = _plan(systems, catalog, tuning, d=d)
    assert out["inputs"]["calibration"]["status"] == "discarded"
    assert out["inputs"]["calibration"]["pools"]["cpu"]["reasons"] == ["calibration_schema_outdated"]
    assert "calibration_schema_outdated" in _codes(out)


def test_pool_absent(systems, catalog, tuning):
    d = doc(systems["rtx3060"], {"gpu": [entry()]})
    out = _plan(systems, catalog, tuning, d=d, sysname="ddr5_64")
    pc = out["inputs"]["calibration"]["pools"]["gpu"]
    assert pc["status"] == "pool_absent" and "calibration_pool_absent" in pc["reasons"]


def test_machine_identity_follows_solver_gpu(systems):
    m = C.machine_identity(systems["rtx3060"])
    assert m["gpu.name"] == "FIXTURE NVIDIA 12GB (perfil 2)" and m["gpu.backend"] == "cuda"
    assert m["platform"] == "linux" and m["cpu.cores_physical"] == 6
    assert tuple(m) == C.MACHINE_FIELDS and "os.kernel" not in m and "memory.speed_mts" not in m
    assert C.measured_with(systems["rtx3060"]) == {"os.kernel": "6.8.0-fixture", "memory.speed_mts": 3200,
                                                   "memory.channels": 2}
    assert C.machine_identity(systems["apple24"])["gpu.name"] is None  # unificada: sem pool gpu


# ---------------------------------------------------------------- software (§3)


@pytest.mark.parametrize("k,expected", [("6.99.0-1-exemplo", "6.99"), ("6.8", "6.8"), ("24.0.0-fixture", "24.0"),
                                        ("10.12.3", "10.12"), ("rolling", "rolling"), ("7", "7")])
def test_kernel_key(k, expected):
    assert C.kernel_key(k) == expected


def test_kernel_patch_change_is_noop_and_minor_change_downgrades(systems, catalog, tuning):
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"os.kernel": "6.8.9-99-generic"})
    out = _plan(systems, catalog, tuning, d=d)
    assert _c(out, "test/dense-7b")["confidence"] == "high"
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"os.kernel": "6.5.0-1"})
    out = _plan(systems, catalog, tuning, d=d)
    c = _c(out, "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_kernel_changed" in c["confidence_reasons"]
    w = next(w for w in out["warnings"] if w["code"] == "calibration_kernel_changed")
    assert w["message"] == ("A calibração foi medida com o kernel 6.5; agora é 6.8. "
                            "Rode `scout bench --run` para recalibrar.")
    assert all(ln["layer"] == 1 for ln in out["inputs"]["calibration"]["match"])
    line = next(ln for ln in out["inputs"]["calibration"]["pools"]["cpu"]["match"] if ln["field"] == "os.kernel")
    assert (line["layer"], line["effect"]) == ("software", "downgrade")
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"os.kernel": None})
    c = _c(_plan(systems, catalog, tuning, d=d), "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_kernel_unknown" in c["confidence_reasons"]


def test_engine_changed_and_unknown(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": [entry(version="0.0.0")]})
    c = _c(out, "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_engine_changed" in c["confidence_reasons"]
    assert "scout bench --run" in next(w for w in out["warnings"] if w["code"] == "calibration_engine_changed")["message"]
    assert out["inputs"]["calibration"]["pools"]["cpu"]["engine"] == {
        "tool": "ollama", "stored_version": "0.0.0", "current_version": "9.9.9", "result": "different"}
    out = _plan(systems, catalog, tuning, {"cpu": [entry()]}, engines={"ollama": None})
    c = _c(out, "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_engine_unknown" in c["confidence_reasons"]
    assert c["tokens_per_s_basis"] == "calibration"  # o número calibrado continua valendo


def test_downgrades_stack_to_low_floor(systems, catalog, tuning):
    d = doc(systems["ddr5_64"], {"cpu": [entry(version="0.0.0")]}, **{"os.kernel": "5.15.0"})
    c = _c(_plan(systems, catalog, tuning, d=d), "test/dense-7b")
    assert c["confidence"] == "low"
    d = doc(systems["ddr5_64"], {"cpu": [entry(version="0.0.0")]}, **{"os.kernel": "5.15.0"})
    c = _c(_plan(systems, catalog, tuning, d=d, power=BATT), "test/dense-7b")
    assert c["confidence"] == "low"  # piso


# ---------------------------------------------------------------- energia (§4)


def test_power_same_entry_chosen_among_states(systems, catalog, tuning):
    pools = {"cpu": [entry(), entry(power=BATT, bw=10_000_000_000, ovh=0.0)]}
    on_ac = _c(_plan(systems, catalog, tuning, pools), "test/dense-7b")
    out = _plan(systems, catalog, tuning, pools, power=BATT)
    batt = _c(out, "test/dense-7b")
    assert out["inputs"]["calibration"]["pools"]["cpu"]["power_match"] == "same"
    assert batt["confidence"] == "high" and batt["tokens_per_s_est"] < on_ac["tokens_per_s_est"] * 0.55
    assert out["inputs"]["calibration"]["pools"]["cpu"]["available_power"] == [AC, BATT]


def test_power_other_state_uses_slowest_and_downgrades(systems, catalog, tuning):
    pools = {"cpu": [entry(), entry(power=BATT, bw=10_000_000_000)]}
    out = _plan(systems, catalog, tuning, pools, power={"on_ac": False, "profile": "performance", "not_applicable": []})
    pc = out["inputs"]["calibration"]["pools"]["cpu"]
    assert pc["power_match"] == "other_state" and pc["calibration"]["bandwidth_bytes_per_s"] == 10_000_000_000
    c = _c(out, "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_power_mismatch" in c["confidence_reasons"]
    w = [w for w in out["warnings"] if w["code"] == "calibration_power_mismatch"]
    assert len(w) == 1 and w[0]["severity"] == "warn"
    assert "Calibrado na bateria, perfil balanced; você está na bateria, perfil performance" in w[0]["message"]
    assert "10.00 GB/s efetivos" in w[0]["message"]
    # o original: calibrado na tomada, plan na bateria
    out = _plan(systems, catalog, tuning, {"cpu": [entry()]}, power=BATT)
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "calibration_power_mismatch")
    assert "na tomada" in msg and "na bateria" in msg


def test_power_unknown_uses_slowest_compatible_and_caps(systems, catalog, tuning):
    pools = {"cpu": [entry(bw=30_000_000_000), entry(power=BATT, bw=10_000_000_000),
                     entry(power={"on_ac": True, "profile": "balanced", "not_applicable": []}, bw=20_000_000_000)]}
    out = _plan(systems, catalog, tuning, pools, power={"on_ac": True, "profile": None, "not_applicable": []})
    pc = out["inputs"]["calibration"]["pools"]["cpu"]
    assert pc["power_match"] == "unknown" and pc["calibration"]["bandwidth_bytes_per_s"] == 20_000_000_000
    c = _c(out, "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_power_unknown" in c["confidence_reasons"]
    w = next(w for w in out["warnings"] if w["code"] == "calibration_power_unknown")
    assert w["severity"] == "info" and "perfil desconhecido" in w["message"]
    # sem leitura nenhuma: tudo desconhecido, nunca não aplicável
    out = _plan(systems, catalog, tuning, {"cpu": [entry()]}, power=None)
    assert out["inputs"]["power"] == {"on_ac": None, "profile": None, "not_applicable": []}
    assert out["inputs"]["calibration"]["pools"]["cpu"]["power_match"] == "unknown"


def test_power_not_applicable_reaches_high(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": [entry(power=NA)]}, power=NA)
    assert out["inputs"]["calibration"]["pools"]["cpu"]["power_match"] == "same"
    assert _c(out, "test/dense-7b")["confidence"] == "high"
    out = _plan(systems, catalog, tuning, {"cpu": [entry(power=NA)]},
                power={"on_ac": None, "profile": None, "not_applicable": ["on_ac"]})
    assert out["inputs"]["calibration"]["pools"]["cpu"]["power_match"] == "unknown"
    assert _c(out, "test/dense-7b")["confidence"] == "medium"


@pytest.mark.parametrize("s,c,expected", [
    (NA, NA, "equal"), (AC, AC, "equal"), (AC, BATT, "different"), (AC, NA, "different"),
    (NA, AC, "different"), ({"on_ac": None, "profile": "x", "not_applicable": []}, AC, "unknown"),
    (AC, {"on_ac": None, "profile": "x", "not_applicable": []}, "unknown"),
    ({"on_ac": None, "profile": "x", "not_applicable": []}, NA, "unknown"),
])
def test_compare_power_attr(s, c, expected):
    assert C.compare_power_attr(s, c, "on_ac") == expected


def test_describe_power():
    assert C.describe_power(AC) == "na tomada, perfil performance"
    assert C.describe_power(BATT) == "na bateria, perfil balanced"
    assert C.describe_power(NA) == "sem bateria"
    assert C.describe_power({"on_ac": True, "profile": None, "not_applicable": []}) == "na tomada, perfil não lido"
    assert C.describe_power(None) == "tomada/bateria não lida, perfil não lido"


def test_choose_entry_tie_breaks():
    a = entry(power=BATT, bw=10, date="2000-01-01")
    b = entry(power=AC, bw=10, date="2001-01-01")
    c = entry(power=AC, bw=10, date="2001-01-01", version="x")
    other = {"on_ac": False, "profile": "performance", "not_applicable": []}
    assert C.choose_entry([a, b, c], other) == (b, "other_state")   # data mais recente, depois ordem
    assert C.choose_entry([a, b], AC) == (b, "same")
    assert C.choose_entry([a, b], {"on_ac": None, "profile": None, "not_applicable": []})[1] == "unknown"


def test_within_5pct_is_integer_and_symmetric():
    assert C.within_5pct(100, 95) and C.within_5pct(95, 100) and not C.within_5pct(100, 94)
    assert C.compare_field("memory.ram_total_bytes", 16_000_000_000, 16_700_000_000) == "equal"
    assert C.compare_field("cpu.cores_physical", 8, None) == "unknown"


# ---------------------------------------------------------------- rev. 4: measured_with por entrada


def _with_kernel(systems, k):
    s = copy.deepcopy(systems)
    s["ddr5_64"]["os"]["kernel"]["value"] = k
    return s


def test_finding1_old_entry_keeps_its_own_kernel_and_unverified_ram(systems, catalog, tuning):
    s = _with_kernel(systems, "6.99.0-3-exemplo")
    old = entry(power=AC, mw={"os.kernel": "6.98.0-1-exemplo", "memory.speed_mts": None, "memory.channels": 2})
    new = entry(power=BATT, mw={"os.kernel": "6.99.0-1-exemplo", "memory.speed_mts": 5600, "memory.channels": 2})
    d = doc(s["ddr5_64"], {"cpu": [old, new]})
    out = _plan(s, catalog, tuning, d=d)  # no estado da entrada antiga (tomada)
    pc = out["inputs"]["calibration"]["pools"]["cpu"]
    assert pc["power_match"] == "same" and pc["calibration"]["measured_with"]["os.kernel"] == "6.98.0-1-exemplo"
    c = _c(out, "test/dense-7b")
    assert {"calibration_kernel_changed", "calibration_memory_unverified"} <= set(c["confidence_reasons"])
    assert c["confidence"] == "low"  # teto medium, depois um nível abaixo; nunca high
    assert {ln["field"]: ln["effect"] for ln in pc["match"]} == {
        "os.kernel": "downgrade", "memory.speed_mts": "cap_medium", "memory.channels": "none"}
    # no estado da entrada nova: tudo confere
    c = _c(_plan(s, catalog, tuning, d=d, power=BATT), "test/dense-7b")
    assert c["confidence"] == "high"


def test_layer2_different_entry_is_ineligible_before_power_choice(systems, catalog, tuning):
    other_ram = {"os.kernel": "6.8.0-fixture", "memory.speed_mts": 4800, "memory.channels": 2}
    d = doc(systems["ddr5_64"], {"cpu": [entry(power=AC, bw=5_000_000_000, mw=other_ram),
                                         entry(power=BATT, bw=10_000_000_000)]})
    out = _plan(systems, catalog, tuning, d=d)  # na tomada: a única que casa é inelegível
    pc = out["inputs"]["calibration"]["pools"]["cpu"]
    assert pc["status"] == "applied" and pc["power_match"] == "other_state"
    assert pc["calibration"]["bandwidth_bytes_per_s"] == 10_000_000_000
    assert pc["available_power"] == [AC, BATT]


def test_ttft_outside_exact_power_match_uses_lowest_eligible_prefill(systems, catalog, tuning):
    other_ram = {"os.kernel": "6.8.0-fixture", "memory.speed_mts": 4800, "memory.channels": 2}
    pools = {"cpu": [entry(power=AC, bw=10_000_000_000, prefill=40.0),
                     entry(power=BATT, bw=30_000_000_000, prefill=20.0),
                     entry(power={"on_ac": False, "profile": "power-saver", "not_applicable": []},
                           bw=40_000_000_000, prefill=5.0, mw=other_ram)]}  # inelegível: não conta
    d = doc(systems["ddr5_64"], pools)
    out = _plan(systems, catalog, tuning, d=d, prompt_tokens=400,
                power={"on_ac": False, "profile": "performance", "not_applicable": []})
    pc = out["inputs"]["calibration"]["pools"]["cpu"]
    assert pc["power_match"] == "other_state" and pc["calibration"]["prefill_tokens_per_s"] == 40.0
    assert _c(out, "test/dense-7b")["ttft_s_est"] == pytest.approx(400 / 20.0)
    out = _plan(systems, catalog, tuning, d=d, prompt_tokens=400)  # casamento exato: a da entrada
    assert _c(out, "test/dense-7b")["ttft_s_est"] == pytest.approx(400 / 40.0)
    out = _plan(systems, catalog, tuning, d=d, prompt_tokens=400, power=None)  # unknown: também o menor
    assert out["inputs"]["calibration"]["pools"]["cpu"]["power_match"] == "unknown"
    assert _c(out, "test/dense-7b")["ttft_s_est"] == pytest.approx(400 / 20.0)


def test_chosen_tools_only_from_applicable_pools(systems):
    s = systems["rtx3060"]
    cpu = entry()
    gpu = dict(entry(), tool="llama-bench")
    assert C.chosen_tools(doc(s, {"cpu": [cpu], "gpu": [gpu]}), s, AC) == {"ollama", "llama-bench"}
    assert C.chosen_tools(doc(s, {"cpu": [cpu], "gpu": [gpu]}, **{"gpu.name": "OUTRA"}), s, AC) == {"ollama"}
    assert C.chosen_tools(doc(s, {"gpu": [gpu]}), systems["ddr5_64"], AC) == set()  # pool ausente
    # entre estados, só a escolhida conta
    batt = dict(entry(power=BATT), tool="llama-bench")
    assert C.chosen_tools(doc(s, {"cpu": [cpu, batt]}), s, BATT) == {"llama-bench"}


# ---------------------------------------------------------------- revisão, rodada 2


def test_identity_messages_show_bytes_with_unit(systems, catalog, tuning):
    ram = systems["ddr5_64"]["memory"]["ram_total_bytes"]["value"]
    d = doc(systems["ddr5_64"], {"cpu": [entry()]}, **{"memory.ram_total_bytes": ram * 2})
    out = _plan(systems, catalog, tuning, d=d)
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "calibration_machine_mismatch")
    assert E.fmt_gb(ram * 2) in msg and E.fmt_gb(ram) in msg
    assert str(ram) not in msg and str(ram * 2) not in msg


def test_discarded_pool_echo_has_no_active_effects(systems, catalog, tuning):
    mw = {"os.kernel": "6.98.0-1-fixture", "memory.speed_mts": None, "memory.channels": None}
    d = doc(systems["ddr5_64"], {"cpu": [entry(mw=mw)]}, **{"cpu.model": "OUTRA CPU"})
    out = _plan(systems, catalog, tuning, d=d)
    pc = out["inputs"]["calibration"]["pools"]["cpu"]
    assert pc["status"] == "discarded"
    assert pc["match"] and all(ln["effect"] == "none" for ln in pc["match"])
    assert "calibration_kernel_changed" not in _codes(out) and "calibration_memory_unverified" not in _codes(out)


def test_slow_prefill_outside_exact_match_cites_lowest_eligible_prefill(systems, catalog, tuning):
    ref = {"tokens_per_s": 11.7, "measured_on": "REFERÊNCIA", "tool": "t", "date": "2026-11-15"}
    pools = {"cpu": [entry(power=AC, bw=10_000_000_000, prefill=40.0),
                     entry(power=BATT, bw=30_000_000_000, prefill=20.0)]}
    out = _plan(systems, catalog, dict(tuning, prefill_reference=ref), pools,
                power={"on_ac": False, "profile": "performance", "not_applicable": []})
    assert out["inputs"]["calibration"]["pools"]["cpu"]["power_match"] == "other_state"
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "slow_prefill_cpu")
    assert "~20 tokens/s" in msg and "~40 tokens/s" not in msg
    out = _plan(systems, catalog, dict(tuning, prefill_reference=ref), pools)   # casamento exato: a escolhida
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "slow_prefill_cpu")
    assert "~40 tokens/s" in msg
