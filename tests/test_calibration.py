"""Consumo da calibração do scout bench pelo plan (schemas/bench_spec.md). Tunings sintéticos em memória."""
from __future__ import annotations

import copy
import json

import pytest

from conftest import CATALOG_PATH, SYSTEM_PATHS, schema_validator
from scout import cli
from scout import estimate as E
from scout.solver import Intent, solve

AC = {"on_ac": True, "profile": "performance"}


def cal(models=("test/dense-7b",), stable=True, power=AC, bw=20_000_000_000, ovh=0.002, prefill=None):
    return {"bandwidth_bytes_per_s": bw, "overhead_s_per_token": ovh, "form": "2p" if ovh else "1p",
            "stable": stable, "power": dict(power), "tool": "sintético", "date": "2026-11-15",
            "models": [{"id": m, "architecture": "testarch", "bytes": 1} for m in models],
            "prefill_tokens_per_s": prefill}


def _c(out: dict, mid: str) -> dict:
    return next(c for c in out["candidates"] if c["model"]["id"] == mid)


def _model(catalog: dict, mid: str) -> dict:
    return next(m for m in catalog["models"] if m["id"] == mid)


def _valid(out: dict) -> None:
    assert list(schema_validator("feasibility.schema.json").iter_errors(out)) == []


def _plan(systems, catalog, tuning, calibration, power=AC, **intent):
    t = dict(tuning, calibration=calibration)
    out = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",), **intent), t, power)
    _valid(out)
    return out


def test_calibrated_cpu_uses_bytes_over_bandwidth_plus_overhead(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": cal()})
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
    assert out["inputs"]["tuning"]["calibration"]["cpu"]["bandwidth_bytes_per_s"] == 20_000_000_000
    assert out["inputs"]["power"] == AC


def test_eta_untouched_and_no_calibration_is_unchanged(systems, catalog, tuning):
    base = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",)), tuning)
    again = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",)), dict(tuning), AC)
    assert base == again
    c = _c(base, "test/dense-7b")
    assert c["confidence"] == "medium" and "eta_default" in c["confidence_reasons"]
    assert c["tokens_per_s_basis"] == "eta"
    out = _plan(systems, catalog, tuning, {"cpu": cal()})
    assert out["inputs"]["tuning"]["eta_cpu"] == tuning["eta_cpu"]


def test_unstable_calibration_caps_medium(systems, catalog, tuning):
    c = _c(_plan(systems, catalog, tuning, {"cpu": cal(stable=False)}), "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_unstable" in c["confidence_reasons"]


def test_power_mismatch_warns_and_downgrades(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": cal()}, power={"on_ac": False, "profile": "performance"})
    c = _c(out, "test/dense-7b")
    assert "calibration_power_mismatch" in c["confidence_reasons"]
    assert c["confidence"] == "medium"  # high rebaixado um nível
    assert [w["code"] for w in out["warnings"]].count("calibration_power_mismatch") == 1
    unstable = _c(_plan(systems, catalog, tuning, {"cpu": cal(stable=False)},
                        power={"on_ac": True, "profile": "power-saver"}), "test/dense-7b")
    assert unstable["confidence"] == "low"  # medium rebaixado um nível


def test_power_unknown_caps_medium_without_warning(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": cal()}, power=None)
    c = _c(out, "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_power_unknown" in c["confidence_reasons"]
    assert "calibration_power_mismatch" not in [w["code"] for w in out["warnings"]]


def test_other_architecture_caps_medium(systems, catalog, tuning):
    c = _c(_plan(systems, catalog, tuning, {"cpu": cal(models=("test/other-4b",))}), "test/dense-7b")
    assert c["confidence"] == "medium" and "calibration_other_arch" in c["confidence_reasons"]
    assert c["tokens_per_s_basis"] == "calibration"


def test_hybrid_with_one_calibrated_pool_is_mixed(systems, catalog, tuning):
    out = solve(systems["rtx3060"], catalog, Intent(tasks=("chat", "code")),
                dict(tuning, calibration={"cpu": cal(models=("test/moe-30b-a3b",))}), AC)
    _valid(out)
    hyb = [c for c in out["candidates"] if c["placement"] == "hybrid"]
    assert hyb, "fixture rtx3060 precisa de um híbrido"
    for c in hyb:
        assert c["tokens_per_s_basis"] == "mixed"
        assert c["provenance"]["tokens_per_s_est"] == "estimated"
        assert "eta_default" in c["confidence_reasons"] and c["confidence"] != "high"


def test_ttft_only_with_prompt_and_own_stable_prefill(systems, catalog, tuning):
    out = _plan(systems, catalog, tuning, {"cpu": cal(prefill=40.0)}, prompt_tokens=532)
    c = _c(out, "test/dense-7b")
    assert c["ttft_s_est"] == pytest.approx(532 / 40.0, abs=0.006)
    assert "ttft_unspecified" not in c["confidence_reasons"]
    assert out["inputs"]["intent"]["prompt_tokens"] == 532
    # sem prompt_tokens, ou prefill de outro modelo, ou sem prefill: null
    assert _c(_plan(systems, catalog, tuning, {"cpu": cal(prefill=40.0)}), "test/dense-7b")["ttft_s_est"] is None
    other = _plan(systems, catalog, tuning, {"cpu": cal(models=("test/other-4b",), prefill=40.0)}, prompt_tokens=532)
    assert _c(other, "test/dense-7b")["ttft_s_est"] is None
    assert _c(_plan(systems, catalog, tuning, {"cpu": cal()}, prompt_tokens=532), "test/dense-7b")["ttft_s_est"] is None


def test_slow_prefill_cites_local_measurement(systems, catalog, tuning):
    ref = {"tokens_per_s": 11.7, "measured_on": "REFERÊNCIA", "tool": "t", "date": "2026-11-15"}
    t = dict(tuning, prefill_reference=ref, calibration={"cpu": cal(prefill=30.0)})
    out = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat",)), t, AC)
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "slow_prefill_cpu")
    assert "nesta máquina" in msg and "~30 tokens/s" in msg and "1800 tokens" in msg
    assert "REFERÊNCIA" not in msg


@pytest.mark.parametrize("bad", [{"bandwidth_bytes_per_s": 0, "overhead_s_per_token": 0},
                                 {"bandwidth_bytes_per_s": 1.5e10, "overhead_s_per_token": 0},
                                 {"bandwidth_bytes_per_s": 10**10, "overhead_s_per_token": -1}])
def test_invalid_calibration_fails_loudly(systems, catalog, tuning, bad):
    with pytest.raises(ValueError):
        solve(systems["cpu8"], catalog, Intent(tasks=("chat",)), dict(tuning, calibration={"cpu": bad}), AC)


def test_embeddings_ignore_calibration(systems, catalog, tuning):
    out = solve(systems["cpu8"], catalog, Intent(tasks=("embeddings",)),
                dict(tuning, calibration={"cpu": cal(models=("test/embed-110m",))}), AC)
    c = _c(out, "test/embed-110m")
    assert c["tokens_per_s_est"] is None and "tokens_per_s_basis" not in c


def test_cli_reads_power_only_with_calibration_and_is_deterministic(tmp_path, monkeypatch, capsys, tuning):
    import scout.bench as bench

    calls = []
    monkeypatch.setattr(bench, "read_power", lambda: calls.append(1) or dict(AC))
    tp = tmp_path / "tuning.json"
    tp.write_text(json.dumps(tuning), encoding="utf-8")
    argv = ["plan", "--system", str(SYSTEM_PATHS["ddr5_64"]), "--catalog", str(CATALOG_PATH),
            "--tuning", str(tp), "--json", "--prompt-tokens", "532"]
    assert cli.main(argv) == 0 and calls == []
    capsys.readouterr()
    tp.write_text(json.dumps(dict(tuning, calibration={"cpu": cal(prefill=40.0)})), encoding="utf-8")
    assert cli.main(argv) == 0
    first = capsys.readouterr().out
    assert cli.main(argv) == 0
    assert capsys.readouterr().out == first and len(calls) == 2
    doc = json.loads(first)
    _valid(doc)
    assert doc["inputs"]["power"] == AC
    assert cli.main([a for a in argv if a != "--json"]) == 0
    md = capsys.readouterr().out
    assert "calibração desta máquina" in md and "primeiro token" in md


def test_architecture_normalized_match(systems, catalog, tuning):
    from scout.solver import norm_arch

    assert norm_arch("qwen3_moe") == norm_arch("qwen3moe") == norm_arch("Qwen3-MoE")
    assert norm_arch(None) is None and norm_arch("") is None
    cat = copy.deepcopy(catalog)
    _model(cat, "test/dense-7b")["architecture"] = "test_arch"
    other = {"cpu": cal(models=("test/other-4b",))}
    other["cpu"]["models"][0]["architecture"] = "TestArch"
    out = solve(systems["ddr5_64"], cat, Intent(tasks=("chat",)), dict(tuning, calibration=other), AC)
    c = _c(out, "test/dense-7b")
    assert "calibration_other_arch" not in c["confidence_reasons"] and c["confidence"] == "high"
    # null em qualquer lado = diferente
    other["cpu"]["models"][0]["architecture"] = None
    out = solve(systems["ddr5_64"], cat, Intent(tasks=("chat",)), dict(tuning, calibration=other), AC)
    assert "calibration_other_arch" in _c(out, "test/dense-7b")["confidence_reasons"]
