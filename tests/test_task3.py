"""Embeddings sem decode e aviso de prefill lento na CPU (decisão pós-validação em hardware real)."""
from __future__ import annotations

import copy

from conftest import schema_validator
from scout import cli
from scout.solver import Intent, solve

REF = {"tokens_per_s": 11.7, "measured_on": "MÁQUINA DE TESTE, CPU only", "tool": "ferramenta de teste",
       "date": "2026-11-15", "power": "AC"}


def _emb(out: dict) -> dict:
    return next(c for c in out["candidates"] if c["model"]["id"] == "test/embed-110m")


def _codes(out: dict) -> list[str]:
    return [w["code"] for w in out["warnings"]]


def test_embeddings_have_no_decode_speed(systems, catalog, tuning):
    for name, system in systems.items():
        out = solve(system, catalog, Intent(tasks=("embeddings",)), tuning)
        c = _emb(out)
        assert c["tokens_per_s_est"] is None and c["ttft_s_est"] is None, name
        assert [n["code"] for n in c.get("notes", [])][-1] == "decode_not_applicable"
        reasons = c["confidence_reasons"]
        assert "decode_not_applicable" in reasons
        assert not {"ranked_by_bytes", "bandwidth_unknown", "eta_default", "bandwidth_estimated"} & set(reasons)
        assert list(schema_validator("feasibility.schema.json").iter_errors(out)) == []


def test_llm_candidates_keep_decode_speed(systems, catalog, tuning):
    out = solve(systems["rtx3060"], catalog, Intent(tasks=("chat", "embeddings")), tuning)
    llm = [c for c in out["candidates"] if c["model"]["id"] != "test/embed-110m"]
    assert llm and all(c["tokens_per_s_est"] is not None for c in llm)
    assert all("decode_not_applicable" not in c["confidence_reasons"] for c in llm)


def test_slow_prefill_warning_uses_measured_reference(systems, catalog, tuning):
    out = solve(systems["cpu8"], catalog, Intent(tasks=("chat",)), dict(tuning, prefill_reference=REF))
    w = [w for w in out["warnings"] if w["code"] == "slow_prefill_cpu"]
    assert len(w) == 1 and w[0]["severity"] == "warn"
    assert "~12 tokens/s" in w[0]["message"] and "702" not in w[0]["message"]
    assert "700 tokens" in w[0]["message"]  # 11.7 * 60 = 702, 2 algarismos significativos
    assert "MÁQUINA DE TESTE" in w[0]["message"]
    assert out["inputs"]["tuning"]["prefill_reference"] == REF
    assert list(schema_validator("feasibility.schema.json").iter_errors(out)) == []


def test_slow_prefill_number_follows_tuning(systems, catalog, tuning):
    ref = dict(REF, tokens_per_s=23.4)  # 1404 por minuto -> 1400
    out = solve(systems["cpu8"], catalog, Intent(tasks=("chat",)), dict(tuning, prefill_reference=ref))
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "slow_prefill_cpu")
    assert "~23 tokens/s" in msg and "1400 tokens" in msg


def test_slow_prefill_without_reference_has_no_number(systems, catalog, tuning):
    out = solve(systems["cpu8"], catalog, Intent(tasks=("chat",)), tuning)
    msg = next(w["message"] for w in out["warnings"] if w["code"] == "slow_prefill_cpu")
    assert "tokens/s" not in msg and "referência" in msg


def test_no_slow_prefill_when_everything_on_gpu(systems, catalog, tuning):
    out = solve(systems["rtx3060"], catalog, Intent(tasks=("chat",)), dict(tuning, prefill_reference=REF))
    if all(c["placement"] == "gpu_full" for c in out["candidates"]):
        assert "slow_prefill_cpu" not in _codes(out)


def test_embeddings_only_on_cpu_do_not_trigger_prefill_warning(systems, catalog, tuning):
    out = solve(systems["cpu8"], catalog, Intent(tasks=("embeddings",)), dict(tuning, prefill_reference=REF))
    assert _emb(out)["placement"] == "cpu_only"
    assert "slow_prefill_cpu" not in _codes(out)


def test_hybrid_with_cpu_majority_triggers_warning(systems, catalog, tuning):
    out = solve(systems["rtx3060"], copy.deepcopy(catalog), Intent(tasks=("chat", "code")), dict(tuning, prefill_reference=REF))
    hyb = [c for c in out["candidates"] if c["placement"] == "hybrid"]
    for c in hyb:
        cpu_major = c["model"]["n_layers"] - c["n_gpu_layers"] > c["n_gpu_layers"]
        msg = " ".join(w["message"] for w in out["warnings"] if w["code"] == "slow_prefill_cpu")
        assert (c["model"]["id"] in msg) == cpu_major


def test_markdown_shows_dash_for_embeddings(capsys):
    from conftest import CATALOG_PATH, SYSTEM_PATHS, FIX

    argv = ["plan", "--system", str(SYSTEM_PATHS["cpu8"]), "--catalog", str(CATALOG_PATH),
            "--task", "embeddings", "--tuning", str(FIX / "tuning_test.json")]
    assert cli.main(argv) == 0
    md = capsys.readouterr().out
    row = next(line for line in md.splitlines() if line.startswith("| [test/embed-110m]"))
    assert "| — |" in row and "decode não se aplica" in row
