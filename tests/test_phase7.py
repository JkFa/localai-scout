"""Fase 7 (schemas/m7_fixes.md): F4 ordem sem tokens/s, Fa limiar de offload, F7 proveniência nos
rejeitados. Contas à mão nos comentários. (F6 e F8 estão em test_cli.py.)"""
from __future__ import annotations

import json

import pytest

from conftest import ALL_TASKS, TUNING, assert_valid, deep
from scout.solver import Intent, solve
from test_quality_preference import mk
from test_solver import bytes_keys, cand, ids, rej, run, sorted_ok

NONE_BW = {"value": None, "source": "none", "confidence": "low"}


def go(system, catalog, models=None, tuning=None, tasks=ALL_TASKS, ctx=8192):
    cat = deep(catalog)
    if models is not None:
        cat["models"] = models
    return solve(deep(system), cat, Intent(tasks=tuple(tasks), ctx=ctx), dict(tuning or TUNING))


# ================================================================ F4
def squeezed_rtx(systems, gpu_bw=None, cpu_bw=51_200_000_000):
    """VRAM usable 9_000_000_000: o 7B Q8_0 não cabe em gpu_full (9_287_581_824 > 0.95*9e9)."""
    s = deep(systems["rtx3060"])
    g = s["gpus"]["value"][0]
    g["vram_free_bytes"]["value"] = 9_600_000_000
    g["bandwidth_bytes_per_s"] = NONE_BW if gpu_bw is None else {"value": gpu_bw, "source": "fixture", "confidence": "medium"}
    s["memory"]["bandwidth_bytes_per_s"] = NONE_BW if cpu_bw is None else {"value": cpu_bw, "source": "fixture", "confidence": "medium"}
    return s


def test_f4_unknown_gpu_bw_picks_least_cpu_bytes(systems, catalog):
    """hybrid (tokens None) e cpu_only (tokens conhecido) viáveis: vence o de MENOR bytes na CPU.
    n_gpu = floor((8_550_000_000 - 1_073_741_824 - 563_840_000)/239_062_500) = 28
    hybrid  : 4 camadas * 239_062_500 =   956_250_000 B/token na CPU
    cpu_only: 32 camadas * 239_062_500 = 7_650_000_000 B/token na CPU"""
    out = go(squeezed_rtx(systems), catalog)
    d = cand(out, "test/dense-7b")
    assert d["quant"]["name"] == "Q8_0" and d["placement"] == "hybrid" and d["n_gpu_layers"] == 28
    assert d["tokens_per_s_est"] is None
    assert "ranked_by_bytes" in d["confidence_reasons"]
    assert bytes_keys(d) == (956_250_000, 7_650_000_000)
    assert_valid("feasibility.schema.json", out)


def test_f4_unknown_gpu_bw_gpu_full_wins_with_zero_cpu_bytes(systems, catalog):
    """gpu_full lê 0 B na CPU: vence cpu_only mesmo este tendo tokens/s conhecido."""
    s = deep(systems["rtx3060"])
    s["gpus"]["value"][0]["bandwidth_bytes_per_s"] = NONE_BW
    d = cand(go(s, catalog), "test/dense-7b")
    assert d["placement"] == "gpu_full" and d["n_gpu_layers"] == 32 and d["tokens_per_s_est"] is None
    assert "ranked_by_bytes" in d["confidence_reasons"]


def test_f4_unknown_cpu_bw_hybrid_over_cpu_only(systems, catalog):
    out = go(squeezed_rtx(systems, gpu_bw=360_000_000_000, cpu_bw=None), catalog)
    d = cand(out, "test/dense-7b")
    # trecho CPU sem banda: hybrid e cpu_only ficam None; menos bytes na CPU => hybrid
    assert d["placement"] == "hybrid" and d["tokens_per_s_est"] is None
    assert "ranked_by_bytes" in d["confidence_reasons"]


def test_f4_order_unchanged_when_tokens_known(systems, catalog):
    for profile in ("cpu8", "rtx3060", "ddr5_64", "apple24", "rx7800xt"):
        out = run(systems[profile], catalog)
        assert all(c["tokens_per_s_est"] is not None for c in out["candidates"])
        assert all("ranked_by_bytes" not in c["confidence_reasons"] for c in out["candidates"]), profile
        sorted_ok(out["candidates"])
    # ddr5_64 conhecido: embed 58.. > 1B > MoE 21.85 > 7B 7.03 (por tokens/s, não por id)
    order = [c["model"]["id"] for c in run(systems["ddr5_64"], catalog)["candidates"]]
    tps = {c["model"]["id"]: c["tokens_per_s_est"] for c in run(systems["ddr5_64"], catalog)["candidates"]}
    assert order == sorted(order, key=lambda i: -tps[i])


def test_f4_known_placement_choice_still_by_tokens(systems, catalog):
    """Todos conhecidos: vence o maior tokens/s (como antes). RTX3060 + MoE: híbrido único viável."""
    out = run(systems["rtx3060"], catalog)
    d = cand(out, "test/dense-7b")
    assert d["placement"] == "gpu_full" and "ranked_by_bytes" not in d["confidence_reasons"]


def test_f4_mixed_known_and_unknown_none_last(systems, catalog):
    s = deep(systems["rtx3060"])
    s["memory"]["bandwidth_bytes_per_s"] = NONE_BW        # só o trecho CPU fica sem banda
    out = go(s, catalog)
    sorted_ok(out["candidates"])
    moe = cand(out, "test/moe-30b-a3b")                    # híbrido usa a RAM => None
    assert moe["tokens_per_s_est"] is None and "ranked_by_bytes" in moe["confidence_reasons"]
    known = [c for c in out["candidates"] if c["tier"] == "confortavel" and c["tokens_per_s_est"] is not None]
    assert known, "gpu_full continua com tokens/s"


# ================================================================ Fa
def fat_model(catalog):
    return mk(catalog, "test/fat-7b", 7_200_000_000, 7_200_000_000, 32,
              [("F16", 16.0, 14_400_000_000), ("Q8_0", 8.5, 7_650_000_000)], kv_heads=8, head_dim=128)


def test_fa_bpw_above_limit_not_hybrid(systems, catalog):
    """F16 (bpw 16 > 8.5): gpu_full não cabe (14.4e9 > 11.21e9), cpu_only não cabe
    (14.4e9+1_073_741_824+563_840_000 = 16_037_581_824 > 0.95*14_179_869_184) e hybrid NÃO é avaliado."""
    out = go(systems["rtx3060"], catalog, [fat_model(catalog)])
    c = cand(out, "test/fat-7b")
    assert c["quant"]["name"] == "Q8_0" and c["placement"] == "gpu_full"
    assert c["skipped_quants"] == [{"format": "GGUF", "name": "F16", "reason_code": "bpw_above_offload_limit"}]
    assert_valid("feasibility.schema.json", out)


def test_fa_limit_is_read_from_tuning(systems, catalog):
    # limiar 16.0: o F16 volta a ser híbrido. bytes_camada 450_000_000;
    # n_gpu = floor((11_210_000_000 - 1_073_741_824 - 563_840_000)/450_000_000) = floor(21.27) = 21
    out = go(systems["rtx3060"], catalog, [fat_model(catalog)], tuning={**TUNING, "max_bpw_offload": 16.0})
    c = cand(out, "test/fat-7b")
    assert c["quant"]["name"] == "F16" and c["placement"] == "hybrid" and c["n_gpu_layers"] == 21
    assert c["skipped_quants"] == []
    # limiar menor que 8.5 também age: 4.0 deixa Q8_0 sem hybrid (aqui ainda cabe em gpu_full)
    out2 = go(systems["rtx3060"], catalog, [fat_model(catalog)], tuning={**TUNING, "max_bpw_offload": 4.0})
    assert cand(out2, "test/fat-7b")["placement"] == "gpu_full"


def moe_q8(catalog):
    return mk(catalog, "test/moe-q8", 30_000_000_000, 3_000_000_000, 48,
              [("Q8_0", 8.5, 31_875_000_000), ("Q6_K", 6.56, 24_600_000_000)], kv_heads=4, head_dim=128)


def test_fa_limit_boundary_is_inclusive(systems, catalog):
    """bpw 8.5 == limiar 8.5 NÃO é 'acima': hybrid avaliado. rx7800xt: Q8_0 31_875_000_000 só cabe híbrido;
    bytes_camada 664_062_500; n_gpu = floor((15_200_000_000 - 1_369_146_368)/664_062_500) = floor(20.83) = 20."""
    out = go(systems["rx7800xt"], catalog, [moe_q8(catalog)])
    c = cand(out, "test/moe-q8")
    assert c["quant"]["name"] == "Q8_0" and c["placement"] == "hybrid" and c["n_gpu_layers"] == 20
    # limiar 8.4: Q8_0 (8.5 > 8.4) sem hybrid, não cabe nos outros => cai para Q6_K híbrido (26 camadas)
    out2 = go(systems["rx7800xt"], catalog, [moe_q8(catalog)], tuning={**TUNING, "max_bpw_offload": 8.4})
    c2 = cand(out2, "test/moe-q8")
    assert c2["quant"]["name"] == "Q6_K" and c2["placement"] == "hybrid" and c2["n_gpu_layers"] == 26
    assert c2["skipped_quants"] == [{"format": "GGUF", "name": "Q8_0", "reason_code": "bpw_above_offload_limit"}]
    assert_valid("feasibility.schema.json", out2)


def test_fa_gpu_full_and_cpu_only_still_evaluated_above_limit(systems, catalog):
    for tun in (TUNING, {**TUNING, "max_bpw_offload": 4.0}):
        # embed F16 (bpw 16): gpu_full no perfil 2, cpu_only no perfil 1
        e2 = cand(go(systems["rtx3060"], catalog, tuning=tun), "test/embed-110m")
        assert e2["quant"]["name"] == "F16" and e2["placement"] == "gpu_full" and e2["n_gpu_layers"] == 12
        e1 = cand(go(systems["cpu8"], catalog, tuning=tun), "test/embed-110m")
        assert e1["quant"]["name"] == "F16" and e1["placement"] == "cpu_only" and e1["n_gpu_layers"] == 0
        assert e1["skipped_quants"] == [] and e2["skipped_quants"] == []
    # cpu_only com bpw 16 em RAM grande: F16 14.4e9 cabe nos 65_719_476_736 B
    f = cand(go(systems["ddr5_64"], catalog, [fat_model(catalog)]), "test/fat-7b")
    assert f["quant"]["name"] == "F16" and f["placement"] == "cpu_only"


# ================================================================ F7
def tiny_fallback_only(catalog):
    m = deep(next(x for x in catalog["models"] if x["id"] == "test/tiny-1b"))
    m["quants"] = [q for q in m["quants"] if q["name"] == "Q8_0"]    # só fallback (file_bytes null)
    return m


def test_f7_missing_bytes_estimated_when_closest_uses_fallback(systems, catalog):
    s = deep(systems["cpu8"])
    s["memory"]["ram_total_bytes"]["value"] = 4_000_000_000          # usable = 1_000_000_000
    out = go(s, catalog, [tiny_fallback_only(catalog)])
    r = rej(out, "test/tiny-1b")
    assert r["reason_code"] == "does_not_fit"
    # fallback: ceil(1.2e9*8.5/8*1.15) = 1_466_250_000 (±1); total = +268_435_456 + 563_840_000 = 2_298_525_456
    # falta = 2_298_525_456 - floor(0.95e9) = 1_348_525_456 (±1)
    assert abs(r["missing_bytes"] - 1_348_525_456) <= 1
    assert r["provenance"]["missing_bytes"] == "estimated"
    assert_valid("feasibility.schema.json", out)


def test_f7_missing_bytes_calculated_when_closest_has_file_bytes(systems, catalog):
    out = run(systems["cpu8"], catalog)
    for mid in ("test/dense-7b", "test/moe-30b-a3b"):
        r = rej(out, mid)
        assert r["missing_bytes"] > 0
        assert r.get("provenance", {}).get("missing_bytes", "calculated") == "calculated"
    assert_valid("feasibility.schema.json", out)


# ================================================================ R3: nota offload_skipped_above_bpw
CODE = "offload_skipped_above_bpw"


def notes(c):
    return c.get("notes") or []


def test_r3_note_when_offload_limit_discards_hybrid_but_model_still_enters(systems, catalog):
    out = go(systems["rtx3060"], catalog, [fat_model(catalog)])
    c = cand(out, "test/fat-7b")
    # F16 só seria híbrida (limiar 8.5); Q8_0 cabe inteiro na GPU
    assert c["quant"]["name"] == "Q8_0" and c["placement"] == "gpu_full"
    ns = [n for n in notes(c) if n["code"] == CODE]
    assert len(ns) == 1 and ns[0]["message"].strip()
    msg = ns[0]["message"]
    assert "F16" in msg and "Q8_0" in msg and ("8,5" in msg or "8.5" in msg)
    assert rej(out, "test/fat-7b") is None and out["rejected"] == []      # sem entrada em rejected
    assert c["skipped_quants"] == [{"format": "GGUF", "name": "F16", "reason_code": "bpw_above_offload_limit"}]
    assert_valid("feasibility.schema.json", out)


def test_r3_note_numbers_come_from_tuning(systems, catalog):
    out = go(systems["rtx3060"], catalog, [fat_model(catalog)], tuning={**TUNING, "max_bpw_offload": 12.0})
    msg = next(n["message"] for n in notes(cand(out, "test/fat-7b")) if n["code"] == CODE)
    assert "12" in msg and "8,5" not in msg and "8.5" not in msg


def test_r3_no_note_without_limit_in_tuning(systems, catalog):
    tun = {k: v for k, v in TUNING.items() if k != "max_bpw_offload"}
    out = go(systems["rtx3060"], catalog, [fat_model(catalog)], tuning=tun)
    c = cand(out, "test/fat-7b")
    assert c["quant"]["name"] == "F16" and c["placement"] == "hybrid" and c["n_gpu_layers"] == 21
    assert notes(c) == []


def test_r3_no_note_when_limit_does_not_act(systems, catalog):
    # catálogo de teste puro: nenhuma quant acima do limiar depende de híbrido
    for profile in ("cpu8", "rtx3060", "ddr5_64", "apple24", "rx7800xt"):
        out = run(systems[profile], catalog)
        assert all(notes(c) == [] for c in out["candidates"]), profile
    # limiar acima de todas as quants do 7B/F16 também não gera nota
    out = go(systems["rtx3060"], catalog, [fat_model(catalog)], tuning={**TUNING, "max_bpw_offload": 16.0})
    assert notes(cand(out, "test/fat-7b")) == []


def test_r3_notes_are_deterministic(systems, catalog):
    a = go(systems["rtx3060"], catalog, [fat_model(catalog)])
    b = go(systems["rtx3060"], catalog, [fat_model(catalog)])
    assert notes(cand(a, "test/fat-7b")) == notes(cand(b, "test/fat-7b"))


def test_r3_markdown_shows_the_note(tmp_path, capsys):
    import json

    from conftest import CATALOG_PATH, SYSTEM_PATHS, TUNING_PATH, load
    from scout.cli import main
    from test_cli import markdown_numbers_missing_from_json

    cat = load(CATALOG_PATH)
    cat["models"] = [fat_model(cat)]
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(cat))
    argv = ["plan", "--system", str(SYSTEM_PATHS["rtx3060"]), "--catalog", str(p), "--tuning", str(TUNING_PATH)]
    assert main([*argv, "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    msg = next(n["message"] for n in notes(doc["candidates"][0]) if n["code"] == CODE)
    assert main(argv) == 0
    md = capsys.readouterr().out
    assert msg in md
    assert markdown_numbers_missing_from_json(md, doc) == []
