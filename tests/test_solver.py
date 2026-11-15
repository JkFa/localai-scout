"""solve() contra o catálogo de TESTE e os 5 perfis sintéticos.

As expectativas vêm de PROMPT.md (M1), schemas/m1_interface.md e DECISIONS.md (Q6, Q7,
A17-A23). A aritmética de conferência é reescrita aqui, independente do código."""
from __future__ import annotations

import json
import math
import socket

import pytest

from conftest import ALL_TASKS, TUNING, assert_valid, deep, load, CATALOG_PATH, SYSTEM_PATHS
from scout.solver import Intent, solve

PROFILES = sorted(SYSTEM_PATHS)
TIER_RANK = {"confortavel": 0, "limite": 1, "inviavel": 2}


# ---------------------------------------------------------------- helpers
def run(system, catalog, tasks=ALL_TASKS, ctx=8192, commercial=False, kv="f16", batch=1):
    intent = Intent(tasks=tuple(tasks), ctx=ctx, commercial_use=commercial, batch=batch, kv_quant=kv)
    return solve(deep(system), deep(catalog), intent, dict(TUNING))


def cand(out, model_id):
    hits = [c for c in out["candidates"] if c["model"]["id"] == model_id]
    assert len(hits) <= 1, "mais de um candidato por modelo"
    return hits[0] if hits else None


def rej(out, model_id):
    hits = [r for r in out["rejected"] if r["model"]["id"] == model_id]
    return hits[0] if hits else None


def ids(rows):
    return {r["model"]["id"] for r in rows}


def models_by_id(catalog):
    return {m["id"]: m for m in catalog["models"]}


def tier_of(total, budget):  # regra inteira de m1_interface.md
    if total * 100 <= 80 * budget:
        return "confortavel"
    if total * 100 <= 95 * budget:
        return "limite"
    return "inviavel"


def kv_of(m, ctx, kv="f16", batch=1):
    bits = {"f16": 16, "q8_0": 8, "q4_0": 4}[kv]
    return 2 * m["n_layers"] * m["n_kv_heads"] * m["head_dim"] * ctx * bits * batch // 8


def buf_of(ctx):
    return 400_000_000 + 20_000 * ctx


def k_ovh_of(p):
    return 1.15 if p <= 3_000_000_000 else 1.10 if p <= 7_000_000_000 else 1.05


def floor_of(p):  # piso de qualidade, PROMPT.md
    for lim, v in ((1_500_000_000, 6.5), (4_000_000_000, 5.5), (9_000_000_000, 4.5), (30_000_000_000, 3.8)):
        if p <= lim:
            return v
    return 2.6


def expected_weights(m, q):
    if q["file_bytes"] is not None:
        return q["file_bytes"], 0
    return math.ceil(m["params_total"] * q["bpw"] / 8 * k_ovh_of(m["params_total"])), 1  # ±1


def sorted_ok(cands):
    """Ordem do solver: tier, depois tokens/s desc (None por último), depois model_id."""
    for a, b in zip(cands, cands[1:]):
        ra, rb = TIER_RANK[a["tier"]], TIER_RANK[b["tier"]]
        assert ra <= rb, "tier fora de ordem"
        if ra < rb:
            continue
        ta, tb = a["tokens_per_s_est"], b["tokens_per_s_est"]
        if ta is None:
            assert tb is None, "None deve vir por último"
            assert a["model"]["id"] <= b["model"]["id"]
        elif tb is not None:
            assert tb <= ta + 0.0051, "tokens/s deve ser decrescente"
            if ta == tb:
                assert a["model"]["id"] <= b["model"]["id"]


def check_candidates(out, system, catalog, intent_kw):
    """Invariantes aritméticos de todo candidato, recalculados de forma independente."""
    ms = models_by_id(catalog)
    ctx, kv, batch = intent_kw.get("ctx", 8192), intent_kw.get("kv", "f16"), intent_kw.get("batch", 1)
    pools = {p["id"]: p for p in out["pools"]}
    perf = system["cpu"]["cores_performance"]["value"]
    sorted_ok(out["candidates"])
    for c in out["candidates"]:
        m = ms[c["model"]["id"]]
        q = next(x for x in m["quants"] if x["name"] == c["quant"]["name"])
        w, tol = expected_weights(m, q)
        assert abs(c["weights_bytes"] - w) <= tol
        assert c["kv_bytes"] == kv_of(m, ctx, kv, batch)
        assert c["buffer_bytes"] == buf_of(ctx)
        assert c["total_bytes"] == c["weights_bytes"] + c["kv_bytes"] + c["buffer_bytes"]
        assert c["n_threads"] == perf  # nunca os núcleos lógicos
        assert c["tier"] != "inviavel"
        assert c["ttft_s_est"] is None  # A17
        assert c["tier"] == max((p["tier"] for p in c["by_pool"]), key=TIER_RANK.get)
        # regra 1 do projeto: nenhum candidato acima do orçamento do seu pool
        for p in c["by_pool"]:
            assert p["budget_bytes"] == pools[p["pool"]]["usable_bytes"]
            assert p["total_bytes"] <= p["budget_bytes"]
            assert p["total_bytes"] * 100 <= 95 * p["budget_bytes"]
            assert p["tier"] == tier_of(p["total_bytes"], p["budget_bytes"])
            assert p["total_bytes"] == p["weights_bytes"] + p["kv_bytes"] + p["buffer_bytes"]
        assert sum(p["total_bytes"] for p in c["by_pool"]) == c["total_bytes"]
        assert sum(p["weights_bytes"] for p in c["by_pool"]) == c["weights_bytes"]
        # piso de qualidade
        assert q["bpw"] >= floor_of(m["params_total"])
        if m["params_total"] <= 3_000_000_000:
            assert q["bpw"] >= 5.5
        # fallback => confidence low
        if q["file_bytes"] is None:
            assert c["confidence"] == "low" and "weights_fallback" in c["confidence_reasons"]
            assert c["provenance"]["weights_bytes"] == "estimated"
        else:
            assert "weights_fallback" not in c["confidence_reasons"]
            assert c["provenance"]["weights_bytes"] == "catalog"
        # parou na MAIOR quant viável: skipped_quants = quants de bpw maior (acima do piso), na ordem
        higher = sorted((x for x in m["quants"] if x["bpw"] > q["bpw"] and x["bpw"] >= floor_of(m["params_total"])),
                        key=lambda x: -x["bpw"])
        assert [s["name"] for s in c["skipped_quants"]] == [x["name"] for x in higher]
        assert all(s["reason_code"] != "below_quality_floor" for s in c["skipped_quants"])
        # placement coerente com n_gpu_layers
        if "unified" in pools and len(pools) == 1:
            assert [p["pool"] for p in c["by_pool"]] == ["unified"]
            continue
        bytes_camada = c["weights_bytes"] // m["n_layers"]
        if c["placement"] == "gpu_full":
            assert c["n_gpu_layers"] == m["n_layers"] and [p["pool"] for p in c["by_pool"]] == ["gpu"]
        elif c["placement"] == "cpu_only":
            assert c["n_gpu_layers"] == 0 and [p["pool"] for p in c["by_pool"]] == ["cpu"]
        else:
            assert c["placement"] == "hybrid" and q["format"] == "GGUF"
            assert 0 < c["n_gpu_layers"] < m["n_layers"]
            g, h = c["by_pool"]
            assert (g["pool"], h["pool"]) == ("gpu", "cpu")
            # n_gpu = floor((floor(0.95*vram_usable) - kv - buffer) / bytes_camada)
            exp = (95 * pools["gpu"]["usable_bytes"] // 100 - c["kv_bytes"] - c["buffer_bytes"]) // bytes_camada
            assert c["n_gpu_layers"] == min(exp, m["n_layers"])
            assert g["weights_bytes"] == c["n_gpu_layers"] * bytes_camada
            # KV inteiro e buffer ficam do lado GPU (A18)
            assert (g["kv_bytes"], g["buffer_bytes"]) == (c["kv_bytes"], c["buffer_bytes"])
            assert (h["kv_bytes"], h["buffer_bytes"]) == (0, 0)


# ---------------------------------------------------------------- schema + invariantes em todos os perfis
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("commercial", [False, True])
def test_output_valid_against_feasibility_schema(profile, commercial, systems, catalog):
    out = run(systems[profile], catalog, commercial=commercial)
    assert_valid("feasibility.schema.json", out)
    assert out["kind"] == "feasibility" and out["schema_version"] == 1
    assert out["inputs"]["intent"]["commercial_use"] is commercial


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("kw", [
    {}, {"ctx": 2048}, {"ctx": 16384}, {"ctx": 32768}, {"kv": "q8_0"}, {"kv": "q4_0"}, {"batch": 2},
])
def test_candidate_arithmetic_and_budget_invariants(profile, kw, systems, catalog):
    """Asserção obrigatória: nenhum candidato com total acima do orçamento do seu pool.
    Mais: kv/buffer/weights/tier/n_gpu recalculados de forma independente."""
    out = run(systems[profile], catalog, **kw)
    assert_valid("feasibility.schema.json", out)
    check_candidates(out, systems[profile], catalog, kw)


@pytest.mark.parametrize("profile", PROFILES)
def test_no_small_model_below_bpw_5_5(profile, systems, catalog):
    """Asserção obrigatória: nenhum modelo <= 3e9 recomendado abaixo de bpw 5.5."""
    for commercial in (False, True):
        out = run(systems[profile], catalog, commercial=commercial)
        for c in out["candidates"]:
            if c["model"]["params_total"] <= 3_000_000_000:
                assert c["quant"]["bpw"] >= 5.5, c["model"]["id"]


def test_quality_floor_beats_fit(systems, catalog):
    """RAM tão curta que só o Q4_K_M (bpw 4.9, abaixo do piso 6.5 do 1B) caberia: o 1B NÃO entra."""
    s = deep(systems["cpu8"])
    s["memory"]["ram_total_bytes"]["value"] = 4_700_000_000   # usable = 1_700_000_000
    # Q4_K_M: 735e6 + 268_435_456 + 563_840_000 = 1_567_275_456 <= 0.95*1.7e9 = 1.615e9 (CABERIA)
    # Q6_K  : 984e6 + 268_435_456 + 563_840_000 = 1_816_275_456  > 1.615e9      (não cabe)
    out = run(s, catalog)
    assert cand(out, "test/tiny-1b") is None
    r = rej(out, "test/tiny-1b")
    assert r is not None and r["reason_code"] in ("does_not_fit", "below_quality_floor", "no_usable_quant")
    assert any(c["quant"]["bpw"] < 5.5 for c in out["candidates"]) is False
    assert cand(out, "test/embed-110m") is not None  # o que cabe acima do piso segue recomendado
    assert_valid("feasibility.schema.json", out)


# ---------------------------------------------------------------- perfis (valores calculados à mão)
def test_profile1_cpu8(systems, catalog):
    out = run(systems["cpu8"], catalog)
    # pool cpu: 8_589_934_592 - 2e9 - 1e9 = 5_589_934_592 ; sem GPU, sem pool gpu
    assert [(p["id"], p["usable_bytes"]) for p in out["pools"]] == [("cpu", 5_589_934_592)]
    # perfil 1: nenhuma difusão e ao menos um embeddings (asserção obrigatória)
    assert not [c for c in out["candidates"] if "diffusion" in c["model"]["tasks"]]
    assert [c for c in out["candidates"] if "embeddings" in c["model"]["tasks"]]
    assert ids(out["candidates"]) == {"test/tiny-1b", "test/embed-110m"}
    assert all(c["placement"] == "cpu_only" and c["n_gpu_layers"] == 0 for c in out["candidates"])

    tiny = cand(out, "test/tiny-1b")
    # Q8_0 sem file_bytes: ceil(1.2e9*8.5/8*1.15) = 1_466_250_000 (±1) => confidence low
    assert abs(tiny["weights_bytes"] - 1_466_250_000) <= 1
    assert tiny["quant"]["name"] == "Q8_0" and tiny["confidence"] == "low"
    assert "weights_fallback" in tiny["confidence_reasons"]
    assert tiny["kv_bytes"] == 268_435_456 and tiny["buffer_bytes"] == 563_840_000
    assert tiny["tier"] == "confortavel"       # 2_298_525_456 <= 0.8 * 5_589_934_592 = 4_471_947_673
    assert tiny["n_threads"] == 4              # núcleos de performance; lógicos seriam 8
    # t = 91_640_625*16 / (21.328e9 * 0.6) = 0.114581 s -> 8.7276 tok/s
    assert tiny["tokens_per_s_est"] == pytest.approx(8.7276, rel=1e-3)

    emb = cand(out, "test/embed-110m")
    assert emb["quant"]["name"] == "F16"       # maior quant viável, não a mais rápida
    assert emb["weights_bytes"] == 220_000_000
    assert emb["tokens_per_s_est"] == pytest.approx(58.17, rel=1e-3)   # 220e6/(21.328e9*0.6) = 0.017192 s

    # 7B: nem o Q4_K_M cabe. 4_410_000_000 + 1_073_741_824 + 563_840_000 = 6_047_581_824
    # falta = 6_047_581_824 - floor(0.95 * 5_589_934_592 = 5_310_437_862.4) = 737_143_962
    r7 = rej(out, "test/dense-7b")
    assert r7["reason_code"] == "does_not_fit"
    assert r7["closest"]["quant"]["name"] == "Q4_K_M" and r7["closest"]["placement"] == "cpu_only"
    assert r7["closest"]["total_bytes"] == 6_047_581_824
    assert r7["missing_bytes"] == 737_143_962
    # Q3_K_M (bpw 3.9 < piso 4.5) é pulada com motivo, nunca escolhida
    assert {"format": "GGUF", "name": "Q3_K_M", "reason_code": "below_quality_floor"} in r7["skipped_quants"]

    # MoE: melhor tentativa Q3_K_M: 14_900_000_000 + 805_306_368 + 563_840_000 = 16_269_146_368
    # falta = 16_269_146_368 - 5_310_437_862 = 10_958_708_506 ; Q2_K abaixo do piso 3.8
    rm = rej(out, "test/moe-30b-a3b")
    assert rm["reason_code"] == "does_not_fit" and rm["missing_bytes"] == 10_958_708_506
    assert rm["closest"]["quant"]["name"] == "Q3_K_M"
    assert {"format": "GGUF", "name": "Q2_K", "reason_code": "below_quality_floor"} in rm["skipped_quants"]


def test_profile2_rtx3060(systems, catalog):
    out = run(systems["rtx3060"], catalog)
    pools = {p["id"]: p for p in out["pools"]}
    # vram_usable = 12_884_901_888 - (12_884_901_888-12_400_000_000) - 600e6 = 11_800_000_000
    assert pools["gpu"]["usable_bytes"] == 11_800_000_000
    assert pools["gpu"]["reserved_bytes"] == 1_084_901_888          # display 484_901_888 + 600e6
    assert pools["cpu"]["usable_bytes"] == 14_179_869_184           # 17_179_869_184 - 3e9
    assert pools["cpu"]["reserved_bytes"] == 3_000_000_000

    # perfil 2: n_gpu_layers == n_layers para o 7B na melhor quant que couber (asserção obrigatória)
    d = cand(out, "test/dense-7b")
    assert d["quant"]["name"] == "Q8_0"                      # maior quant: cabe inteira
    assert d["placement"] == "gpu_full" and d["n_gpu_layers"] == 32 == 32
    # 7_650_000_000 + 1_073_741_824 + 563_840_000 = 9_287_581_824 <= 0.8 * 11.8e9 = 9.44e9
    assert d["total_bytes"] == 9_287_581_824 and d["tier"] == "confortavel"
    assert d["skipped_quants"] == []
    # 7.65e9 / (360e9 * 0.75) = 0.0283333 s -> 35.2941 tok/s
    assert d["tokens_per_s_est"] == pytest.approx(35.2941, rel=1e-3)

    # MoE: Q6_K (24.6e9) não cabe em nenhum placement; Q4_K_M híbrido com 25 camadas na GPU
    m = cand(out, "test/moe-30b-a3b")
    assert m["quant"]["name"] == "Q4_K_M" and m["placement"] == "hybrid" and m["n_gpu_layers"] == 25
    assert m["skipped_quants"] == [{"format": "GGUF", "name": "Q6_K", "reason_code": "does_not_fit"}]
    g, h = m["by_pool"]
    # gpu: 25 * 381_250_000 = 9_531_250_000 ; + kv 805_306_368 + buffer 563_840_000 = 10_900_396_368
    assert (g["weights_bytes"], g["total_bytes"]) == (9_531_250_000, 10_900_396_368)
    assert g["tier"] == "limite"           # > 0.8*11.8e9 mas <= 0.95*11.8e9 = 11.21e9 (Q6: offload sai "limite")
    # cpu: 18.3e9 - 9_531_250_000 = 8_768_750_000 <= 0.8 * 14_179_869_184
    assert (h["weights_bytes"], h["total_bytes"], h["tier"]) == (8_768_750_000, 8_768_750_000, "confortavel")
    assert m["tier"] == "limite"           # pior dos dois pools (A18)
    # MoE: só a fração ativa é lida por token => 31.1777 tok/s (usar params_total daria ~3,1)
    assert m["tokens_per_s_est"] == pytest.approx(31.1777, rel=1e-3)

    assert ids(out["candidates"]) == {"test/tiny-1b", "test/dense-7b", "test/moe-30b-a3b", "test/embed-110m"}
    assert cand(out, "test/tiny-1b")["placement"] == "gpu_full"
    assert cand(out, "test/tiny-1b")["n_gpu_layers"] == 16
    assert cand(out, "test/embed-110m")["quant"]["name"] == "F16"


def test_profile3_ddr5_moe_viable(systems, catalog):
    out = run(systems["ddr5_64"], catalog)
    # ram_usable = 68_719_476_736 - 3e9 = 65_719_476_736 ; sem GPU
    assert [(p["id"], p["usable_bytes"]) for p in out["pools"]] == [("cpu", 65_719_476_736)]
    m = cand(out, "test/moe-30b-a3b")
    assert m is not None, "perfil 3: o MoE tem de aparecer como viável"   # asserção obrigatória
    assert m["quant"]["name"] == "Q6_K" and m["skipped_quants"] == []     # maior quant do modelo
    assert m["placement"] == "cpu_only" and m["n_gpu_layers"] == 0 and m["tier"] == "confortavel"
    # 24.6e9 + 805_306_368 + 563_840_000 = 25_969_146_368 <= 0.8 * 65_719_476_736
    assert m["total_bytes"] == 25_969_146_368
    # bytes ativos por camada 51_250_000: t = 2_460_000_000/(89.6e9*0.6) -> 21.8537 tok/s
    assert m["tokens_per_s_est"] == pytest.approx(21.8537, rel=1e-3)
    assert m["n_threads"] == 16
    d = cand(out, "test/dense-7b")
    # denso Q8_0: 7.65e9/(89.6e9*0.6) = 0.14230 s -> 7.0275 tok/s
    assert d["quant"]["name"] == "Q8_0" and d["tokens_per_s_est"] == pytest.approx(7.0275, rel=1e-3)
    # o MoE de 30B com 3B ativas é ~3x mais rápido que o denso de 7B: prova de que a fração ativa entra
    assert m["tokens_per_s_est"] > 2.5 * d["tokens_per_s_est"]


def test_profile4_apple_unified_single_pool(systems, catalog):
    out = run(systems["apple24"], catalog)
    # Q7: orçamento = 50% da RAM total = 25_769_803_776 // 2 = 12_884_901_888, um pool só
    assert [(p["id"], p["usable_bytes"]) for p in out["pools"]] == [("unified", 12_884_901_888)]
    for c in out["candidates"]:
        assert [p["pool"] for p in c["by_pool"]] == ["unified"]
        assert c["tokens_per_s_est"] is not None and c["tokens_per_s_est"] > 0
        assert c["n_threads"] == 8                     # núcleos de performance, não 12
    d = cand(out, "test/dense-7b")
    # 9_287_581_824 <= 0.8 * 12_884_901_888 = 10_307_921_510
    assert d["quant"]["name"] == "Q8_0" and d["tier"] == "confortavel"
    # MoE: Q3_K_M 16_269_146_368 - floor(0.95*12_884_901_888 = 12_240_656_793.6) = 4_028_489_575
    rm = rej(out, "test/moe-30b-a3b")
    assert rm["reason_code"] == "does_not_fit" and rm["missing_bytes"] == 4_028_489_575
    assert rm["closest"]["quant"]["name"] == "Q3_K_M"


def test_profile5_rx7800xt_rocm(systems, catalog):
    out = run(systems["rx7800xt"], catalog)
    pools = {p["id"]: p for p in out["pools"]}
    # 17_179_869_184 - (17_179_869_184 - 16_600_000_000) - 600e6 = 16_000_000_000
    assert pools["gpu"]["usable_bytes"] == 16_000_000_000
    assert pools["cpu"]["usable_bytes"] == 31_359_738_368          # 34_359_738_368 - 3e9
    d = cand(out, "test/dense-7b")
    assert d["placement"] == "gpu_full" and d["n_gpu_layers"] == 32 and d["quant"]["name"] == "Q8_0"
    assert d["tokens_per_s_est"] == pytest.approx(61.1765, rel=1e-3)  # 7.65e9/(624e9*0.75)=0.016346 s
    # MoE Q6_K híbrido: floor(0.95*16e9)=15.2e9 - 805_306_368 - 563_840_000 = 13_830_853_632
    # 13_830_853_632 / 512_500_000 = 26.987 -> 26 (27 camadas pediriam 13_837_500_000)
    m = cand(out, "test/moe-30b-a3b")
    assert m["quant"]["name"] == "Q6_K" and m["placement"] == "hybrid" and m["n_gpu_layers"] == 26
    assert m["by_pool"][0]["weights_bytes"] == 26 * 512_500_000
    # gpu = 13_325_000_000 + 1_369_146_368 = 14_694_146_368 : > 0.8*16e9, <= 0.95*16e9 => limite
    assert m["by_pool"][0]["tier"] == "limite" and m["by_pool"][1]["tier"] == "confortavel"
    # 26*51_250_000/(624e9*0.75) + 22*51_250_000/(96e9*0.6) = 0.0028472 + 0.0195747 -> 44.5993
    assert m["tokens_per_s_est"] == pytest.approx(44.5993, rel=1e-3)


def test_swap_is_only_a_warning(systems, catalog):
    """Regra 6: nunca recomendar aumentar swap. Há aviso, e swap não entra em nenhum orçamento."""
    out = run(systems["cpu8"], catalog)                      # swap de 4 GiB na fixture
    assert "swap_not_a_fix" in {w["code"] for w in out["warnings"]}
    assert out["pools"][0]["usable_bytes"] == 5_589_934_592  # nada de swap somado
    low = deep(systems["cpu8"])
    low["memory"]["swap_total_bytes"]["value"] = 0
    assert run(low, catalog)["pools"] == out["pools"]


# ---------------------------------------------------------------- licença comercial
@pytest.mark.parametrize("profile", PROFILES)
def test_commercial_never_primary_for_noncommercial(profile, systems, catalog):
    """Asserção obrigatória: com --commercial, nenhum modelo não-comercial como principal."""
    ms = models_by_id(catalog)
    base = run(systems[profile], catalog, commercial=False)
    out = run(systems[profile], catalog, commercial=True)
    for c in out["candidates"]:
        ok = ms[c["model"]["id"]]["commercial_ok"] is True
        assert c["blocked"] is (not ok)
        if not ok:
            assert c["primary_for"] == [] and c["blocked_reason"] == "non_commercial"
        else:
            assert c["blocked_reason"] is None
    for c in base["candidates"]:     # sem --commercial nada é bloqueado
        assert c["blocked"] is False and c["blocked_reason"] is None
    # principal = primeiro candidato NÃO bloqueado que serve à tarefa, na ordem do solver
    for out_ in (base, out):
        for task in ALL_TASKS:
            expected = next((c for c in out_["candidates"] if not c["blocked"] and task in c["model"]["tasks"]), None)
            holders = [c for c in out_["candidates"] if task in c["primary_for"]]
            assert [h["model"]["id"] for h in holders] == ([expected["model"]["id"]] if expected else [])


def test_commercial_blocks_noncommercial_but_keeps_it_in_report(systems, catalog):
    out = run(systems["rtx3060"], catalog, commercial=True)
    tiny = cand(out, "test/tiny-1b")                     # licença test-noncommercial, o mais rápido de chat
    assert tiny is not None, "candidato bloqueado continua no relatório"
    assert tiny["blocked"] is True and tiny["blocked_reason"] == "non_commercial"
    chat_primary = [c for c in out["candidates"] if "chat" in c["primary_for"]]
    assert [c["model"]["id"] for c in chat_primary] in (["test/dense-7b"], ["test/moe-30b-a3b"])
    # sem --commercial o 1B não está bloqueado
    assert cand(run(systems["rtx3060"], catalog), "test/tiny-1b")["blocked"] is False
    assert_valid("feasibility.schema.json", out)


def test_unknown_license_is_blocked_under_commercial(systems, catalog):
    """Q4: commercial_ok null => bloqueado sob --commercial (license_unknown), sem tornar incompleto."""
    cat = deep(catalog)
    next(m for m in cat["models"] if m["id"] == "test/dense-7b")["commercial_ok"] = None
    out = run(systems["rtx3060"], cat, commercial=True)
    d = cand(out, "test/dense-7b")
    assert d["blocked"] is True and d["blocked_reason"] == "license_unknown" and d["primary_for"] == []
    assert cand(run(systems["rtx3060"], cat), "test/dense-7b")["blocked"] is False


# ---------------------------------------------------------------- rejeições e filtros
def test_mla_rejected_with_reason(systems, catalog):
    cat = deep(catalog)
    next(m for m in cat["models"] if m["id"] == "test/dense-7b")["kv_model"] = "mla"
    out = run(systems["rtx3060"], cat)
    assert cand(out, "test/dense-7b") is None
    r = rej(out, "test/dense-7b")
    assert r["reason_code"] == "kv_model_unsupported" and r["message"].strip()
    assert r["closest"] is None and r["missing_bytes"] is None   # rejeição anterior ao cálculo
    assert_valid("feasibility.schema.json", out)


def test_ctx_above_ctx_max_rejected_with_reason(systems, catalog):
    out = run(systems["rtx3060"], catalog, ctx=16384)            # embed-110m tem ctx_max 8192
    assert cand(out, "test/embed-110m") is None
    r = rej(out, "test/embed-110m")
    assert r["reason_code"] == "ctx_exceeds_max" and r["message"].strip()
    assert r["closest"] is None and r["missing_bytes"] is None
    assert cand(out, "test/dense-7b") is not None                # ctx_max 32768: segue valendo
    assert_valid("feasibility.schema.json", out)


def test_ctx_equal_to_ctx_max_is_accepted(systems, catalog):
    out = run(systems["rtx3060"], catalog, ctx=8192)             # == ctx_max do embed
    assert cand(out, "test/embed-110m") is not None


def test_ctx_above_every_ctx_max_rejects_everything(systems, catalog):
    out = run(systems["rtx3060"], catalog, ctx=40000)
    assert out["candidates"] == []
    assert {r["reason_code"] for r in out["rejected"]} == {"ctx_exceeds_max"}
    assert ids(out["rejected"]) == {m["id"] for m in catalog["models"]}
    assert_valid("feasibility.schema.json", out)


def test_incomplete_models_never_appear(systems, catalog):
    """complete:false nunca aparece (regra 3): pulado em silêncio, só contagem em excluded."""
    cat = deep(catalog)
    for m in cat["models"]:
        if m["id"] in ("test/embed-110m", "test/tiny-1b"):
            m["complete"], m["ctx_max"], m["incomplete_fields"] = False, None, ["ctx_max"]
    for profile in PROFILES:
        out = run(systems[profile], cat)
        shown = ids(out["candidates"]) | ids(out["rejected"])
        assert not shown & {"test/embed-110m", "test/tiny-1b"}
        assert out["excluded"]["incomplete"] == 2
        assert_valid("feasibility.schema.json", out)


@pytest.mark.parametrize("tasks,survivors,mismatch", [
    (("embeddings",), {"test/embed-110m"}, 3),
    (("code",), {"test/dense-7b", "test/moe-30b-a3b"}, 2),
    (("chat", "code"), {"test/tiny-1b", "test/dense-7b", "test/moe-30b-a3b"}, 1),
])
def test_task_filter(tasks, survivors, mismatch, systems, catalog):
    out = run(systems["rtx3060"], catalog, tasks=tasks)
    assert ids(out["candidates"]) == survivors
    assert out["rejected"] == []
    assert out["excluded"]["task_mismatch"] == mismatch
    assert out["inputs"]["intent"]["tasks"] == list(tasks)


def test_stops_at_largest_viable_quant(systems, catalog):
    """PARA na maior quant viável: qualidade máxima que cabe, não a menor que cabe."""
    out = run(systems["rtx3060"], catalog)
    # 7B: Q8_0 e Q6_K caberiam; escolhe Q8_0
    assert cand(out, "test/dense-7b")["quant"]["name"] == "Q8_0"
    # embeddings: F16 e Q8_0 caberiam; escolhe F16
    assert cand(out, "test/embed-110m")["quant"]["name"] == "F16"
    # MoE: Q4_K_M é a maior que cabe, e registra a(s) que foram tentadas antes
    assert cand(out, "test/moe-30b-a3b")["quant"]["name"] == "Q4_K_M"
    # menos memória livre => cai para a quant seguinte, nunca pula para uma menor que a viável
    s = deep(systems["rtx3060"])
    s["gpus"]["value"][0]["vram_free_bytes"]["value"] = 9_600_000_000   # usable = 9_000_000_000
    out2 = run(s, catalog)
    d = cand(out2, "test/dense-7b")
    # Q8_0 gpu_full: 9_287_581_824 > 0.95*9e9 = 8.55e9 ; híbrido Q8_0: gpu 0.95*9e9 -> poucas camadas,
    # cpu pool suficiente => Q8_0 ainda viável em híbrido (mantém a maior quant)
    # n_gpu = floor((8_550_000_000 - 1_073_741_824 - 563_840_000) / 239_062_500) = floor(28.91) = 28
    assert d["quant"]["name"] == "Q8_0" and d["placement"] == "hybrid" and d["n_gpu_layers"] == 28
    check_candidates(out2, s, catalog, {})


def test_kv_quant_and_batch_change_memory(systems, catalog):
    base = cand(run(systems["rtx3060"], catalog), "test/dense-7b")
    q8 = cand(run(systems["rtx3060"], catalog, kv="q8_0"), "test/dense-7b")
    q4 = cand(run(systems["rtx3060"], catalog, kv="q4_0"), "test/dense-7b")
    b2 = cand(run(systems["rtx3060"], catalog, batch=2), "test/dense-7b")
    assert (base["kv_bytes"], q8["kv_bytes"], q4["kv_bytes"], b2["kv_bytes"]) == (
        1_073_741_824, 536_870_912, 268_435_456, 2_147_483_648)
    # batch 2: 7_650_000_000 + 2_147_483_648 + 563_840_000 = 10_361_325_568 > 9.44e9 => limite
    assert b2["tier"] == "limite"


# ---------------------------------------------------------------- campos de hardware ausentes (A23)
def test_gpu_without_vram_free_forms_no_pool(systems, catalog):
    s = deep(systems["rtx3060"])
    s["gpus"]["value"][0]["vram_free_bytes"] = {"value": None, "source": "fixture", "confidence": "low"}
    out = run(s, catalog)
    assert [p["id"] for p in out["pools"]] == ["cpu"]
    assert "hardware_field_missing" in {w["code"] for w in out["warnings"]}
    assert all(c["placement"] == "cpu_only" for c in out["candidates"])
    assert_valid("feasibility.schema.json", out)


def test_vram_usable_clamped_to_zero_with_warning(systems, catalog):
    s = deep(systems["rtx3060"])
    # free 500e6 => display 12_384_901_888 ; usable = 12_884_901_888 - 12_384_901_888 - 600e6 = -100e6 -> 0
    s["gpus"]["value"][0]["vram_free_bytes"]["value"] = 500_000_000
    out = run(s, catalog)
    assert "vram_usable_clamped" in {w["code"] for w in out["warnings"]}
    for p in out["pools"]:
        if p["id"] == "gpu":
            assert p["usable_bytes"] == 0
    for c in out["candidates"]:
        assert all(bp["pool"] != "gpu" or bp["weights_bytes"] == 0 for bp in c["by_pool"])
    assert_valid("feasibility.schema.json", out)


def test_missing_ram_total_fails_loudly(systems, catalog):
    s = deep(systems["rtx3060"])
    s["memory"]["ram_total_bytes"] = {"value": None, "source": "fixture", "confidence": "low"}
    with pytest.raises(Exception):
        run(s, catalog)


def test_unknown_bandwidth_gives_none_tokens_sorted_last(systems, catalog):
    s = deep(systems["ddr5_64"])
    s["memory"]["bandwidth_bytes_per_s"] = {"value": None, "source": "none", "confidence": "low"}
    out = run(s, catalog)
    assert out["candidates"], "sem banda o modelo continua viável (cabe na memória)"
    assert all(c["tokens_per_s_est"] is None for c in out["candidates"])
    assert all(c["ttft_s_est"] is None for c in out["candidates"])
    sorted_ok(out["candidates"])
    ordered = [(TIER_RANK[c["tier"]], c["model"]["id"]) for c in out["candidates"]]
    assert ordered == sorted(ordered)            # só sobra tier, depois model_id asc
    assert_valid("feasibility.schema.json", out)


def test_none_tokens_come_after_known_in_same_tier(systems, catalog):
    s = deep(systems["rtx3060"])
    s["memory"]["bandwidth_bytes_per_s"] = {"value": None, "source": "none", "confidence": "low"}
    out = run(s, catalog)
    sorted_ok(out["candidates"])                 # gpu_full tem tokens/s; híbrido (usa a RAM) fica None
    by_tier = {}
    for c in out["candidates"]:
        by_tier.setdefault(c["tier"], []).append(c["tokens_per_s_est"])
    for vals in by_tier.values():
        seen_none = False
        for v in vals:
            seen_none = seen_none or v is None
            assert not (seen_none and v is not None)


def test_multi_gpu_uses_largest_vram_and_warns(systems, catalog):
    s = deep(systems["rtx3060"])
    big = deep(s["gpus"]["value"][0])
    big["index"] = 1
    big["vram_total_bytes"]["value"] = 25_769_803_776
    big["vram_free_bytes"]["value"] = 25_000_000_000
    s["gpus"]["value"].append(big)
    out = run(s, catalog)
    assert "multi_gpu_ignored" in {w["code"] for w in out["warnings"]}
    g = next(p for p in out["pools"] if p["id"] == "gpu")
    assert g["gpu_index"] == 1
    # usable = total - (total - free) - 600e6 = free - 600e6 = 25_000_000_000 - 600_000_000
    assert g["usable_bytes"] == 24_400_000_000
    assert_valid("feasibility.schema.json", out)


def test_multi_gpu_tie_picks_lowest_index(systems, catalog):
    """Empate de vram_total: vence o menor `index` (DECISIONS), mesmo com a lista fora de ordem."""
    s = deep(systems["rtx3060"])
    s["gpus"]["value"][0]["index"] = 3
    twin = deep(s["gpus"]["value"][0])
    twin["index"] = 4
    s["gpus"]["value"].insert(0, twin)           # posição 0 = index 4 ; posição 1 = index 3
    g = next(p for p in run(s, catalog)["pools"] if p["id"] == "gpu")
    assert g["gpu_index"] in (1, 3)              # posição na lista (1) ou campo index (3): ambos = a GPU index 3


# ---------------------------------------------------------------- determinismo e pureza
@pytest.mark.parametrize("profile", PROFILES)
def test_solve_is_deterministic_and_does_not_mutate_inputs(profile, systems, catalog):
    s, c = deep(systems[profile]), deep(catalog)
    intent = Intent(tasks=ALL_TASKS, ctx=8192)
    a = solve(s, c, intent, dict(TUNING))
    b = solve(s, c, intent, dict(TUNING))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert s == systems[profile] and c == catalog     # solver não muta as entradas


@pytest.mark.parametrize("profile", PROFILES)
def test_solver_never_touches_the_network(profile, systems, catalog, monkeypatch):
    """Regra 5: o solver nunca toca a rede."""
    def boom(*a, **k):
        raise AssertionError("acesso à rede durante solve()")

    import urllib.request

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = run(systems[profile], catalog)
    assert out["candidates"] or out["rejected"]


def test_inputs_block_describes_the_run(systems, catalog):
    out = run(systems["rtx3060"], catalog, ctx=4096, commercial=True, kv="q8_0", batch=2)
    assert out["inputs"]["intent"] == {"tasks": list(ALL_TASKS), "ctx": 4096, "commercial_use": True,
                                       "batch": 2, "kv_quant": "q8_0"}
    assert out["inputs"]["tuning"]["eta_gpu"] == 0.75 and out["inputs"]["tuning"]["eta_cpu"] == 0.6
    assert out["inputs"]["catalog"]["model_count"] == 4
    assert out["inputs"]["system"] == load(SYSTEM_PATHS["rtx3060"])


def test_candidates_cite_only_catalog_models_with_source_url(systems, catalog):
    """Regra 2: nenhum modelo citado fora do catálogo, sempre com source_url."""
    ms = models_by_id(catalog)
    for profile in PROFILES:
        out = run(systems[profile], catalog)
        for row in out["candidates"] + out["rejected"]:
            assert row["model"]["id"] in ms
            assert row["model"]["source_url"] == ms[row["model"]["id"]]["source_url"]
            assert row["model"]["source_url"].startswith("https://")
