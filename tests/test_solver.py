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


def bytes_keys(c):
    """Critério de ordem da F4 com tokens/s null, recalculado só do candidato:
    (bytes_cpu_por_token, bytes_por_token). bytes_por_token = weights*active//total;
    por camada = // n_layers; no CPU = camadas fora da GPU."""
    m = c["model"]
    bpt = c["weights_bytes"] * m["params_active"] // m["params_total"]
    per_layer = bpt // m["n_layers"]
    return per_layer * (m["n_layers"] - c["n_gpu_layers"]), bpt


def sorted_ok(cands):
    """Ordem do solver (F4): tier, tokens/s desc (None por último), depois
    bytes_cpu_por_token asc, bytes_por_token asc, model_id asc (só entre tokens/s None)."""
    for a, b in zip(cands, cands[1:]):
        ra, rb = TIER_RANK[a["tier"]], TIER_RANK[b["tier"]]
        assert ra <= rb, "tier fora de ordem"
        if ra < rb:
            continue
        ta, tb = a["tokens_per_s_est"], b["tokens_per_s_est"]
        if ta is None:
            assert tb is None, "None deve vir por último"
            assert (*bytes_keys(a), a["model"]["id"]) <= (*bytes_keys(b), b["model"]["id"])
        elif tb is not None:
            assert tb <= ta + 0.0051, "tokens/s deve ser decrescente"


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
        # embeddings como única tarefa avaliada: ctx efetivo = min(intent.ctx, ctx_max) (DECISIONS)
        only_emb = set(m["tasks"]) & set(intent_kw.get("tasks", ALL_TASKS)) == {"embeddings"}
        eff = min(ctx, m["ctx_max"]) if only_emb else ctx
        if ctx > m["ctx_max"]:
            assert only_emb and c.get("ctx_effective") == m["ctx_max"]
        assert c.get("ctx_effective", eff) == eff
        assert c["kv_bytes"] == kv_of(m, eff, kv, batch)
        assert c["buffer_bytes"] == buf_of(eff)
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


def assert_embeddings_no_decode(c):
    """Candidato cuja única tarefa avaliada é embeddings: sem tokens/s de decode."""
    assert c["tokens_per_s_est"] is None
    assert "decode_not_applicable" in c["confidence_reasons"]
    for bad in ("ranked_by_bytes", "eta_default", "bandwidth_estimated", "bandwidth_unknown"):
        assert bad not in c["confidence_reasons"], bad
    assert "decode_not_applicable" in {n["code"] for n in c.get("notes", [])}


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
    # decode no fim da janela de 8192 (posição 7680): KV lido = 2*16*8*64*7680*2 = 251_658_240
    # t = (91_640_625*16 + 251_658_240) / (21.328e9 * 0.6) = 0.134243 s -> 7.4491 tok/s
    assert tiny["tokens_per_s_est"] == pytest.approx(7.4491, rel=1e-3)

    emb = cand(out, "test/embed-110m")
    assert emb["quant"]["name"] == "F16"       # maior quant viável, não a mais rápida
    assert emb["weights_bytes"] == 220_000_000
    # embeddings não fazem decode: tokens/s null + nota/motivo decode_not_applicable (PROMPT.md, Throughput)
    assert_embeddings_no_decode(emb)

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
    # KV na posição 7680: 2*32*8*128*7680*2 = 1_006_632_960
    # (7.65e9 + 1_006_632_960) / (360e9 * 0.75) = 0.0320616 s -> 31.1900 tok/s
    assert d["tokens_per_s_est"] == pytest.approx(31.1900, rel=1e-3)

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
    # MoE: só a fração ativa dos PESOS é lida (38_125_000/camada); KV inteiro, 2*48*4*128*7680*2 =
    # 754_974_720 (15_728_640/camada), lido onde a camada está:
    # (25*53_853_640)/(360e9*0.75) + (23*53_853_640)/(51.2e9*0.6) -> 22.0719 tok/s
    assert m["tokens_per_s_est"] == pytest.approx(22.0719, rel=1e-3)

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
    # bytes ativos por camada 51_250_000 + KV 754_974_720: t = 3_214_974_720/(89.6e9*0.6) -> 16.7217 tok/s
    assert m["tokens_per_s_est"] == pytest.approx(16.7217, rel=1e-3)
    assert m["n_threads"] == 16
    d = cand(out, "test/dense-7b")
    # denso Q8_0: (7.65e9 + 1_006_632_960)/(89.6e9*0.6) = 0.16102 s -> 6.2103 tok/s
    assert d["quant"]["name"] == "Q8_0" and d["tokens_per_s_est"] == pytest.approx(6.2103, rel=1e-3)
    # o MoE de 30B com 3B ativas é bem mais rápido que o denso de 7B: prova de que a fração ativa entra
    assert m["tokens_per_s_est"] > 2.5 * d["tokens_per_s_est"]


def test_profile4_apple_unified_single_pool(systems, catalog):
    out = run(systems["apple24"], catalog)
    # Q7: orçamento = 50% da RAM total = 25_769_803_776 // 2 = 12_884_901_888, um pool só
    assert [(p["id"], p["usable_bytes"]) for p in out["pools"]] == [("unified", 12_884_901_888)]
    for c in out["candidates"]:
        assert [p["pool"] for p in c["by_pool"]] == ["unified"]
        if c["model"]["tasks"] == ["embeddings"]:
            assert_embeddings_no_decode(c)
        else:
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
    assert d["tokens_per_s_est"] == pytest.approx(54.0626, rel=1e-3)  # (7.65e9+1_006_632_960)/(624e9*0.75)
    # MoE Q6_K híbrido: floor(0.95*16e9)=15.2e9 - 805_306_368 - 563_840_000 = 13_830_853_632
    # 13_830_853_632 / 512_500_000 = 26.987 -> 26 (27 camadas pediriam 13_837_500_000)
    m = cand(out, "test/moe-30b-a3b")
    assert m["quant"]["name"] == "Q6_K" and m["placement"] == "hybrid" and m["n_gpu_layers"] == 26
    assert m["by_pool"][0]["weights_bytes"] == 26 * 512_500_000
    # gpu = 13_325_000_000 + 1_369_146_368 = 14_694_146_368 : > 0.8*16e9, <= 0.95*16e9 => limite
    assert m["by_pool"][0]["tier"] == "limite" and m["by_pool"][1]["tier"] == "confortavel"
    # por camada: 51_250_000 de pesos ativos + 15_728_640 de KV (posição 7680) = 66_978_640
    # 26*66_978_640/(624e9*0.75) + 22*66_978_640/(96e9*0.6) -> 34.1260
    assert m["tokens_per_s_est"] == pytest.approx(34.1260, rel=1e-3)


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
    """Modelo avaliado para outra tarefa (chat) continua rejeitado com ctx_exceeds_max."""
    cat = deep(catalog)
    next(m for m in cat["models"] if m["id"] == "test/tiny-1b")["ctx_max"] = 4096
    out = run(systems["rtx3060"], cat, ctx=8192)
    assert cand(out, "test/tiny-1b") is None
    r = rej(out, "test/tiny-1b")
    assert r["reason_code"] == "ctx_exceeds_max" and r["message"].strip()
    assert r["closest"] is None and r["missing_bytes"] is None
    assert cand(out, "test/dense-7b") is not None                # ctx_max 32768: segue valendo
    assert_valid("feasibility.schema.json", out)


def test_ctx_equal_to_ctx_max_is_accepted(systems, catalog):
    out = run(systems["rtx3060"], catalog, ctx=8192)             # == ctx_max do embed
    c = cand(out, "test/embed-110m")
    assert c is not None and c.get("ctx_effective", 8192) == 8192
    cat = deep(catalog)
    next(m for m in cat["models"] if m["id"] == "test/tiny-1b")["ctx_max"] = 8192
    assert cand(run(systems["rtx3060"], cat, ctx=8192), "test/tiny-1b") is not None


def test_embeddings_ctx_above_ctx_max_uses_effective_ctx(systems, catalog):
    """Embeddings (única tarefa): ctx 16384 > ctx_max 8192 vira candidato com ctx_effective 8192."""
    out = run(systems["rtx3060"], catalog, ctx=16384)
    c = cand(out, "test/embed-110m")
    assert c is not None and rej(out, "test/embed-110m") is None
    assert c["ctx_effective"] == 8192
    # KV com ctx 8192 (não 16384): 2*12*12*64*8192*2 bytes = 301_989_888 ; buffer 400e6 + 20e3*8192
    assert c["kv_bytes"] == 301_989_888 and c["buffer_bytes"] == 563_840_000
    assert c["total_bytes"] == 220_000_000 + 301_989_888 + 563_840_000     # F16 = 220_000_000
    assert c["quant"]["name"] == "F16"
    assert c["model"]["id"] in ids(out["candidates"])
    # os demais modelos usam o ctx pedido (16384): 7B kv = 2*32*8*128*16384*2 = 2_147_483_648
    d = cand(out, "test/dense-7b")
    assert d["kv_bytes"] == 2_147_483_648 and d["buffer_bytes"] == 400_000_000 + 20_000 * 16384
    assert "ctx_effective" not in d
    assert_valid("feasibility.schema.json", out)


def test_embeddings_ctx_below_ctx_max_effective_equals_intent(systems, catalog):
    out = run(systems["rtx3060"], catalog, ctx=4096)
    c = cand(out, "test/embed-110m")
    assert c.get("ctx_effective", 4096) == 4096                 # presente (4096) ou ausente: ambos valem
    # 2*12*12*64*4096*2 = 150_994_944 ; buffer 400e6 + 20e3*4096 = 481_920_000
    assert c["kv_bytes"] == 150_994_944 and c["buffer_bytes"] == 481_920_000


def test_embeddings_multi_task_model_still_rejected_above_ctx_max(systems, catalog):
    """Só vale quando a ÚNICA tarefa avaliada (model.tasks ∩ intent.tasks) é embeddings."""
    cat = deep(catalog)
    next(m for m in cat["models"] if m["id"] == "test/embed-110m")["tasks"] = ["embeddings", "chat"]
    out = run(systems["rtx3060"], cat, ctx=16384)               # intent com chat => avaliado também para chat
    assert cand(out, "test/embed-110m") is None
    assert rej(out, "test/embed-110m")["reason_code"] == "ctx_exceeds_max"
    out2 = run(systems["rtx3060"], cat, tasks=("embeddings",), ctx=16384)   # interseção = {embeddings}
    c = cand(out2, "test/embed-110m")
    assert c is not None and c["ctx_effective"] == 8192 and c["kv_bytes"] == 301_989_888


def test_ctx_above_every_ctx_max_rejects_all_but_embeddings(systems, catalog):
    out = run(systems["rtx3060"], catalog, ctx=40000)
    assert ids(out["candidates"]) == {"test/embed-110m"}
    assert out["candidates"][0]["ctx_effective"] == 8192
    assert {r["reason_code"] for r in out["rejected"]} == {"ctx_exceeds_max"}
    assert ids(out["rejected"]) == {"test/tiny-1b", "test/dense-7b", "test/moe-30b-a3b"}
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


def test_unknown_bandwidth_gives_none_tokens_ordered_by_bytes(systems, catalog):
    s = deep(systems["ddr5_64"])
    s["memory"]["bandwidth_bytes_per_s"] = {"value": None, "source": "none", "confidence": "low"}
    out = run(s, catalog)
    assert out["candidates"], "sem banda o modelo continua viável (cabe na memória)"
    assert all(c["tokens_per_s_est"] is None for c in out["candidates"])
    assert all(c["ttft_s_est"] is None for c in out["candidates"])
    sorted_ok(out["candidates"])
    # F4: sem tokens/s a ordem é por bytes lidos na CPU por token (cpu_only: bytes_camada_lidos * n_layers)
    #   embed F16  220_000_000 // 12 * 12 = 219_999_996
    #   1B  Q8_0   1_466_250_000 // 16 * 16 = 1_466_250_000
    #   MoE Q6_K   24.6e9*3e9//30e9 = 2_460_000_000
    #   7B  Q8_0   7_650_000_000
    assert [c["model"]["id"] for c in out["candidates"]] == [
        "test/embed-110m", "test/tiny-1b", "test/moe-30b-a3b", "test/dense-7b"]
    for c in out["candidates"]:
        if c["model"]["tasks"] == ["embeddings"]:
            assert_embeddings_no_decode(c)          # exceção: motivo é decode_not_applicable
        else:
            assert "ranked_by_bytes" in c["confidence_reasons"]
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


# ---- KV lido no decode previsto (decisão B: resposta de 1024 tokens no fim da janela)


def test_larger_context_predicts_lower_decode(systems, catalog):
    tps = [cand(run(systems["ddr5_64"], catalog, tasks=("chat",), ctx=ctx), "test/dense-7b")["tokens_per_s_est"]
           for ctx in (512, 2048, 8192, 32768)]
    assert tps == sorted(tps, reverse=True) and len(set(tps)) == 4
    # ctx 2048 -> posição 1536: (7.65e9 + 2*32*8*128*1536*2 = 201_326_592) / (89.6e9*0.6) = 0.146043 s -> 6.8473
    assert tps[1] == pytest.approx(6.8473, rel=1e-3)


def test_q8_kv_reads_less_than_f16(systems, catalog):
    f16 = cand(run(systems["ddr5_64"], catalog, tasks=("chat",), kv="f16"), "test/dense-7b")
    q8 = cand(run(systems["ddr5_64"], catalog, tasks=("chat",), kv="q8_0"), "test/dense-7b")
    assert q8["tokens_per_s_est"] > f16["tokens_per_s_est"]
    # q8_0: KV na posição 7680 = 503_316_480 -> (7.65e9 + 503_316_480)/(89.6e9*0.6) -> 6.5937
    assert q8["tokens_per_s_est"] == pytest.approx(6.5937, rel=1e-3)


def test_decode_assumption_is_echoed_and_rendered(systems, catalog):
    from scout.cli import render_markdown

    out = run(systems["ddr5_64"], catalog, tasks=("chat",), ctx=8192)
    assert out["inputs"]["decode"] == {"window_tokens": 8192, "answer_tokens": 1024, "position_tokens": 7680,
                                       "start_position_tokens": 512}
    assert "tokens/s no fim de uma janela de 8192 tokens" in render_markdown(out)
    small = run(systems["ddr5_64"], catalog, tasks=("chat",), ctx=600)
    assert small["inputs"]["decode"]["position_tokens"] == 300
    assert "resposta ocupando a janela inteira" in render_markdown(small)


# ---- opção C: tokens/s no início da conversa, só exibido (schemas/decode_range_spec.md)


def test_start_of_conversation_dense_cpu_hand_computed(systems, catalog):
    out = run(systems["ddr5_64"], catalog, tasks=("chat",), ctx=8192)
    assert_valid("feasibility.schema.json", out)
    assert out["inputs"]["decode"] == {"window_tokens": 8192, "answer_tokens": 1024, "position_tokens": 7680,
                                       "start_position_tokens": 512}
    d = cand(out, "test/dense-7b")
    # início: KV na posição 512 = 2*32*8*128*512*2 = 67_108_864
    # (7.65e9 + 67_108_864) / (89.6e9 * 0.6) = 0.143547 s -> 6.9663 tok/s
    assert d["tokens_per_s_est_start"] == pytest.approx(6.9663, rel=1e-3)
    # fim (inalterado): (7.65e9 + 1_006_632_960) / (89.6e9 * 0.6) -> 6.2103
    assert d["tokens_per_s_est"] == pytest.approx(6.2103, rel=1e-3)
    assert d["provenance"]["tokens_per_s_est_start"] == d["provenance"]["tokens_per_s_est"] == "estimated"


@pytest.mark.parametrize("profile", sorted(SYSTEM_PATHS))
@pytest.mark.parametrize("ctx", [512, 1024, 2048, 8192, 32768])
@pytest.mark.parametrize("kv", ["f16", "q8_0"])
def test_start_invariants_on_every_profile(systems, catalog, profile, ctx, kv):
    out = run(systems[profile], catalog, ctx=ctx, kv=kv)
    assert_valid("feasibility.schema.json", out)
    dec = out["inputs"]["decode"]
    assert 1 <= dec["start_position_tokens"] <= dec["position_tokens"]
    for c in out["candidates"]:
        start, end = c["tokens_per_s_est_start"], c["tokens_per_s_est"]
        assert (start is None) == (end is None)
        assert c["provenance"]["tokens_per_s_est_start"] == c["provenance"]["tokens_per_s_est"]
        if end is not None:
            assert start >= end
            if ctx <= 1024:
                assert start == end


def _two_model_catalog(catalog):
    """kvheavy: pesos menores (Q4_K_M 4.41e9) e KV grande (64 cabeças KV, 1_048_576 B/posição);
    dense-7b: pesos maiores (Q8_0 7.65e9) e KV pequeno (131_072 B/posição)."""
    cat = deep(catalog)
    dense = next(m for m in cat["models"] if m["id"] == "test/dense-7b")
    heavy = deep(dense)
    heavy.update(id="test/kvheavy-7b", source_url="https://example.invalid/test/kvheavy-7b", n_heads=64,
                 n_kv_heads=64, quants=[q for q in dense["quants"] if q["name"] == "Q4_K_M"])
    cat["models"] = [dense, heavy]
    return cat


def test_start_never_orders_nor_picks_primary(systems, catalog):
    out = run(systems["ddr5_64"], _two_model_catalog(catalog), tasks=("chat",), ctx=8192)
    assert_valid("feasibility.schema.json", out)
    h, d = cand(out, "test/kvheavy-7b"), cand(out, "test/dense-7b")
    # kvheavy, início: (4.41e9 + 2*32*64*128*512*2 = 536_870_912) / 53.76e9 -> 10.8675
    #          fim:    (4.41e9 + 2*32*64*128*7680*2 = 8_053_063_680) / 53.76e9 -> 4.3135
    # dense-7b, início 6.9663, fim 6.2103: no início o kvheavy é mais rápido, no fim é o mais lento
    assert h["tokens_per_s_est_start"] == pytest.approx(10.8675, rel=1e-3)
    assert h["tokens_per_s_est"] == pytest.approx(4.3135, rel=1e-3)
    assert h["tokens_per_s_est_start"] > d["tokens_per_s_est_start"] and h["tokens_per_s_est"] < d["tokens_per_s_est"]
    assert [c["model"]["id"] for c in out["candidates"]] == ["test/dense-7b", "test/kvheavy-7b"]  # ordem do fim
    assert d["primary_for"] == ["chat"] and h["primary_for"] == []


def test_start_hybrid_moe_same_placement_hand_computed(systems, catalog):
    out = run(systems["rtx3060"], catalog)
    m = cand(out, "test/moe-30b-a3b")
    assert m["placement"] == "hybrid" and m["n_gpu_layers"] == 25  # placement escolhido pelo fim
    # início: KV na posição 512 = 2*48*4*128*512*2 = 50_331_648, dividido pelas camadas de cada trecho
    # gpu: 25*38_125_000 + 50_331_648*25//48 (26_214_400) = 979_339_400 / (360e9*0.75)
    # cpu: 23*38_125_000 + 50_331_648*23//48 (24_117_248) = 900_992_248 / (51.2e9*0.6)
    # t = 0.0036272 + 0.0293292 = 0.0329564 s -> 30.3432 tok/s (fim: 22.0719)
    assert m["tokens_per_s_est_start"] == pytest.approx(30.3432, rel=1e-3)
    assert m["tokens_per_s_est_start"] > m["tokens_per_s_est"]


def test_start_embeddings_null_with_provenance(systems, catalog):
    out = run(systems["cpu8"], catalog)
    e = cand(out, "test/embed-110m")
    assert e["tokens_per_s_est_start"] is None and "tokens_per_s_est_start" in e["provenance"]


def test_start_markdown_line_and_phrase(systems, catalog):
    from scout.cli import render_markdown

    md = render_markdown(run(systems["ddr5_64"], catalog, ctx=8192))
    assert "no início de uma conversa: ~6.97 tokens/s" in md
    assert "Esse é o número que ordena e escolhe a recomendação." in md
    assert "resposta de 1024 tokens começando com o contexto vazio (posição média 512)" in md
    emb = next(ln for ln in md.splitlines() if ln.startswith("| [test/embed-110m]"))
    assert "no início" not in emb
    md600 = render_markdown(run(systems["ddr5_64"], catalog, ctx=600))
    assert "no início" not in md600  # posições iguais: um número só (D7)


def test_markdown_of_020_shape_keeps_old_sentence(systems, catalog):
    from scout.cli import render_markdown

    out = run(systems["ddr5_64"], catalog, ctx=8192)
    del out["inputs"]["decode"]["start_position_tokens"]
    for c in out["candidates"]:
        del c["tokens_per_s_est_start"]
        del c["provenance"]["tokens_per_s_est_start"]
    assert_valid("feasibility.schema.json", out)
    md = render_markdown(out)
    assert "Conversas mais curtas geram mais rápido." in md
    assert "no início de uma conversa" not in md and "Esse é o número que ordena" not in md
