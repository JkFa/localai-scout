"""Funções puras de scout/estimate.py. Cada valor esperado foi calculado À MÃO a partir de
PROMPT.md (M1) e schemas/m1_interface.md; a conta está no comentário. Nada aqui é
copiado da implementação."""
from __future__ import annotations

import math

import pytest

from scout import estimate as E


# ---------------------------------------------------------------- orçamento
def test_vram_display():
    # display = total - free = 12_884_901_888 - 12_400_000_000 = 484_901_888
    assert E.vram_display(12_884_901_888, 12_400_000_000) == 484_901_888
    assert E.vram_display(8_000_000_000, 8_000_000_000) == 0


def test_vram_usable():
    # usable = total - display - 600e6 = 12_884_901_888 - 484_901_888 - 600_000_000 = 11_800_000_000
    v = E.vram_usable(12_884_901_888, 12_400_000_000)
    assert v == 11_800_000_000 and isinstance(v, int)
    # sem uso de display (free == total): 16_000_000_000 - 0 - 600_000_000
    assert E.vram_usable(16_000_000_000, 16_000_000_000) == 15_400_000_000
    # display 2e9: 8e9 - 2e9 - 0.6e9 = 5.4e9
    assert E.vram_usable(8_000_000_000, 6_000_000_000) == 5_400_000_000


def test_ram_usable():
    # 8 GiB: 8_589_934_592 - 2e9 - 1e9 = 5_589_934_592
    v = E.ram_usable(8_589_934_592)
    assert v == 5_589_934_592 and isinstance(v, int)
    # 64 GiB: 68_719_476_736 - 3e9 = 65_719_476_736
    assert E.ram_usable(68_719_476_736) == 65_719_476_736


def test_unified_budget_is_half_of_ram():
    # Q7: 50% da RAM total. 24 GiB = 25_769_803_776 // 2 = 12_884_901_888
    assert E.unified_budget(25_769_803_776) == 12_884_901_888
    # inteiro, divisão por piso: 25 // 2 = 12
    assert E.unified_budget(25) == 12


# ---------------------------------------------------------------- k_ovh e pesos
@pytest.mark.parametrize("params,expected", [
    (1, 1.15), (1_200_000_000, 1.15), (3_000_000_000, 1.15),          # <= 3e9 (fronteira inclusiva)
    (3_000_000_001, 1.10), (7_000_000_000, 1.10),                     # 3e9 < p <= 7e9 (7e9 inclusivo)
    (7_000_000_001, 1.05), (30_000_000_000, 1.05), (70_000_000_000, 1.05),  # > 7e9
])
def test_k_ovh_bands_and_boundaries(params, expected):
    assert E.k_ovh(params) == pytest.approx(expected)


def test_weights_from_file_bytes_wins():
    # file_bytes existe: weights = file_bytes, sem fallback, independe de bpw e params
    assert E.weights_bytes(7_200_000_000, 8.5, 7_650_000_000) == (7_650_000_000, False)
    assert E.weights_bytes(7_200_000_000, 1.0, 7_650_000_000) == (7_650_000_000, False)


@pytest.mark.parametrize("params,bpw,expected", [
    # 1.2e9 * 8.5 / 8 = 1_275_000_000 ; * 1.15 (<=3e9)  = 1_466_250_000
    (1_200_000_000, 8.5, 1_466_250_000),
    # 3e9 * 5.5 / 8 = 2_062_500_000 ; * 1.15 (fronteira 3e9) = 2_371_875_000
    (3_000_000_000, 5.5, 2_371_875_000),
    # 3_000_000_001 usa 1.10: 3_000_000_001*4.5/8 = 1_687_500_000.5625 ; *1.10 = 1_856_250_000.62 -> ceil 1_856_250_001
    (3_000_000_001, 4.5, 1_856_250_001),
    # 7e9 * 4.0 / 8 = 3_500_000_000 ; * 1.10 (fronteira 7e9) = 3_850_000_000
    (7_000_000_000, 4.0, 3_850_000_000),
    # 7.2e9 * 4.9 / 8 = 4_410_000_000 ; * 1.05 = 4_630_500_000
    (7_200_000_000, 4.9, 4_630_500_000),
    # 30e9 * 4.88 / 8 = 18_300_000_000 ; * 1.05 = 19_215_000_000
    (30_000_000_000, 4.88, 19_215_000_000),
])
def test_weights_fallback_formula(params, bpw, expected):
    w, used = E.weights_bytes(params, bpw, None)
    assert used is True
    assert isinstance(w, int)
    assert abs(w - expected) <= 1  # ±1 byte: ceil em float (m1_interface.md)


def test_fallback_is_ceil_not_floor():
    # 1 param, bpw 8.0: 1*8/8*1.15 = 1.15 -> ceil = 2 (floor/round dariam 1)
    w, used = E.weights_bytes(1, 8.0, None)
    assert used and w == 2


# ---------------------------------------------------------------- KV
def test_kv_f16_gqa():
    # 2 * 32 * 8 * 128 * 8192 * 16 bits * 1 // 8
    #   = 2*32*8*128 = 65_536 ; * 8192 = 536_870_912 ; * 2 bytes = 1_073_741_824
    assert E.kv_bytes(32, 8, 128, 8192, "f16", 1) == 1_073_741_824
    assert E.kv_bytes(32, 8, 128, 8192) == 1_073_741_824  # default = f16, batch 1


def test_kv_q8_0_and_q4_0():
    assert E.kv_bytes(32, 8, 128, 8192, "q8_0") == 536_870_912    # 1 byte/elem
    assert E.kv_bytes(32, 8, 128, 8192, "q4_0") == 268_435_456    # 0.5 byte/elem


def test_kv_batch_and_ctx_linear():
    assert E.kv_bytes(32, 8, 128, 8192, "f16", 2) == 2_147_483_648
    assert E.kv_bytes(32, 8, 128, 1, "f16") == 131_072            # 2*32*8*128*1*2
    assert E.kv_bytes(32, 8, 128, 16384) == 2 * E.kv_bytes(32, 8, 128, 8192)


def test_kv_gqa_is_four_times_smaller_than_mha():
    # n_kv_heads 8 (GQA) vs 32 (MHA): fator exato 4 (errar isso erra o orçamento em várias vezes)
    assert E.kv_bytes(32, 32, 128, 8192) == 4 * E.kv_bytes(32, 8, 128, 8192)


def test_kv_other_shapes():
    # MoE do catálogo de teste: 2*48*4*128*8192*2 = 805_306_368
    assert E.kv_bytes(48, 4, 128, 8192) == 805_306_368
    # 1B: 2*16*8*64*8192*2 = 268_435_456
    assert E.kv_bytes(16, 8, 64, 8192) == 268_435_456
    assert isinstance(E.kv_bytes(48, 4, 128, 8192), int)


# ---------------------------------------------------------------- buffer
def test_buffer_bytes():
    # 400e6 + 20e3 * 8192 = 400_000_000 + 163_840_000 = 563_840_000
    assert E.buffer_bytes(8192) == 563_840_000
    assert E.buffer_bytes(0) == 400_000_000
    assert E.buffer_bytes(1) == 400_020_000
    assert isinstance(E.buffer_bytes(8192), int)


# ---------------------------------------------------------------- tier
@pytest.mark.parametrize("total,budget,expected", [
    # budget 1e9: 80% = 800_000_000, 95% = 950_000_000 (ambos inclusivos)
    (0, 1_000_000_000, "confortavel"),
    (800_000_000, 1_000_000_000, "confortavel"),
    (800_000_001, 1_000_000_000, "limite"),
    (950_000_000, 1_000_000_000, "limite"),
    (950_000_001, 1_000_000_000, "inviavel"),
    # budget não redondo 1_000_000_001: 80% = 800_000_000.8 ; 95% = 950_000_000.95
    (800_000_000, 1_000_000_001, "confortavel"),
    (800_000_001, 1_000_000_001, "limite"),   # 800_000_001*100 = 80_000_000_100 > 80*1_000_000_001 = 80_000_000_080
    (950_000_000, 1_000_000_001, "limite"),
    (950_000_001, 1_000_000_001, "inviavel"), # 95_000_000_100 > 95*1_000_000_001 = 95_000_000_095
    (1, 0, "inviavel"),
    (5_000_000_000, 12_000_000_000, "confortavel"),
])
def test_tier_boundaries(total, budget, expected):
    assert E.tier(total, budget) == expected


# ---------------------------------------------------------------- piso de qualidade
@pytest.mark.parametrize("params,expected", [
    (1, 6.5), (1_500_000_000, 6.5),                  # <= 1.5e9 -> Q6_K
    (1_500_000_001, 5.5), (4_000_000_000, 5.5),      # <= 4e9  -> Q5_K_M
    (4_000_000_001, 4.5), (9_000_000_000, 4.5),      # <= 9e9  -> Q4_K_M
    (9_000_000_001, 3.8), (30_000_000_000, 3.8),     # <= 30e9 -> Q3_K_M
    (30_000_000_001, 2.6), (70_000_000_000, 2.6),    # acima   -> IQ2_M
])
def test_min_bpw_boundaries(params, expected):
    assert E.min_bpw(params) == pytest.approx(expected)


def test_min_bpw_is_monotonic_non_increasing():
    sizes = [10**8, 10**9, 2 * 10**9, 5 * 10**9, 8 * 10**9, 20 * 10**9, 40 * 10**9, 10**11]
    vals = [E.min_bpw(s) for s in sizes]
    assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------- n_gpu_layers
def test_n_gpu_layers_hybrid_moe_case():
    # weights 18.3e9, 48 camadas -> bytes_camada = 381_250_000
    # floor(0.95 * 11_800_000_000) = 11_210_000_000
    # 11_210_000_000 - kv 805_306_368 - buffer 563_840_000 = 9_840_853_632
    # 9_840_853_632 / 381_250_000 = 25.81 -> 25
    assert E.n_gpu_layers(18_300_000_000, 48, 11_800_000_000, 805_306_368, 563_840_000) == 25


def test_n_gpu_layers_capped_at_n_layers():
    # 7B Q8_0: (11_210_000_000 - 1_073_741_824 - 563_840_000) / 239_062_500 = 40.04 -> limitado a 32
    assert E.n_gpu_layers(7_650_000_000, 32, 11_800_000_000, 1_073_741_824, 563_840_000) == 32


def test_n_gpu_layers_floor_at_zero():
    # 1_000_000_000*0.95 - 805_306_368 - 563_840_000 < 0 -> 0 (nunca negativo)
    assert E.n_gpu_layers(18_300_000_000, 48, 1_000_000_000, 805_306_368, 563_840_000) == 0
    assert E.n_gpu_layers(18_300_000_000, 48, 0, 0, 0) == 0


def test_n_gpu_layers_ceiling_095_is_applied():
    # weights 4000 / 40 camadas = 100 B/camada; vram_usable 2000.
    # com teto 0,95: floor(1900)/100 = 19 ; sem teto seriam 20.
    assert E.n_gpu_layers(4000, 40, 2000, 0, 0) == 19
    # kv 1 byte: (1900 - 1) / 100 = 18.99 -> 18
    assert E.n_gpu_layers(4000, 40, 2000, 1, 0) == 18
    # kv 100 + buffer 50: (1900 - 150)/100 = 17.5 -> 17
    assert E.n_gpu_layers(4000, 40, 2000, 100, 50) == 17


def test_n_gpu_layers_uses_floor_division_bytes_per_layer():
    # weights 4099 // 40 = 102 B/camada: 1900 // 102 = 18 (18*102=1836 ; 19*102=1938 > 1900)
    assert E.n_gpu_layers(4099, 40, 2000, 0, 0) == 18


# ---------------------------------------------------------------- MoE e throughput
def test_bytes_per_token():
    # MoE: 18_300_000_000 * 3e9 // 30e9 = 1_830_000_000
    assert E.bytes_per_token(18_300_000_000, 3_000_000_000, 30_000_000_000) == 1_830_000_000
    # denso: active == total -> weights inteiro
    assert E.bytes_per_token(7_650_000_000, 7_200_000_000, 7_200_000_000) == 7_650_000_000
    # piso: 7 * 2 // 3 = 14 // 3 = 4
    assert E.bytes_per_token(7, 2, 3) == 4
    assert isinstance(E.bytes_per_token(10, 1, 3), int) and E.bytes_per_token(10, 1, 3) == 3


def test_tokens_per_s_two_segments():
    # bytes_camada 1e9, 10 camadas, 4 na GPU: bw_gpu 100e9*0.5 = 50e9 ; bw_cpu 20e9*0.5 = 10e9
    #   t = 4e9/50e9 + 6e9/10e9 = 0.08 + 0.6 = 0.68 s  ->  1/0.68 = 1.470588...
    r = E.tokens_per_s(1_000_000_000, 10, 4, 100_000_000_000, 20_000_000_000, 0.5, 0.5)
    assert r == pytest.approx(1 / 0.68, rel=1e-9)


def test_tokens_per_s_full_gpu_and_cpu_only():
    # gpu_full: t = 10e9/50e9 = 0.2 -> 5.0 tok/s (o trecho CPU tem 0 camadas)
    assert E.tokens_per_s(1_000_000_000, 10, 10, 100_000_000_000, 20_000_000_000, 0.5, 0.5) == pytest.approx(5.0)
    # cpu_only: t = 10e9/10e9 = 1.0 -> 1.0 tok/s
    assert E.tokens_per_s(1_000_000_000, 10, 0, 100_000_000_000, 20_000_000_000, 0.5, 0.5) == pytest.approx(1.0)


def test_tokens_per_s_default_etas_and_rtx_profile():
    # MoE hybrid do perfil 2: bytes_camada ativo = 381_250_000 * 3e9/30e9 = 38_125_000
    #   gpu: 38_125_000*25 / (360e9*0.75) = 953_125_000/270e9 = 0.00353009...
    #   cpu: 38_125_000*23 / (51.2e9*0.60) = 876_875_000/30.72e9 = 0.02854..
    #   t = 0.0320746 -> 31.1777 tok/s
    r = E.tokens_per_s(38_125_000, 48, 25, 360_000_000_000, 51_200_000_000, 0.75, 0.60)
    assert r == pytest.approx(31.1777, rel=1e-4)


def test_tokens_per_s_moe_uses_active_fraction_only():
    # perfil 3, cpu_only, MoE Q6_K: bytes_camada ativo = 512_500_000 * 0.1 = 51_250_000
    #   t = 51_250_000*48 / (89.6e9*0.6) = 2_460_000_000/53.76e9 = 0.045759 -> 21.8537 tok/s
    r = E.tokens_per_s(51_250_000, 48, 0, None, 89_600_000_000, 0.75, 0.60)
    assert r == pytest.approx(21.8537, rel=1e-4)


def test_tokens_per_s_none_when_bandwidth_unknown():
    args = (1_000_000_000, 10)
    # trecho GPU em uso sem banda de GPU
    assert E.tokens_per_s(*args, 10, None, 20_000_000_000, 0.5, 0.5) is None
    assert E.tokens_per_s(*args, 4, None, 20_000_000_000, 0.5, 0.5) is None
    # trecho CPU em uso sem banda de CPU
    assert E.tokens_per_s(*args, 0, 100_000_000_000, None, 0.5, 0.5) is None
    assert E.tokens_per_s(*args, 4, 100_000_000_000, None, 0.5, 0.5) is None
    assert E.tokens_per_s(*args, 4, None, None, 0.5, 0.5) is None


def test_tokens_per_s_unused_segment_may_have_no_bandwidth():
    # cpu_only sem GPU: bw_gpu None não deve zerar o resultado (perfis 1 e 3 não têm GPU)
    # t = 10e9/(20e9*0.5) = 1.0 -> 1.0 tok/s
    assert E.tokens_per_s(1_000_000_000, 10, 0, None, 20_000_000_000, 0.5, 0.5) == pytest.approx(1.0)
    # gpu_full com bw_cpu None: t = 10e9/(100e9*0.5) = 0.2 -> 5.0
    assert E.tokens_per_s(1_000_000_000, 10, 10, 100_000_000_000, None, 0.5, 0.5) == pytest.approx(5.0)


def test_tokens_per_s_monotonic_in_bandwidth_and_eta():
    base = E.tokens_per_s(1_000_000_000, 10, 0, None, 20_000_000_000, 0.5, 0.5)
    assert E.tokens_per_s(1_000_000_000, 10, 0, None, 40_000_000_000, 0.5, 0.5) == pytest.approx(2 * base)
    assert E.tokens_per_s(1_000_000_000, 10, 0, None, 20_000_000_000, 0.5, 1.0) == pytest.approx(2 * base)


# ---------------------------------------------------------------- apresentação
@pytest.mark.parametrize("n,expected", [
    # Adendo 2026-11-15: base BINÁRIA, n / 2**30, 2 casas, rótulo GiB
    (0, "0.00 GiB"),
    (1_073_741_824, "1.00 GiB"),     # 2**30
    (1_000_000_000, "0.93 GiB"),     # 1e9 / 1_073_741_824 = 0.9313
    (17_179_869_184, "16.00 GiB"),   # 16 * 2**30
    (7_650_000_000, "7.12 GiB"),     # 7.65e9 / 1_073_741_824 = 7.1247
    (7_620_000_000, "7.10 GiB"),     # 7.0968
    (12_884_901_888, "12.00 GiB"),   # 12 * 2**30
    (536_870_912, "0.50 GiB"),       # 2**29
    (1_610_612_736, "1.50 GiB"),     # 1.5 * 2**30
    (1_073_741_823, "1.00 GiB"),     # 0.99999999907 arredonda para 1.00
    (12_345_678_901, "11.50 GiB"),   # 11.4978
])
def test_fmt_gb(n, expected):
    assert E.fmt_gb(n) == expected


def test_fmt_gb_is_not_decimal_gb():
    assert "GiB" in E.fmt_gb(5) and " GB" not in E.fmt_gb(5)
    assert E.fmt_gb(1_000_000_000) != "1.00 GB"
