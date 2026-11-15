"""Regra 3 do Adendo (m1_interface.md): preferência de qualidade abaixo de 7e9 em primary_for,
e catalog_stale fora do solver. Casos sintéticos montados AQUI a partir do catálogo de teste.

Cenário base (perfil ddr5_64, cpu_only, ctx 8192, tarefa chat):
  - BIG  = MoE 6e9 total / 1e9 ativos, Q4_K_M bpw 4.9 (3_675_000_000 B): lê só 1/6 dos bytes por token => o mais rápido
  - SMALL = denso 2e9, Q8_0 bpw 8.5 (2_125_000_000 B): piso 5.5, bpw 8.5 >= 6.5
Ambos confortáveis (cabem folgados em 65_719_476_736 B) => SMALL tem tier "não pior" e vem DEPOIS na tabela."""
from __future__ import annotations

import json

import pytest

from conftest import ALL_TASKS, TUNING, deep
from scout.solver import Intent, solve
from test_solver import sorted_ok

CHAT = ("chat",)


def mk(base, id_, params, active, layers, quants, commercial=True, kv_heads=4, head_dim=64, tasks=("chat",)):
    """Modelo sintético derivado do 7B do catálogo de teste (mesmo formato, valores novos)."""
    m = deep(next(x for x in base["models"] if x["id"] == "test/dense-7b"))
    m.update(id=id_, source_url=f"https://example.invalid/{id_}", gguf_repo=None,
             license_source=f"https://example.invalid/{id_}", commercial_ok=commercial,
             tasks=list(tasks), params_total=params, params_active=active, n_layers=layers,
             n_heads=kv_heads, n_kv_heads=kv_heads, head_dim=head_dim)
    m["quants"] = [{"format": "GGUF", "name": n, "bpw": b, "bpw_source": "file_bytes",
                    "file_bytes": fb, "file_url": None} for n, b, fb in quants]
    return m


def big(base, params=6_000_000_000, quants=(("Q4_K_M", 4.9, 3_675_000_000),), **kw):
    return mk(base, "test/big-moe", params, 1_000_000_000, 24, quants, **kw)


def small(base, id_="test/small-2b", params=2_000_000_000, quants=(("Q8_0", 8.5, 2_125_000_000),), **kw):
    return mk(base, id_, params, params, 24, quants, **kw)


def go(system, base, models, commercial=False, tasks=CHAT):
    cat = deep(base)
    cat["models"] = models
    return solve(deep(system), cat, Intent(tasks=tasks, ctx=8192, commercial_use=commercial), dict(TUNING))


def primary(out, task="chat"):
    return [c["model"]["id"] for c in out["candidates"] if task in c["primary_for"]]


def order(out):
    return [c["model"]["id"] for c in out["candidates"]]


def flagged(out):
    return "quality_preference" in json.dumps(out)


# ---------------------------------------------------------------- dispara
def test_preference_triggers_small_high_bpw_becomes_primary(systems, catalog):
    out = go(systems["ddr5_64"], catalog, [big(catalog), small(catalog)])
    # regra 2 daria o BIG (mais rápido, 1o na tabela); regra 3: BIG tem 6e9 < 7e9 e bpw 4.9 < 5.5,
    # SMALL tem params menor, bpw 8.5 >= 6.5 e tier igual => principal passa a ser o SMALL
    assert order(out) == ["test/big-moe", "test/small-2b"]          # ordem da tabela INALTERADA
    sorted_ok(out["candidates"])
    big_c, small_c = out["candidates"]
    assert big_c["tier"] == small_c["tier"] == "confortavel"
    assert big_c["tokens_per_s_est"] > small_c["tokens_per_s_est"]
    assert big_c["quant"]["bpw"] < 5.5 and small_c["quant"]["bpw"] >= 6.5
    assert primary(out) == ["test/small-2b"]
    assert big_c["primary_for"] == [] and small_c["primary_for"] == ["chat"]
    assert flagged(out), "aviso/nota quality_preference esperado"


def test_preference_among_several_picks_first_in_final_order(systems, catalog):
    other = small(catalog, "test/small-3b", 3_000_000_000, (("Q8_0", 8.5, 3_187_500_000),))
    out = go(systems["ddr5_64"], catalog, [other, small(catalog), big(catalog)])
    assert order(out) == ["test/big-moe", "test/small-2b", "test/small-3b"]   # 2b lê menos bytes => antes
    assert primary(out) == ["test/small-2b"]
    assert flagged(out)


def test_preference_only_changes_primary_for_not_table_order(systems, catalog):
    on = go(systems["ddr5_64"], catalog, [big(catalog), small(catalog)])
    # mesmo par, sem disparo (BIG com bpw 5.5): a ordem da tabela é a mesma; só primary_for muda
    off = go(systems["ddr5_64"], catalog, [big(catalog, quants=(("Q5_K_M", 5.5, 4_125_000_000),)), small(catalog)])
    assert order(on) == order(off) == ["test/big-moe", "test/small-2b"]
    assert primary(on) == ["test/small-2b"] and primary(off) == ["test/big-moe"]


# ---------------------------------------------------------------- NÃO dispara
def test_no_trigger_when_larger_has_params_ge_7e9(systems, catalog):
    # fronteira exata: params 7_000_000_000 não é < 7e9 ; e 8e9
    for p in (7_000_000_000, 8_000_000_000):
        out = go(systems["ddr5_64"], catalog, [big(catalog, params=p), small(catalog)])
        assert order(out) == ["test/big-moe", "test/small-2b"]
        assert primary(out) == ["test/big-moe"], p
        assert not flagged(out)


def test_no_trigger_when_smaller_bpw_below_6_5(systems, catalog):
    # Q5_K_M bpw 5.7 (>= piso 5.5 do 2B, mas < 6.5): 2e9*5.7/8 = 1_425_000_000
    s = small(catalog, quants=(("Q5_K_M", 5.7, 1_425_000_000),))
    out = go(systems["ddr5_64"], catalog, [big(catalog), s])
    assert primary(out) == ["test/big-moe"] and not flagged(out)


def test_trigger_boundary_smaller_bpw_exactly_6_5(systems, catalog):
    s = small(catalog, quants=(("Q6_5", 6.5, 1_625_000_000),))    # 2e9*6.5/8
    out = go(systems["ddr5_64"], catalog, [big(catalog), s])
    assert primary(out) == ["test/small-2b"] and flagged(out)


def test_no_trigger_when_larger_bpw_not_below_5_5(systems, catalog):
    # BIG com bpw exatamente 5.5 (6e9*5.5/8 = 4_125_000_000): 5.5 não é < 5.5
    b = big(catalog, quants=(("Q5_K_M", 5.5, 4_125_000_000),))
    out = go(systems["ddr5_64"], catalog, [b, small(catalog)])
    assert primary(out) == ["test/big-moe"] and not flagged(out)


def test_no_trigger_when_smaller_has_worse_tier(systems, catalog):
    # SMALL com KV gigante (80 camadas, 64 cabeças KV, head_dim 128):
    #   kv = 2*80*64*128*8192*2 = 21_474_836_480 ; total = 2_125_000_000 + kv + 563_840_000 = 24_163_676_480
    # RAM 31e9 => orçamento 28e9: 0.8*28e9 = 22.4e9 < 24.16e9 <= 26.6e9 => SMALL "limite"; BIG fica "confortavel"
    s = small(catalog)
    s.update(n_layers=80, n_kv_heads=64, n_heads=64, head_dim=128)
    sysd = deep(systems["ddr5_64"])
    sysd["memory"]["ram_total_bytes"]["value"] = 31_000_000_000
    out = go(sysd, catalog, [big(catalog), s])
    tiers = {c["model"]["id"]: c["tier"] for c in out["candidates"]}
    assert tiers == {"test/big-moe": "confortavel", "test/small-2b": "limite"}
    assert primary(out) == ["test/big-moe"] and not flagged(out)


def test_no_trigger_when_smaller_is_blocked_under_commercial(systems, catalog):
    s = small(catalog, commercial=False)
    # sem --commercial: dispara (contraste)
    assert primary(go(systems["ddr5_64"], catalog, [big(catalog), s])) == ["test/small-2b"]
    # com --commercial: SMALL bloqueado nunca vira principal; BIG fica
    out = go(systems["ddr5_64"], catalog, [big(catalog), s], commercial=True)
    sm = next(c for c in out["candidates"] if c["model"]["id"] == "test/small-2b")
    assert sm["blocked"] is True and sm["primary_for"] == []
    assert primary(out) == ["test/big-moe"] and not flagged(out)


def test_no_trigger_when_smaller_has_not_smaller_params(systems, catalog):
    # "params_total menor": 6e9 contra 6e9 não conta
    s = small(catalog, params=6_000_000_000, quants=(("Q8_0", 8.5, 6_375_000_000),))
    out = go(systems["ddr5_64"], catalog, [big(catalog), s])
    first = out["candidates"][0]["model"]["id"]
    assert primary(out) == [first] and not flagged(out)


def test_preference_is_per_task(systems, catalog):
    b = big(catalog, tasks=("chat", "code"))
    s = small(catalog)                                   # só chat
    out = go(systems["ddr5_64"], catalog, [b, s], tasks=("chat", "code"))
    assert primary(out, "chat") == ["test/small-2b"]
    assert primary(out, "code") == ["test/big-moe"]      # não há menor de qualidade para code


# ---------------------------------------------------------------- catalog_stale fica fora do solver
@pytest.mark.parametrize("profile", ["cpu8", "rtx3060", "ddr5_64", "apple24", "rx7800xt"])
def test_solver_never_emits_catalog_stale(profile, systems, catalog):
    old = deep(catalog)
    for m in old["models"]:
        m["verified_at"] = "2020-01-01"          # muito além dos 60 dias
    for cat in (catalog, old):
        out = solve(deep(systems[profile]), deep(cat), Intent(tasks=ALL_TASKS), dict(TUNING))
        assert "catalog_stale" not in {w["code"] for w in out["warnings"]}
        assert "catalog_stale" not in json.dumps(out)
