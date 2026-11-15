"""Fixtures são dados sintéticos e válidos contra schemas/ (o `qa` é dono delas)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CATALOG_PATH, SYSTEM_PATHS, assert_valid, load

HERE = Path(__file__).resolve().parent
NULLABLE = ["license", "license_source", "tasks", "params_total", "params_active", "n_layers",
            "n_heads", "n_kv_heads", "head_dim", "ctx_max", "kv_model"]


@pytest.mark.parametrize("name", sorted(SYSTEM_PATHS))
def test_system_fixture_valid(name):
    doc = load(SYSTEM_PATHS[name])
    assert_valid("system.schema.json", doc)
    assert doc["fixture"]["synthetic"] is True


def test_catalog_fixture_valid():
    doc = load(CATALOG_PATH)
    assert_valid("catalog.schema.json", doc)
    assert doc["fixture"]["synthetic"] is True


def test_catalog_shape_requested_by_spec():
    models = {m["id"]: m for m in load(CATALOG_PATH)["models"]}
    assert len(models) == 4 and all(i.startswith("test/") for i in models)
    assert all(m["source_url"].startswith("https://example.invalid/") for m in models.values())
    assert any(m["commercial_ok"] is False for m in models.values())
    # 1B, 7B denso, 30B MoE com 3B ativos, embeddings
    by = {m["params_total"]: m for m in models.values()}
    assert by[1_200_000_000]["params_active"] == 1_200_000_000
    assert by[7_200_000_000]["params_active"] == 7_200_000_000
    assert by[30_000_000_000]["params_active"] == 3_000_000_000
    assert any(m["tasks"] == ["embeddings"] for m in models.values())
    # ao menos uma quant sem file_bytes (exercita o fallback) e ao menos uma com
    quants = [q for m in models.values() for q in m["quants"]]
    assert any(q["file_bytes"] is None for q in quants) and any(q["file_bytes"] for q in quants)
    assert {q["format"] for q in quants} == {"GGUF"}


def test_catalog_internal_coherence():
    """Coerência inversa que o schema não impõe: incomplete_fields bate com os campos null;
    bpw de quant com file_bytes == file_bytes*8/params_total."""
    for m in load(CATALOG_PATH)["models"]:
        nulls = sorted(f for f in NULLABLE if m[f] is None)
        assert sorted(m["incomplete_fields"]) == nulls
        assert m["complete"] == (not nulls)
        for q in m["quants"]:
            if q["file_bytes"] is not None:
                assert q["bpw"] == pytest.approx(q["file_bytes"] * 8 / m["params_total"], rel=1e-3)
        assert m["head_dim"] * m["n_kv_heads"] > 0 and m["n_kv_heads"] <= m["n_heads"]


def test_system_fixtures_have_bandwidth_and_ints():
    """tokens/s não pode sair null por falta de banda; tudo em bytes inteiros."""
    for name, path in SYSTEM_PATHS.items():
        d = load(path)
        assert isinstance(d["memory"]["ram_total_bytes"]["value"], int), name
        assert d["memory"]["bandwidth_bytes_per_s"]["value"] > 0, name
        for g in d["gpus"]["value"]:
            assert g["bandwidth_bytes_per_s"]["value"] > 0, name


def test_system_profiles_match_spec():
    s = {k: load(p) for k, p in SYSTEM_PATHS.items()}
    assert s["cpu8"]["gpus"]["value"] == []
    assert s["cpu8"]["memory"]["channels"]["value"] == 1 and s["cpu8"]["memory"]["type"]["value"] == "DDR4"
    assert s["cpu8"]["disks"]["value"][0]["kind"]["value"] == "hdd"
    assert s["rtx3060"]["memory"]["channels"]["value"] == 2
    assert s["rtx3060"]["gpus"]["value"][0]["vram_total_bytes"]["value"] == 12_884_901_888
    assert s["ddr5_64"]["memory"]["type"]["value"] == "DDR5" and s["ddr5_64"]["gpus"]["value"] == []
    assert s["apple24"]["memory"]["unified"]["value"] is True
    assert s["apple24"]["memory"]["ram_total_bytes"]["value"] == 25_769_803_776
    assert s["rx7800xt"]["gpus"]["value"][0]["backend"]["value"] == "rocm"
    for k in ("cpu8", "rtx3060", "ddr5_64", "rx7800xt"):
        assert s[k]["memory"]["unified"]["value"] is False


def test_no_test_reads_real_catalog():
    """Regra 7: nenhum teste referencia o catálogo real."""
    needle = "data" + "/catalog" + ".json"
    for p in HERE.glob("test_*.py"):
        assert needle not in p.read_text(encoding="utf-8"), p.name
