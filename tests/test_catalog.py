"""M3: mapeamento API/config do Hugging Face -> catálogo. Sem rede: JSONs
sintéticos em tests/fixtures/hf/ e scout.hf substituído por monkeypatch."""
from __future__ import annotations

import copy
import json

import pytest

from conftest import FIX, SCHEMAS
from scout import catalog as C
from scout import cli, hf

HF = FIX / "hf"
POLICY = {"licenses": {"apache-2.0": True, "cc-by-nc-4.0": False}}
TODAY = "2026-11-15"


def j(name: str) -> dict:
    return json.loads((HF / name).read_text(encoding="utf-8"))


def schema_errors(doc: dict) -> list[str]:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / "catalog.schema.json").read_text(encoding="utf-8"))
    return [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(doc)]


def as_catalog(*entries: dict) -> dict:
    return {"schema_version": 1, "kind": "catalog", "models": list(entries)}


# ---------------------------------------------------------------- map_config


def test_dense_config_maps_table_fields():
    a = C.map_config(j("config_dense.json"))
    assert (a["n_layers"], a["n_heads"], a["n_kv_heads"], a["ctx_max"]) == (24, 16, 4, 32768)
    assert a["head_dim"] == 2048 // 16
    assert a["kv_model"] == "standard"
    assert a["is_moe"] is False and a["null_reasons"] == {}


def test_missing_kv_heads_means_mha():
    cfg = j("config_dense.json")
    del cfg["num_key_value_heads"]
    assert C.map_config(cfg)["n_kv_heads"] == 16


def test_explicit_head_dim_wins_over_division():
    cfg = j("config_dense.json")
    cfg["head_dim"] = 256
    assert C.map_config(cfg)["head_dim"] == 256


@pytest.mark.parametrize("hidden", [2050, None])
def test_head_dim_null_when_not_exact_or_missing(hidden):
    cfg = j("config_dense.json")
    if hidden is None:
        del cfg["hidden_size"]
    else:
        cfg["hidden_size"] = hidden
    a = C.map_config(cfg)
    assert a["head_dim"] is None
    assert "head_dim" in a["null_reasons"]


def test_null_head_dim_key_falls_back_to_division():
    cfg = j("config_dense.json")
    cfg["head_dim"] = None
    assert C.map_config(cfg)["head_dim"] == 128


def test_config_none_nulls_every_arch_field():
    a = C.map_config(None)
    assert all(a[f] is None for f in C.ARCH_FIELDS)
    assert set(a["null_reasons"]) == set(C.ARCH_FIELDS)


def test_kv_lora_rank_means_mla():
    assert C.map_config(j("config_mla.json"))["kv_model"] == "mla"


def test_vlm_reads_text_config_only():
    a = C.map_config(j("config_vlm.json"))
    assert (a["n_layers"], a["n_heads"], a["n_kv_heads"], a["head_dim"], a["ctx_max"]) == (34, 8, 4, 256, 131072)


def test_vlm_text_config_missing_field_is_null_not_top_level():
    cfg = j("config_vlm.json")
    del cfg["text_config"]["num_hidden_layers"]
    a = C.map_config(cfg)
    assert a["n_layers"] is None  # não pega os 27 do nível de cima (visão)
    assert "n_layers" in a["null_reasons"]


# ---------------------------------------------------------------- MoE


def test_moe_inactive_params_qwen_style():
    a = C.map_config(j("config_moe.json"))
    assert a["is_moe"] is True
    assert a["moe_inactive_params"] == (16 - 2) * 8 * 3 * 1024 * 512


def test_moe_first_k_dense_and_mlp_only_layers_reduce_moe_layers():
    a = C.map_config(j("config_mla.json"))
    assert a["moe_inactive_params"] == (64 - 6) * (27 - 1) * 3 * 2048 * 1408
    cfg = j("config_moe.json")
    cfg["mlp_only_layers"] = [0, 7]
    assert C.map_config(cfg)["moe_inactive_params"] == 14 * 6 * 3 * 1024 * 512


def test_moe_mixtral_style_uses_intermediate_size():
    cfg = j("config_dense.json")
    cfg.update(num_local_experts=8, num_experts_per_tok=2)
    assert C.map_config(cfg)["moe_inactive_params"] == 6 * 24 * 3 * 2048 * 8192


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("moe_intermediate_size"),
    lambda c: c.pop("num_experts_per_tok"),
    lambda c: c.update(hidden_act="gelu"),
    lambda c: c.update(decoder_sparse_step=2),
])
def test_moe_without_enough_data_gives_null_active(mutate):
    cfg = j("config_moe.json")
    mutate(cfg)
    a = C.map_config(cfg)
    assert a["is_moe"] is True and a["moe_inactive_params"] is None
    assert "params_active" in a["null_reasons"]
    info = copy.deepcopy(j("info_dense.json"))
    entry = C.build_entry("test/dense-2b", info, cfg, j("gguf_dense.json"), POLICY, TODAY)
    assert entry["params_active"] is None and entry["complete"] is False
    assert "params_active" in entry["incomplete_fields"]


def test_moe_params_active_from_safetensors_total():
    info = j("info_dense.json")
    info["safetensors"]["total"] = 400_000_000
    e = C.build_entry("test/dense-2b", info, j("config_moe.json"), j("gguf_dense.json"), POLICY, TODAY)
    assert e["params_active"] == 400_000_000 - 14 * 8 * 3 * 1024 * 512
    assert e["params_total"] == 400_000_000


def test_moe_inactive_larger_than_total_is_null():
    info = j("info_dense.json")
    info["safetensors"]["total"] = 100_000_000
    e = C.build_entry("test/dense-2b", info, j("config_moe.json"), None, POLICY, TODAY)
    assert e["params_active"] is None


def test_single_expert_key_is_dense():
    cfg = j("config_dense.json")
    cfg["num_local_experts"] = 1
    assert C.map_config(cfg)["is_moe"] is False


# ---------------------------------------------------------------- GGUF


def test_gguf_quants_sizes_bpw_and_shards():
    qs = {q["name"]: q for q in C.gguf_quants(j("gguf_dense.json"), 2_000_000_000)}
    assert set(qs) == {"Q8_0", "Q4_K_M", "F16", "IQ2_M"}  # Q6_K com shard faltando e mmproj fora
    assert qs["Q8_0"]["file_bytes"] == 2_125_000_000 and qs["Q8_0"]["bpw"] == 8.5
    assert qs["Q4_K_M"]["bpw"] == 4.9 and qs["Q4_K_M"]["bpw_source"] == "file_bytes"
    assert qs["F16"]["file_bytes"] == 4_000_000_000 and qs["F16"]["bpw"] == 16.0
    assert qs["F16"]["file_url"].endswith("/test/dense-2b-gguf/resolve/main/F16/dense-2b-F16-00001-of-00002.gguf")
    assert qs["IQ2_M"]["file_bytes"] is None and qs["IQ2_M"]["bpw"] is None
    assert qs["IQ2_M"]["bpw_source"] is None


def test_gguf_bpw_null_without_params_total():
    qs = C.gguf_quants(j("gguf_dense.json"), None)
    assert all(q["bpw"] is None and q["bpw_source"] is None for q in qs)
    assert any(q["file_bytes"] for q in qs)


def test_gguf_quant_order_is_deterministic():
    info = j("gguf_dense.json")
    rev = copy.deepcopy(info)
    rev["siblings"].reverse()
    assert C.gguf_quants(info, 2_000_000_000) == C.gguf_quants(rev, 2_000_000_000)


# ---------------------------------------------------------------- licença e tarefas


def test_license_fields_from_card_and_policy():
    lic = C.license_fields(j("info_dense.json"), POLICY)
    assert lic == {"license": "apache-2.0", "license_source": "https://huggingface.co/test/dense-2b",
                   "commercial_ok": True}


def test_license_outside_policy_is_null_commercial_but_complete():
    info = j("info_dense.json")
    info["cardData"]["license"] = "other"
    e = C.build_entry("test/dense-2b", info, j("config_dense.json"), j("gguf_dense.json"), POLICY, TODAY)
    assert e["license"] == "other" and e["commercial_ok"] is None
    assert e["complete"] is True


def test_missing_license_makes_incomplete():
    info = j("info_dense.json")
    del info["cardData"]
    e = C.build_entry("test/dense-2b", info, j("config_dense.json"), j("gguf_dense.json"), POLICY, TODAY)
    assert e["license"] is None and e["license_source"] is None and e["commercial_ok"] is None
    assert {"license", "license_source"} <= set(e["incomplete_fields"])
    assert schema_errors(as_catalog(e)) == []


def test_unknown_pipeline_tag_makes_tasks_null():
    info = j("info_dense.json")
    info["pipeline_tag"] = "test-unknown-tag"
    e = C.build_entry("test/dense-2b", info, j("config_dense.json"), j("gguf_dense.json"), POLICY, TODAY)
    assert e["tasks"] is None and "tasks" in e["incomplete_fields"]


# ---------------------------------------------------------------- build_entry


def test_build_entry_complete_dense_is_schema_valid():
    e = C.build_entry("test/dense-2b", j("info_dense.json"), j("config_dense.json"),
                      j("gguf_dense.json"), POLICY, TODAY)
    assert e["complete"] is True and e["incomplete_fields"] == []
    assert e["params_total"] == e["params_active"] == 2_000_000_000
    assert e["source_url"] == "https://huggingface.co/test/dense-2b"
    assert e["gguf_repo"] == "test/dense-2b-gguf" and e["verified_at"] == TODAY
    assert e["tasks"] == ["chat"]
    assert schema_errors(as_catalog(e)) == []


def test_build_entry_without_safetensors_total():
    info = j("info_dense.json")
    del info["safetensors"]
    e = C.build_entry("test/dense-2b", info, j("config_dense.json"), j("gguf_dense.json"), POLICY, TODAY)
    assert e["params_total"] is None and e["params_active"] is None
    assert {"params_total", "params_active"} <= set(e["incomplete_fields"])
    assert all(q["bpw"] is None for q in e["quants"])
    assert schema_errors(as_catalog(e)) == []


def test_build_entry_without_gguf_or_config_is_incomplete_and_valid():
    e = C.build_entry("test/dense-2b", j("info_dense.json"), None, None, POLICY, TODAY)
    assert e["complete"] is False
    assert set(C.ARCH_FIELDS) | {"quants"} <= set(e["incomplete_fields"])
    assert schema_errors(as_catalog(e)) == []
    assert C.check_catalog(as_catalog(e), TODAY) == ([], [])


# ---------------------------------------------------------------- check / upsert


def complete_entry() -> dict:
    return C.build_entry("test/dense-2b", j("info_dense.json"), j("config_dense.json"),
                         j("gguf_dense.json"), POLICY, TODAY)


def test_check_flags_incoherent_complete():
    e = complete_entry()
    e["head_dim"] = None
    errors, _ = C.check_catalog(as_catalog(e), TODAY)
    assert errors


def test_check_flags_incomplete_fields_mismatch():
    e = C.build_entry("test/dense-2b", j("info_dense.json"), None, None, POLICY, TODAY)
    e["incomplete_fields"] = ["quants"]
    errors, _ = C.check_catalog(as_catalog(e), TODAY)
    assert any("incomplete_fields" in x for x in errors)


def test_check_warns_stale_after_60_days():
    cat = as_catalog(complete_entry())
    assert C.check_catalog(cat, "2027-01-14") == ([], [])  # 60 dias: ainda ok
    errors, warnings = C.check_catalog(cat, "2027-01-15")
    assert errors == [] and len(warnings) == 1


def test_upsert_replaces_by_id_and_sorts():
    a = complete_entry()
    b = copy.deepcopy(a)
    b["id"] = "test/aaa-1b"
    cat = C.upsert(C.upsert(as_catalog(), a), b)
    assert [m["id"] for m in cat["models"]] == ["test/aaa-1b", "test/dense-2b"]
    a2 = copy.deepcopy(a)
    a2["verified_at"] = "2026-12-01"
    cat = C.upsert(cat, a2)
    assert len(cat["models"]) == 2 and cat["models"][1]["verified_at"] == "2026-12-01"


# ---------------------------------------------------------------- CLI sem rede


@pytest.fixture
def fake_hf(monkeypatch):
    calls = []

    def info(repo, *, blobs=False, cache=None, offline=False):
        calls.append(("info", repo, blobs, offline))
        return j("gguf_dense.json") if blobs else j("info_dense.json")

    def config(repo, *, cache=None, offline=False):
        calls.append(("config", repo, offline))
        return j("config_dense.json")

    monkeypatch.setattr(hf, "get_model_info", info)
    monkeypatch.setattr(hf, "get_config", config)
    monkeypatch.setattr(cli, "_utc_today", lambda: TODAY)
    return calls


def test_cli_catalog_add_then_check(tmp_path, fake_hf, capsys):
    out = tmp_path / "catalog.json"
    argv = ["catalog", "add", "test/dense-2b", "--gguf-repo", "test/dense-2b-gguf", "--catalog", str(out)]
    assert cli.main(argv) == 0
    first = out.read_bytes()
    assert cli.main(argv) == 0
    assert out.read_bytes() == first  # upsert idempotente, saída determinística
    doc = json.loads(first)
    assert [m["id"] for m in doc["models"]] == ["test/dense-2b"] and doc["models"][0]["complete"]
    assert cli.main(["catalog", "check", "--catalog", str(out), "--today", TODAY]) == 0


def test_cli_catalog_add_offline_is_forwarded(tmp_path, fake_hf):
    out = tmp_path / "catalog.json"
    assert cli.main(["catalog", "add", "test/dense-2b", "--catalog", str(out), "--offline"]) == 0
    assert all(c[-1] is True for c in fake_hf)


def test_cli_catalog_add_hf_error_exits_nonzero(tmp_path, monkeypatch):
    def boom(repo, **kw):
        raise hf.HFError("https://huggingface.co/api/models/test/x", None, "network")

    monkeypatch.setattr(hf, "get_model_info", boom)
    out = tmp_path / "catalog.json"
    assert cli.main(["catalog", "add", "test/x", "--catalog", str(out)]) == 2
    assert not out.exists()


def test_cli_catalog_check_fails_on_incoherent(tmp_path):
    e = complete_entry()
    e["n_layers"] = None
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(as_catalog(e)), encoding="utf-8")
    assert cli.main(["catalog", "check", "--catalog", str(p), "--today", TODAY]) != 0


# ---------------------------------------------------------------- arquivo único + shards


def test_single_file_and_shards_of_same_quant_give_one_quant():
    warnings: list[str] = []
    qs = C.gguf_quants(j("gguf_single_and_shards.json"), 7_200_000_000, warnings)
    by = {q["name"]: q for q in qs}
    assert sorted(by) == ["FP16", "Q4_K_M", "Q5_0", "Q8_0"]
    base = "https://huggingface.co/test/dense-7b-gguf/resolve/main/"
    # arquivo único vence: file_bytes e file_url dele
    assert by["Q4_K_M"]["file_bytes"] == 4_410_000_000
    assert by["Q4_K_M"]["file_url"] == base + "dense-7b-q4_k_m.gguf"
    assert by["FP16"]["file_bytes"] == 15_000_000_000
    # só shards: soma, URL do primeiro shard
    assert by["Q5_0"]["file_bytes"] == 5_000_000_000
    assert by["Q5_0"]["file_url"] == base + "dense-7b-q5_0-00001-of-00002.gguf"
    # divergência > 1% (7.00e9 vs 7.65e9): fica o arquivo único, com aviso
    assert by["Q8_0"]["file_bytes"] == 7_650_000_000
    assert len(warnings) == 1 and "dense-7b-q8_0" in warnings[0]


def test_single_and_shards_warning_reaches_build_entry():
    _, _, warnings = C.build_entry_with_reasons(
        "test/dense-2b", j("info_dense.json"), j("config_dense.json"),
        j("gguf_single_and_shards.json"), POLICY, TODAY)
    assert sum("divergem" in w for w in warnings) == 1


# ---------------------------------------------------------------- overrides, remove, teto de bpw

OVERRIDES = {"overrides": {"test/dense-2b": {"add_tasks": ["code", "chat"], "reason": "r",
                                             "source": "https://huggingface.co/test/dense-2b"}}}


def test_task_override_only_adds():
    e = complete_entry()
    out, msgs = C.apply_task_overrides(e, OVERRIDES)
    assert out["tasks"] == ["chat", "code"]
    assert {k: v for k, v in out.items() if k != "tasks"} == {k: v for k, v in e.items() if k != "tasks"}
    assert e["tasks"] == ["chat"]  # entrada original intacta
    assert len(msgs) == 1 and "code" in msgs[0]
    assert schema_errors(as_catalog(out)) == []


def test_task_override_other_repo_or_null_tasks_untouched():
    other = dict(complete_entry(), id="test/other-1b")
    assert C.apply_task_overrides(other, OVERRIDES) == (other, [])
    null_tasks = dict(complete_entry(), tasks=None)
    out, msgs = C.apply_task_overrides(null_tasks, OVERRIDES)
    assert out["tasks"] is None and msgs


def test_task_override_ignores_unknown_task():
    ov = {"overrides": {"test/dense-2b": {"add_tasks": ["telepathy"]}}}
    out, msgs = C.apply_task_overrides(complete_entry(), ov)
    assert out["tasks"] == ["chat"] and any("telepathy" in m for m in msgs)


def test_cli_add_applies_override_file(tmp_path, fake_hf, monkeypatch):
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps(OVERRIDES), encoding="utf-8")
    monkeypatch.setattr(cli, "OVERRIDES_PATH", ov)
    out = tmp_path / "catalog.json"
    assert cli.main(["catalog", "add", "test/dense-2b", "--gguf-repo", "x/y", "--catalog", str(out)]) == 0
    assert json.loads(out.read_text())["models"][0]["tasks"] == ["chat", "code"]


def test_cli_catalog_remove(tmp_path):
    a = complete_entry()
    b = dict(a, id="test/aaa-1b")
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(C.upsert(C.upsert(as_catalog(), a), b)), encoding="utf-8")
    assert cli.main(["catalog", "remove", "test/aaa-1b", "--catalog", str(p)]) == 0
    assert [m["id"] for m in json.loads(p.read_text())["models"]] == ["test/dense-2b"]
    assert cli.main(["catalog", "remove", "test/aaa-1b", "--catalog", str(p)]) != 0


def test_bpw_above_32_kept_above_40_null():
    info = {"id": "test/x-gguf", "siblings": [
        {"rfilename": "x-F32.gguf", "lfs": {"size": 4_100_000_000}},   # 32.8 bpw
        {"rfilename": "x-Q8_0.gguf", "lfs": {"size": 5_100_000_000}},  # 40.8 bpw
    ]}
    warnings: list[str] = []
    qs = {q["name"]: q for q in C.gguf_quants(info, 1_000_000_000, warnings)}
    assert qs["F32"]["bpw"] == 32.8
    assert qs["Q8_0"]["bpw"] is None and qs["Q8_0"]["bpw_source"] is None
    assert len(warnings) == 1 and "x-Q8_0" in warnings[0]
