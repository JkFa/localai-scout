"""Embeddings: ctx efetivo = min(intent.ctx, ctx_max), sem ctx_exceeds_max (DECISIONS.md)."""
from __future__ import annotations

import copy

from conftest import schema_validator
from scout import estimate as E
from scout.solver import Intent, solve


def _by_id(f: dict, mid: str) -> dict | None:
    return next((c for c in f["candidates"] if c["model"]["id"] == mid), None)


def test_embeddings_ctx_is_clamped_not_rejected(systems, catalog, tuning):
    f = solve(systems["cpu8"], catalog, Intent(tasks=("embeddings",), ctx=32768), tuning)
    c = _by_id(f, "test/embed-110m")
    assert c is not None
    assert not any(r["model"]["id"] == "test/embed-110m" for r in f["rejected"])
    assert c["ctx_effective"] == 8192
    m = next(m for m in catalog["models"] if m["id"] == "test/embed-110m")
    assert c["kv_bytes"] == E.kv_bytes(m["n_layers"], m["n_kv_heads"], m["head_dim"], 8192)
    assert c["buffer_bytes"] == E.buffer_bytes(8192)
    assert list(schema_validator("feasibility.schema.json").iter_errors(f)) == []


def test_embeddings_ctx_below_max_is_kept(systems, catalog, tuning):
    f = solve(systems["cpu8"], catalog, Intent(tasks=("embeddings",), ctx=512), tuning)
    assert _by_id(f, "test/embed-110m")["ctx_effective"] == 512


def test_other_tasks_still_rejected_and_no_ctx_effective(systems, catalog, tuning):
    f = solve(systems["ddr5_64"], catalog, Intent(tasks=("chat", "embeddings"), ctx=65536), tuning)
    codes = {r["model"]["id"]: r["reason_code"] for r in f["rejected"]}
    assert codes["test/dense-7b"] == "ctx_exceeds_max"
    assert _by_id(f, "test/embed-110m")["ctx_effective"] == 8192
    assert all("ctx_effective" not in c for c in f["candidates"] if c["model"]["id"] != "test/embed-110m")


def test_mixed_task_model_evaluated_for_chat_is_rejected(systems, catalog, tuning):
    cat = copy.deepcopy(catalog)
    m = next(m for m in cat["models"] if m["id"] == "test/embed-110m")
    m["tasks"] = ["chat", "embeddings"]
    f = solve(systems["ddr5_64"], cat, Intent(tasks=("chat", "embeddings"), ctx=16384), tuning)
    assert {r["model"]["id"]: r["reason_code"] for r in f["rejected"]}["test/embed-110m"] == "ctx_exceeds_max"
