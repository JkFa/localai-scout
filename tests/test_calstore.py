"""Armazenamento local da calibração (calibration_store_spec.md §1, §3, §4, §6). Tudo sintético."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import CATALOG_PATH, SYSTEM_PATHS, TUNING_PATH, assert_valid, deep, load
from scout import bench as B
from scout import calstore as S
from scout.calibration import CalibrationInput, machine_identity
from scout.cli import InputError, load_tuning, main, run_plan
from scout.solver import Intent

SYS = load(SYSTEM_PATHS["ddr5_64"])
ID = machine_identity(SYS)
AC = {"on_ac": True, "profile": "performance", "not_applicable": []}
BAT = {"on_ac": False, "profile": "balanced", "not_applicable": []}
H = Path("/home/u")


def entry(power=AC, bw=20_000_000_000, tool="ollama"):
    return {"bandwidth_bytes_per_s": bw, "overhead_s_per_token": 0.0, "form": "1p", "stable": True, "power": power,
            "tool": tool, "tool_version": "9.9.9", "date": "2000-01-01", "prefill_tokens_per_s": 10.0,
            "prefill_reason": None, "models": [{"id": "x/y", "architecture": None, "bytes": 2_000_000_000}],
            "fit": {"two_param": None, "one_param": {"bandwidth_bytes_per_s": bw, "overhead_s_per_token": 0.0},
                    "discard_reason": "um único modelo"}}


def doc(pools=None, machine=ID):
    d, _ = S.merge_calibration(None, machine, {}, "0.0.0")
    d["pools"] = {k: (v if isinstance(v, list) else [v]) for k, v in (pools or {"cpu": entry()}).items()}
    return d


# ---- caminho
@pytest.mark.parametrize("flag,env,platform,home,expected", [
    ("/f.json", {"SCOUT_CALIBRATION": "/e.json"}, "linux", H, ("/f.json", "flag")),
    (None, {"SCOUT_CALIBRATION": "/e.json"}, "linux", H, ("/e.json", "env")),
    (None, {"SCOUT_CALIBRATION": ""}, "linux", H, ("/home/u/.local/state/localai-scout/calibration.json", "default")),
    (None, {"XDG_STATE_HOME": "/xdg"}, "linux", H, ("/xdg/localai-scout/calibration.json", "default")),
    (None, {"XDG_STATE_HOME": "relativo"}, "linux", H, ("/home/u/.local/state/localai-scout/calibration.json", "default")),
    (None, {}, "freebsd14", H, ("/home/u/.local/state/localai-scout/calibration.json", "default")),
    (None, {}, "darwin", H, ("/home/u/Library/Application Support/localai-scout/calibration.json", "default")),
    (None, {"LOCALAPPDATA": "/la"}, "win32", H, ("/la/localai-scout/calibration.json", "default")),
    (None, {}, "win32", H, ("/home/u/AppData/Local/localai-scout/calibration.json", "default")),
    (None, {}, "linux", None, (None, "default")),
    (None, {}, "darwin", None, (None, "default")),
    (None, {}, "win32", None, (None, "default")),
])
def test_resolve_calibration_path(flag, env, platform, home, expected):
    path, origin = S.resolve_calibration_path(flag, env, platform, home)
    assert (None if path is None else str(path), origin) == expected


# ---- leitura
def test_read_absent_and_flag(tmp_path):
    p = tmp_path / "c.json"
    assert S.read_calibration(p, "env") == CalibrationInput("env", None)
    assert S.read_calibration(p, "default").doc is None
    with pytest.raises(S.CalibrationError):
        S.read_calibration(p, "flag")
    assert S.read_calibration(p, "disabled") == CalibrationInput("disabled", None)
    assert S.read_calibration(None, "default").doc is None


def test_read_fails_loud_on_malformed(tmp_path):
    p = tmp_path / "c.json"
    for text in ("{", "[]", json.dumps({**doc(), "schema_version": 2}), json.dumps({**doc(), "pools": {}}),
                 json.dumps({**doc(), "pools": {"cpu": [{"bandwidth_bytes_per_s": 0}]}})):
        p.write_text(text)
        with pytest.raises(S.CalibrationError):
            S.read_calibration(p, "env")
    p.write_bytes(b"\xff\xfe")
    with pytest.raises(S.CalibrationError):
        S.read_calibration(p, "env")


def test_read_keeps_other_calibration_schema_version(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({**doc(), "calibration_schema_version": 99}))
    assert S.read_calibration(p, "env").doc["calibration_schema_version"] == 99  # o plan descarta


def test_written_doc_matches_json_schema():
    d = doc({"cpu": [entry(AC), entry(BAT, 10_000_000_000)], "unified": entry({"on_ac": None, "profile": None, "not_applicable": ["on_ac", "profile"]})})
    assert S.validate_doc(d) == []
    assert_valid("calibration.schema.json", d)


# ---- escrita
def test_write_is_deterministic_and_atomic(tmp_path, monkeypatch):
    p = tmp_path / "sub" / "c.json"
    d = doc()
    S.write_calibration_file(p, d)
    first = p.read_bytes()
    assert first.endswith(b"\n") and json.loads(first) == d
    assert (p.parent.stat().st_mode & 0o777) == 0o700
    S.write_calibration_file(p, deep(d))
    assert p.read_bytes() == first
    assert not [f for f in p.parent.iterdir() if f.name != "c.json"]

    def boom(src, dst):
        raise OSError(28, "disco cheio")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(S.CalibrationError):
        S.write_calibration_file(p, doc({"cpu": entry(bw=1)}))
    assert p.read_bytes() == first
    assert [f.name for f in p.parent.iterdir()] == ["c.json"]  # temporário removido


def test_write_refuses_invalid_doc(tmp_path):
    with pytest.raises(S.CalibrationError):
        S.write_calibration_file(tmp_path / "c.json", {"kind": "calibration"})
    assert not (tmp_path / "c.json").exists()


# ---- merge
def test_merge_replaces_same_power_key_and_keeps_others():
    old = doc({"cpu": [entry(AC, 20), entry(BAT, 10)]})
    new, notes = S.merge_calibration(old, ID, {"cpu": entry(AC, 30)}, "1")
    assert [e["bandwidth_bytes_per_s"] for e in new["pools"]["cpu"]] == [30, 10]
    assert any("substituída" in n for n in notes)
    new2, _ = S.merge_calibration(old, ID, {"cpu": entry({**AC, "profile": "power-saver"}, 5)}, "1")
    assert len(new2["pools"]["cpu"]) == 3
    assert len(old["pools"]["cpu"]) == 2  # entrada não muta


def test_merge_not_applicable_state_is_one_key():
    na = {"on_ac": None, "profile": None, "not_applicable": ["on_ac", "profile"]}
    new, _ = S.merge_calibration(doc({"cpu": entry(na, 1)}), ID, {"cpu": entry(na, 2)}, "1")
    assert [e["bandwidth_bytes_per_s"] for e in new["pools"]["cpu"]] == [2]
    # conhecido de um lado e não aplicável do outro: chaves diferentes
    new, _ = S.merge_calibration(doc({"cpu": entry(na, 1)}), ID, {"cpu": entry(AC, 2)}, "1")
    assert len(new["pools"]["cpu"]) == 2


def test_merge_drops_old_pools_on_identity_change():
    old = doc({"cpu": entry(), "gpu": entry()})
    changed = {**ID, "cpu.model": "Outra CPU"}
    new, notes = S.merge_calibration(old, changed, {"cpu": entry(BAT)}, "1")
    assert len(new["pools"]["cpu"]) == 1 and "gpu" not in new["pools"]
    assert sum("descartado" in n for n in notes) == 2 and "cpu.model" in notes[0]
    assert new["machine"] == changed


def test_merge_gpu_field_does_not_drop_cpu_pool_and_kernel_never_drops():
    old_machine = {**ID, "gpu.name": "GPU A"}
    old = doc({"cpu": entry(), "gpu": entry()}, old_machine)
    cur = {**ID, "gpu.name": "GPU B", "os.kernel": "99.1.0"}
    new, _ = S.merge_calibration(old, cur, {}, "1")
    assert set(new["pools"]) == {"cpu"} and new["machine"]["os.kernel"] == "99.1.0"


def test_merge_layer2_unknown_keeps_known_different_drops():
    old = doc({"cpu": entry()}, {**ID, "memory.speed_mts": None})
    new, _ = S.merge_calibration(old, {**ID, "memory.speed_mts": 5600}, {}, "1")
    assert "cpu" in new["pools"]
    old = doc({"cpu": entry()}, {**ID, "memory.speed_mts": 4800})
    new, _ = S.merge_calibration(old, {**ID, "memory.speed_mts": 5600}, {}, "1")
    assert "cpu" not in new["pools"]


def test_merge_drops_everything_on_schema_version_change():
    old = {**doc(), "calibration_schema_version": 7}
    new, notes = S.merge_calibration(old, ID, {"gpu": entry()}, "1")
    assert set(new["pools"]) == {"gpu"} and "calibration_schema_version" in notes[0]


# ---- recusas
def test_identity_refusal_cases():
    assert S.identity_refusal(ID, ID, {"cpu"}) is None
    assert S.identity_refusal({**ID, "cpu.model": None}, ID, {"cpu"})[0] == "identity_incomplete"
    assert S.identity_refusal({**ID, "gpu.name": None}, ID, {"gpu"})[0] == "identity_incomplete"
    assert S.identity_refusal({**ID, "gpu.name": None}, ID, {"cpu"}) is None  # gpu.* só vale no pool gpu
    assert S.identity_refusal(ID, {**ID, "cpu.model": "x"}, {"cpu"})[0] == "system_not_this_machine"
    assert S.identity_refusal(ID, None, {"cpu"})[0] == "system_not_this_machine"
    assert S.identity_refusal({**ID, "memory.speed_mts": None}, {**ID, "memory.speed_mts": 1}, {"cpu"}) is None
    assert S.mixed_tools([("cpu", "ollama"), ("cpu", "llama-bench"), ("gpu", "ollama")]) == "cpu"
    assert S.mixed_tools([("cpu", "ollama"), ("cpu", None), ("gpu", "llama-bench")]) is None


# ---- CLI: plan, serve e tuning
def _plan_argv(*extra):
    return ["plan", "--system", str(SYSTEM_PATHS["ddr5_64"]), "--catalog", str(CATALOG_PATH),
            "--tuning", str(TUNING_PATH), "--json", *extra]


def test_plan_reports_absent_calibration(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("SCOUT_CALIBRATION", str(tmp_path / "nao.json"))
    assert main(_plan_argv()) == 0
    cal = json.loads(capsys.readouterr().out)["inputs"]["calibration"]
    assert cal["status"] == "absent" and cal["origin"] == "env"
    assert not (tmp_path / "nao.json").exists()


def test_plan_flag_missing_file_is_error_and_flags_are_exclusive(capsys, tmp_path):
    assert main(_plan_argv("--calibration", str(tmp_path / "nao.json"))) == 2
    assert "não existe" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(_plan_argv("--calibration", "x", "--no-calibration"))


def test_plan_malformed_file_fails_loud(capsys, tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{")
    assert main(_plan_argv("--calibration", str(p))) == 2
    assert "não é JSON" in capsys.readouterr().err


def test_plan_no_calibration_reads_nothing(capsys, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("não deveria ler")
    monkeypatch.setattr(B, "read_power", boom)
    monkeypatch.setattr(B, "engine_version", boom)
    p = tmp_path / "c.json"
    p.write_text(S.serialize(doc()))
    assert main(_plan_argv("--no-calibration")) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inputs"]["calibration"]["status"] == "disabled" and "power" not in out["inputs"]


def test_plan_with_file_reads_power_and_engine_once_per_tool(capsys, tmp_path, fixed_env, monkeypatch):
    calls = []
    monkeypatch.setattr(B, "engine_version", lambda t: calls.append(t) or "9.9.9")
    p = tmp_path / "c.json"
    p.write_text(S.serialize(doc({"cpu": [entry(AC), entry(BAT, 5)]})))
    assert main(_plan_argv("--calibration", str(p))) == 0
    out = json.loads(capsys.readouterr().out)
    assert calls == ["ollama"] and out["inputs"]["calibration"]["status"] == "used"
    assert out["inputs"]["calibration"]["pools"]["cpu"]["power_match"] == "same"
    assert_valid("feasibility.schema.json", out)


def test_run_plan_without_doc_reads_nothing(monkeypatch, catalog):
    def boom(*a, **k):
        raise AssertionError
    monkeypatch.setattr(B, "read_power", boom)
    monkeypatch.setattr(B, "engine_version", boom)
    raw = CATALOG_PATH.read_bytes()
    for cal in (None, CalibrationInput("env", None), CalibrationInput("disabled", None)):
        run_plan(SYS, catalog, raw, load(TUNING_PATH), Intent(tasks=("chat",), ctx=8192), calibration=cal)


def test_load_tuning_rejects_old_calibration_key(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({**load(TUNING_PATH), "calibration": {"cpu": {}}}))
    with pytest.raises(InputError) as e:
        load_tuning(p)
    assert "scout bench --run" in str(e.value) and "localai-scout" in str(e.value)


def test_serve_gets_calibration_at_startup(tmp_path, fixed_env):
    from scout.server import ScoutServer

    cal = CalibrationInput("env", doc())
    s = ScoutServer(0, "t", CATALOG_PATH, tuning_path=TUNING_PATH, calibration=cal)
    try:
        assert s.calibration is cal
    finally:
        s.server_close()
