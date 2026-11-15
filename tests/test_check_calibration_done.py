"""Matriz das 10 checagens do "Critério de pronto" (schemas/calibration_store_spec.md).

Roda o próprio `schemas/check_calibration_done.py` contra um `$T` sintético montado em tmp_path: máquina
fictícia da fixture ddr5_64, catálogo de um modelo só, `cal.json` escrito à mão (identidade copiada do
system.json, não por `machine_identity`). Nenhum dado real do catálogo e nenhuma rede.

Além do caminho feliz, há sabotagens: o script tem de falhar (código ≠ 0, linha FALHA) quando o `$T` ou o
produto estão errados, senão as 10 linhas OK não provam nada.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FIX, ROOT, SYSTEM_PATHS, load

SCRIPT = ROOT / "schemas" / "check_calibration_done.py"
MODEL = "test/synth-4b"
GB = 1_000_000_000
BW = 40 * GB
BYTES = 3_280_000_000  # 4e9 params x 6,56 bpw / 8; medido = 40e9 / 3,28e9 = 12,2 tok/s
CHECK_IDS = list(range(1, 11))


def _v(n):
    return n["value"]


def _catalog() -> dict:
    # Um modelo sintético de 4B derivado do fixture, com um único quant de bytes conhecidos (Q6_K, bpw 6,56: acima do piso de qualidade).
    m = copy.deepcopy(next(m for m in load(FIX / "catalog_test.json")["models"] if m["id"] == "test/dense-7b"))
    m.update(id=MODEL, params_total=4 * GB, params_active=4 * GB, tasks=["chat"],
             source_url="https://example.invalid/test/synth-4b", gguf_repo="test/synth-4b-gguf",
             license_source="https://example.invalid/test/synth-4b")
    m["quants"] = [{"format": "GGUF", "name": "Q6_K", "bpw": 6.56, "bpw_source": "file_bytes",
                    "file_bytes": BYTES, "file_url": None}]
    return {"schema_version": 1, "kind": "catalog", "models": [m]}


def _identity(s: dict) -> dict:
    return {"platform": s["platform"], "os.arch": _v(s["os"]["arch"]), "os.kernel": _v(s["os"]["kernel"]),
            "cpu.model": _v(s["cpu"]["model"]), "cpu.cores_physical": _v(s["cpu"]["cores_physical"]),
            "memory.ram_total_bytes": _v(s["memory"]["ram_total_bytes"]),
            "gpu.name": None, "gpu.backend": None, "gpu.vram_total_bytes": None,
            "memory.speed_mts": _v(s["memory"]["speed_mts"]), "memory.channels": _v(s["memory"]["channels"])}


def _cal(s: dict) -> dict:
    return {"schema_version": 1, "kind": "calibration", "scout_version": "0.0.0", "calibration_schema_version": 1,
            "machine": _identity(s),
            "pools": {"cpu": [{
                "bandwidth_bytes_per_s": BW, "overhead_s_per_token": 0.0, "form": "1p", "stable": True,
                "power": {"on_ac": True, "profile": "performance", "not_applicable": []},
                "tool": "ollama", "tool_version": "9.9.9", "date": "2000-01-01",
                "models": [{"id": MODEL, "architecture": "testarch", "bytes": BYTES}],
                "prefill_tokens_per_s": None}]}}


def make_T(root: Path, *, mutate_cal=None, mutate_system=None) -> Path:
    T = root / "T"
    T.mkdir()
    system = load(SYSTEM_PATHS["ddr5_64"])
    if mutate_system:
        mutate_system(system)
    cal = _cal(load(SYSTEM_PATHS["ddr5_64"]))  # a calibração nasce da máquina ORIGINAL
    if mutate_cal:
        mutate_cal(cal)
    (T / "system.json").write_text(json.dumps(system), encoding="utf-8")
    (T / "cal.json").write_text(json.dumps(cal, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (T / "catalog.json").write_text(json.dumps(_catalog()), encoding="utf-8")
    return T


def args(T: Path, *extra: str) -> list[str]:
    return [str(T), "--catalog", str(T / "catalog.json"), "--tuning", str(FIX / "tuning_test.json"),
            "--other-system", str(SYSTEM_PATHS["rtx3060"]), "--model", MODEL, *extra]


def run(T: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args(T, *extra)], capture_output=True, text=True,
                          cwd=T.parent, timeout=120)


def verdicts(stdout: str) -> dict[int, str]:
    return {int(m.group(2)): m.group(1) for m in re.finditer(r"^(OK|FALHA)\s+(\d+)\.", stdout, re.M)}


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    T = make_T(tmp_path_factory.mktemp("good"))
    before = {p.name: p.read_bytes() for p in T.iterdir()}
    return T, run(T), before


# ------------------------------------------------------------------ caminho feliz


def test_good_T_passes_with_exit_zero_and_ten_ok_lines(good):
    T, r, _ = good
    assert r.returncode == 0, r.stdout + r.stderr
    assert verdicts(r.stdout) == {n: "OK" for n in CHECK_IDS}
    assert r.stdout.strip().splitlines()[-1] == "10/10 checagens OK"


@pytest.mark.parametrize("n", CHECK_IDS)
def test_matrix_each_criterion_has_its_own_ok_line(good, n):
    _, r, _ = good
    assert verdicts(r.stdout).get(n) == "OK", r.stdout


def test_script_does_not_touch_cal_or_system_and_writes_variations_only_in_T(good):
    T, _, before = good
    for name, content in before.items():
        assert (T / name).read_bytes() == content, f"{name} foi alterado"
    new = {p.name for p in T.iterdir()} - set(before)
    assert new and all(re.match(r"(cal|system|plan)_.+\.json$", n) for n in new), new
    assert {"cal_schema99.json", "system_mem_unknown.json", "plan_disabled.json"} <= new


def test_system_flag_overrides_default_location(tmp_path):
    T = make_T(tmp_path)
    elsewhere = tmp_path / "outro_system.json"
    elsewhere.write_bytes((T / "system.json").read_bytes())
    (T / "system.json").unlink()
    assert run(T).returncode == 2  # sem system.json em $T e sem --system: não serve
    r = run(T, "--system", str(elsewhere))
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ sabotagem do $T


def _speed_null(cal):
    cal["machine"]["memory.speed_mts"] = None


def _bytes_x4(cal):
    cal["pools"]["cpu"][0]["models"][0]["bytes"] = 4 * BYTES  # previsto (catálogo) e medido (banda/bytes) divergem


def _other_cpu(cal):
    cal["machine"]["cpu.model"] = "OUTRA CPU"


def _on_battery(cal):
    cal["pools"]["cpu"][0]["power"] = {"on_ac": False, "profile": "balanced", "not_applicable": []}


@pytest.mark.parametrize("mutate,must_fail", [
    (_bytes_x4, {1}),            # a checagem 1 compara com banda x bytes do próprio cal.json
    (_other_cpu, {1, 2, 5, 6}),  # camada 1 diverge: o arquivo nunca é aplicado
    (_speed_null, {1, 3}),       # sem MT/s gravado: base em medium e a divergência de 4800 não é possível
    (_on_battery, {7}),          # pré-condição da checagem 7 (entrada base na tomada/performance)
])
def test_sabotaged_T_fails_loudly(tmp_path, mutate, must_fail):
    r = run(make_T(tmp_path, mutate_cal=mutate))
    assert r.returncode == 1, r.stdout + r.stderr
    v = verdicts(r.stdout)
    assert len(v) == 10 and all(v[n] == "FALHA" for n in must_fail), r.stdout
    assert f"{10 - sum(1 for x in v.values() if x == 'FALHA')}/10 checagens OK" in r.stdout


def test_missing_cal_is_a_setup_error(tmp_path):
    T = make_T(tmp_path)
    (T / "cal.json").unlink()
    assert run(T).returncode == 2


def test_model_not_in_cal_is_a_setup_error(tmp_path):
    assert run(make_T(tmp_path), "--model", "test/nao-existe").returncode == 2


# ------------------------------------------------------------------ sabotagem do produto (em processo)


def _load_script():
    spec = importlib.util.spec_from_file_location("check_calibration_done_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_in_process(T, capsys):
    mod = _load_script()
    rc = mod.main(args(T))
    return rc, verdicts(capsys.readouterr().out)


def test_in_process_baseline_matches_subprocess(tmp_path, capsys):
    rc, v = _run_in_process(make_T(tmp_path), capsys)
    assert rc == 0 and v == {n: "OK" for n in CHECK_IDS}


def test_mutant_kernel_key_without_major_minor_is_caught_by_check_5(tmp_path, capsys, monkeypatch):
    from scout import calibration as C

    monkeypatch.setattr(C, "kernel_key", lambda s: s)  # patch passa a divergir
    rc, v = _run_in_process(make_T(tmp_path), capsys)
    assert rc == 1 and v[5] == "FALHA"
    assert [n for n, x in v.items() if x == "FALHA"] == [5]


def test_mutant_power_always_equal_is_caught(tmp_path, capsys, monkeypatch):
    from scout import calibration as C

    monkeypatch.setattr(C, "compare_power_attr", lambda stored, current, attr: "equal")
    rc, v = _run_in_process(make_T(tmp_path), capsys)
    assert rc == 1 and {7, 8} <= {n for n, x in v.items() if x == "FALHA"}


def test_script_restores_bench_functions_and_never_opens_a_socket(tmp_path, capsys, monkeypatch):
    import socket

    from scout import bench

    orig = (bench.read_power, bench.engine_version)

    def no_net(*a, **k):
        raise AssertionError("rede acessada")

    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.setattr(socket, "create_connection", no_net)
    rc, v = _run_in_process(make_T(tmp_path), capsys)
    assert rc == 0 and len(v) == 10
    assert (bench.read_power, bench.engine_version) == orig
