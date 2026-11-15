"""scout bench sem rede e sem ferramentas reais: Ollama e llama-bench simulados por monkeypatch."""
from __future__ import annotations

import json

import pytest

from conftest import CATALOG_PATH, FIX, SYSTEM_PATHS, load
from scout import bench as B
from scout.cli import main

TUNING = load(FIX / "tuning_test.json")
SYS = load(SYSTEM_PATHS["ddr5_64"])
CAT = load(CATALOG_PATH)
AC = {"on_ac": True, "profile": "performance"}


def target(model_id="test/dense-7b"):
    return B.build_target(SYS, CAT, model_id, 8192, TUNING)


def seq_measure(values, prefill=None):
    """measure(n) que devolve values[n] (n=0 é o aquecimento)."""
    pre = prefill if prefill is not None else [10.0] * len(values)

    def m(n):
        return B.Round(values[n], pre[n], 532)
    return m


# ---- estabilização

def test_stabilizes_on_fourth_round():
    # aquecimento 2.0 (descartado); rodadas 5.0, 6.0, 7.0, 7.2 -> par (3,4) a 2,8%
    res = B.stabilize(seq_measure([2.0, 5.0, 6.0, 7.0, 7.2, 99.0, 99.0, 99.0]))
    assert len(res["rounds"]) == 4 and res["stable"] is True
    assert [r["used_decode"] for r in res["rounds"]] == [False, False, True, True]
    assert res["decode_tps"] == pytest.approx(7.1)
    assert res["warmup"]["discarded"] is True and res["warmup"]["decode_tps"] == 2.0
    assert res["prefill_stable"] is True and res["prefill_tps"] == 10.0


def test_does_not_stabilize_in_six():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # cada rodada sobe >5%
    res = B.stabilize(seq_measure(vals))
    assert len(res["rounds"]) == 6 and res["stable"] is False
    assert res["decode_tps"] == pytest.approx(6.5)  # duas últimas
    assert [r["used_decode"] for r in res["rounds"]] == [False] * 4 + [True, True]


def test_prefill_unstable_is_null_but_decode_stable():
    dec = [7.0] * 7
    pre = [None, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    res = B.stabilize(seq_measure(dec, pre))
    assert len(res["rounds"]) == 6  # roda até o teto esperando o prefill
    assert res["stable"] is True and res["prefill_stable"] is False and res["prefill_tps"] is None


def test_first_stable_pair_tolerance():
    assert B.first_stable_pair([100.0, 95.0]) == 1      # 5,0%
    assert B.first_stable_pair([100.0, 94.0]) is None   # 6%
    assert B.first_stable_pair([1.0]) is None


# ---- ajuste

def test_fit_2p_recovers_bandwidth_and_overhead():
    bw, ovh = 20e9, 0.05
    pts = [(b, 1 / (b / bw + ovh)) for b in (2_500_000_000, 8_000_000_000, 4_000_000_000)]
    f = B.fit_calibration(pts)
    assert f["form"] == "2p" and abs(f["bandwidth_bytes_per_s"] - bw) / bw < 1e-6
    assert f["overhead_s_per_token"] == pytest.approx(ovh) and f["discard_reason"] is None
    assert f["two_param"]["bandwidth_bytes_per_s"] == f["bandwidth_bytes_per_s"]


def test_fit_falls_back_to_1p_when_overhead_under_5_percent():
    bw, ovh = 20e9, 0.002  # t do menor = 0,127 s -> 5% = 0,0063
    pts = [(b, 1 / (b / bw + ovh)) for b in (2_500_000_000, 8_000_000_000)]
    f = B.fit_calibration(pts)
    assert f["form"] == "1p" and f["overhead_s_per_token"] == 0.0
    assert "5%" in f["discard_reason"] and f["two_param"]["overhead_s_per_token"] == pytest.approx(ovh)
    assert isinstance(f["bandwidth_bytes_per_s"], int)


def test_fit_negative_overhead_and_single_model():
    pts = [(2_000_000_000, 10.0), (8_000_000_000, 2.4)]  # 4x bytes, mais de 4x mais rápido
    f = B.fit_calibration(pts)
    assert f["two_param"]["overhead_s_per_token"] < 0 and f["form"] == "1p" and f["discard_reason"] == "overhead negativo"
    one = B.fit_calibration([(2_500_000_000, 7.94)])
    assert one["form"] == "1p" and one["overhead_s_per_token"] == 0.0
    assert one["bandwidth_bytes_per_s"] == round(2_500_000_000 * 7.94)  # ~19,8 GB/s
    assert B.fit_calibration([(1_000, 5.0), (1_000, 6.0)])["discard_reason"] == "modelos do mesmo tamanho"
    with pytest.raises(ValueError):
        B.fit_calibration([(0, 1.0)])


# ---- alvo

def test_target_matches_plan_and_bytes():
    t = target()
    assert (t.quant, t.n_gpu_layers, t.pool, t.threads, t.ctx) == ("Q8_0", 0, "cpu", 16, 8192)
    assert t.bytes_per_token == 7_650_000_000 and t.file_bytes == 7_650_000_000
    moe = target("test/moe-30b-a3b")
    assert moe.bytes_per_token < moe.file_bytes // 5  # só a fração ativa é lida por token


def test_target_refusals():
    with pytest.raises(B.BenchError) as e:
        target("test/nao-existe")
    assert e.value.reason == "model_not_in_catalog"
    with pytest.raises(B.BenchError) as e:
        B.build_target(load(SYSTEM_PATHS["cpu8"]), CAT, "test/dense-7b", 8192, TUNING)
    assert e.value.reason == "not_a_candidate"
    with pytest.raises(B.BenchError) as e:
        target("test/tiny-1b")  # quant escolhida sem file_bytes
    assert e.value.reason == "no_file_bytes"
    with pytest.raises(B.BenchError) as e:
        target("test/embed-110m")
    assert e.value.reason == "decode_not_applicable"


# ---- Ollama simulado

class FakeOllama:
    def __init__(self, models, decode, prefill=None, blob_delta=0):
        self.models, self.decode, self.prefill = models, list(decode), list(prefill or [])
        self.blob_delta, self.calls, self.generated = blob_delta, [], []

    def __call__(self, method, path, body=None, timeout=600.0):
        self.calls.append((method, path))
        if path == "/api/tags":
            return {"models": [{"name": n, "size": s + self.blob_delta} for n, s in self.models.items()]}
        if path == "/api/show":
            return {"model_info": {"general.architecture": "qwen3"}}
        if path == "/api/version":
            return {"version": "0.0.test"}
        assert path == "/api/generate"
        self.generated.append(body)
        i = len(self.generated) - 1
        tps = self.decode[min(i, len(self.decode) - 1)]
        pre = self.prefill[min(i, len(self.prefill) - 1)] if self.prefill else 12.0
        return {"eval_count": 128, "eval_duration": int(128 / tps * 1e9),
                "prompt_eval_count": 532, "prompt_eval_duration": int(532 / pre * 1e9)}


def test_ollama_measure_end_to_end(monkeypatch):
    t = target()
    fake = FakeOllama({"dense:7b": t.file_bytes}, [1.0, 0.8, 0.9, 1.0, 1.01])
    monkeypatch.setattr(B, "_ollama_request", fake)
    r = B.measure_model(t, "ollama", None, "dense:7b", AC, now=lambda: __import__("datetime").datetime(2026, 11, 15))
    assert r["tool"] == "ollama" and r["tool_version"] == "0.0.test" and r["architecture"] == "qwen3"
    assert r["stable"] is True and len(r["rounds"]) == 4
    assert r["effective_bandwidth_bytes_per_s"] == round(t.bytes_per_token * r["decode_tps"])
    assert r["prompt_tokens"] == 532 and r["gen_tokens"] == 128 and r["power"] == AC
    # parâmetros da medição: temperatura 0, semente fixa, 128 tokens, threads de performance, prefixo por rodada
    g = fake.generated
    assert all(b["options"]["temperature"] == 0 and b["options"]["seed"] == B.SEED
               and b["options"]["num_predict"] == 128 and b["options"]["num_thread"] == 16
               and b["options"]["num_gpu"] == 0 and b["options"]["num_ctx"] == 8192 for b in g)
    assert len({b["prompt"][:12] for b in g}) == len(g)  # prefixo diferente em toda rodada (sem cache de prompt)


def test_ollama_quant_mismatch_refuses_without_measuring(monkeypatch):
    t = target()
    fake = FakeOllama({"dense:7b": t.file_bytes}, [1.0], blob_delta=5 * B.MIB)
    monkeypatch.setattr(B, "_ollama_request", fake)
    with pytest.raises(B.BenchError) as e:
        B.measure_model(t, "ollama", None, "dense:7b", AC)
    assert e.value.reason == "quant_mismatch" and not fake.generated
    fake2 = FakeOllama({"dense:7b": t.file_bytes}, [1.0], blob_delta=B.MIB)  # no limite: casa
    monkeypatch.setattr(B, "_ollama_request", fake2)
    B.ollama_check_quant("dense:7b", t)
    fake3 = FakeOllama({}, [1.0])
    monkeypatch.setattr(B, "_ollama_request", fake3)
    with pytest.raises(B.BenchError) as e:
        B.ollama_check_quant("dense:7b", t)
    assert e.value.reason == "ollama_model_missing"


def test_ollama_short_generation(monkeypatch):
    t = target()

    def short(method, path, body=None, timeout=600.0):
        return {"eval_count": 3, "eval_duration": 10**9, "prompt_eval_count": 5, "prompt_eval_duration": 10**9}
    monkeypatch.setattr(B, "_ollama_request", short)
    with pytest.raises(B.BenchError) as e:
        B.ollama_round("x", t, "p", 1)
    assert e.value.reason == "short_generation"


def test_ollama_unreachable(monkeypatch):
    def boom(*a, **k):
        raise OSError("sem rede")
    monkeypatch.setattr(B.urllib.request, "urlopen", boom)
    monkeypatch.setattr(B.urllib.request, "urlopen", boom)
    with pytest.raises(B.BenchError) as e:
        B._ollama_request("GET", "/api/tags")
    assert e.value.reason == "ollama_unreachable"


# ---- llama-bench simulado

LB = json.dumps([{"build_commit": "abc123", "build_number": 42, "n_prompt": 512, "n_gen": 0, "avg_ts": 40.5},
                 {"build_commit": "abc123", "build_number": 42, "n_prompt": 0, "n_gen": 128, "avg_ts": 6.25}])


def test_parse_llama_bench():
    assert B.parse_llama_bench("aviso\n" + LB) == (6.25, 40.5, "42 (abc123)")
    with pytest.raises(B.BenchError):
        B.parse_llama_bench("não é json")
    with pytest.raises(B.BenchError):
        B.parse_llama_bench(json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0}]))


def test_choose_tool(monkeypatch, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_text("x")
    monkeypatch.setattr(B.shutil, "which", lambda b: "/usr/bin/llama-bench" if b == "llama-bench" else None)
    assert B.choose_tool(str(f), "o:1") == ("llama-bench", [])
    assert B.choose_tool(str(tmp_path / "falta.gguf"), "o:1")[0] == "ollama"
    tool, missing = B.choose_tool(str(tmp_path / "falta.gguf"), None)
    assert tool is None and any("GGUF" in m for m in missing)
    monkeypatch.setattr(B.shutil, "which", lambda b: None)
    tool, missing = B.choose_tool(str(f), None)
    assert tool is None and any("PATH" in m for m in missing)


def test_llama_bench_measure_uses_plan_parameters(monkeypatch):
    t = target()
    cmds = []

    def fake_exec(cmd, timeout):
        cmds.append(cmd)
        return LB
    monkeypatch.setattr(B, "_exec", fake_exec)
    r = B.measure_model(t, "llama-bench", "/x/m.gguf", None, AC)
    assert r["tool"] == "llama-bench" and r["tool_version"] == "42 (abc123)" and r["stable"] is True
    c = cmds[0]
    assert c[c.index("-ngl") + 1] == "0" and c[c.index("-t") + 1] == "16" and c[c.index("-n") + 1] == "128"
    assert r["decode_tps"] == 6.25 and r["prompt_tokens"] == 512


# ---- energia

def test_read_power_battery(monkeypatch):
    monkeypatch.setattr(B.C, "collect_power", lambda system, probes: {
        "has_battery": B.C.field(True, "x"), "on_battery": B.C.field(True, "x")})
    monkeypatch.setattr(B.C, "read_text", lambda path, probes, probe=None, limit=0: "power-saver\n")
    assert B.read_power() == {"on_ac": False, "profile": "power-saver"}
    monkeypatch.setattr(B.C, "collect_power", lambda system, probes: {
        "has_battery": B.C.field(True, "x"), "on_battery": B.C.missing("?")})
    monkeypatch.setattr(B.C, "read_text", lambda *a, **k: None)
    monkeypatch.setattr(B.C, "run_cmd", lambda *a, **k: "balanced\n")
    assert B.read_power() == {"on_ac": None, "profile": "balanced"}


# ---- calibração e CLI

def _result(model_id, bytes_pt, tps, pool="cpu", stable=True, prefill=11.0, power=AC):
    return {"model_id": model_id, "bytes_per_token": bytes_pt, "decode_tps": tps, "pool": pool, "stable": stable,
            "prefill_tps": prefill, "power": power, "tool": "ollama", "architecture": "qwen3"}


def test_build_calibration_per_pool():
    cal = B.build_calibration([_result("a", 2_500_000_000, 7.94), _result("b", 100, 5.0, pool="gpu", stable=False, prefill=None)],
                              "2026-11-15")
    c = cal["cpu"]
    assert c["form"] == "1p" and c["overhead_s_per_token"] == 0 and c["bandwidth_bytes_per_s"] == round(2.5e9 * 7.94)
    assert c["stable"] is True and c["power"] == AC and c["prefill_tokens_per_s"] == 11.0 and c["prefill_reason"] is None
    assert c["models"] == [{"id": "a", "architecture": "qwen3", "bytes": 2_500_000_000}] and c["date"] == "2026-11-15"
    g = cal["gpu"]
    assert g["stable"] is False and g["prefill_tokens_per_s"] is None and g["prefill_reason"]
    mixed = B.build_calibration([_result("a", 1_000, 5.0), _result("b", 4_000, 1.0, power={"on_ac": False, "profile": None})], "d")
    assert mixed["cpu"]["power"] == {"on_ac": None, "profile": None}


def test_write_calibration_never_touches_eta(tmp_path):
    p = tmp_path / "tuning.json"
    p.write_text(json.dumps(TUNING), encoding="utf-8")
    B.write_calibration(p, {"cpu": {"bandwidth_bytes_per_s": 1}})
    out = json.loads(p.read_text(encoding="utf-8"))
    assert out["eta_gpu"] == TUNING["eta_gpu"] and out["eta_cpu"] == TUNING["eta_cpu"]
    assert out["max_bpw_offload"] == TUNING["max_bpw_offload"] and out["calibration"]["cpu"] == {"bandwidth_bytes_per_s": 1}
    B.write_calibration(p, {"gpu": {"bandwidth_bytes_per_s": 2}})
    assert set(json.loads(p.read_text(encoding="utf-8"))["calibration"]) == {"cpu", "gpu"}


def _cli_setup(monkeypatch, tmp_path, power, delta=0):
    t = target()
    tuning = tmp_path / "tuning.json"
    tuning.write_text(json.dumps(TUNING), encoding="utf-8")
    monkeypatch.setattr(B, "_ollama_request", FakeOllama({"dense:7b": t.file_bytes}, [1.0], blob_delta=delta))
    monkeypatch.setattr(B, "read_power", lambda: power)
    return tuning, ["bench", "--model", "test/dense-7b", "--ollama-model", "dense:7b", "--system",
                    str(SYSTEM_PATHS["ddr5_64"]), "--catalog", str(CATALOG_PATH), "--tuning", str(tuning),
                    "--out", str(tmp_path / "measured.json")]


def test_cli_dry_run_measures_nothing_and_writes_nothing(monkeypatch, tmp_path, capsys):
    tuning, argv = _cli_setup(monkeypatch, tmp_path, AC)
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Nada foi medido" in out and "dense:7b" in out
    assert not (tmp_path / "measured.json").exists() and "calibration" not in json.loads(tuning.read_text())
    assert not B._ollama_request.generated


def test_cli_run_on_battery_warns_and_records(monkeypatch, tmp_path, capsys):
    tuning, argv = _cli_setup(monkeypatch, tmp_path, {"on_ac": False, "profile": "power-saver"})
    assert main(argv + ["--run"]) == 0
    cap = capsys.readouterr()
    assert "BATERIA" in cap.err
    doc = json.loads((tmp_path / "measured.json").read_text(encoding="utf-8"))
    assert doc["on_battery_warning"] is True and doc["power"]["profile"] == "power-saver"
    m = doc["models"][0]
    assert m["quant"] == "Q8_0" and m["rounds"] and m["pool"] == "cpu" and m["on_battery_warning"] is True
    t = json.loads(tuning.read_text())
    assert t["calibration"]["cpu"]["power"]["on_ac"] is False and t["calibration"]["cpu"]["form"] == "1p"
    assert t["eta_cpu"] == TUNING["eta_cpu"]


def test_cli_quant_mismatch_writes_no_calibration(monkeypatch, tmp_path, capsys):
    tuning, argv = _cli_setup(monkeypatch, tmp_path, AC, delta=10 * B.MIB)
    assert main(argv + ["--run"]) == 1
    assert "não casa com Q8_0" in capsys.readouterr().err
    assert "calibration" not in json.loads(tuning.read_text())


def test_cli_run_unstable_is_flagged(monkeypatch, tmp_path, capsys):
    tuning, argv = _cli_setup(monkeypatch, tmp_path, AC)
    monkeypatch.setattr(B, "_ollama_request", FakeOllama({"dense:7b": target().file_bytes}, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert main(argv + ["--run"]) == 0
    assert "NÃO ESTABILIZOU" in capsys.readouterr().out
    c = json.loads(tuning.read_text())["calibration"]["cpu"]
    assert c["stable"] is False
