"""scout bench sem rede e sem ferramentas reais: Ollama e llama-bench simulados por monkeypatch."""
from __future__ import annotations

import json

import pytest

from conftest import CATALOG_PATH, FIX, SYSTEM_PATHS, load
from scout import bench as B
from scout import calstore
from scout.calibration import CALIBRATION_SCHEMA_VERSION, machine_identity
from scout.cli import main

TUNING = load(FIX / "tuning_test.json")
SYS = load(SYSTEM_PATHS["ddr5_64"])
CAT = load(CATALOG_PATH)
AC = {"on_ac": True, "profile": "performance", "not_applicable": []}


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
        self.blob_delta, self.calls, self.generated, self.unloaded = blob_delta, [], [], []

    def __call__(self, method, path, body=None, timeout=600.0):
        self.calls.append((method, path))
        if path == "/api/tags":
            return {"models": [{"name": n, "size": s + self.blob_delta} for n, s in self.models.items()]}
        if path == "/api/show":
            return {"model_info": {"general.architecture": "qwen3"}}
        if path == "/api/version":
            return {"version": "0.0.test"}
        assert path == "/api/generate"
        if body.get("keep_alive") == 0 and "prompt" not in body:
            self.unloaded.append(body["model"])
            return {}
        self.generated.append(body)
        i = len(self.generated) - 1
        tps = self.decode[min(i, len(self.decode) - 1)]
        pre = self.prefill[min(i, len(self.prefill) - 1)] if self.prefill else 12.0
        return {"eval_count": B.GEN_TOKENS, "eval_duration": int(B.GEN_TOKENS / tps * 1e9),
                "prompt_eval_count": 532, "prompt_eval_duration": int(532 / pre * 1e9)}


def test_ollama_measure_end_to_end(monkeypatch, fixed_env):
    t = target()
    fake = FakeOllama({"dense:7b": t.file_bytes}, [1.0, 0.8, 0.9, 1.0, 1.01])
    monkeypatch.setattr(B, "_ollama_request", fake)
    r = B.measure_model(t, "ollama", None, "dense:7b", AC, now=lambda: __import__("datetime").datetime(2026, 11, 15))
    assert r["tool"] == "ollama" and r["tool_version"] == "9.9.9" and r["architecture"] == "qwen3"
    assert r["stable"] is True and len(r["rounds"]) == 4
    # bytes por token = pesos + KV f16 na posição média P + N/2 = 532 + 512 = 1044:
    # 2*32*8*128*1044*2 = 136_839_168 -> 7_650_000_000 + 136_839_168
    assert r["weights_bytes_per_token"] == t.bytes_per_token == 7_650_000_000
    assert r["bytes_per_token"] == 7_786_839_168 and r["kv_quant_assumed"] == "f16"
    assert (r["decode_context_tokens"], r["decode_gen_tokens"]) == (532, B.GEN_TOKENS)
    assert r["effective_bandwidth_bytes_per_s"] == round(7_786_839_168 * r["decode_tps"])
    assert r["prompt_tokens"] == 532 and r["gen_tokens"] == B.GEN_TOKENS and r["power"] == AC
    # parâmetros da medição: temperatura 0, semente fixa, GEN_TOKENS tokens, threads de performance, prefixo por rodada
    g = fake.generated
    assert all(b["options"]["temperature"] == 0 and b["options"]["seed"] == B.SEED
               and b["options"]["num_predict"] == B.GEN_TOKENS and b["options"]["num_thread"] == 16
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
    monkeypatch.setattr(B.urllib.request.OpenerDirector, "open", boom)
    with pytest.raises(B.BenchError) as e:
        B._ollama_request("GET", "/api/tags")
    assert e.value.reason == "ollama_unreachable"


# ---- llama-bench simulado

LB = json.dumps([{"build_commit": "abc123", "build_number": 42, "n_prompt": 512, "n_gen": 0, "avg_ts": 40.5},
                 {"build_commit": "abc123", "build_number": 42, "n_prompt": 0, "n_gen": B.GEN_TOKENS, "avg_ts": 6.25}])


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


def test_llama_bench_measure_uses_plan_parameters(monkeypatch, fixed_env):
    t = target()
    cmds = []

    def fake_exec(cmd, timeout):
        cmds.append(cmd)
        return LB
    monkeypatch.setattr(B, "_exec", fake_exec)
    r = B.measure_model(t, "llama-bench", "/x/m.gguf", None, AC)
    assert r["tool"] == "llama-bench" and r["tool_version"] == "b9999" and r["stable"] is True
    c = cmds[0]
    assert c[c.index("-ngl") + 1] == "0" and c[c.index("-t") + 1] == "16" and c[c.index("-n") + 1] == str(B.GEN_TOKENS)
    assert r["decode_tps"] == 6.25 and r["prompt_tokens"] == 512
    # tg do llama-bench parte do contexto vazio: posição média 0 + 1024/2 = 512 -> KV 2*32*8*128*512*2
    assert (r["decode_context_tokens"], r["decode_gen_tokens"]) == (0, B.GEN_TOKENS)
    assert r["bytes_per_token"] == 7_650_000_000 + 67_108_864


# ---- energia

def test_decide_power_not_applicable_vs_unknown():
    d = B.decide_power
    assert d(False, None, None, False) == {"on_ac": None, "profile": None, "not_applicable": ["on_ac", "profile"]}
    # com bateria e leitura falhando: desconhecido, NÃO não aplicável
    assert d(True, None, None, True) == {"on_ac": None, "profile": None, "not_applicable": []}
    assert d(True, True, "balanced", True) == {"on_ac": False, "profile": "balanced", "not_applicable": []}
    assert d(False, None, "performance", True) == {"on_ac": None, "profile": "performance", "not_applicable": ["on_ac"]}
    # nada observado (plataforma sem leitura): tudo desconhecido
    assert d(None, None, None, None) == {"on_ac": None, "profile": None, "not_applicable": []}


def _fake_sys(tmp_path, supplies=(), profile=None):
    root = tmp_path / "sys"
    (root / "class" / "power_supply").mkdir(parents=True)
    for name, attrs in supplies:
        d = root / "class" / "power_supply" / name
        d.mkdir()
        for k, v in attrs.items():
            (d / k).write_text(v)
    if profile is not None:
        (root / "firmware" / "acpi").mkdir(parents=True)
        (root / "firmware" / "acpi" / "platform_profile").write_text(profile)
    return str(root)


def test_read_power_desktop_has_nothing_applicable(monkeypatch, tmp_path):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    root = _fake_sys(tmp_path, [("AC", {"type": "Mains", "online": "1"})])
    assert B.read_power(root, lambda b: None) == {"on_ac": None, "profile": None, "not_applicable": ["on_ac", "profile"]}


def test_read_power_battery_by_type_not_name(monkeypatch, tmp_path):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    root = _fake_sys(tmp_path, [("macsmc-battery", {"type": "Battery", "status": "Discharging", "scope": "System"}),
                                ("macsmc-ac", {"type": "Mains", "online": "0"})], profile="balanced\n")
    assert B.read_power(root, lambda b: None) == {"on_ac": False, "profile": "balanced", "not_applicable": []}


def test_read_power_peripheral_battery_is_not_the_system_battery(monkeypatch, tmp_path):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    root = _fake_sys(tmp_path, [("hidpp_battery_0", {"type": "Battery", "status": "Discharging", "scope": "Device"}),
                                ("AC", {"type": "Mains", "online": "1"})])
    assert B.read_power(root, lambda b: None)["not_applicable"] == ["on_ac", "profile"]


def test_read_power_unreadable_type_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    root = _fake_sys(tmp_path, [("X", {"status": "?"})])
    got = B.read_power(root, lambda b: None)
    assert "on_ac" not in got["not_applicable"]


def test_ollama_request_ignores_proxy_env(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    seen = {}

    def fake_open(self, req, data=None, timeout=None):
        seen["handlers"] = [type(h).__name__ for h in self.handlers]
        seen["proxies"] = {k: v for h in self.handlers if hasattr(h, "proxies") for k, v in h.proxies.items()}
        raise OSError("sem rede")
    monkeypatch.setattr(B.urllib.request.OpenerDirector, "open", fake_open)
    with pytest.raises(B.BenchError):
        B._ollama_request("GET", "/api/version", timeout=1.0)
    assert seen["proxies"] == {}


def test_read_power_laptop_battery_and_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    root = _fake_sys(tmp_path, [("BAT0", {"type": "Battery", "status": "Discharging", "scope": "System"}),
                                ("AC", {"type": "Mains", "online": "0"})], profile="low-power\n")
    assert B.read_power(root, lambda b: None) == {"on_ac": False, "profile": "low-power", "not_applicable": []}


def test_read_power_daemon_present_but_read_fails_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    monkeypatch.setattr(B.C, "run_cmd", lambda *a, **k: None)  # powerprofilesctl existe e falha
    root = _fake_sys(tmp_path, [("AC", {"type": "Mains", "online": "1"})])
    got = B.read_power(root, lambda b: "/usr/bin/powerprofilesctl" if b == "powerprofilesctl" else None)
    assert got == {"on_ac": None, "profile": None, "not_applicable": ["on_ac"]}


def test_read_power_unlistable_sysfs_is_unknown_not_na(monkeypatch, tmp_path):
    monkeypatch.setattr(B.platform, "system", lambda: "Linux")
    got = B.read_power(str(tmp_path / "nao-existe"), lambda b: None)
    assert got["on_ac"] is None and "on_ac" not in got["not_applicable"]


def test_read_power_other_platforms_never_crash(monkeypatch):
    monkeypatch.setattr(B.platform, "system", lambda: "Windows")
    assert B.read_power() == {"on_ac": None, "profile": None, "not_applicable": []}


def test_engine_version_failures_are_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("sem rede")
    monkeypatch.setattr(B.urllib.request.OpenerDirector, "open", boom)
    assert B.engine_version("ollama") is None
    monkeypatch.setattr(B.subprocess, "run", boom)
    assert B.engine_version("llama-bench") is None
    assert B.engine_version("outra") is None


def test_engine_version_parses(monkeypatch):
    monkeypatch.setattr(B, "_ollama_request", lambda m, p, body=None, timeout=600.0: {"version": " 1.2.3 "})
    assert B.engine_version("ollama") == "1.2.3"
    monkeypatch.setattr(B, "_ollama_request", lambda *a, **k: {"version": 3})
    assert B.engine_version("ollama") is None

    class CP:
        stdout, stderr = "", "\nversion: 42 (abc)\nresto\n"
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: CP)
    assert B.engine_version("llama-bench") == "version: 42 (abc)"


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
    assert mixed["cpu"]["power"] == {"on_ac": None, "profile": None, "not_applicable": []}


CAL = "calibration.json"


def _cli_setup(monkeypatch, tmp_path, power, delta=0):
    t = target()
    tuning = tmp_path / "tuning.json"
    tuning.write_text(json.dumps(TUNING), encoding="utf-8")
    monkeypatch.setattr(B, "_ollama_request", FakeOllama({"dense:7b": t.file_bytes}, [1.0], blob_delta=delta))
    monkeypatch.setattr(B, "read_power", lambda *a, **k: power)
    monkeypatch.setattr(B, "engine_version", lambda tool: "9.9.9")
    monkeypatch.setattr(B, "live_system", lambda: SYS)
    cal = tmp_path / CAL
    return tuning, cal, ["bench", "--model", "test/dense-7b", "--ollama-model", "dense:7b", "--system",
                         str(SYSTEM_PATHS["ddr5_64"]), "--catalog", str(CATALOG_PATH), "--tuning", str(tuning),
                         "--out", str(tmp_path / "measured.json"), "--calibration", str(cal)]


def test_cli_dry_run_measures_nothing_and_writes_nothing(monkeypatch, tmp_path, capsys):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Nada foi medido" in out and "dense:7b" in out
    assert not (tmp_path / "measured.json").exists() and not cal.exists()
    assert not B._ollama_request.generated


def test_cli_run_on_battery_warns_and_records(monkeypatch, tmp_path, capsys):
    bat = {"on_ac": False, "profile": "power-saver", "not_applicable": []}
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, bat)
    before = tuning.read_bytes()
    assert main(argv + ["--run"]) == 0
    cap = capsys.readouterr()
    assert "BATERIA" in cap.err and str(cal) in cap.out
    doc = json.loads((tmp_path / "measured.json").read_text(encoding="utf-8"))
    assert doc["on_battery_warning"] is True and doc["power"]["profile"] == "power-saver"
    m = doc["models"][0]
    assert m["quant"] == "Q8_0" and m["rounds"] and m["pool"] == "cpu" and m["tool_version"] == "9.9.9"
    c = json.loads(cal.read_text(encoding="utf-8"))
    e = c["pools"]["cpu"][0]
    assert e["power"] == bat and e["form"] == "1p" and e["tool"] == "ollama" and e["tool_version"] == "9.9.9"
    assert c["machine"] == machine_identity(SYS) and c["calibration_schema_version"] == CALIBRATION_SCHEMA_VERSION == 2
    assert tuning.read_bytes() == before  # tuning nunca é gravado
    assert calstore.validate_doc(c) == []


def test_tuning_test_json_bytes_unchanged_by_bench(monkeypatch, tmp_path):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    fixture_before = (FIX / "tuning_test.json").read_bytes()
    assert main(argv[:argv.index("--tuning")] + ["--tuning", str(FIX / "tuning_test.json")]
                + argv[argv.index("--out"):] + ["--run"]) == 0
    assert (FIX / "tuning_test.json").read_bytes() == fixture_before and cal.exists()


def test_cli_quant_mismatch_writes_no_calibration(monkeypatch, tmp_path, capsys):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC, delta=10 * B.MIB)
    assert main(argv + ["--run"]) == 1
    assert "não casa com Q8_0" in capsys.readouterr().err
    assert not cal.exists()


def test_cli_run_unstable_is_flagged(monkeypatch, tmp_path, capsys):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    monkeypatch.setattr(B, "_ollama_request", FakeOllama({"dense:7b": target().file_bytes}, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert main(argv + ["--run"]) == 0
    assert "NÃO ESTABILIZOU" in capsys.readouterr().out
    assert json.loads(cal.read_text())["pools"]["cpu"][0]["stable"] is False


def test_cli_refuses_incomplete_identity_before_measuring(monkeypatch, tmp_path, capsys):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    bad = json.loads(json.dumps(SYS))
    bad["cpu"]["model"] = {"value": None, "source": "x", "confidence": "low"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    argv[argv.index("--system") + 1] = str(p)
    assert main(argv + ["--run"]) == 2
    assert "identity_incomplete" in capsys.readouterr().err
    assert not B._ollama_request.generated and not cal.exists()


def test_cli_refuses_system_from_another_machine(monkeypatch, tmp_path, capsys):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    other = load(SYSTEM_PATHS["cpu8"])
    monkeypatch.setattr(B, "live_system", lambda: other)
    assert main(argv + ["--run"]) == 2
    assert "system_not_this_machine" in capsys.readouterr().err
    monkeypatch.setattr(B, "live_system", lambda: None)
    assert main(argv + ["--run"]) == 2
    assert not B._ollama_request.generated and not cal.exists()


def test_cli_refuses_mixed_tools_in_a_pool(monkeypatch, tmp_path, capsys):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    g = tmp_path / "m.gguf"
    g.write_text("x")
    monkeypatch.setattr(B.shutil, "which", lambda b: "/x/llama-bench")
    argv += ["--model", "test/moe-30b-a3b", "--ollama-model", "moe:30b", "--gguf", "-", "--gguf", str(g)]
    # modelo 1: gguf "-" não existe -> ollama; modelo 2: gguf existe -> llama-bench; mesmo pool cpu
    assert main(argv + ["--run"]) == 2
    assert "mixed_tools" in capsys.readouterr().err and not B._ollama_request.generated


def test_cli_run_merges_into_existing_file(monkeypatch, tmp_path):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    assert main(argv + ["--run"]) == 0
    bat = {"on_ac": False, "profile": "balanced", "not_applicable": []}
    monkeypatch.setattr(B, "read_power", lambda *a, **k: bat)
    monkeypatch.setattr(B, "_ollama_request", FakeOllama({"dense:7b": target().file_bytes}, [0.5]))
    assert main(argv + ["--run"]) == 0
    entries = json.loads(cal.read_text())["pools"]["cpu"]
    assert [e["power"]["on_ac"] for e in entries] == [True, False]


# ---- condições (diagnóstico)

def _fs(files):
    return (lambda p: files.get(p)), (lambda p: sorted({k[len(p) + 1:].split("/")[0] for k in files if k.startswith(p + "/")}) or None)


def test_read_conditions_normal_missing_and_garbage():
    c = "/sys/devices/system/cpu"
    files = {f"{c}/cpu0/cpufreq/scaling_driver": "amd-pstate-epp\n", f"{c}/cpu0/cpufreq/scaling_governor": "powersave\n",
             f"{c}/cpu0/cpufreq/energy_performance_preference": "performance\n",
             f"{c}/cpu0/cpufreq/scaling_cur_freq": "2000000\n", f"{c}/cpu1/cpufreq/scaling_cur_freq": "3000000\n",
             f"{c}/cpu2/cpufreq/scaling_cur_freq": "lixo", f"{c}/cpufreq/x": "1",
             "/sys/class/thermal/thermal_zone0/temp": "45000\n", "/sys/class/thermal/thermal_zone0/type": "acpitz\n",
             "/sys/class/thermal/thermal_zone1/temp": "71000\n", "/sys/class/thermal/thermal_zone1/type": "x86_pkg\n",
             "/sys/class/thermal/thermal_zone2/temp": "???",
             "/proc/loadavg": "0.50 1.25 2.00 1/300 999\n",
             "/proc/meminfo": "MemTotal: 100 kB\\nMemAvailable:    2048 kB\\nSwapFree:  4 kB\\n".replace("\\n", "\n")}
    r, l = _fs(files)
    got = B.read_conditions("linux", r, l)
    assert got["scaling_driver"] == "amd-pstate-epp" and got["energy_performance_preference"] == "performance"
    assert got["freq_khz_mean"] == 2_500_000 and got["freq_khz_max"] == 3_000_000
    assert got["temp_max_millic"] == 71000 and got["temp_max_zone_type"] == "x86_pkg"
    assert got["loadavg"] == [0.5, 1.25, 2.0]
    assert got["mem_available_bytes"] == 2048 * 1024 and got["swap_free_bytes"] == 4096
    r, l = _fs({"/proc/loadavg": "lixo", "/proc/meminfo": "MemAvailable: abc kB"})
    bad = B.read_conditions("linux", r, l)
    assert all(v is None for v in bad.values())
    assert all(v is None for v in B.read_conditions("darwin").values())

    def boom(p):
        raise RuntimeError
    assert all(v is None for v in B.read_conditions("linux", boom, boom).values())


def test_swap_counters_and_delta():
    r = lambda p: "nr_free 1\npswpin 10\npswpout 20\n"  # noqa: E731
    assert B.read_swap_counters(r) == (10, 20)
    assert B.read_swap_counters(lambda p: "pswpin x") is None and B.read_swap_counters(lambda p: None) is None
    assert B.swap_delta((10, 20), (15, 20)) == {"swap_in_pages": 5, "swap_out_pages": 0}
    assert B.swap_delta(None, (1, 1)) == {"swap_in_pages": None, "swap_out_pages": None}


def test_measure_records_conditions_and_swap_per_round(monkeypatch, fixed_env):
    t = target()
    monkeypatch.setattr(B, "_ollama_request", FakeOllama({"dense:7b": t.file_bytes}, [1.0, 1.0, 1.0]))
    monkeypatch.setattr(B, "read_conditions", lambda *a, **k: {"scaling_governor": "g"})
    ctr = iter(range(0, 1000, 3))
    monkeypatch.setattr(B, "read_swap_counters", lambda *a, **k: (next(ctr), 0))
    r = B.measure_model(t, "ollama", None, "dense:7b", AC)
    assert r["conditions"] == {"before": {"scaling_governor": "g"}, "after": {"scaling_governor": "g"}}
    assert r["warmup"]["swap_in_pages"] == 3 and all(x["swap_in_pages"] == 3 and x["swap_out_pages"] == 0 for x in r["rounds"])


def test_cli_run_records_live_kernel_not_system_file_kernel(monkeypatch, tmp_path, capsys):
    tuning, cal, argv = _cli_setup(monkeypatch, tmp_path, AC)
    live = json.loads(json.dumps(SYS))
    live["os"]["kernel"]["value"] = "6.97.0-9-fixture-ao-vivo"
    monkeypatch.setattr(B, "live_system", lambda: live)
    assert main(argv + ["--run"]) == 0
    entry = json.loads(cal.read_text())["pools"]["cpu"][0]
    assert entry["measured_with"]["os.kernel"] == "6.97.0-9-fixture-ao-vivo"
    assert "6.97.0-9-fixture-ao-vivo" in capsys.readouterr().err


@pytest.mark.parametrize("count,ok", [(B.MIN_GEN_TOKENS - 1, False), (B.MIN_GEN_TOKENS, True), (B.GEN_TOKENS, True)])
def test_ollama_generation_must_reach_three_quarters_of_gen_tokens(monkeypatch, count, ok):
    """Resposta que para cedo mede o decode noutra posição da resposta (DECISIONS.md, 2026-11-15)."""
    def fake(method, path, body=None, timeout=600.0):
        return {"eval_count": count, "eval_duration": 10**9, "prompt_eval_count": 5, "prompt_eval_duration": 10**9}
    monkeypatch.setattr(B, "_ollama_request", fake)
    assert B.MIN_GEN_TOKENS * 4 == B.GEN_TOKENS * 3
    if ok:
        assert B.ollama_round("x", target(), "p", 1).decode_tps == pytest.approx(count)
    else:
        with pytest.raises(B.BenchError):
            B.ollama_round("x", target(), "p", 1)


def test_each_ollama_model_is_unloaded_after_measuring(monkeypatch, fixed_env):
    """Com dois modelos na mesma rodada, o segundo não pode carregar com o primeiro na RAM (swap)."""
    t = target()
    fake = FakeOllama({"dense:7b": t.file_bytes}, [1.0])
    monkeypatch.setattr(B, "_ollama_request", fake)
    B.measure_model(t, "ollama", None, "dense:7b", AC)
    assert fake.unloaded == ["dense:7b"]
    assert fake.generated, "a medição rodou antes de descarregar"


def test_unload_failure_does_not_lose_measurement(monkeypatch, fixed_env):
    t = target()
    fake = FakeOllama({"dense:7b": t.file_bytes}, [1.0])

    def flaky(method, path, body=None, timeout=600.0):
        if path == "/api/generate" and body.get("keep_alive") == 0 and "prompt" not in body:
            raise B.BenchError("ollama_unreachable", "x")
        return fake(method, path, body, timeout)
    monkeypatch.setattr(B, "_ollama_request", flaky)
    r = B.measure_model(t, "ollama", None, "dense:7b", AC)
    assert r["decode_tps"] > 0


def test_ollama_model_without_tag_matches_latest(monkeypatch, fixed_env):
    """`ollama create x` registra "x:latest"; o bench recebe "x" e precisa achar o modelo."""
    t = target()
    fake = FakeOllama({"local-model:latest": t.file_bytes}, [1.0])
    monkeypatch.setattr(B, "_ollama_request", fake)
    assert B.ollama_check_quant("local-model", t)["architecture"] == "qwen3"
    with pytest.raises(B.BenchError) as e:
        B.ollama_check_quant("outro-modelo", t)
    assert e.value.reason == "ollama_model_missing"


def test_calibration_with_several_models_keeps_lowest_prefill():
    """A média do prefill de modelos de tamanhos diferentes não descreve nenhum deles e superestima o maior."""
    cal = B.build_calibration([_result("grande", 2_500_000_000, 7.95, prefill=19.9),
                               _result("pequeno", 640_000_000, 25.1, prefill=146.8)], "2026-11-15")
    assert cal["cpu"]["prefill_tokens_per_s"] == 19.9 and cal["cpu"]["prefill_reason"] is None


def test_decode_position_averages_used_rounds_in_integer_tokens():
    rs = [B.Round(5.0, None, 500, 1000, 500), B.Round(5.0, None, 531, 1023, 531)]
    assert B._decode_position(rs) == (516, 1012)  # (1031/2 -> 516, 2023/2 -> 1012), meio arredonda para cima
    assert B._decode_position([B.Round(5.0, None, 500)]) is None  # sem contagem: não inventa posição
