"""Coleta: parsers puros sobre saídas capturadas (tests/fixtures/collect). Nenhum comando real roda."""
from __future__ import annotations

import json
import subprocess

import pytest

from conftest import FIX, assert_valid
from scout import collect as C
from scout import parsers as P

CF = FIX / "collect"


def txt(name: str) -> str:
    return (CF / name).read_text(encoding="utf-8")


def test_os_release():
    kv = P.parse_os_release(txt("os_release.txt"))
    assert kv["NAME"] == "Ubuntu" and kv["VERSION_ID"] == "24.04"


def test_lscpu_amd_not_hybrid():
    i = P.parse_lscpu(txt("lscpu_amd.txt"))
    assert (i["logical"], i["physical"], i["vendor"]) == (8, 4, "AuthenticAMD")
    isa = P.isa_from_flags(i["flags"], i["arch"])
    assert isa == {"avx2": True, "avx512": False, "amx": False, "neon": False, "sve": False}


def test_lscpu_intel_hybrid_separates_p_and_e_cores():
    i = P.parse_lscpu(txt("lscpu_intel_hybrid.txt"))
    assert i["logical"] == 20 and i["physical"] == 14
    split = P.hybrid_split(P.parse_lscpu_p(txt("lscpu_p_hybrid.txt")),
                           P.parse_cpu_list(txt("sys_cpu_core_cpus.txt")),
                           P.parse_cpu_list(txt("sys_cpu_atom_cpus.txt")))
    assert split == (6, 8)  # 6 P-cores com SMT (12 threads) + 8 E-cores
    isa = P.isa_from_flags(i["flags"], i["arch"])
    assert isa["avx512"] and isa["amx"]


def test_hybrid_split_without_map_is_none():
    assert P.hybrid_split({}, {0}, {1}) is None
    assert P.hybrid_split({0: 0, 1: 1}, set(), {1}) is None


def test_cpu_list():
    assert P.parse_cpu_list("0-2,5,7-8\n") == {0, 1, 2, 5, 7, 8}
    assert P.parse_cpu_list("lixo") == set()


def test_cpuinfo_fallback():
    i = P.parse_cpuinfo(txt("cpuinfo_x86.txt"))
    assert i["logical"] == 4 and i["physical"] == 2 and i["model"] == "Test CPU"
    assert P.parse_cpuinfo("")["logical"] is None


def test_meminfo_bytes_and_no_free():
    kv = P.parse_meminfo(txt("meminfo.txt"))
    assert kv == {"MemTotal": 16384000 * 1024, "SwapTotal": 2097152 * 1024}


def test_nvidia_smi_csv_and_caps():
    g = P.parse_nvidia_smi_csv(txt("nvidia_smi.csv"))
    assert g[0]["vram_total"] == 12288 * 1024 * 1024 and g[0]["cc"] == "8.6"
    assert P.gpu_caps_from_cc("8.6") == {"bf16": True, "fp8": False, "flash_attention": True}
    assert P.gpu_caps_from_cc("8.9")["fp8"] is True
    assert P.gpu_caps_from_cc("6.1") == {"bf16": False, "fp8": False, "flash_attention": False}
    assert P.gpu_caps_from_cc(None) == {"bf16": None, "fp8": None, "flash_attention": None}
    old = P.parse_nvidia_smi_csv(txt("nvidia_smi_no_cc.csv"))
    assert old[0]["cc"] is None and old[0]["driver"] == "470.82.00"
    assert P.parse_nvidia_smi_csv("") == [] and P.parse_nvidia_smi_csv("No devices were found") == []


def test_rocm_smi_json():
    g = P.parse_rocm_smi_json(txt("rocm_smi.json"))
    assert g == [{"name": "Mendocino", "vram_total": 536870912, "vram_free": 536870912 - 468795392,
                  "driver": "6.8.0-test", "cc": None}]
    assert P.parse_rocm_smi_json("lixo") == []


def test_drm_cards():
    g = P.parse_drm_cards([{"vendor": "0x1002", "vram_total": "100", "vram_used": "40"},
                           {"vendor": "0x8086", "vram_total": None, "vram_used": None}])
    assert g[0]["vendor"] == "amd" and g[0]["vram_free"] == 60
    assert g[1]["vendor"] == "intel" and g[1]["vram_total"] is None


def test_dmidecode_ddr5():
    d = P.parse_dmidecode(txt("dmidecode_ddr5.txt"))
    assert d["type"] == "DDR5" and d["populated"] == 2
    assert d["speed_mts"] == 4400  # o módulo mais lento limita
    assert d["channels"] == 2 and d["channels_exact"] is True


def test_dmidecode_denied_has_no_data():
    d = P.parse_dmidecode(txt("dmidecode_denied.txt"))
    assert not d["populated"] and d["type"] is None and d["channels"] is None


def test_ram_bandwidth_policy():
    table = json.loads((C.DATA / "bandwidth_tables.json").read_text(encoding="utf-8"))
    assert P.ram_bandwidth(3200, 2, "DDR4", table)[:3] == (51_200_000_000, "estimate:mts*channels*8", "medium")
    bw, src, conf, _ = P.ram_bandwidth(None, None, "DDR5", table)
    assert bw == 76_800_000_000 and src.startswith("estimate:") and conf == "low"
    bw, src, conf, _ = P.ram_bandwidth(None, None, None, table)
    assert bw is None and conf == "low"
    assert P.ram_bandwidth(None, None, "DDR9", table)[0] is None
    assert P.ram_bandwidth(None, None, "DDR4", None)[0] is None
    assert all(v["confidence"] == "low" for v in table["ram"].values())


def test_lsblk_disks():
    disks = {d["disk"]: d for d in P.parse_lsblk_json(txt("lsblk.json"))}
    assert set(disks) == {"sda", "nvme0n1", "sdb"}  # loop e zram fora
    assert P.disk_kind(disks["nvme0n1"]) == ("nvme", "high", None)
    assert P.disk_kind(disks["sdb"])[0] == "hdd"
    assert P.disk_kind(disks["sda"])[1] == "low"  # USB
    assert P.disk_kind({"disk": "sdz", "rota": None, "tran": None})[0] is None
    assert disks["nvme0n1"]["mounts"] == ["/", "/boot/efi"]
    assert not P.user_visible_mount("/media/alguem/PEN") and not P.user_visible_mount("/boot/efi")
    assert P.user_visible_mount("/") and P.user_visible_mount("/home") and P.user_visible_mount("/mnt/dados")
    assert P.parse_lsblk_json("não é json") == []


def test_power():
    assert P.parse_power([{"type": "Mains", "status": None}]) == (False, False)
    assert P.parse_power([{"type": "Battery", "status": "Discharging"}]) == (True, True)
    assert P.parse_power([{"type": "Battery", "status": "Charging"}, {"type": "Mains"}]) == (True, False)
    assert P.parse_power([{"type": "Battery", "status": "Unknown"}]) == (True, None)


def test_ollama_list():
    assert P.parse_ollama_list(txt("ollama_list.txt")) == [
        {"name": "qwen3:4b-instruct", "size_bytes": 2_500_000_000},
        {"name": "tiny:latest", "size_bytes": 637_000_000}]


def test_field_null_forces_low():
    assert C.field(None, "x", "high")["confidence"] == "low"
    assert C.missing("motivo")["value"] is None


@pytest.mark.parametrize("exc,status", [
    (FileNotFoundError(), "not_found"), (PermissionError(), "permission_denied"),
    (subprocess.TimeoutExpired("x", 1), "timeout"),
    (subprocess.CalledProcessError(1, "x", output="", stderr="Permission denied"), "permission_denied"),
    (subprocess.CalledProcessError(2, "x", output="", stderr="boom"), "error"),
    (OSError(5, "io"), "error")])
def test_run_cmd_failures_never_raise(monkeypatch, exc, status):
    def boom(*a, **k):
        assert k.get("timeout") and not k.get("shell")
        raise exc
    monkeypatch.setattr(subprocess, "run", boom)
    probes = C.Probes()
    assert C.run_cmd(["x"], probes) is None
    assert probes.items[0]["status"] == status


def _no_tools(monkeypatch):
    def nf(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(subprocess, "run", nf)
    monkeypatch.setattr(C.shutil, "which", lambda b: None)


def test_collect_without_any_tool_is_valid_and_low(monkeypatch):
    _no_tools(monkeypatch)
    monkeypatch.setattr(C, "_read_drm", lambda probes: None)
    doc = C.collect()
    assert_valid("system.schema.json", doc)
    assert doc["gpus"]["value"] is None and doc["gpus"]["confidence"] == "low"
    assert doc["memory"]["type"]["value"] is None
    assert doc["memory"]["bandwidth_bytes_per_s"]["value"] is None
    assert doc["runtimes"]["ollama"]["value"] is False
    assert doc["runtimes"]["ollama_models"]["confidence"] == "low"
    assert any(p["probe"] == "dmidecode" and p["status"] == "not_found" for p in doc["probes"])


def test_collect_no_gpu_is_empty_list_not_null(monkeypatch):
    _no_tools(monkeypatch)
    monkeypatch.setattr(C, "_read_drm", lambda probes: [])
    doc = C.collect()
    assert_valid("system.schema.json", doc)
    assert doc["gpus"]["value"] == []


def test_collect_nvidia_end_to_end(monkeypatch):
    _no_tools(monkeypatch)
    real = C.run_cmd

    def fake(cmd, probes, timeout=5.0, probe=None):
        if cmd[0] == "nvidia-smi":
            probes.add("nvidia-smi", "ok")
            return txt("nvidia_smi.csv")
        if cmd[0] == "dmidecode":
            probes.add("dmidecode", "ok")
            return txt("dmidecode_ddr5.txt")
        return real(cmd, probes, timeout, probe)
    monkeypatch.setattr(C, "run_cmd", fake)
    doc = C.collect()
    assert_valid("system.schema.json", doc)
    gpus = doc["gpus"]["value"]
    assert [g["index"] for g in gpus] == [0, 1]
    assert gpus[0]["bf16"]["value"] is True and gpus[0]["backend"]["value"] == "cuda"
    assert gpus[0]["bandwidth_bytes_per_s"]["value"] is None  # sem tabela por nome de GPU
    assert doc["memory"]["bandwidth_bytes_per_s"]["value"] == 4400 * 2 * 8 * 10**6
    assert doc["memory"]["bandwidth_bytes_per_s"]["source"].startswith("estimate:")


def test_cli_collect_writes_valid_file(tmp_path, monkeypatch):
    _no_tools(monkeypatch)
    from scout.cli import main
    out = tmp_path / "system.json"
    assert main(["collect", "--out", str(out)]) == 0
    assert_valid("system.schema.json", json.loads(out.read_text(encoding="utf-8")))


def test_collect_dmidecode_file(monkeypatch, tmp_path):
    _no_tools(monkeypatch)
    f = tmp_path / "mem.txt"
    f.write_text(txt("dmidecode_ddr5.txt"), encoding="utf-8")
    doc = C.collect(str(f))
    assert_valid("system.schema.json", doc)
    m = doc["memory"]
    assert m["type"]["value"] == "DDR5" and m["type"]["source"] == "dmidecode_file"
    assert m["bandwidth_bytes_per_s"]["value"] == 4400 * 2 * 8 * 10**6


def test_collect_dmidecode_file_denied_or_missing(monkeypatch, tmp_path):
    _no_tools(monkeypatch)
    f = tmp_path / "mem.txt"
    f.write_text(txt("dmidecode_denied.txt"), encoding="utf-8")
    for path in (str(f), str(tmp_path / "nao_existe.txt")):
        doc = C.collect(path)
        assert_valid("system.schema.json", doc)
        assert doc["memory"]["type"]["value"] is None


def test_denied_dmidecode_gives_hint(monkeypatch, capsys, tmp_path):
    def denied(*a, **k):
        raise subprocess.CalledProcessError(1, "dmidecode", output="", stderr="Permission denied")
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(C.shutil, "which", lambda b: None)
    doc = C.collect()
    p = next(p for p in doc["probes"] if p["probe"] == "dmidecode")
    assert p["status"] == "permission_denied" and "sudo scout collect" in p["message"] and "--dmidecode" in p["message"]
    from scout.cli import main
    main(["collect", "--out", str(tmp_path / "s.json")])
    assert "--dmidecode mem.txt" in capsys.readouterr().err


# ---- Fase 7: F2 (híbrida sem sysfs), F3 (canais), F4b (tabela de GPU)

def test_hybrid_by_freq():
    kind, split = P.hybrid_by_freq(P.parse_lscpu_p_mhz(txt("lscpu_p_mhz_hybrid.txt")))
    assert (kind, split) == ("hybrid", (6, 8))
    assert P.hybrid_by_freq(P.parse_lscpu_p_mhz(txt("lscpu_p_mhz_uniform.txt")))[0] == "uniform"
    assert P.hybrid_by_freq(P.parse_lscpu_p_mhz(txt("lscpu_p_mhz_empty.txt")))[0] == "unknown"
    assert P.hybrid_by_freq([])[0] == "unknown"


def _cpu(monkeypatch, lscpu_txt, mhz_txt):
    def fake(cmd, probes, timeout=5.0, probe=None):
        probes.add(probe or cmd[0], "ok")
        if cmd == ["lscpu"]:
            return txt(lscpu_txt)
        if cmd[0] == "lscpu":
            return txt(mhz_txt) if mhz_txt else None
        return None
    monkeypatch.setattr(C, "run_cmd", fake)
    monkeypatch.setattr(C, "_sysfs_cpus", lambda name, probes: set())
    return C.collect_cpu_linux(C.Probes())


def test_f2_hybrid_without_sysfs_is_medium(monkeypatch):
    cpu = _cpu(monkeypatch, "lscpu_intel_hybrid.txt", "lscpu_p_mhz_hybrid.txt")
    assert cpu["cores_performance"]["value"] == 6 and cpu["cores_efficiency"]["value"] == 8
    assert cpu["cores_performance"]["confidence"] == "medium"


def test_f2_intel_inconclusive_is_low_and_never_high(monkeypatch):
    cpu = _cpu(monkeypatch, "lscpu_intel_hybrid.txt", "lscpu_p_mhz_empty.txt")
    assert cpu["cores_performance"]["confidence"] == "low"
    assert "hybrid_undetermined" in cpu["cores_performance"]["note"]
    assert cpu["cores_efficiency"]["value"] is None
    cpu = _cpu(monkeypatch, "lscpu_intel_hybrid.txt", None)
    assert cpu["cores_performance"]["confidence"] == "low"


def test_f2_amd_without_sysfs_never_high(monkeypatch):
    cpu = _cpu(monkeypatch, "lscpu_amd.txt", "lscpu_p_mhz_uniform.txt")
    assert cpu["cores_performance"]["value"] == 4 and cpu["cores_performance"]["confidence"] == "medium"
    assert cpu["cores_efficiency"]["value"] == 0 and cpu["cores_efficiency"]["confidence"] == "medium"


def test_f3_controller_is_part_of_channel_key():
    d = P.parse_dmidecode(txt("dmidecode_two_controllers.txt"))
    assert d["channels"] == 2 and d["channels_exact"] is True
    # mesmo ChannelA em controladores distintos não colapsa em 1


def test_f3_count_fallback_lowers_bandwidth_confidence():
    d = P.parse_dmidecode(txt("dmidecode_no_locator.txt"))
    assert d["channels"] == 2 and d["channels_exact"] is False
    mem = C._memory_rest({"ram_total_bytes": C.missing("x"), "swap_total_bytes": C.missing("x")}, d, False, "p")
    assert mem["channels"]["confidence"] == "low"
    assert mem["bandwidth_bytes_per_s"]["value"] == 3200 * 2 * 8 * 10**6
    assert mem["bandwidth_bytes_per_s"]["confidence"] == "low"


def test_f3_bandwidth_never_more_confident_than_inputs():
    assert P.ram_bandwidth(3200, 2, "DDR4", None, "medium", "medium")[2] == "medium"
    assert P.ram_bandwidth(3200, 2, "DDR4", None, "low", "medium")[2] == "low"
    assert P.ram_bandwidth(3200, 2, "DDR4", None, "high", "high")[2] == "medium"


GT = {"reviewed": False, "entries": [
    {"match": "rtx 3060", "bandwidth_bytes_per_s": 360_000_000_000, "source_url": "https://www.nvidia.com/x", "verified_at": "2026-11-15"},
    {"match": "rtx 3060 ti", "bandwidth_bytes_per_s": 448_000_000_000, "source_url": "https://www.nvidia.com/y", "verified_at": "2026-11-15"},
    {"match": "rx 7800 xt", "bandwidth_bytes_per_s": 624_000_000_000, "source_url": "https://www.amd.com/z", "verified_at": "2026-11-15"},
    {"match": "rx 6600", "bandwidth_bytes_per_s": 224_000_000_000, "source_url": "https://evil.example/q", "verified_at": "2026-11-15"}]}


def test_gpu_table_lookup():
    L = P.gpu_bandwidth_lookup
    assert L("NVIDIA GeForce RTX 3060", GT)[0] == 360_000_000_000
    assert L("AMD Radeon RX 7800 XT", GT)[0] == 624_000_000_000
    assert L("NVIDIA GeForce RTX 3060 Ti", GT)[2] == "ambiguous_table_match"  # casa 2 entradas
    assert L("AMD Radeon RX 6600", GT)[2] == "no_table_match"  # fonte não oficial é ignorada
    assert L("Mendocino", GT)[2] == "no_table_match"
    assert L("x", None)[0] is None and L(None, GT)[0] is None
    assert L("x", {"entries": [{"match": ""}]})[0] is None


def test_gpu_table_used_in_collect(monkeypatch):
    _no_tools(monkeypatch)
    real = C.run_cmd

    def fake(cmd, probes, timeout=5.0, probe=None):
        if cmd[0] == "nvidia-smi":
            probes.add("nvidia-smi", "ok")
            return txt("nvidia_smi.csv")
        return real(cmd, probes, timeout, probe)
    monkeypatch.setattr(C, "run_cmd", fake)
    monkeypatch.setattr(C, "_load_gpu_table", lambda: GT)
    doc = C.collect()
    assert_valid("system.schema.json", doc)
    g0, g1 = doc["gpus"]["value"]
    assert g0["bandwidth_bytes_per_s"]["value"] == 360_000_000_000
    assert g0["bandwidth_bytes_per_s"]["source"] == "table" and g0["bandwidth_bytes_per_s"]["confidence"] == "medium"
    assert g1["bandwidth_bytes_per_s"]["value"] is None  # RTX 4090 fora da tabela


def test_shipped_gpu_table_entries_have_official_source():
    t = json.loads((C.DATA / "gpu_bandwidth.json").read_text(encoding="utf-8"))
    assert t["reviewed"] is False
    for e in t["entries"]:
        assert P._official(e["source_url"]) and isinstance(e["bandwidth_bytes_per_s"], int)
