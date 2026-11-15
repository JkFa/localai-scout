"""scout collect: lê o hardware e produz system.json (schemas/system.schema.json).

Nunca toca a rede (ollama list é local). Todo subprocess tem timeout, lista de
argumentos e tratamento de falha; dmidecode nunca roda com sudo. Campo não
detectado = value null + confidence low + motivo em `source`; nunca 0.
Linux é o alvo testado; macOS e Windows são best effort.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scout import __version__
from scout import parsers as P

DATA = Path(__file__).resolve().parent.parent / "data"
Field = dict[str, Any]


def field(value: Any, source: str, confidence: str = "high", note: str | None = None) -> Field:
    """Monta {value, source, confidence}. value None força confidence low (schema)."""
    f: Field = {"value": value, "source": source[:200], "confidence": "low" if value is None else confidence}
    if note:
        f["note"] = note[:500]
    return f


def missing(reason: str) -> Field:
    return field(None, f"none: {reason}", "low")


class Probes:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, probe: str, status: str, message: str = "") -> int:
        self.items.append({"probe": probe, "status": status, "message": message})
        return len(self.items) - 1

    def set(self, idx: int, status: str, message: str) -> None:
        self.items[idx].update(status=status, message=message)


def run_cmd(cmd: list[str], probes: Probes, timeout: float = 5.0, probe: str | None = None) -> str | None:
    """Roda sem shell, com timeout. Devolve stdout ou None (a causa vai para probes)."""
    name = probe or cmd[0]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
                            check=True, stdin=subprocess.DEVNULL, env={**os.environ, "LC_ALL": "C"})
    except FileNotFoundError:
        probes.add(name, "not_found", f"{cmd[0]} não está no PATH")
    except PermissionError:
        probes.add(name, "permission_denied", f"sem permissão para executar {cmd[0]}")
    except subprocess.TimeoutExpired:
        probes.add(name, "timeout", f"{cmd[0]} excedeu {timeout:g}s")
    except subprocess.CalledProcessError as e:
        blob = f"{e.stdout or ''}{e.stderr or ''}".lower()
        if "permission denied" in blob or "must be root" in blob:
            probes.add(name, "permission_denied", f"{cmd[0]} exige privilégios")
        else:
            probes.add(name, "error", f"{cmd[0]} saiu com código {e.returncode}")
    except OSError as e:
        probes.add(name, "error", f"{cmd[0]}: {e.strerror or type(e).__name__}")
    else:
        probes.add(name, "ok")
        return cp.stdout
    return None


def read_text(path: str, probes: Probes, probe: str | None = None, limit: int = 1_000_000) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read(limit)
    except FileNotFoundError:
        probes.add(probe or path, "not_found", "arquivo ausente")
    except PermissionError:
        probes.add(probe or path, "permission_denied", "sem permissão de leitura")
    except OSError as e:
        probes.add(probe or path, "error", e.strerror or type(e).__name__)
    else:
        probes.add(probe or path, "ok")
        return text
    return None


def _guard(probes: Probes, name: str, fn: Callable[[], Any]) -> Any:
    """Nenhuma seção derruba a coleta: exceção inesperada vira probe de erro."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a coleta nunca pode dar crash
        probes.add(name, "error", f"falha inesperada: {type(e).__name__}")
        return None


def _load_bw_table() -> dict | None:
    try:
        return json.loads((DATA / "bandwidth_tables.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- seções


def collect_os(system: str, probes: Probes) -> dict:
    name = version = missing("os-release indisponível")
    if system == "linux":
        text = read_text("/etc/os-release", probes, "os-release")
        if text:
            kv = P.parse_os_release(text)
            name = field(kv.get("NAME"), "/etc/os-release:NAME") if kv.get("NAME") else missing("NAME ausente")
            version = (field(kv.get("VERSION_ID"), "/etc/os-release:VERSION_ID") if kv.get("VERSION_ID")
                       else missing("VERSION_ID ausente"))
    elif system == "darwin":
        out = run_cmd(["sw_vers", "-productVersion"], probes)
        name = field("macOS", "platform.system", "medium")
        version = field(out.strip(), "sw_vers", "high") if out and out.strip() else missing("sw_vers falhou")
    elif system == "windows":
        name = field("Windows", "platform.system", "medium")
        v = platform.version()
        version = field(v, "platform.version", "medium") if v else missing("platform.version vazio")
    kernel, arch = platform.release(), platform.machine()
    return {"name": name, "version": version,
            "kernel": field(kernel, "platform.release") if kernel else missing("platform.release vazio"),
            "arch": field(arch, "platform.machine") if arch else missing("platform.machine vazio")}


def _cpu_empty() -> dict:
    m = missing("não detectado")
    return {"model": dict(m), "vendor": dict(m), "cores_logical": dict(m), "cores_physical": dict(m),
            "cores_performance": dict(m), "cores_efficiency": dict(m),
            "isa": {k: dict(m) for k in ("avx2", "avx512", "amx", "neon", "sve")}}


def _sysfs_cpus(name: str, probes: Probes) -> set[int]:
    path = f"/sys/devices/{name}/cpus"
    if not os.path.exists(path):
        return set()
    return P.parse_cpu_list(read_text(path, probes, f"sysfs:{name}") or "")


def collect_cpu_linux(probes: Probes) -> dict:
    cpu = _cpu_empty()
    info: dict[str, Any] | None = None
    src = "lscpu"
    out = run_cmd(["lscpu"], probes)
    if out:
        info = P.parse_lscpu(out)
    if not info or not (info.get("model") or info.get("logical")):
        text = read_text("/proc/cpuinfo", probes, "cpuinfo")
        info = P.parse_cpuinfo(text) if text else None
        src = "/proc/cpuinfo"
    arch = platform.machine()
    conf = "high" if src == "lscpu" else "medium"
    info = info or {"model": None, "vendor": None, "logical": None, "physical": None, "flags": None, "arch": None}
    why = "lscpu e /proc/cpuinfo indisponíveis"
    for key, k in (("model", "model"), ("vendor", "vendor")):
        cpu[key] = field(info.get(k), src, conf) if info.get(k) else missing(why)
    logical = info.get("logical")
    if logical:
        cpu["cores_logical"] = field(logical, src, conf)
    elif os.cpu_count():
        cpu["cores_logical"] = field(os.cpu_count(), "os.cpu_count", "medium")
    physical = info.get("physical")
    if physical:
        cpu["cores_physical"] = field(physical, src, conf)
    if physical:
        hybrid = None
        p_cpus, e_cpus = _sysfs_cpus("cpu_core", probes), _sysfs_cpus("cpu_atom", probes)
        if p_cpus and e_cpus:
            lp = run_cmd(["lscpu", "-p=CPU,CORE"], probes, probe="lscpu-p")
            hybrid = P.hybrid_split(P.parse_lscpu_p(lp), p_cpus, e_cpus) if lp else None
            if hybrid is None:
                cpu["cores_performance"] = missing("CPU híbrida detectada, mas lscpu -p falhou")
                cpu["cores_efficiency"] = missing("CPU híbrida detectada, mas lscpu -p falhou")
        if hybrid:
            perf, eff = hybrid
            cpu["cores_physical"] = field(perf + eff, "sysfs:cpu_core+cpu_atom", "high")
            cpu["cores_performance"] = field(perf, "sysfs:cpu_core", "high", "E-cores fora de n_threads")
            cpu["cores_efficiency"] = field(eff, "sysfs:cpu_atom", "high")
        elif not (p_cpus and e_cpus):
            cpu["cores_performance"] = field(physical, "derived:cores_physical", conf, "CPU não híbrida")
            cpu["cores_efficiency"] = field(0, "derived:cpu_not_hybrid", "medium",
                                            "sem /sys/devices/cpu_core e cpu_atom")
    else:
        cpu["cores_physical"] = missing(why)
        cpu["cores_performance"] = missing("núcleos físicos não detectados")
        cpu["cores_efficiency"] = missing("núcleos físicos não detectados")
    flags = info.get("flags")
    isa = P.isa_from_flags(flags, arch)
    for k, v in isa.items():
        if v is None:
            cpu["isa"][k] = missing("flags de CPU indisponíveis")
        elif flags is None:
            cpu["isa"][k] = field(v, "platform.machine", "high")
        else:
            cpu["isa"][k] = field(v, f"{src}:flags", "high")
    return cpu


DMI_HINT = ("dmidecode exige root: rode `sudo scout collect` ou gere `sudo dmidecode -t memory > mem.txt` "
            "e use `scout collect --dmidecode mem.txt`")


def collect_memory_linux(probes: Probes, dmidecode_file: str | None = None) -> dict:
    mem = {"ram_total_bytes": missing("/proc/meminfo indisponível"), "swap_total_bytes": missing("/proc/meminfo indisponível")}
    text = read_text("/proc/meminfo", probes, "meminfo")
    kv = P.parse_meminfo(text) if text else {}
    if "MemTotal" in kv:
        mem["ram_total_bytes"] = field(kv["MemTotal"], "/proc/meminfo:MemTotal")
    else:
        try:
            mem["ram_total_bytes"] = field(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "os.sysconf", "medium")
        except (ValueError, OSError, AttributeError):
            pass
    if "SwapTotal" in kv:
        mem["swap_total_bytes"] = field(kv["SwapTotal"], "/proc/meminfo:SwapTotal")
    dmi = None
    dmi_src = "dmidecode"
    if dmidecode_file:  # saída gerada pelo próprio usuário; nunca invocamos sudo
        dmi_src = "dmidecode_file"
        out = read_text(dmidecode_file, probes, "dmidecode_file", limit=2_000_000)
        idx = len(probes.items) - 1
    else:
        out = run_cmd(["dmidecode", "-t", "memory"], probes, timeout=5.0)
        idx = len(probes.items) - 1
    if out is not None:
        dmi = P.parse_dmidecode(out)
        if not dmi.get("populated"):
            low = out.lower()
            denied = "permission denied" in low
            probes.set(idx, "permission_denied" if denied else "error",
                       DMI_HINT if denied and not dmidecode_file else "dmidecode sem dados de memória")
            dmi = None
    elif not dmidecode_file and probes.items[idx]["status"] == "permission_denied":
        probes.set(idx, "permission_denied", DMI_HINT)
    return _memory_rest(mem, dmi, unified=False, platform_note="derived:platform", dmi_src=dmi_src)


def _memory_rest(mem: dict, dmi: dict | None, unified: bool | None, platform_note: str,
                 dmi_src: str = "dmidecode") -> dict:
    why = "dmidecode indisponível (exige root)"
    t = dmi.get("type") if dmi else None
    mts = dmi.get("speed_mts") if dmi else None
    ch = dmi.get("channels") if dmi else None
    mem["type"] = field(t, dmi_src, "medium") if t else missing(why)
    mem["speed_mts"] = field(mts, dmi_src, "medium") if mts else missing(why)
    if ch:
        exact = dmi.get("channels_exact")
        mem["channels"] = field(ch, dmi_src, "medium" if exact else "low",
                                None if exact else "estimado pelo nº de módulos populados")
    else:
        mem["channels"] = missing(why)
    bw, src, conf, note = P.ram_bandwidth(mts, ch, t, _load_bw_table())
    mem["bandwidth_bytes_per_s"] = field(bw, src, conf, note)
    mem["unified"] = field(unified, platform_note, "high") if unified is not None else missing("plataforma desconhecida")
    return mem


def _gpu_entry(i: int, vendor: str | None, g: dict, src: str, vendor_src: str) -> dict:
    cc = g.get("cc")
    caps = P.gpu_caps_from_cc(cc) if vendor == "nvidia" else {"bf16": None, "fp8": None, "flash_attention": None}
    no_cc = "compute capability indisponível" if vendor == "nvidia" else "só derivável de CUDA compute capability"
    backend = {"nvidia": "cuda", "amd": "rocm"}.get(vendor or "")
    bw_reason = "sem sonda local de clock de memória e largura de barramento; sem tabela por nome de GPU"
    entry = {
        "index": i,
        "vendor": field(vendor, vendor_src, "high") if vendor else missing("fabricante não detectado"),
        "name": field(g["name"], src, "high") if g.get("name") else missing(f"{src} não informou o nome"),
        "vram_total_bytes": (field(g["vram_total"], f"{src}:memory.total", "high") if g.get("vram_total") is not None
                             else missing(f"{src} não informou a VRAM")),
        "vram_free_bytes": (field(g["vram_free"], f"{src}:memory.free", "high") if g.get("vram_free") is not None
                            else missing(f"{src} não informou VRAM livre")),
        "bandwidth_bytes_per_s": missing(bw_reason),
        "driver_version": field(g["driver"], src, "medium" if src == "rocm-smi" else "high") if g.get("driver")
        else missing(f"{src} não informou o driver"),
        "compute_capability": field(cc, f"{src}:compute_cap", "high") if cc else missing(no_cc),
        "backend": (field(backend, "derived:vendor", "high" if src != "sysfs:drm" else "medium")
                    if backend else missing("backend não derivável do fabricante")),
    }
    for k in ("bf16", "fp8", "flash_attention"):
        v = caps[k]
        entry[k] = field(v, "derived:compute_capability", "medium" if k == "flash_attention" else "high") \
            if v is not None else missing(no_cc)
    return entry


def collect_gpus(system: str, probes: Probes) -> Field:
    gpus: list[tuple[str | None, dict, str, str]] = []
    nv = run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                  "--format=csv,noheader,nounits"], probes, timeout=10.0)
    if nv is None and probes.items[-1]["status"] == "error":  # driver antigo sem compute_cap
        probes.items.pop()
        nv = run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
                      "--format=csv,noheader,nounits"], probes, timeout=10.0)
    for g in P.parse_nvidia_smi_csv(nv or ""):
        gpus.append(("nvidia", g, "nvidia-smi", "nvidia-smi"))
    rc = run_cmd(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showdriverversion", "--json"],
                 probes, timeout=10.0)
    for g in P.parse_rocm_smi_json(rc or ""):
        gpus.append(("amd", g, "rocm-smi", "rocm-smi"))
    if not gpus and system == "linux":
        cards = _read_drm(probes)
        if cards is not None:
            for g in P.parse_drm_cards(cards):
                gpus.append((g["vendor"], g, "sysfs:drm", "sysfs:drm"))
            if not gpus:
                return field([], "sysfs:drm", "medium", "nenhuma GPU em /sys/class/drm e sem nvidia-smi/rocm-smi")
    if system == "darwin" and not gpus:
        gpus = _gpus_darwin(probes)
    if system == "windows" and not gpus:
        gpus = _gpus_windows(probes)
    if not gpus:
        return field(None, "none: nvidia-smi, rocm-smi e fontes do sistema indisponíveis", "low")
    src = "+".join(sorted({g[2] for g in gpus}))
    return field([_gpu_entry(i, v, g, s, vs) for i, (v, g, s, vs) in enumerate(gpus)], src, "high")


def _read_drm(probes: Probes) -> list[dict[str, str | None]] | None:
    root = "/sys/class/drm"
    try:
        names = sorted((n for n in os.listdir(root) if re.fullmatch(r"card\d+", n)), key=lambda n: int(n[4:]))
    except OSError:
        probes.add("sysfs:drm", "not_found", "/sys/class/drm indisponível")
        return None
    probes.add("sysfs:drm", "ok")
    cards = []
    for n in names:
        def rd(f: str) -> str | None:
            try:
                with open(f"{root}/{n}/device/{f}", encoding="utf-8") as fh:
                    return fh.read(64).strip()
            except OSError:
                return None
        cards.append({"vendor": rd("vendor"), "vram_total": rd("mem_info_vram_total"), "vram_used": rd("mem_info_vram_used")})
    return cards


def _gpus_darwin(probes: Probes) -> list:
    out = run_cmd(["system_profiler", "SPDisplaysDataType", "-json"], probes, timeout=15.0)
    try:
        items = json.loads(out or "{}").get("SPDisplaysDataType", [])
    except ValueError:
        return []
    res = []
    for it in items:
        name = it.get("sppci_model")
        if name:
            vendor_raw = str(it.get("spdisplays_vendor", "")).lower()
            vendor = "apple" if "apple" in vendor_raw or platform.machine() == "arm64" else None
            res.append((vendor, {"name": name, "vram_total": None, "vram_free": None, "driver": None, "cc": None},
                        "system_profiler", "system_profiler"))
    return res


def _gpus_windows(probes: Probes) -> list:
    out = run_cmd(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"],
                  probes, timeout=15.0, probe="powershell:gpu")
    try:
        doc = json.loads(out or "[]")
    except ValueError:
        return []
    res = []
    for it in [doc] if isinstance(doc, dict) else doc:
        name = (it or {}).get("Name") or ""
        low = name.lower()
        vendor = "nvidia" if "nvidia" in low else "amd" if ("amd" in low or "radeon" in low) else "intel" if "intel" in low else None
        # AdapterRAM é 32 bits e satura em 4 GiB: não confiável, fica null.
        res.append((vendor, {"name": name or None, "vram_total": None, "vram_free": None,
                             "driver": it.get("DriverVersion"), "cc": None}, "powershell", "powershell"))
    return res


def collect_disks(system: str, probes: Probes) -> Field:
    def stat(mount: str) -> tuple[Field, Field]:
        try:
            if hasattr(os, "statvfs"):
                s = os.statvfs(mount)
                return (field(s.f_blocks * s.f_frsize, "os.statvfs"), field(s.f_bavail * s.f_frsize, "os.statvfs"))
            u = shutil.disk_usage(mount)
            return field(u.total, "shutil.disk_usage"), field(u.free, "shutil.disk_usage")
        except OSError:
            return missing("statvfs falhou"), missing("statvfs falhou")

    entries: dict[str, dict] = {}
    if system == "linux":
        out = run_cmd(["lsblk", "-b", "-J", "-o", "NAME,ROTA,TRAN,TYPE,MOUNTPOINTS"], probes, timeout=5.0)
        if out is None and probes.items[-1]["status"] == "error":
            probes.items.pop()
            out = run_cmd(["lsblk", "-b", "-J", "-o", "NAME,ROTA,TRAN,TYPE,MOUNTPOINT"], probes, timeout=5.0)
        for d in P.parse_lsblk_json(out or ""):
            kind, conf, note = P.disk_kind(d)
            for m in d["mounts"]:
                if P.user_visible_mount(m) and m not in entries:
                    entries[m] = {"kind": field(kind, "lsblk:rota+tran", conf, note) if kind else missing("lsblk não informou ROTA")}
    root = "/" if system != "windows" else os.environ.get("SystemDrive", "C:") + "\\"
    if system == "windows":
        entries = {"C:\\": {}}
        root = "C:\\"
    if "/" not in entries and root not in entries:
        entries[root] = {}
    disks = []
    for m in sorted(entries):
        total, free = stat(m)
        disks.append({"mount": m, "kind": entries[m].get("kind") or missing("lsblk indisponível"),
                      "total_bytes": total, "free_bytes": free})
    lsblk_ok = any(p["probe"] == "lsblk" and p["status"] == "ok" for p in probes.items)
    return field(disks, "os.statvfs+lsblk" if lsblk_ok else "os.statvfs", "high" if lsblk_ok else "medium",
                 None if lsblk_ok else "tipo de disco não detectado")


def collect_power(system: str, probes: Probes) -> dict:
    if system == "linux":
        root = "/sys/class/power_supply"
        try:
            names = sorted(os.listdir(root))
        except OSError:
            probes.add("power_supply", "not_found", "/sys/class/power_supply indisponível")
            return {"has_battery": missing("power_supply indisponível"), "on_battery": missing("power_supply indisponível")}
        probes.add("power_supply", "ok")
        sup = []
        for n in names:
            d: dict[str, str | None] = {}
            for k in ("type", "status"):
                try:
                    with open(f"{root}/{n}/{k}", encoding="utf-8") as fh:
                        d[k] = fh.read(32).strip()
                except OSError:
                    d[k] = None
            sup.append(d)
        has, on = P.parse_power(sup)
        return {"has_battery": field(has, "/sys/class/power_supply"),
                "on_battery": field(on, "/sys/class/power_supply") if on is not None else missing("estado da bateria desconhecido")}
    if system == "darwin":
        out = run_cmd(["pmset", "-g", "batt"], probes)
        if out:
            has = "InternalBattery" in out
            return {"has_battery": field(has, "pmset"), "on_battery": field("'Battery Power'" in out, "pmset")}
    return {"has_battery": missing("não coletado nesta plataforma"), "on_battery": missing("não coletado nesta plataforma")}


RUNTIME_BINS = {"ollama": ("ollama",), "llama_cli": ("llama-cli",), "llama_server": ("llama-server",),
                "llama_bench": ("llama-bench",), "lm_studio": ("lm-studio", "lmstudio", "lms"), "comfyui": ("comfyui",)}


def collect_runtimes(probes: Probes) -> dict:
    rt: dict[str, Any] = {}
    for key, bins in RUNTIME_BINS.items():
        rt[key] = field(any(shutil.which(b) for b in bins), "PATH")
    if rt["ollama"]["value"]:
        out = run_cmd(["ollama", "list"], probes, timeout=5.0, probe="ollama")
        if out is not None:
            rt["ollama_models"] = field(P.parse_ollama_list(out), "ollama list", "medium",
                                        "tamanhos arredondados pelo ollama (unidades decimais)")
        else:
            rt["ollama_models"] = missing("ollama list falhou (servidor parado?)")
    else:
        probes.add("ollama", "not_found", "ollama não está no PATH")
        rt["ollama_models"] = missing("ollama ausente do PATH")
    return rt


# ---------------------------------------------------------------- macOS / Windows (best effort)


def _sysctl(key: str, probes: Probes) -> str | None:
    out = run_cmd(["sysctl", "-n", key], probes, probe=f"sysctl:{key}")
    return out.strip() if out and out.strip() else None


def collect_darwin(probes: Probes) -> tuple[dict, dict]:
    cpu = _cpu_empty()
    brand = _sysctl("machdep.cpu.brand_string", probes)
    arm = platform.machine() == "arm64"
    if brand:
        cpu["model"] = field(brand, "sysctl", "high")
        cpu["vendor"] = field("Apple" if "Apple" in brand else "GenuineIntel" if "Intel" in brand else brand.split()[0], "sysctl", "medium")
    for key, k in (("cores_logical", "hw.logicalcpu"), ("cores_physical", "hw.physicalcpu")):
        v = _sysctl(k, probes)
        if v and v.isdigit() and int(v) > 0:
            cpu[key] = field(int(v), "sysctl")
    p, e = _sysctl("hw.perflevel0.physicalcpu", probes), _sysctl("hw.perflevel1.physicalcpu", probes)
    if p and p.isdigit() and int(p) > 0:
        cpu["cores_performance"] = field(int(p), "sysctl:hw.perflevel0")
        cpu["cores_efficiency"] = field(int(e), "sysctl:hw.perflevel1") if e and e.isdigit() else field(0, "derived:no_perflevel1", "medium")
    elif cpu["cores_physical"]["value"]:
        cpu["cores_performance"] = field(cpu["cores_physical"]["value"], "derived:cores_physical", "medium", "CPU não híbrida")
        cpu["cores_efficiency"] = field(0, "derived:cpu_not_hybrid", "medium")
    for k, v in (("avx2", False), ("avx512", False), ("amx", False), ("neon", arm), ("sve", False)):
        cpu["isa"][k] = field(v, "platform.machine", "medium")
    mem = {"ram_total_bytes": missing("sysctl hw.memsize falhou"), "swap_total_bytes": missing("não coletado no macOS")}
    v = _sysctl("hw.memsize", probes)
    if v and v.isdigit():
        mem["ram_total_bytes"] = field(int(v), "sysctl:hw.memsize")
    mem = _memory_rest(mem, None, unified=arm, platform_note="derived:platform")
    mem["type"] = missing("tipo de memória não coletado no macOS")
    return cpu, mem


def collect_windows(probes: Probes) -> tuple[dict, dict]:
    cpu = _cpu_empty()
    out = run_cmd(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"],
                  probes, timeout=15.0, probe="powershell:cpu")
    try:
        doc = json.loads(out or "null")
    except ValueError:
        doc = None
    if isinstance(doc, list) and doc:
        doc = doc[0]
    if isinstance(doc, dict):
        if doc.get("Name"):
            cpu["model"] = field(str(doc["Name"]).strip(), "powershell")
        if doc.get("Manufacturer"):
            cpu["vendor"] = field(doc["Manufacturer"], "powershell")
        if isinstance(doc.get("NumberOfLogicalProcessors"), int) and doc["NumberOfLogicalProcessors"] > 0:
            cpu["cores_logical"] = field(doc["NumberOfLogicalProcessors"], "powershell")
        if isinstance(doc.get("NumberOfCores"), int) and doc["NumberOfCores"] > 0:
            cpu["cores_physical"] = field(doc["NumberOfCores"], "powershell")
            cpu["cores_performance"] = missing("híbrido P/E não distinguido no Windows")
            cpu["cores_efficiency"] = missing("híbrido P/E não distinguido no Windows")
    mem = {"ram_total_bytes": missing("Win32_ComputerSystem falhou"), "swap_total_bytes": missing("não coletado no Windows")}
    out = run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                  probes, timeout=15.0, probe="powershell:ram")
    if out and out.strip().isdigit():
        mem["ram_total_bytes"] = field(int(out.strip()), "powershell")
    return cpu, _memory_rest(mem, None, unified=False, platform_note="derived:platform")


# ---------------------------------------------------------------- orquestração


def collect(dmidecode_file: str | None = None) -> dict:
    probes = Probes()
    system = {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}.get(platform.system(), "other")
    why = "seção falhou"
    os_ = _guard(probes, "os", lambda: collect_os(system, probes)) or {
        k: missing(why) for k in ("name", "version", "kernel", "arch")}
    cpu = mem = None
    if system == "linux":
        cpu = _guard(probes, "cpu", lambda: collect_cpu_linux(probes))
        mem = _guard(probes, "memory", lambda: collect_memory_linux(probes, dmidecode_file))
    elif system == "darwin":
        r = _guard(probes, "cpu+memory", lambda: collect_darwin(probes))
        cpu, mem = r if r else (None, None)
    elif system == "windows":
        r = _guard(probes, "cpu+memory", lambda: collect_windows(probes))
        cpu, mem = r if r else (None, None)
    cpu = cpu or _cpu_empty()
    if mem is None:
        mem = _memory_rest({"ram_total_bytes": missing("não coletado"), "swap_total_bytes": missing("não coletado")},
                           None, unified=None, platform_note="derived:platform")
    gpus = _guard(probes, "gpus", lambda: collect_gpus(system, probes)) or field(None, "none: coleta de GPU falhou", "low")
    disks = _guard(probes, "disks", lambda: collect_disks(system, probes)) or field(None, "none: coleta de disco falhou", "low")
    power = _guard(probes, "power", lambda: collect_power(system, probes)) or {
        "has_battery": missing(why), "on_battery": missing(why)}
    runtimes = _guard(probes, "runtimes", lambda: collect_runtimes(probes)) or {
        **{k: missing(why) for k in RUNTIME_BINS}, "ollama_models": missing(why)}
    return {
        "schema_version": 1, "kind": "system", "scout_version": __version__,
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": system, "os": os_, "cpu": cpu, "memory": mem, "gpus": gpus, "disks": disks,
        "power": power, "runtimes": runtimes, "probes": probes.items,
    }
