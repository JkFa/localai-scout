"""Parsers puros da coleta: texto entra, estruturas saem. Nada aqui executa comando,
lê arquivo ou toca a rede. Tudo que não é encontrado volta como None (nunca 0)."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

MIB = 1024 * 1024
KIB = 1024


def parse_os_release(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def parse_kv_colon(text: str) -> dict[str, str]:
    """`Chave: valor` (lscpu, /proc/cpuinfo do primeiro bloco). Primeira ocorrência vence."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip() and k.strip() not in out:
            out[k.strip()] = v.strip()
    return out


def _int(s: str | None) -> int | None:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def parse_lscpu(text: str) -> dict[str, Any]:
    kv = parse_kv_colon(text)
    threads = _int(kv.get("Thread(s) per core"))
    cores = _int(kv.get("Core(s) per socket"))
    sockets = _int(kv.get("Socket(s)"))
    physical = cores * sockets if cores and sockets else None
    flags_s = kv.get("Flags") or kv.get("Features")
    return {
        "model": kv.get("Model name") or None,
        "vendor": kv.get("Vendor ID") or None,
        "logical": _int(kv.get("CPU(s)")),
        "physical": physical,
        "threads_per_core": threads,
        "arch": kv.get("Architecture") or None,
        "flags": set(flags_s.split()) if flags_s else None,
    }


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """Fallback x86/ARM de /proc/cpuinfo."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    procs = [parse_kv_colon(b) for b in blocks]
    procs = [p for p in procs if "processor" in p]
    if not procs:
        return {"model": None, "vendor": None, "logical": None, "physical": None, "flags": None}
    first = procs[0]
    cores_per = _int(first.get("cpu cores"))
    pkgs = {p.get("physical id") for p in procs if p.get("physical id") is not None}
    physical = cores_per * max(len(pkgs), 1) if cores_per else None
    flags_s = first.get("flags") or first.get("Features")
    return {
        "model": first.get("model name") or first.get("Model") or None,
        "vendor": first.get("vendor_id") or None,
        "logical": len(procs),
        "physical": physical,
        "flags": set(flags_s.split()) if flags_s else None,
    }


def parse_cpu_list(text: str) -> set[int]:
    """'0-3,8,10-11' -> {0,1,2,3,8,10,11}. Texto inválido -> conjunto vazio."""
    out: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        try:
            lo, hi = int(a), int(b) if b else int(a)
        except ValueError:
            return set()
        out.update(range(lo, hi + 1))
    return out


def parse_lscpu_p(text: str) -> dict[int, int]:
    """`lscpu -p=CPU,CORE` -> {cpu: core}."""
    out: dict[int, int] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        a, _, b = line.partition(",")
        ia, ib = _int(a), _int(b)
        if ia is not None and ib is not None:
            out[ia] = ib
    return out


def hybrid_split(cpu_core: dict[int, int], p_cpus: set[int], e_cpus: set[int]) -> tuple[int, int] | None:
    """(núcleos físicos de performance, núcleos de eficiência) numa CPU híbrida Intel.
    p_cpus/e_cpus vêm de /sys/devices/cpu_core|cpu_atom/cpus. Sem mapa CPU->core, None."""
    if not p_cpus or not e_cpus or not cpu_core:
        return None
    p = {cpu_core[c] for c in p_cpus if c in cpu_core}
    e = {cpu_core[c] for c in e_cpus if c in cpu_core}
    if not p or not e:
        return None
    return len(p), len(e)


def isa_from_flags(flags: set[str] | None, arch: str | None) -> dict[str, bool | None]:
    """Flags de ISA. Sem lista de flags, só neon/sve de arquiteturas sem ambiguidade."""
    arm = (arch or "").lower() in ("aarch64", "arm64", "armv8l", "armv7l")
    x86 = (arch or "").lower() in ("x86_64", "amd64", "i686", "i386")
    if flags is None:
        return {"avx2": None, "avx512": None, "amx": None,
                "neon": True if arm else (False if x86 else None), "sve": False if x86 else None}
    return {
        "avx2": "avx2" in flags,
        "avx512": any(f.startswith("avx512") for f in flags),
        "amx": any(f.startswith("amx") for f in flags),
        "neon": bool(flags & {"asimd", "neon"}) or (arm and "asimd" in flags),
        "sve": any(f.startswith("sve") for f in flags),
    }


def parse_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo -> bytes inteiros. Só MemTotal e SwapTotal (RAM livre nunca entra)."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^(MemTotal|SwapTotal):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * KIB
    return out


_DMI_BAD = {"", "unknown", "other", "<out of spec>", "not specified", "undefined", "n/a"}


def parse_dmidecode(text: str) -> dict[str, Any]:
    """`dmidecode -t memory` -> {populated, type, speed_mts, channels}. Campos ausentes = None."""
    devices: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for line in text.splitlines():
        if line.startswith("Memory Device"):
            cur = {}
            devices.append(cur)
        elif cur is not None and ":" in line and line.startswith(("\t", " ")):
            k, _, v = line.partition(":")
            cur.setdefault(k.strip(), v.strip())
    pop = [d for d in devices if re.match(r"^\d+\s*(MB|GB|TB|kB)", d.get("Size", ""))]
    if not pop:
        return {"populated": 0 if devices else None, "type": None, "speed_mts": None, "channels": None,
                "data_width_bits": None}
    types = {d.get("Type", "") for d in pop} - _DMI_BAD
    type_ = next(iter(types)) if len(types) == 1 else None
    speeds: set[int] = set()
    for d in pop:
        for key in ("Configured Memory Speed", "Speed"):
            m = re.match(r"^(\d+)\s*MT/s", d.get(key, ""))
            if m:
                speeds.add(int(m.group(1)))
                break
    # Canais: localizadores distintos de canal; sem padrão, nº de módulos (estimativa).
    chans = set()
    for d in pop:
        loc = d.get("Locator", "") + " " + d.get("Bank Locator", "")
        m = re.search(r"channel\s*([a-z0-9]+)", loc, re.I)
        if m:
            c = re.search(r"controller\s*([a-z0-9]+)", loc, re.I)  # Controller0-ChannelA != Controller1-ChannelA
            chans.add(((c.group(1).upper() if c else ""), m.group(1).upper()))
    channels = len(chans) if chans else len(pop)
    widths = {_int(re.sub(r"\D", "", d.get("Data Width", "")) or None) for d in pop}  # Data Width, nunca Total Width (ECC)
    width = next(iter(widths)) if len(widths) == 1 else None
    return {"populated": len(pop), "type": type_, "data_width_bits": width,
            "speed_mts": min(speeds) if speeds else None,  # o mais lento limita o barramento
            "channels": channels, "channels_exact": bool(chans)}


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    """--query-gpu=index,name,memory.total,memory.free,driver_version[,compute_cap] --format=csv,noheader,nounits."""
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(text)):
        row = [c.strip() for c in row]
        if len(row) < 5:
            continue

        def mib(x: str) -> int | None:
            n = _int(x)
            return n * MIB if n is not None else None

        cc = row[5] if len(row) > 5 and re.fullmatch(r"\d+\.\d+", row[5]) else None
        gpus.append({"name": row[1] or None, "vram_total": mib(row[2]), "vram_free": mib(row[3]),
                     "driver": row[4] if row[4] and not row[4].startswith("[") else None, "cc": cc})
    return gpus


def _json_obj(text: str) -> Any:
    i = text.find("{")
    if i < 0:
        return None
    try:
        return json.loads(text[i:text.rfind("}") + 1])
    except json.JSONDecodeError:
        return None


def parse_rocm_smi_json(text: str) -> list[dict[str, Any]]:
    """rocm-smi --showproductname --showmeminfo vram --showdriverversion --json."""
    doc = _json_obj(text)
    if not isinstance(doc, dict):
        return []
    driver = None
    sysd = doc.get("system")
    if isinstance(sysd, dict):
        driver = sysd.get("Driver version") or None
    gpus = []
    for key in sorted((k for k in doc if re.fullmatch(r"card\d+", k)), key=lambda k: int(k[4:])):
        c = doc[key]
        if not isinstance(c, dict):
            continue
        used = _int(c.get("VRAM Total Used Memory (B)"))
        total = _int(c.get("VRAM Total Memory (B)"))
        name = c.get("Card series") or c.get("Card SKU")
        gpus.append({"name": name if name and name != "N/A" else None, "vram_total": total,
                     "vram_free": total - used if total is not None and used is not None and used <= total else None,
                     "driver": driver, "cc": None})
    return gpus


_PCI_VENDOR = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}


def parse_drm_cards(cards: list[dict[str, str | None]]) -> list[dict[str, Any]]:
    """Último recurso: /sys/class/drm/cardN/device/{vendor,mem_info_vram_total,mem_info_vram_used}."""
    out = []
    for c in cards:
        v = (c.get("vendor") or "").strip().lower()
        total, used = _int(c.get("vram_total")), _int(c.get("vram_used"))
        out.append({"vendor": _PCI_VENDOR.get(v, "other" if v else None), "name": None,
                    "vram_total": total,
                    "vram_free": total - used if total is not None and used is not None and used <= total else None,
                    "driver": None, "cc": None})
    return out


def gpu_caps_from_cc(cc: str | None) -> dict[str, bool | None]:
    """bf16 (cc >= 8.0), fp8 (cc >= 8.9), flash attention (cc >= 8.0, Ampere+). Sem cc, None."""
    if cc is None:
        return {"bf16": None, "fp8": None, "flash_attention": None}
    major, minor = (int(x) for x in cc.split("."))
    v = (major, minor)
    return {"bf16": v >= (8, 0), "fp8": v >= (8, 9), "flash_attention": v >= (8, 0)}


def parse_lsblk_json(text: str, require_mount: bool = True) -> list[dict[str, Any]]:
    """lsblk -b -J -o NAME,ROTA,TRAN,TYPE,MOUNTPOINTS -> [{disk, rota, tran, mounts}] (sem loop/zram/rom)."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    for dev in doc.get("blockdevices", []) if isinstance(doc, dict) else []:
        name = str(dev.get("name", ""))
        if dev.get("type") != "disk" or name.startswith(("zram", "ram", "loop")):
            continue
        mounts: set[str] = set()

        def walk(d: dict) -> None:
            mps = d.get("mountpoints")
            if mps is None and d.get("mountpoint") is not None:
                mps = [d["mountpoint"]]
            mounts.update(m for m in (mps or []) if m and m.startswith("/"))
            for ch in d.get("children") or []:
                walk(ch)

        walk(dev)
        rota = dev.get("rota")
        if isinstance(rota, str):
            rota = {"1": True, "0": False}.get(rota)
        out.append({"disk": name, "rota": rota if isinstance(rota, bool) else None,
                    "tran": (dev.get("tran") or None), "mounts": sorted(mounts)})
    return out


def disk_kind(disk: dict[str, Any]) -> tuple[str | None, str, str | None]:
    """-> (kind, confidence, note)."""
    tran, rota = disk.get("tran"), disk.get("rota")
    if tran == "nvme" or str(disk.get("disk", "")).startswith("nvme"):
        return "nvme", "high", None
    if rota is None:
        return None, "low", "lsblk não informou ROTA"
    if tran == "usb":
        return ("hdd" if rota else "ssd"), "low", "USB: ROTA não é confiável"
    return ("hdd" if rota else "ssd"), "medium", None


def user_visible_mount(m: str) -> bool:
    """Mounts de sistema apenas: nunca caminhos que carreguem nome de usuário."""
    if m in ("/", "/home"):
        return True
    return not m.startswith(("/snap", "/boot", "/home/", "/media", "/run", "/var/lib", "/sys", "/proc", "/dev"))


def parse_power(supplies: list[dict[str, str | None]]) -> tuple[bool, bool | None]:
    """[{type,status,online,scope}] de /sys/class/power_supply -> (has_battery, on_battery).
    A tomada vem de `online` dos supplies Mains/USB*, nunca do status da bateria sozinho:
    qualquer um online=1 => AC; todos online=0 => bateria; sem Mains legível, só `Discharging` prova bateria."""
    def t(s: dict) -> str:
        return (s.get("type") or "").strip()
    bats = [s for s in supplies if t(s) == "Battery" and (s.get("scope") or "").strip() != "Device"]
    if not bats:
        return False, False
    states = [(s.get("status") or "").strip() for s in bats]
    online = [(s.get("online") or "").strip() for s in supplies if t(s) == "Mains" or t(s).startswith("USB")]
    online = [o for o in online if o in ("0", "1")]
    if "1" in online:
        return True, False
    if online:  # todos offline
        return True, (None if "Charging" in states else True)
    if "Discharging" in states:
        return True, True
    return True, None


_UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}


def parse_ollama_list(text: str) -> list[dict[str, Any]]:
    """`ollama list` -> [{name, size_bytes}]. Tamanho vem arredondado (unidades decimais): aproximado."""
    out = []
    for line in text.splitlines()[1:]:
        m = re.match(r"^(\S+)\s+\S+\s+(\d+(?:\.\d+)?)\s*([KMGT]?B)\b", line)
        if m:
            out.append({"name": m.group(1), "size_bytes": int(round(float(m.group(2)) * _UNITS[m.group(3)]))})
        elif line.strip():
            out.append({"name": line.split()[0], "size_bytes": None})
    return out


_RANK = {"low": 0, "medium": 1, "high": 2}


def ram_bandwidth(mts: int | None, channels: int | None, mem_type: str | None,
                  table: dict | None, mts_conf: str = "medium", channels_conf: str = "medium",
                  data_width_bits: int | None = None) -> tuple[int | None, str, str, str | None]:
    """-> (bytes/s, source, confidence, note).
    banda = MT/s * canais * data_width_bits / 8 (Data Width do dmidecode, nunca Total Width). Largura fora de {32, 64}
    ou ausente: assume 64 e rebaixa a confiança a low. Tipo de memória não entra na fórmula.
    Sem MT/s ou canais: tabela por geração (low); senão None. Confiança nunca passa da menor entre as entradas (teto medium)."""
    if mts and channels:
        width, wconf = (data_width_bits, "medium") if data_width_bits in (32, 64) else (64, "low")
        conf = min(mts_conf, channels_conf, wconf, "medium", key=_RANK.__getitem__)
        note = f"{mts} MT/s * {channels} canais * {width} bits / 8"
        if wconf == "low":
            note += "; Data Width ausente ou fora de {32, 64}: assumido 64"
        return (mts * channels * width // 8 * 10**6, "estimate:mts*channels*data_width/8", conf, note)
    t = ((table or {}).get("ram") or {}).get((mem_type or "").upper())
    if t and isinstance(t.get("bytes_per_s"), int):
        return (t["bytes_per_s"], "estimate:table:data/bandwidth_tables.json", "low",
                "dado de referência da geração, não detectado; " + str(t.get("note", "")))
    why = "tipo de memória e MT/s não detectados" if not mem_type else f"geração {mem_type} fora da tabela"
    return None, f"none: {why}", "low", None


_STD_MAX_MTS = {"DDR3": 2133, "DDR4": 3200}


def type_divergence(mem_type: str | None, mts: int | None, data_width_bits: int | None) -> str | None:
    """Tipo reportado incompatível com MT/s/largura (ex.: DDR4 a 5500 MT/s em canal de 32 bits = padrão LPDDR5)."""
    t = (mem_type or "").upper()
    if t not in _STD_MAX_MTS:
        return None
    if mts and mts > _STD_MAX_MTS[t]:
        return f"{t} reportado, mas {mts} MT/s excede o padrão {t} ({_STD_MAX_MTS[t]})"
    if data_width_bits == 32:
        return f"{t} reportado, mas Data Width de 32 bits não é típico de {t} (DIMM de 64 bits)"
    return None


def hybrid_by_freq(rows: list[tuple[int, int, float | None]]) -> tuple[str, tuple[int, int] | None, float | None]:
    """`lscpu -p=CPU,CORE,MAXMHZ` sem sysfs. -> ("hybrid", (perf, eff), separação %) | ("uniform", None, None) | ("unknown", None, None).
    Dois grupos de MAXMHZ separados por >15%: o de maior frequência é o de performance."""
    per_core: dict[int, float] = {}
    for _cpu, core, mhz in rows:
        if mhz is None or mhz <= 0:
            return "unknown", None, None
        per_core[core] = max(per_core.get(core, 0.0), mhz)
    if not per_core:
        return "unknown", None, None
    hi = max(per_core.values())
    fast = [c for c, v in per_core.items() if v >= 0.85 * hi]
    slow = [c for c in per_core if c not in fast]
    if not slow:
        return "uniform", None, None
    slow_hi = max(per_core[c] for c in slow)
    if any(per_core[c] < 0.85 * slow_hi for c in slow):
        return "unknown", None, None  # mais de dois grupos
    return "hybrid", (len(fast), len(slow)), round((1 - slow_hi / hi) * 100, 1)


def parse_lscpu_p_mhz(text: str) -> list[tuple[int, int, float | None]]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(",")
        cpu, core = _int(parts[0]), _int(parts[1]) if len(parts) > 1 else None
        if cpu is None or core is None:
            continue
        try:
            mhz = float(parts[2]) if len(parts) > 2 and parts[2].strip() else None
        except ValueError:
            mhz = None
        rows.append((cpu, core, mhz))
    return rows


OFFICIAL_DOMAINS = ("nvidia.com", "amd.com", "intel.com")


def _official(url: str) -> bool:
    m = re.match(r"^https://([^/]+)/", url or "")
    host = m.group(1).lower() if m else ""
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def gpu_table_lookup(name: str | None, vram_total: int | None, table: dict | None) -> tuple[int | None, list[str]]:
    """Tabela `gpu` de bandwidth_tables.json, chave exata `nome|vram_bytes`. -> (bytes/s, source_urls).
    Entrada sem inteiro positivo ou sem fonte oficial (nvidia.com/amd.com/intel.com) é ignorada.
    Sem casamento exato: (None, []). Nenhuma fórmula de fallback."""
    if not name or vram_total is None or not isinstance(table, dict):
        return None, []
    e = (table.get("gpu") or {}).get(f"{name}|{vram_total}")
    if not isinstance(e, dict):
        return None, []
    bw, urls = e.get("bandwidth_bytes_s"), e.get("source_url")
    urls = [urls] if isinstance(urls, str) else (urls if isinstance(urls, list) else [])
    if not isinstance(bw, int) or isinstance(bw, bool) or bw <= 0 or not urls or not all(_official(u) for u in urls):
        return None, []
    return bw, urls
