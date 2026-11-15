#!/usr/bin/env python3
"""Critério de pronto da Tarefa 5 (schemas/calibration_store_spec.md, "Critério de pronto").

Uso:
    python3 schemas/check_calibration_done.py "$T" [--system ARQ] [--catalog ARQ] [--tuning ARQ]
                                                    [--other-system ARQ] [--model ID] [--task T] [--ctx N]

`$T` contém o `cal.json` gravado por um `scout bench --run` real. O `system.json` base é
`$T/system.json`, a menos que `--system` aponte outro.

Sem rede e sem medir nada. Roda as 11 checagens chamando `scout plan --json` (a função
`scout.cli.main`, em processo) com variações de `cal.json` e `system.json` gravadas em `$T`
(`cal_<tag>.json`, `system_<tag>.json`, `plan_<tag>.json`).

Energia e versão do motor: a CLI não tem como injetá-las (o plan as lê com
`bench.read_power` e `bench.engine_version` dentro de `run_plan`), então este script troca essas
duas funções do módulo `scout.bench` por versões fixas durante cada chamada e as restaura depois.
Nenhuma linha de `scout/` muda. Na checagem 10 as duas funções são trocadas por versões que
levantam exceção, para provar que `--no-calibration` não lê nada.

Catálogo e tuning: por padrão `data/catalog.json` e `data/tuning.json` do repositório, porque o
modelo medido pelo bench real precisa existir no catálogo. A suíte de testes passa fixtures.

Saída: uma linha OK/FALHA por checagem; código de saída 1 se alguma falhar, 2 se o `$T` não servir.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYNTH_STATE = {"on_ac": False, "profile": "balanced", "not_applicable": []}
NA_STATE = {"on_ac": None, "profile": None, "not_applicable": ["on_ac", "profile"]}


class Setup(Exception):
    """O `$T` não permite montar as variações (arquivo ausente, pool cpu vazio...)."""


class PlanError(Exception):
    pass


def jload(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Setup(f"{path}: {e}") from e


class Ctx:
    def __init__(self, a: argparse.Namespace):
        self.T = Path(a.T)
        self.system_path = Path(a.system) if a.system else self.T / "system.json"
        self.system = jload(self.system_path)
        self.cal = jload(self.T / "cal.json")
        self.other = jload(Path(a.other_system))
        self.catalog = Path(a.catalog)
        self.tuning = Path(a.tuning)
        self.model, self.task, self.ctx_tokens = a.model, a.task, a.ctx
        pools = self.cal.get("pools") or {}
        if not pools.get("cpu"):
            raise Setup("cal.json não tem entrada no pool cpu")
        self.entry = pools["cpu"][0]
        self.mw = self.entry.get("measured_with") or {}
        self.power0 = copy.deepcopy(self.entry["power"])
        self.engines0 = {self.entry["tool"]: self.entry["tool_version"]}
        hit = [m for m in self.entry["models"] if m["id"] == self.model]
        if not hit:
            raise Setup(f"{self.model} não está em models da entrada cpu do cal.json")
        self.measured_bytes = hit[0]["bytes"]

    # ------------------------------------------------------------------ montagem

    def write(self, name: str, doc: Any) -> Path:
        p = self.T / name
        p.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return p

    def sys_with(self, **fields: Any) -> dict:
        """Cópia do system base com `memory.<campo>.value` / `os.<campo>.value` trocados."""
        s = copy.deepcopy(self.system)
        for dotted, value in fields.items():
            sect, name = dotted.split("__")
            s[sect][name] = {**s[sect].get(name, {}), "value": value}
        return s

    def cal_with(self, **kw: Any) -> dict:
        c = copy.deepcopy(self.cal)
        c.update(kw)
        return c

    # ------------------------------------------------------------------ execução

    def plan(self, tag: str, system: dict | None = None, cal: dict | None = None,
             power: dict | None = None, engines: dict | None = None, no_cal: bool = False) -> dict:
        from scout import bench, cli

        sys_path = self.write(f"system_{tag}.json", system if system is not None else self.system)
        argv = ["plan", "--json", "--system", str(sys_path), "--catalog", str(self.catalog),
                "--tuning", str(self.tuning), "--task", self.task, "--ctx", str(self.ctx_tokens)]
        if no_cal:
            argv.append("--no-calibration")

            def boom(*_a: Any, **_k: Any) -> Any:
                raise AssertionError("--no-calibration não pode ler energia nem motor")

            fake_power: Callable = boom
            fake_engine: Callable = boom
        else:
            cal_path = self.write(f"cal_{tag}.json", cal if cal is not None else self.cal)
            argv += ["--calibration", str(cal_path)]
            p = copy.deepcopy(self.power0 if power is None else power)
            e = dict(self.engines0 if engines is None else engines)

            def fake_power(*_a: Any, **_k: Any) -> dict:
                return copy.deepcopy(p)

            def fake_engine(tool: str) -> str | None:
                return e.get(tool)

        orig = (bench.read_power, bench.engine_version)
        bench.read_power, bench.engine_version = fake_power, fake_engine
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(argv)
        finally:
            bench.read_power, bench.engine_version = orig
        if rc != 0:
            raise PlanError(f"scout plan ({tag}) saiu com {rc}: {err.getvalue().strip()[:300]}")
        doc = json.loads(out.getvalue())
        self.write(f"plan_{tag}.json", doc)
        return doc

    # ------------------------------------------------------------------ leitura do resultado

    def cand(self, out: dict) -> dict:
        for c in out["candidates"]:
            if c["model"]["id"] == self.model:
                return c
        raise PlanError(f"{self.model} não é candidato (task {self.task})")

    @staticmethod
    def echo(out: dict) -> dict:
        return out["inputs"]["calibration"]

    def pool(self, out: dict) -> dict:
        return self.echo(out)["pools"]["cpu"]

    @staticmethod
    def codes(out: dict) -> list[str]:
        return [w["code"] for w in out["warnings"]]

    @staticmethod
    def msgs(out: dict, code: str) -> list[str]:
        return [w["message"] for w in out["warnings"] if w["code"] == code]

    def tps(self, out: dict) -> float:
        return self.cand(out)["tokens_per_s_est"]


def need(problems: list[str], cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)


def pool_is(c: Ctx, out: dict, problems: list[str], status: str, label: str = "") -> None:
    need(problems, c.pool(out)["status"] == status,
         f"{label}pool cpu status={c.pool(out)['status']!r}, esperado {status!r}")


# ---------------------------------------------------------------------------- checagens


def check1(c: Ctx) -> list[str]:
    p: list[str] = []
    out = c.plan("base")
    need(p, c.echo(out)["status"] == "used", f"status={c.echo(out)['status']!r}, esperado 'used'")
    pool_is(c, out, p, "applied")
    need(p, c.pool(out)["power_match"] == "same", f"power_match={c.pool(out)['power_match']!r}")
    cand = c.cand(out)
    e = c.entry
    measured = 1.0 / (c.measured_bytes / e["bandwidth_bytes_per_s"] + e.get("overhead_s_per_token", 0.0))
    pred = cand["tokens_per_s_est"]
    need(p, pred is not None and abs(pred - measured) <= 0.30 * measured,
         f"previsto {pred} tok/s fora de 30% do medido {measured:.2f} (banda/bytes do cal.json)")
    need(p, cand["confidence"] == "high",
         f"confiança {cand['confidence']!r} {cand['confidence_reasons']}, esperado 'high'")
    return p


def check2(c: Ctx) -> list[str]:
    p: list[str] = []
    out = c.plan("mem_unknown", system=c.sys_with(memory__speed_mts=None))
    pool_is(c, out, p, "applied")
    cand = c.cand(out)
    need(p, cand["confidence"] == "medium", f"confiança {cand['confidence']!r}, esperado 'medium'")
    need(p, "calibration_memory_unverified" in cand["confidence_reasons"],
         f"razão ausente em {cand['confidence_reasons']}")
    need(p, any("dmidecode" in m for m in c.msgs(out, "calibration_memory_unverified")),
         "nenhum aviso calibration_memory_unverified cita dmidecode")
    return p


def check3(c: Ctx) -> list[str]:
    p: list[str] = []
    stored = c.mw.get("memory.speed_mts")
    if stored is None:
        return ["pré-condição: cal.json sem memory.speed_mts, não há como divergir"]
    other_mts = 4800 if stored != 4800 else 3200
    sys3 = c.sys_with(memory__speed_mts=other_mts)
    out = c.plan("mem_diff", system=sys3)
    pool_is(c, out, p, "discarded")
    need(p, "calibration_machine_mismatch" in c.codes(out), f"avisos {c.codes(out)}")
    eta = c.plan("mem_diff_eta", system=sys3, no_cal=True)
    need(p, c.tps(out) == c.tps(eta), f"tokens/s {c.tps(out)} != estimativa por eta {c.tps(eta)}")
    return p


def check4(c: Ctx) -> list[str]:
    p: list[str] = []
    out = c.plan("layer1_gpu", system=c.other)
    need(p, c.echo(out)["status"] == "discarded", f"status={c.echo(out)['status']!r}")
    pool_is(c, out, p, "discarded")
    return p


def _kernel_variants(stored: str) -> tuple[str, str]:
    m = re.match(r"^(\d+)\.(\d+)", stored)
    if not m:
        raise Setup(f"os.kernel gravado ({stored!r}) sem major.minor")
    major, minor = int(m.group(1)), int(m.group(2))
    return f"{major}.{minor}.999-qa-patch", f"{major}.{minor + 1}.0-qa-minor"


def check5(c: Ctx) -> list[str]:
    p: list[str] = []
    stored = c.mw.get("os.kernel")
    if not isinstance(stored, str):
        return ["pré-condição: cal.json sem os.kernel"]
    patch, minor = _kernel_variants(stored)
    sys_p = c.sys_with(os__kernel=patch)
    cal_p = c.cal  # o kernel gravado vive no measured_with da entrada (rev. 4)
    out = c.plan("kernel_patch", system=sys_p, cal=cal_p)
    pool_is(c, out, p, "applied", "patch: ")
    line = next((m for m in c.pool(out)["match"] if m["field"] == "os.kernel"), None)
    need(p, line is not None and line["effect"] == "none", f"patch: linha os.kernel={line}")
    need(p, "calibration_kernel_changed" not in c.codes(out), "patch: calibration_kernel_changed apareceu")
    need(p, c.cand(out)["confidence"] == "high", f"patch: confiança {c.cand(out)['confidence']!r}")
    out = c.plan("kernel_minor", system=c.sys_with(os__kernel=minor), cal=cal_p)
    pool_is(c, out, p, "applied", "minor: ")
    need(p, "calibration_kernel_changed" in c.codes(out), f"minor: avisos {c.codes(out)}")
    need(p, any("scout bench --run" in m for m in c.msgs(out, "calibration_kernel_changed")),
         "minor: mensagem sem `scout bench --run`")
    cand = c.cand(out)
    need(p, cand["confidence"] == "medium", f"minor: confiança {cand['confidence']!r}, esperado 'medium'")
    need(p, "calibration_kernel_changed" in cand["confidence_reasons"], "minor: razão ausente no candidato")
    return p


def check6(c: Ctx) -> list[str]:
    p: list[str] = []
    cal = copy.deepcopy(c.cal)
    for e in cal["pools"]["cpu"]:
        e["tool_version"] = "0.0.0"
    out = c.plan("engine_changed", cal=cal)
    pool_is(c, out, p, "applied", "0.0.0: ")
    need(p, "calibration_engine_changed" in c.codes(out), f"0.0.0: avisos {c.codes(out)}")
    need(p, any("scout bench --run" in m for m in c.msgs(out, "calibration_engine_changed")),
         "0.0.0: mensagem sem `scout bench --run`")
    need(p, c.cand(out)["confidence"] == "medium", f"0.0.0: confiança {c.cand(out)['confidence']!r}")
    out = c.plan("engine_unknown", engines={c.entry["tool"]: None})
    pool_is(c, out, p, "applied", "null: ")
    need(p, "calibration_engine_unknown" in c.codes(out), f"null: avisos {c.codes(out)}")
    need(p, c.cand(out)["confidence"] == "medium", f"null: confiança {c.cand(out)['confidence']!r}")
    return p


def check7(c: Ctx) -> list[str]:
    p: list[str] = []
    if not (c.power0.get("on_ac") is True and c.power0.get("profile") == "performance"
            and not c.power0.get("not_applicable")):
        return [f"pré-condição: entrada base deveria ser tomada/performance, é {c.power0}"]
    half = copy.deepcopy(c.entry)
    half["power"] = copy.deepcopy(SYNTH_STATE)
    half["bandwidth_bytes_per_s"] = c.entry["bandwidth_bytes_per_s"] // 2
    for k in ("one_param", "two_param"):
        fit = (half.get("fit") or {}).get(k)
        if fit and "bandwidth_bytes_per_s" in fit:
            fit["bandwidth_bytes_per_s"] = fit["bandwidth_bytes_per_s"] // 2
    cal = copy.deepcopy(c.cal)
    cal["pools"]["cpu"] = [c.entry, half]
    base = c.tps(c.plan("base"))
    out = c.plan("power_same_batt", cal=cal, power=SYNTH_STATE)
    need(p, c.pool(out)["power_match"] == "same", f"bateria/balanced: power_match={c.pool(out)['power_match']!r}")
    ratio = c.tps(out) / base
    need(p, 0.45 <= ratio <= 0.55, f"bateria/balanced: tokens/s {c.tps(out)} é {ratio:.2f}x do base, esperado ~0,5x")
    out = c.plan("power_other", cal=cal, power={"on_ac": False, "profile": "performance", "not_applicable": []})
    pool = c.pool(out)
    need(p, pool["power_match"] == "other_state", f"bateria/performance: power_match={pool['power_match']!r}")
    need(p, pool["calibration"]["bandwidth_bytes_per_s"] == half["bandwidth_bytes_per_s"],
         f"entrada escolhida tem banda {pool['calibration']['bandwidth_bytes_per_s']}, esperado a menor "
         f"({half['bandwidth_bytes_per_s']})")
    need(p, c.cand(out)["confidence"] == "medium", f"confiança {c.cand(out)['confidence']!r}, esperado 'medium'")
    # O texto do critério ("na tomada" e "na bateria") só vale quando a entrada escolhida é a de
    # tomada; com a sintética (bateria/balanced) a mais lenta, a §4 manda nomear ESTA entrada e o
    # estado atual. Por isso há dois cenários: o texto da §4 com as duas entradas, e o do critério
    # (entrada única de tomada, plan na bateria).
    ms = c.msgs(out, "calibration_power_mismatch")
    need(p, any("Calibrado na bateria, perfil balanced" in m and "você está na bateria, perfil performance" in m
                for m in ms), f"duas entradas: mensagem não nomeia os dois estados: {ms}")
    out = c.plan("power_other_single", power={"on_ac": False, "profile": "performance", "not_applicable": []})
    need(p, c.pool(out)["power_match"] == "other_state", f"entrada única: power_match={c.pool(out)['power_match']!r}")
    ms = c.msgs(out, "calibration_power_mismatch")
    need(p, any("na tomada" in m and "na bateria" in m for m in ms),
         f"entrada única: mensagem sem 'na tomada'/'na bateria': {ms}")
    need(p, c.cand(out)["confidence"] == "medium", f"entrada única: confiança {c.cand(out)['confidence']!r}")
    return p


def check8(c: Ctx) -> list[str]:
    p: list[str] = []
    entry = copy.deepcopy(c.entry)
    entry["power"] = copy.deepcopy(NA_STATE)
    cal = copy.deepcopy(c.cal)
    cal["pools"]["cpu"] = [entry]
    out = c.plan("power_na", cal=cal, power=NA_STATE)
    need(p, c.pool(out)["power_match"] == "same", f"n/a: power_match={c.pool(out)['power_match']!r}")
    need(p, c.cand(out)["confidence"] == "high", f"n/a: confiança {c.cand(out)['confidence']!r}")
    out = c.plan("power_na_unknown", cal=cal,
                 power={"on_ac": None, "profile": None, "not_applicable": ["on_ac"]})
    need(p, c.pool(out)["power_match"] == "unknown", f"profile null: power_match={c.pool(out)['power_match']!r}")
    need(p, c.cand(out)["confidence"] == "medium", f"profile null: confiança {c.cand(out)['confidence']!r}")
    return p


def check9(c: Ctx) -> list[str]:
    p: list[str] = []
    out = c.plan("schema99", cal=c.cal_with(calibration_schema_version=99))
    need(p, c.echo(out)["status"] == "discarded", f"status={c.echo(out)['status']!r}")
    pool_is(c, out, p, "discarded")
    need(p, "calibration_schema_outdated" in c.codes(out), f"avisos {c.codes(out)}")
    return p


def check11(c: Ctx) -> list[str]:
    """Merge (rev. 4): o measured_with da entrada antiga sobrevive a um segundo bench com outro kernel e MT/s."""
    from scout import calstore
    from scout.calibration import measured_with

    p: list[str] = []
    old_mw = {"os.kernel": "6.98.0-qa-old", "memory.speed_mts": None, "memory.channels": None}
    old_entry = copy.deepcopy(c.entry)
    old_entry["measured_with"] = old_mw
    old = c.cal_with(pools={"cpu": [old_entry]})
    now = measured_with(c.sys_with(os__kernel="6.99.0-qa-new"))
    now["memory.speed_mts"] = now["memory.speed_mts"] or 4000
    now["memory.channels"] = now["memory.channels"] or 2
    new_entry = copy.deepcopy(c.entry)
    new_entry["power"] = copy.deepcopy(SYNTH_STATE)
    merged, _ = calstore.merge_calibration(old, c.cal["machine"], {"cpu": new_entry}, "0.0.0", now)
    entries = merged["pools"]["cpu"]
    need(p, len(entries) == 2, f"merge deixou {len(entries)} entradas, esperado 2")
    kept = next((e for e in entries if e["power"] == c.power0), None)
    need(p, kept is not None and kept["measured_with"] == old_mw, f"measured_with antigo reescrito: {kept and kept['measured_with']}")
    need(p, any(e["measured_with"] == now for e in entries), "entrada nova sem o measured_with do bench atual")
    sys_now = c.sys_with(os__kernel="6.99.0-qa-new", memory__speed_mts=now["memory.speed_mts"],
                         memory__channels=now["memory.channels"])
    out = c.plan("merge_rev4", system=sys_now, cal=merged)
    pool_is(c, out, p, "applied")
    need(p, c.pool(out)["power_match"] == "same", f"power_match={c.pool(out)['power_match']!r}")
    need(p, "calibration_kernel_changed" in c.codes(out), f"avisos {c.codes(out)}")
    need(p, "calibration_memory_unverified" in c.codes(out), f"avisos {c.codes(out)}")
    need(p, c.cand(out)["confidence"] != "high", f"confiança {c.cand(out)['confidence']!r}, nunca high")
    return p


def check10(c: Ctx) -> list[str]:
    p: list[str] = []
    out = c.plan("disabled", no_cal=True)
    need(p, c.echo(out)["status"] == "disabled", f"status={c.echo(out)['status']!r}")
    need(p, "power" not in out["inputs"], "inputs.power presente")
    return p


CHECKS: list[tuple[int, str, Callable[[Ctx], list[str]]]] = [
    (1, "plano normal: used, cpu applied, power same, Qwen a até 30% do medido, high", check1),
    (2, "camada 2 desconhecida: applied, medium, calibration_memory_unverified, aviso com dmidecode", check2),
    (3, "camada 2 divergente (4800 MT/s): discarded, calibration_machine_mismatch, volta ao eta", check3),
    (4, "camada 1 (sys_rtx3060): discarded", check4),
    (5, "kernel: só patch sem efeito; outro major.minor rebaixa um nível com calibration_kernel_changed", check5),
    (6, "motor: 0.0.0 rebaixa (engine_changed); versão atual null dá medium (engine_unknown)", check6),
    (7, "energia, outro estado: same com ~metade; other_state usa a menor banda, rebaixa e diz o estado", check7),
    (8, "energia não aplicável: same e high; profile null fora de not_applicable: unknown e medium", check8),
    (9, "calibration_schema_version 99: discarded e calibration_schema_outdated", check9),
    (10, "--no-calibration: disabled e sem inputs.power", check10),
    (11, "merge: measured_with antigo preservado; plan no estado dele não sai high", check11),
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("T", help="diretório com o cal.json do bench real (e system.json)")
    ap.add_argument("--system", default=None, help="system.json base (padrão: $T/system.json)")
    ap.add_argument("--catalog", default=str(ROOT / "data" / "catalog.json"))
    ap.add_argument("--tuning", default=str(ROOT / "data" / "tuning.json"))
    ap.add_argument("--other-system", default=str(ROOT / "schemas" / "examples" / "system_rtx3060.json"),
                    help="system.json de outra máquina, para a checagem 4")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="modelo medido pelo bench")
    ap.add_argument("--task", default="chat")
    ap.add_argument("--ctx", type=int, default=8192)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        c = Ctx(a)
    except Setup as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 2
    failed = 0
    for n, desc, fn in CHECKS:
        try:
            problems = fn(c)
        except Setup as e:
            problems = [f"montagem: {e}"]
        except Exception as e:  # uma checagem quebrada não esconde as outras
            problems = [f"{type(e).__name__}: {e}"]
        if problems:
            failed += 1
            print(f"FALHA {n:>2}. {desc}")
            for pr in problems:
                print(f"        - {pr}")
        else:
            print(f"OK    {n:>2}. {desc}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checagens OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
