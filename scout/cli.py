"""CLI do LocalAI Scout. Fase 1: só `scout plan`.

Saída determinística: sem timestamp, JSON com sort_keys. Todo número do
markdown sai também no --json (bytes no JSON, GiB via fmt_gb e GB/s via fmt_gbps no markdown).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from scout import __version__
from scout.estimate import fmt_gb, fmt_gbps
from scout import calstore
from scout.calibration import CalibrationInput, chosen_tools, describe_power, machine_identity, measured_with
from scout.solver import TASKS, Intent, solve

TUNING_PATH = Path(__file__).resolve().parent.parent / "data" / "tuning.json"


class InputError(Exception):
    pass


def _read_json(path: Path, kind: str | None) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise InputError(f"não foi possível ler {path}: {e.strerror or e}") from e
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InputError(f"{path} não é JSON válido: {e}") from e
    if kind is not None:
        if not isinstance(doc, dict) or doc.get("kind") != kind:
            raise InputError(f"{path}: esperado kind={kind!r}")
        if doc.get("schema_version") != 1:
            raise InputError(f"{path}: schema_version {doc.get('schema_version')!r} não suportado")
    return doc, raw


def _parse_tasks(arg: str | None, catalog: dict) -> tuple[str, ...]:
    if arg is None:
        present = {t for m in catalog.get("models", []) for t in (m.get("tasks") or [])}
        tasks = tuple(t for t in TASKS if t in present)
        return tasks or TASKS
    wanted = [t.strip() for t in arg.split(",") if t.strip()]
    unknown = sorted(t for t in wanted if t not in TASKS)
    if unknown or not wanted:
        raise InputError(f"--task inválido: {', '.join(unknown) or '(vazio)'}; use {','.join(TASKS)}")
    return tuple(t for t in TASKS if t in wanted)


def _fmt_tps(x: float | None, prov: dict | None = None) -> str:
    """tokens/s com ~ salvo quando vem de calibração medida (proveniência calculated)."""
    if x is None:
        return "?"
    return ("" if (prov or {}).get("tokens_per_s_est") == "calculated" else "~") + f"{x:.2f}"


def _est(text: str, prov: dict, field: str) -> str:
    """Prefixo ~ quando a proveniência do número é estimated (m7_fixes.md F6)."""
    return ("~" if prov.get(field) == "estimated" else "") + text


def _warn(text: str, c: dict) -> str:
    """⚠ ao lado do número quando o candidato tem confiança low."""
    return text + (" ⚠" if c["confidence"] == "low" else "")


def render_markdown(f: dict) -> str:
    intent = f["inputs"]["intent"]
    lines: list[str] = ["# LocalAI Scout — plano", ""]
    if "fixture" in f:
        lines += ["> DADOS SINTÉTICOS: " + f["fixture"]["description"], ""]
    lines += [
        "Legenda: `~` = estimado (constante empírica ou dado de baixa confiança); "
        "`⚠` = candidato com confiança baixa.",
        "",
        f"Tarefas: {', '.join(intent['tasks'])} · contexto: {intent['ctx']} tokens · "
        f"KV: {intent['kv_quant']} · batch: {intent['batch']} · "
        f"uso comercial: {'sim' if intent['commercial_use'] else 'não'}",
        "",
        "## Orçamento de memória",
        "",
        "| pool | total | reservado | utilizável | banda |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in f["pools"]:
        pv = p["provenance"]
        bw = ("desconhecida" if p["bandwidth_bytes_per_s"] is None
              else _est(fmt_gbps(p["bandwidth_bytes_per_s"]), pv, "bandwidth_bytes_per_s"))
        lines.append(f"| {p['id']} | {_est(fmt_gb(p['total_bytes']), pv, 'total_bytes')} "
                     f"| {_est(fmt_gb(p['reserved_bytes']), pv, 'reserved_bytes')} "
                     f"| {_est(fmt_gb(p['usable_bytes']), pv, 'usable_bytes')} | {bw} |")

    lines += ["", "## Candidatos", ""]
    if not f["candidates"]:
        lines.append("Nenhum modelo cabe com a intenção pedida.")
    else:
        lines += [
            "| modelo | quant | placement | camadas na GPU | memória total | tier | tokens/s | confiança | nota |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        passed_over = {w["field"] for w in f["warnings"] if w["code"] == "quality_preference"}
        for i, c in enumerate(f["candidates"]):
            q = c["quant"]
            notes: list[str] = []
            if f"candidates[{i}]" in passed_over:
                notes.append("preterido por qualidade (ver avisos)")
            if c["primary_for"]:
                notes.append("principal: " + ", ".join(c["primary_for"]))
            notes += [n["message"] for n in c.get("notes", [])]
            if c.get("tokens_per_s_basis") == "calibration":
                notes.append("tokens/s pela calibração desta máquina (scout bench)")
            elif c.get("tokens_per_s_basis") == "mixed":
                notes.append("tokens/s: um trecho calibrado, outro por eta")
            if c["ttft_s_est"] is not None:
                notes.append(f"tempo até o primeiro token: {c['ttft_s_est']:.2f} s")
            if "ctx_effective" in c:
                notes.append(f"contexto efetivo: {c['ctx_effective']} tokens")
            if c["blocked"]:
                notes.append("bloqueado: " + c["blocked_reason"])
            lines.append(
                f"| [{c['model']['id']}]({c['model']['source_url']}) | {q['format']} {q['name']} "
                f"| {c['placement']} | {c['n_gpu_layers']}/{c['model']['n_layers']} "
                f"| {_warn(_est(fmt_gb(c['total_bytes']), c['provenance'], 'total_bytes'), c)} | {c['tier']} "
                f"| {'—' if 'decode_not_applicable' in c['confidence_reasons'] else _warn(_fmt_tps(c['tokens_per_s_est'], c['provenance']), c)} "
                f"| {c['confidence']} | {'; '.join(notes)} |")
        lines += ["", "Memória por pool:", ""]
        pool_prov = {p["id"]: p["provenance"] for p in f["pools"]}
        for c in f["candidates"]:
            cp = c["provenance"]
            parts = [f"{u['pool']} {_est(fmt_gb(u['total_bytes']), cp, 'total_bytes')} de "
                     f"{_est(fmt_gb(u['budget_bytes']), pool_prov.get(u['pool'], {}), 'usable_bytes')} ({u['tier']})"
                     for u in c["by_pool"]]
            lines.append(f"- {c['model']['id']}: pesos {_est(fmt_gb(c['weights_bytes']), cp, 'weights_bytes')}, "
                         f"KV {fmt_gb(c['kv_bytes'])}, buffer {fmt_gb(c['buffer_bytes'])}; " + "; ".join(parts))

    if f["rejected"]:
        lines += ["", "## Rejeitados", ""]
        for r in f["rejected"]:
            extra = ("" if r["missing_bytes"] is None
                     else f" Faltam {_est(fmt_gb(r['missing_bytes']), r.get('provenance', {}), 'missing_bytes')}.")
            lines.append(f"- {r['model']['id']} — `{r['reason_code']}`: {r['message']}{extra}")

    lines += ["", *_calibration_lines(f["inputs"].get("calibration"), f["inputs"].get("power"))]

    ex = f["excluded"]
    lines += ["", f"Excluídos sem avaliação: {ex['incomplete']} incompletos, "
                  f"{ex['task_mismatch']} sem a tarefa pedida.", "", "## Avisos", ""]
    for w in f["warnings"]:
        lines.append(f"- [{w['severity']}] {w['message']}")
    return "\n".join(lines) + "\n"


_CAL_STATUS = {"disabled": "desligada", "absent": "nenhum arquivo", "used": "usada",
               "discarded": "descartada (tokens/s pelo eta)"}
_CAL_POOL_STATUS = {"applied": "aplicada", "discarded": "descartada", "pool_absent": "pool ausente neste sistema"}
_CAL_MATCH = {"same": "mesmo estado de energia", "other_state": "outro estado de energia (usada a medição mais lenta)",
              "unknown": "estado de energia atual não provado (usada a medição mais lenta compatível)"}


def _calibration_lines(cal: dict | None, power: dict | None) -> list[str]:
    """Linha de calibração do markdown, com o estado de energia por extenso."""
    if not cal:
        return []
    head = f"Calibração local: {_CAL_STATUS[cal['status']]} (origem: {cal['origin']})"
    if power is not None:
        head += f"; energia agora: {describe_power(power)}"
    out = [head + "."]
    for pid, pc in sorted(cal.get("pools", {}).items()):
        e, eng = pc["calibration"], pc["engine"]
        states = "; ".join(describe_power(x) for x in pc["available_power"])
        out.append(f"- pool {pid}: {_CAL_POOL_STATUS[pc['status']]}; medida {describe_power(e['power'])} "
                   f"em {e['date']} ({eng['tool']} {eng['stored_version'] or 'versão não lida'}), "
                   f"{fmt_gbps(e['bandwidth_bytes_per_s'])} efetivos; {_CAL_MATCH[pc['power_match']]}; "
                   f"estados gravados: {states}.")
    return out


def load_tuning(path: Path | None = None) -> dict:
    path = TUNING_PATH if path is None else path
    tuning, _ = _read_json(path, None)
    if not isinstance(tuning, dict) or "eta_gpu" not in tuning or "eta_cpu" not in tuning:
        raise InputError(f"{path}: faltam eta_gpu/eta_cpu")
    if "calibration" in tuning:
        new, _ = calstore.resolve_calibration_path(None, {}, sys.platform, calstore.find_home(os.environ))
        raise InputError(
            f"{path}: traz a chave `calibration`, formato antigo sem identidade de hardware, que não é mais "
            f"lido. A calibração agora vive em {new or 'calibration.json no diretório de estado do usuário'} "
            "(ou em --calibration / SCOUT_CALIBRATION). Remova a chave do tuning e rode `scout bench --run` de novo.")
    return tuning


def resolve_calibration(args: argparse.Namespace, env: dict[str, str] | None = None) -> tuple[Path | None, str]:
    """--no-calibration -> (None, disabled); senão flag > SCOUT_CALIBRATION > padrão do SO."""
    if getattr(args, "no_calibration", False):
        return None, "disabled"
    env = os.environ if env is None else env
    return calstore.resolve_calibration_path(getattr(args, "calibration", None), env, sys.platform, calstore.find_home(env))


def load_calibration(args: argparse.Namespace) -> CalibrationInput:
    """Resolve e lê o arquivo. Falha alto se malformado ou ausente com --calibration."""
    path, origin = resolve_calibration(args)
    try:
        return calstore.read_calibration(path, origin)
    except calstore.CalibrationError as e:
        raise InputError(str(e)) from e


def run_plan(system: dict, catalog: dict, catalog_raw: bytes, tuning: dict, intent: Intent,
             power: dict | None = None, calibration: CalibrationInput | None = None,
             engines: dict[str, str | None] | None = None) -> dict:
    """solve + identificação do catálogo pelos bytes exatos do arquivo lido (schema).

    Com um calibration.json lido, lê o estado de energia ATUAL (mesma regra do
    bench) para o solver comparar com o da calibração. Sem calibração não lê
    nada, e a saída não depende do estado da máquina.
    O arquivo já vem lido em `calibration` (plan: uma vez; serve: na partida).
    Só com doc lê a energia e a versão de cada motor das entradas (loopback).
    """
    if calibration is not None and calibration.doc is not None:
        from scout import bench

        if power is None:
            power = bench.read_power()
        if engines is None:
            engines = {t: bench.engine_version(t) for t in sorted(chosen_tools(calibration.doc, system, power))}
    result = solve(system, catalog, intent, tuning, power, calibration, engines)
    result["inputs"]["catalog"]["sha256"] = hashlib.sha256(catalog_raw).hexdigest()
    return result


def _cmd_plan(args: argparse.Namespace) -> int:
    system, _ = _read_json(Path(args.system), "system")
    catalog, catalog_raw = _read_json(Path(args.catalog), "catalog")
    tuning = load_tuning(Path(args.tuning))
    if args.ctx < 1:
        raise InputError("--ctx precisa ser >= 1")
    if args.prompt_tokens is not None and args.prompt_tokens < 1:
        raise InputError("--prompt-tokens precisa ser >= 1")
    intent = Intent(tasks=_parse_tasks(args.task, catalog), ctx=args.ctx, commercial_use=args.commercial,
                    prompt_tokens=args.prompt_tokens)
    calibration = load_calibration(args)
    try:
        result = run_plan(system, catalog, catalog_raw, tuning, intent, calibration=calibration)
    except ValueError as e:
        raise InputError(str(e)) from e
    if args.json:
        sys.stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    else:
        sys.stdout.write(render_markdown(result))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from scout import server

    calibration = load_calibration(args)
    try:
        return server.serve(args.port, Path(args.catalog), Path(args.tuning), calibration)
    except OSError as e:
        raise InputError(f"não foi possível abrir 127.0.0.1:{args.port}: {e.strerror or e}") from e


def _cmd_collect(args: argparse.Namespace) -> int:
    from scout.collect import collect

    doc = collect(args.dmidecode)
    for p in doc["probes"]:
        if p["probe"] == "dmidecode" and p["status"] == "permission_denied":
            sys.stderr.write(f"scout: dica: {p['message']}\n")
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
        return 0
    try:
        Path(args.out).write_text(text, encoding="utf-8")
    except OSError as e:
        raise InputError(f"não foi possível escrever {args.out}: {e.strerror or e}") from e
    sys.stderr.write(f"scout: perfil de hardware gravado em {args.out}\n")
    return 0


CATALOG_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "license_policy.json"
OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "data" / "task_overrides.json"
CATALOG_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "catalog.schema.json"


def _utc_today() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


def _cmd_catalog_add(args: argparse.Namespace) -> int:
    # Único comando que toca a rede, e só via scout.hf.
    from scout import hf
    from scout.catalog import apply_task_overrides, build_entry_with_reasons, upsert

    policy, _ = _read_json(POLICY_PATH, None)
    path = Path(args.catalog)
    if path.exists():
        catalog, _ = _read_json(path, "catalog")
    else:
        catalog = {"schema_version": 1, "kind": "catalog", "models": []}
    try:
        info = hf.get_model_info(args.repo, offline=args.offline)
        try:
            config = hf.get_config(args.repo, offline=args.offline)
        except hf.HFError as e:
            if e.reason not in ("gated", "not_found"):
                raise
            sys.stderr.write(f"scout: aviso: config.json de {args.repo} indisponível ({e.reason}); "
                             "campos de arquitetura ficam nulos\n")
            config = None
        gguf = hf.get_model_info(args.gguf_repo, blobs=True, offline=args.offline) if args.gguf_repo else None
    except hf.HFError as e:
        raise InputError(f"Hugging Face: {e.reason} em {e.url}") from e

    entry, reasons, add_warnings = build_entry_with_reasons(args.repo, info, config, gguf, policy, _utc_today())
    if OVERRIDES_PATH.exists():
        overrides, _ = _read_json(OVERRIDES_PATH, None)
        entry, override_msgs = apply_task_overrides(entry, overrides)
        add_warnings = add_warnings + override_msgs
    text = json.dumps(upsert(catalog, entry), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise InputError(f"não foi possível escrever {path}: {e.strerror or e}") from e
    state = "completo" if entry["complete"] else "INCOMPLETO (nunca será recomendado)"
    sys.stderr.write(f"scout: {entry['id']} gravado em {path}: {state}, {len(entry['quants'])} quants\n")
    for field in sorted(reasons):
        sys.stderr.write(f"  {field}: {reasons[field]}\n")
    for w in add_warnings:
        sys.stderr.write(f"  aviso: {w}\n")
    if entry["license"] is not None and entry["commercial_ok"] is None:
        sys.stderr.write(f"  aviso: licença {entry['license']!r} fora de {POLICY_PATH.name}; commercial_ok=null\n")
    return 0


def _cmd_catalog_remove(args: argparse.Namespace) -> int:
    from scout.catalog import remove

    path = Path(args.catalog)
    catalog, _ = _read_json(path, "catalog")
    try:
        catalog = remove(catalog, args.id)
    except KeyError:
        raise InputError(f"{args.id} não está em {path}") from None
    text = json.dumps(catalog, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise InputError(f"não foi possível escrever {path}: {e.strerror or e}") from e
    sys.stderr.write(f"scout: {args.id} removido de {path}\n")
    return 0


def _cmd_catalog_check(args: argparse.Namespace) -> int:
    from scout.catalog import check_catalog

    catalog, _ = _read_json(Path(args.catalog), "catalog")
    today = args.today or _utc_today()
    try:
        errors, warnings = check_catalog(catalog, today)
    except ValueError as e:
        raise InputError(f"--today inválido: {e}") from e
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        sys.stderr.write("scout: jsonschema ausente; validação de schema pulada\n")
    else:
        schema = json.loads(CATALOG_SCHEMA_PATH.read_text(encoding="utf-8"))
        for e in sorted(Draft202012Validator(schema).iter_errors(catalog), key=lambda e: e.json_path):
            errors.append(f"schema: {e.json_path}: {e.message}")
    for w in warnings:
        sys.stdout.write(f"aviso: {w}\n")
    for e in errors:
        sys.stdout.write(f"erro: {e}\n")
    n = len(catalog.get("models", []))
    sys.stdout.write(f"{n} modelos, {len(errors)} erros, {len(warnings)} avisos\n")
    return 1 if errors else 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from scout import bench as B

    system, _ = _read_json(Path(args.system), "system")
    catalog, _ = _read_json(Path(args.catalog), "catalog")
    tuning_path = Path(args.tuning)
    tuning = load_tuning(tuning_path)
    models = args.model
    if not models:
        raise InputError("informe ao menos um --model <id>")
    pair = lambda xs, i: xs[i] if i < len(xs) else None  # noqa: E731
    plans = []
    for i, mid in enumerate(models):
        try:
            target = B.build_target(system, catalog, mid, args.ctx, tuning)
        except B.BenchError as e:
            raise InputError(str(e)) from e
        gguf, om = pair(args.gguf, i), pair(args.ollama_model, i)
        tool, missing = B.choose_tool(gguf, om)
        plans.append((target, gguf, om, tool, missing))
    if not args.run:
        for target, gguf, om, tool, missing in plans:
            sys.stdout.write(f"{target.model_id} ({target.quant}, ngl {target.n_gpu_layers}, ctx {target.ctx}, "
                             f"threads {target.threads}, pool {target.pool})\n")
            if tool == "llama-bench":
                sys.stdout.write("  " + " ".join(B.llama_bench_cmd(gguf, target.n_gpu_layers, target.threads)) + "\n")
            elif tool == "ollama":
                sys.stdout.write(f"  API local do Ollama, modelo {om} (quant conferida pelo tamanho do blob)\n")
            for m in missing:
                sys.stdout.write(f"  falta: {m}\n")
        sys.stdout.write("Nada foi medido. Use --run para medir e gravar a calibração.\n")
        return 0
    cal_path, cal_origin = resolve_calibration(args)
    if cal_path is None:
        raise InputError("sem caminho para a calibração (sem home); use --calibration ou SCOUT_CALIBRATION")
    try:
        old = calstore.read_calibration(cal_path, "default").doc  # ausente não é erro para quem grava
    except calstore.CalibrationError as e:
        raise InputError(str(e)) from e
    identity = machine_identity(system)
    pools = {t.pool for t, *_ in plans}
    bad_pool = calstore.mixed_tools([(t.pool, tool) for t, _, _, tool, _ in plans])
    if bad_pool:
        raise InputError(f"mixed_tools: o pool {bad_pool} misturaria ferramentas (ollama e llama-bench) numa "
                         "entrada; meça cada ferramenta numa rodada separada.")
    live = B.live_system()
    refusal = calstore.identity_refusal(identity, machine_identity(live) if live is not None else None, pools)
    if refusal:
        raise InputError(f"{refusal[0]}: {refusal[1]}")
    power = B.read_power()
    if power["on_ac"] is False:
        sys.stderr.write("scout: AVISO: máquina NA BATERIA. A medição reflete a bateria e a calibração "
                         "só vale nesse estado de energia.\n")
    results, failed = [], False
    for target, gguf, om, tool, missing in plans:
        if tool is None:
            sys.stderr.write(f"scout: {target.model_id}: não medido ({'; '.join(missing)})\n")
            failed = True
            continue
        try:
            r = B.measure_model(target, tool, gguf, om, power)
        except B.BenchError as e:
            sys.stderr.write(f"scout: {target.model_id}: não medido ({e})\n")
            failed = True
            continue
        results.append(r)
        sys.stdout.write(f"{r['model_id']} [{tool}]: decode {r['decode_tps']:.2f} tok/s, banda efetiva "
                         f"{r['effective_bandwidth_bytes_per_s'] / 1e9:.1f} GB/s, "
                         f"{'estável' if r['stable'] else 'NÃO ESTABILIZOU em 6 rodadas (confiança rebaixada)'}\n")
        sys.stderr.write(conditions_line(r) + "\n")
        if r["prefill_stable"] is False:
            sys.stdout.write("  prefill NÃO estabilizou: não entra na calibração\n")
    doc = {"schema_version": 1, "kind": "measured", "scout_version": __version__, "power": power,
           "on_battery_warning": power["on_ac"] is False, "models": results}
    Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(f"medições gravadas em {args.out}\n")
    if failed or not results:
        sys.stderr.write("scout: medição incompleta: calibração NÃO gravada\n")
        return 1
    cal = B.build_calibration(results, results[0]["measured_at"][:10])
    new_doc, notes = calstore.merge_calibration(old, identity, cal, __version__, measured_with(system))
    for n in notes:
        sys.stderr.write(f"scout: calibração: {n}\n")
    try:
        calstore.write_calibration_file(cal_path, new_doc)
    except calstore.CalibrationError as e:
        raise InputError(str(e)) from e
    for pool, c in sorted(cal.items()):
        sys.stdout.write(f"calibração {pool}: banda {c['bandwidth_bytes_per_s'] / 1e9:.1f} GB/s, overhead "
                         f"{c['overhead_s_per_token']:.4f} s/token, forma {c['form']}, "
                         f"{'estável' if c['stable'] else 'não estável'}, {describe_power(c['power'])}\n")
    sys.stdout.write(f"calibração gravada em {cal_path}\n")
    return 0


def _add_calibration_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--calibration", default=None, metavar="PATH",
                   help="calibration.json do scout bench (erro se não existir); senão SCOUT_CALIBRATION ou o padrão do SO")
    g.add_argument("--no-calibration", action="store_true",
                   help="não lê calibração nem consulta motor nenhum (tokens/s pelo eta)")


def conditions_line(r: dict) -> str:
    """Linha de diagnóstico (nunca afeta a calibração): governor/EPP, freq e temp antes->depois, swap usado."""
    b, a = r["conditions"]["before"], r["conditions"]["after"]
    f = lambda x, d=1: "?" if x is None else f"{x / d:.0f}"  # noqa: E731
    used = [x for x in r["rounds"] if x["used_decode"] or x["used_prefill"]]
    sw = [x.get("swap_in_pages") for x in used], [x.get("swap_out_pages") for x in used]
    tot = ["?" if any(v is None for v in col) else str(sum(col)) for col in sw]
    return (f"  condições: governor {b['scaling_governor'] or '?'}/EPP {b['energy_performance_preference'] or '?'}; "
            f"freq média {f(b['freq_khz_mean'], 1000)}->{f(a['freq_khz_mean'], 1000)} MHz; "
            f"temp máx {f(b['temp_max_millic'], 1000)}->{f(a['temp_max_millic'], 1000)} C; "
            f"swap in/out nas rodadas usadas {tot[0]}/{tot[1]} páginas")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scout", description="Recomenda modelos de IA locais que cabem nesta máquina.")
    parser.add_argument("--version", action="version", version=f"scout {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="calcula o que roda nesta máquina")
    plan.add_argument("--system", default="system.json", help="system.json gerado por scout collect")
    plan.add_argument("--catalog", default="data/catalog.json", help="catálogo de modelos")
    plan.add_argument("--task", default=None, help="tarefas separadas por vírgula (padrão: todas do catálogo)")
    plan.add_argument("--ctx", type=int, default=8192, help="contexto em tokens")
    plan.add_argument("--commercial", action="store_true", help="uso comercial: bloqueia licenças não comerciais")
    plan.add_argument("--offline", action="store_true", help="sem efeito: plan nunca usa a rede")
    plan.add_argument("--json", action="store_true", help="imprime feasibility.json")
    plan.add_argument("--tuning", default=str(TUNING_PATH), help="constantes empíricas (eta, max_bpw_offload)")
    plan.add_argument("--prompt-tokens", type=int, default=None,
                      help="tamanho do prompt em tokens; com prefill calibrado, estima o tempo até o 1º token")
    _add_calibration_flags(plan)
    plan.set_defaults(func=_cmd_plan)
    col = sub.add_parser("collect", help="lê o hardware desta máquina e grava system.json")
    col.add_argument("--out", default="system.json", help="arquivo de saída ('-' imprime no stdout)")
    col.add_argument("--dmidecode", default=None, metavar="ARQUIVO",
                     help="saída de `sudo dmidecode -t memory` gerada por você (tipo, MT/s e canais da RAM)")
    col.set_defaults(func=_cmd_collect)
    ben = sub.add_parser("bench", help="mede velocidade real e calibra a banda efetiva (sem --run só mostra o plano)")
    ben.add_argument("--model", action="append", default=[], help="id do catálogo (repetível)")
    ben.add_argument("--ollama-model", action="append", default=[], help="nome no Ollama, pareado com --model na ordem")
    ben.add_argument("--gguf", action="append", default=[], help="arquivo GGUF p/ llama-bench, pareado com --model na ordem")
    ben.add_argument("--run", action="store_true", help="executa a medição e grava a calibração")
    ben.add_argument("--ctx", type=int, default=8192, help="contexto em tokens")
    ben.add_argument("--system", default="system.json")
    ben.add_argument("--catalog", default=str(CATALOG_DEFAULT))
    ben.add_argument("--tuning", default=str(TUNING_PATH))
    ben.add_argument("--out", default="measured.json")
    ben.add_argument("--calibration", default=None, metavar="PATH",
                     help="arquivo onde gravar a calibração (padrão: SCOUT_CALIBRATION ou diretório de estado do usuário)")
    ben.set_defaults(func=_cmd_bench)

    srv = sub.add_parser("serve", help="backend local para o frontend (só 127.0.0.1, com token)")
    srv.add_argument("--port", type=int, default=8765, help="porta em 127.0.0.1 (padrão 8765; falha se ocupada)")
    srv.add_argument("--catalog", default=str(CATALOG_DEFAULT), help="catálogo de modelos")
    srv.add_argument("--tuning", default=str(TUNING_PATH), help="constantes empíricas (eta, max_bpw_offload)")
    _add_calibration_flags(srv)
    srv.set_defaults(func=_cmd_serve)
    cat = sub.add_parser("catalog", help="mantém o catálogo a partir da API do Hugging Face")
    csub = cat.add_subparsers(dest="catalog_command", required=True)
    add = csub.add_parser("add", help="adiciona ou atualiza um modelo pelo repo do Hugging Face")
    add.add_argument("repo", help="repo HF <org>/<nome>")
    add.add_argument("--gguf-repo", default=None, help="repo com os arquivos GGUF")
    add.add_argument("--catalog", default="data/catalog.json")
    add.add_argument("--offline", action="store_true", help="só usa o cache local, sem rede")
    add.set_defaults(func=_cmd_catalog_add)
    rm = csub.add_parser("remove", help="remove um modelo do catálogo pelo id")
    rm.add_argument("id", help="id do modelo (<org>/<nome>)")
    rm.add_argument("--catalog", default="data/catalog.json")
    rm.set_defaults(func=_cmd_catalog_remove)
    chk = csub.add_parser("check", help="valida o catálogo e avisa dados velhos")
    chk.add_argument("--catalog", default="data/catalog.json")
    chk.add_argument("--today", default=None, help="data de referência YYYY-MM-DD (padrão: hoje, UTC)")
    chk.set_defaults(func=_cmd_catalog_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except InputError as e:
        sys.stderr.write(f"scout: erro: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
