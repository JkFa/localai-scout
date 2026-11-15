"""CLI do LocalAI Scout. Fase 1: só `scout plan`.

Saída determinística: sem timestamp, JSON com sort_keys. Todo número do
markdown sai também no --json (bytes no JSON, GiB via fmt_gb e GB/s via fmt_gbps no markdown).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from scout import __version__
from scout.estimate import fmt_gb, fmt_gbps
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


def _fmt_tps(x: float | None) -> str:
    return "?" if x is None else f"{x:.2f}"


def render_markdown(f: dict) -> str:
    intent = f["inputs"]["intent"]
    lines: list[str] = ["# LocalAI Scout — plano", ""]
    if "fixture" in f:
        lines += ["> DADOS SINTÉTICOS: " + f["fixture"]["description"], ""]
    lines += [
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
        bw = "desconhecida" if p["bandwidth_bytes_per_s"] is None else fmt_gbps(p["bandwidth_bytes_per_s"])
        lines.append(f"| {p['id']} | {fmt_gb(p['total_bytes'])} | {fmt_gb(p['reserved_bytes'])} | "
                     f"{fmt_gb(p['usable_bytes'])} | {bw} |")

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
            if c["blocked"]:
                notes.append("bloqueado: " + c["blocked_reason"])
            lines.append(
                f"| [{c['model']['id']}]({c['model']['source_url']}) | {q['format']} {q['name']} "
                f"| {c['placement']} | {c['n_gpu_layers']}/{c['model']['n_layers']} "
                f"| {fmt_gb(c['total_bytes'])} | {c['tier']} | {_fmt_tps(c['tokens_per_s_est'])} "
                f"| {c['confidence']} | {'; '.join(notes)} |")
        lines += ["", "Memória por pool:", ""]
        for c in f["candidates"]:
            parts = [f"{u['pool']} {fmt_gb(u['total_bytes'])} de {fmt_gb(u['budget_bytes'])} ({u['tier']})"
                     for u in c["by_pool"]]
            lines.append(f"- {c['model']['id']}: pesos {fmt_gb(c['weights_bytes'])}, "
                         f"KV {fmt_gb(c['kv_bytes'])}, buffer {fmt_gb(c['buffer_bytes'])}; " + "; ".join(parts))

    if f["rejected"]:
        lines += ["", "## Rejeitados", ""]
        for r in f["rejected"]:
            extra = "" if r["missing_bytes"] is None else f" Faltam {fmt_gb(r['missing_bytes'])}."
            lines.append(f"- {r['model']['id']} — `{r['reason_code']}`: {r['message']}{extra}")

    ex = f["excluded"]
    lines += ["", f"Excluídos sem avaliação: {ex['incomplete']} incompletos, "
                  f"{ex['task_mismatch']} sem a tarefa pedida.", "", "## Avisos", ""]
    for w in f["warnings"]:
        lines.append(f"- [{w['severity']}] {w['message']}")
    return "\n".join(lines) + "\n"


def _cmd_plan(args: argparse.Namespace) -> int:
    system, _ = _read_json(Path(args.system), "system")
    catalog, catalog_raw = _read_json(Path(args.catalog), "catalog")
    tuning, _ = _read_json(TUNING_PATH, None)
    if not isinstance(tuning, dict) or "eta_gpu" not in tuning or "eta_cpu" not in tuning:
        raise InputError(f"{TUNING_PATH}: faltam eta_gpu/eta_cpu")
    if args.ctx < 1:
        raise InputError("--ctx precisa ser >= 1")
    intent = Intent(tasks=_parse_tasks(args.task, catalog), ctx=args.ctx, commercial_use=args.commercial)
    try:
        result = solve(system, catalog, intent, tuning)
    except ValueError as e:
        raise InputError(str(e)) from e
    # Identifica o catálogo pelos bytes exatos do arquivo lido (schema).
    result["inputs"]["catalog"]["sha256"] = hashlib.sha256(catalog_raw).hexdigest()
    if args.json:
        sys.stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    else:
        sys.stdout.write(render_markdown(result))
    return 0


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
    plan.set_defaults(func=_cmd_plan)
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
