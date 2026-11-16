"""Verificação da Fase 0: valida schemas/examples/* contra schemas/*.schema.json.

Uso: python schemas/validate_examples.py  (requer jsonschema; não é runtime do scout)
Sai com código 1 se um exemplo válido falhar ou um caso negativo passar.
"""
import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).parent
SCHEMAS = {p.name: json.loads(p.read_text()) for p in HERE.glob("*.schema.json")}
STORE = {s["$id"]: s for s in SCHEMAS.values()}


def validator(name: str) -> Draft202012Validator:
    schema = SCHEMAS[name]
    Draft202012Validator.check_schema(schema)
    try:  # jsonschema >= 4.18
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        registry = Registry().with_resources(
            [(i, Resource.from_contents(s, default_specification=DRAFT202012)) for i, s in STORE.items()]
        )
        return Draft202012Validator(schema, registry=registry)
    except ImportError:  # jsonschema antigo
        from jsonschema import RefResolver

        return Draft202012Validator(schema, resolver=RefResolver.from_schema(schema, store=STORE))


def load(name: str) -> dict:
    return json.loads((HERE / "examples" / name).read_text())


def set_path(doc: dict, path: list, value) -> dict:
    doc = copy.deepcopy(doc)
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return doc


sysx, catx, feax = load("system_rtx3060.json"), load("catalog_example.json"), load("feasibility_example.json")
calx = load("calibration_example.json")


def without_start(doc: dict) -> dict:
    """Forma de um feasibility.json de scout <= 0.2.0 (decode_range_spec.md §3)."""
    doc = copy.deepcopy(doc)
    doc["inputs"]["decode"].pop("start_position_tokens")
    for c in doc["candidates"]:
        c.pop("tokens_per_s_est_start")
        c["provenance"].pop("tokens_per_s_est_start")
    return doc


feax_v020 = without_start(feax)
POSITIVE = [("system.schema.json", "system_rtx3060.json", sysx),
            ("catalog.schema.json", "catalog_example.json", catx),
            ("feasibility.schema.json", "feasibility_example.json", feax),
            ("error.schema.json", "erro inline", {"error": {"code": "unauthorized", "message": "token ausente"}}),
            ("feasibility.schema.json", "arquivo de scout 0.2.0, sem tokens/s do início", feax_v020),
            ("calibration.schema.json", "calibration_example.json", calx),
            ("calibration.schema.json", "desktop: energia não aplicável, kernel null",
             set_path(set_path(calx, ["pools", "cpu", 0, "power"], {"on_ac": None, "profile": None, "not_applicable": ["on_ac", "profile"]}),
                      ["pools", "cpu", 0, "measured_with", "os.kernel"], None))]
RAM = ["memory", "ram_total_bytes", "value"]
NEGATIVE = [
    ("system.schema.json", "ram_total como float", set_path(sysx, RAM, 17179869184.5)),
    ("system.schema.json", "ram_total como '16 GB'", set_path(sysx, RAM, "16 GB")),
    ("system.schema.json", "null com confidence high", set_path(sysx, ["memory", "ram_total_bytes"], {"value": None, "source": "x", "confidence": "high"})),
    ("system.schema.json", "campo extra hostname", set_path(sysx, ["hostname"], "maquina")),
    ("catalog.schema.json", "complete:true com licença null", set_path(catx, ["models", 0, "license"], None)),
    ("catalog.schema.json", "complete:false sem incomplete_fields", set_path(catx, ["models", 4, "incomplete_fields"], [])),
    ("catalog.schema.json", "source_url null", set_path(catx, ["models", 0, "source_url"], None)),
    ("feasibility.schema.json", "candidato inviavel", set_path(feax, ["candidates", 0, "tier"], "inviavel")),
    ("feasibility.schema.json", "kv_bytes float", set_path(feax, ["candidates", 0, "kv_bytes"], 1.5e9 + 0.5)),
    ("feasibility.schema.json", "system embutido inválido", set_path(feax, ["inputs", "system", "kind"], "catalog")),
    ("feasibility.schema.json", "sem inputs.calibration",
     set_path(feax, ["inputs"], {k: v for k, v in feax["inputs"].items() if k != "calibration"})),
    ("feasibility.schema.json", "inputs.power com calibração ausente",
     set_path(feax, ["inputs", "power"], {"on_ac": True, "profile": None, "not_applicable": []})),
    ("feasibility.schema.json", "eco com caminho do arquivo", set_path(feax, ["inputs", "calibration", "path"], "/home/x")),
    ("feasibility.schema.json", "tuning.calibration antigo",
     set_path(feax, ["inputs", "tuning", "calibration"], {"cpu": calx["pools"]["cpu"][0]})),
    ("feasibility.schema.json", "start_position_tokens sem o campo num candidato",
     set_path(feax, ["candidates", 1], {k: v for k, v in feax["candidates"][1].items() if k != "tokens_per_s_est_start"})),
    ("feasibility.schema.json", "tokens_per_s_est_start sem start_position_tokens",
     set_path(feax_v020, ["candidates", 0, "tokens_per_s_est_start"], 29.16)),
    ("feasibility.schema.json", "tokens_per_s_est null e início numérico",
     set_path(feax, ["candidates", 0, "tokens_per_s_est"], None)),
    ("feasibility.schema.json", "início sem proveniência",
     set_path(feax, ["candidates", 0, "provenance"],
              {k: v for k, v in feax["candidates"][0]["provenance"].items() if k != "tokens_per_s_est_start"})),
    ("error.schema.json", "código fora do enum", {"error": {"code": "oops", "message": "x"}}),
    ("calibration.schema.json", "calibração sem identidade", {k: v for k, v in calx.items() if k != "machine"}),
    ("calibration.schema.json", "sem calibration_schema_version",
     {k: v for k, v in calx.items() if k != "calibration_schema_version"}),
    ("calibration.schema.json", "identidade com hostname", set_path(calx, ["machine", "hostname"], "maquina")),
    ("calibration.schema.json", "camada 1 null (cpu.model)", set_path(calx, ["machine", "cpu.model"], None)),
    ("calibration.schema.json", "não aplicável com valor",
     set_path(calx, ["pools", "cpu", 0, "power", "not_applicable"], ["on_ac"])),
    ("calibration.schema.json", "power sem not_applicable",
     set_path(calx, ["pools", "cpu", 0, "power"], {"on_ac": True, "profile": "performance"})),
    ("calibration.schema.json", "pool gpu sem gpu.name",
     set_path(set_path(calx, ["pools", "gpu"], calx["pools"]["cpu"]), ["machine", "gpu.name"], None)),
    ("calibration.schema.json", "kernel no nível da máquina", set_path(calx, ["machine", "os.kernel"], "6.99.0-1")),
    ("calibration.schema.json", "entrada sem measured_with",
     {**calx, "pools": {"cpu": [{k: v for k, v in calx["pools"]["cpu"][0].items() if k != "measured_with"}]}}),
    ("calibration.schema.json", "pool como objeto, não lista", set_path(calx, ["pools", "cpu"], calx["pools"]["cpu"][0])),
    ("calibration.schema.json", "ferramentas misturadas", set_path(calx, ["pools", "cpu", 0, "tool"], "llama-bench+ollama")),
    ("calibration.schema.json", "ram_total como float", set_path(calx, ["machine", "memory.ram_total_bytes"], 7.5e9 + 0.5)),
]

failed = 0
for schema, label, doc in POSITIVE:
    errors = list(validator(schema).iter_errors(doc))
    failed += bool(errors)
    print(("OK  " if not errors else "FAIL"), f"{label} vs {schema}", *(f"\n    {e.json_path}: {e.message}" for e in errors[:5]))
for schema, label, doc in NEGATIVE:
    rejected = not validator(schema).is_valid(doc)
    failed += not rejected
    print(("OK  " if rejected else "FAIL"), f"rejeita {label}")
sys.exit(1 if failed else 0)
