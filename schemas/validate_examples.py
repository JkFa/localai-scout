"""Verificação da Fase 0: valida schemas/examples/* contra schemas/*.schema.json.

Uso: python schemas/validate_examples.py  (requer jsonschema; não é runtime do scout)
Sai com código 1 se um exemplo válido falhar ou um caso negativo passar.
"""
import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, RefResolver

HERE = Path(__file__).parent
SCHEMAS = {p.name: json.loads(p.read_text()) for p in HERE.glob("*.schema.json")}
STORE = {s["$id"]: s for s in SCHEMAS.values()}


def validator(name: str) -> Draft202012Validator:
    schema = SCHEMAS[name]
    Draft202012Validator.check_schema(schema)
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
POSITIVE = [("system.schema.json", "system_rtx3060.json", sysx),
            ("catalog.schema.json", "catalog_example.json", catx),
            ("feasibility.schema.json", "feasibility_example.json", feax),
            ("error.schema.json", "erro inline", {"error": {"code": "unauthorized", "message": "token ausente"}})]
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
    ("error.schema.json", "código fora do enum", {"error": {"code": "oops", "message": "x"}}),
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
