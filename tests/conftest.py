"""Infra compartilhada da suíte. Nenhum teste lê o diretório de dados reais: só as fixtures daqui."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
SCHEMAS = ROOT / "schemas"

if str(ROOT) not in sys.path:  # permite rodar sem `pip install -e .`
    sys.path.insert(0, str(ROOT))

CATALOG_PATH = FIX / "catalog_test.json"
SYSTEM_PATHS = {
    "cpu8": FIX / "sys_cpu8.json",
    "rtx3060": FIX / "sys_rtx3060.json",
    "ddr5_64": FIX / "sys_ddr5_64.json",
    "apple24": FIX / "sys_apple24.json",
    "rx7800xt": FIX / "sys_rx7800xt.json",
}
TUNING_PATH = FIX / "tuning_test.json"  # F8: tuning de teste, nunca o real
ALL_TASKS = ("chat", "code", "embeddings")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


TUNING = load(TUNING_PATH)  # eta_gpu 0.75, eta_cpu 0.60, max_bpw_offload 8.5


@pytest.fixture
def catalog() -> dict:
    return load(CATALOG_PATH)


@pytest.fixture
def systems() -> dict:
    return {k: load(p) for k, p in SYSTEM_PATHS.items()}


@pytest.fixture
def tuning() -> dict:
    return dict(TUNING)


def schema_validator(name: str):
    """Validador jsonschema (importorskip: a suíte passa sem jsonschema, DECISIONS Q1)."""
    jsonschema = pytest.importorskip("jsonschema")
    schemas = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in SCHEMAS.glob("*.schema.json")}
    schema = schemas[name]
    try:  # jsonschema >= 4.18
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        registry = Registry().with_resources(
            [(s["$id"], Resource.from_contents(s, default_specification=DRAFT202012)) for s in schemas.values()]
        )
        return jsonschema.Draft202012Validator(schema, registry=registry)
    except ImportError:  # jsonschema antigo
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = {s["$id"]: s for s in schemas.values()}
            resolver = jsonschema.RefResolver.from_schema(schema, store=store)
            return jsonschema.Draft202012Validator(schema, resolver=resolver)


def assert_valid(name: str, doc: dict) -> None:
    errors = sorted(schema_validator(name).iter_errors(doc), key=lambda e: list(e.path))
    assert not errors, "; ".join(f"{list(e.path)}: {e.message[:120]}" for e in errors[:5])


def deep(doc: dict) -> dict:
    return copy.deepcopy(doc)
