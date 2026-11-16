"""As fixtures do frontend (web/fixtures) ficam coerentes com os schemas.

O CI não validava web/: uma fixture regenerada podia sair inválida, ou uma nova
podia nascer no formato antigo sem ninguém perceber. As fixtures antigas que
provam compatibilidade do modo ARQUIVO ficam numa lista explícita; qualquer
outra precisa validar no schema atual.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import schema_validator

WEB_FIXTURES = Path(__file__).resolve().parent.parent / "web" / "fixtures"

# Formato anterior a inputs.calibration / inputs.decode, mantido de propósito:
# provam que arquivo antigo ainda abre no modo ARQUIVO (DECISIONS.md, 2026-11-16).
LEGACY = {
    "demo-feasibility.json",
    "demo-feasibility-falta.json",
    "demo-feasibility-nota.json",
    "demo-feasibility-rejeitados.json",
}
# Formato da 0.2.0 (inputs.decode sem start_position_tokens): válido no schema atual.
V020 = "demo-feasibility-calibracao-v020.json"

FEASIBILITY = sorted(p.name for p in WEB_FIXTURES.glob("demo-feasibility*.json"))


def _load(name: str) -> dict:
    return json.loads((WEB_FIXTURES / name).read_text(encoding="utf-8"))


def _errors(validator, doc: dict) -> list[str]:
    return [f"{list(e.path)}: {e.message[:160]}" for e in validator.iter_errors(doc)][:3]


def test_fixture_lists_match_the_directory():
    assert LEGACY <= set(FEASIBILITY), "fixture antiga listada mas ausente"
    assert V020 in FEASIBILITY
    assert (WEB_FIXTURES / "demo-system.json").exists()


@pytest.mark.parametrize("name", [n for n in FEASIBILITY if n not in LEGACY])
def test_current_feasibility_fixtures_validate(name):
    v = schema_validator("feasibility.schema.json")
    assert _errors(v, _load(name)) == []


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_legacy_fixtures_stay_in_the_old_format(name):
    """Se uma fixture antiga passar a ter inputs.calibration ou inputs.decode, ela
    deixou de provar compatibilidade e precisa sair da lista LEGACY."""
    inputs = _load(name)["inputs"]
    assert "calibration" not in inputs and "decode" not in inputs


def test_v020_fixture_has_no_start_position():
    doc = _load(V020)
    assert "start_position_tokens" not in doc["inputs"]["decode"]
    assert all("tokens_per_s_est_start" not in c for c in doc["candidates"])


def test_system_fixture_validates():
    v = schema_validator("system.schema.json")
    assert _errors(v, _load("demo-system.json")) == []
