"""scout plan via scout.cli.main(argv) (capsys, sem subprocess)."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from conftest import ALL_TASKS, CATALOG_PATH, SYSTEM_PATHS, assert_valid, load
from scout.cli import main
from scout.estimate import fmt_gb

PROFILES = sorted(SYSTEM_PATHS)


def plan(capsys, profile, *extra, as_json=False):
    argv = ["plan", "--system", str(SYSTEM_PATHS[profile]), "--catalog", str(CATALOG_PATH), *extra]
    if as_json:
        argv.append("--json")
    try:
        rc = main(argv)
    except SystemExit as e:  # argparse / sys.exit
        rc = e.code or 0
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def plan_json(capsys, profile, *extra):
    rc, out, _ = plan(capsys, profile, *extra, as_json=True)
    assert rc == 0
    return json.loads(out), out


# ---------------------------------------------------------------- saída básica
@pytest.mark.parametrize("profile", PROFILES)
def test_plan_markdown_runs(profile, capsys):
    rc, out, err = plan(capsys, profile, "--ctx", "8192")
    assert rc == 0 and out.strip()
    assert "test/" in out          # cita modelos do catálogo de teste
    assert "Traceback" not in err


@pytest.mark.parametrize("profile", PROFILES)
def test_plan_json_valid_and_canonical(profile, capsys):
    doc, raw = plan_json(capsys, profile, "--ctx", "8192")
    assert_valid("feasibility.schema.json", doc)
    # sort_keys=True, indent=2 (ensure_ascii livre)
    canon = {json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=a) for a in (True, False)}
    assert raw.rstrip("\n") in canon


def test_plan_json_reflects_arguments(capsys):
    doc, _ = plan_json(capsys, "rtx3060", "--ctx", "4096", "--task", "chat,code", "--commercial")
    assert doc["inputs"]["intent"]["ctx"] == 4096
    assert doc["inputs"]["intent"]["tasks"] == ["chat", "code"]
    assert doc["inputs"]["intent"]["commercial_use"] is True
    doc2, _ = plan_json(capsys, "rtx3060")
    # --task omitido = todas as tarefas do catálogo; ctx default 8192; sem --commercial
    assert set(doc2["inputs"]["intent"]["tasks"]) == set(ALL_TASKS)
    assert doc2["inputs"]["intent"]["ctx"] == 8192 and doc2["inputs"]["intent"]["commercial_use"] is False


def test_plan_json_identifies_catalog_and_tuning(capsys):
    doc, _ = plan_json(capsys, "rtx3060")
    cat = doc["inputs"]["catalog"]
    assert cat["sha256"] == hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    assert cat["model_count"] == 4 and cat["oldest_verified_at"] == "2026-11-15"
    # data/tuning.json é lido pela CLI (não pelo solver): eta_gpu 0.75, eta_cpu 0.60
    assert doc["inputs"]["tuning"]["eta_gpu"] == 0.75 and doc["inputs"]["tuning"]["eta_cpu"] == 0.6


def test_plan_json_has_no_timestamp_of_its_own(capsys):
    doc, raw = plan_json(capsys, "rtx3060")
    for key in ("generated_at", "timestamp", "created_at", "date", "now"):
        assert key not in doc
    # o único instante presente é o collected_at do system de entrada (que vem da fixture)
    stamps = set(re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", raw))
    assert stamps <= {"2026-11-15T12:00:00"}
    _, md, _ = plan(capsys, "rtx3060")
    assert not re.search(r"\d{2}:\d{2}:\d{2}", md)


# ---------------------------------------------------------------- determinismo (o mais revelador)
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("extra", [(), ("--commercial",), ("--ctx", "16384"), ("--task", "code")])
def test_two_runs_are_byte_identical_markdown(profile, extra, capsys):
    """Asserção obrigatória: duas execuções seguidas dão saída idêntica (leitura de RAM livre
    ou aritmética fora das funções puras quebraria isto)."""
    a = plan(capsys, profile, *extra)
    b = plan(capsys, profile, *extra)
    assert a == b and a[1]


@pytest.mark.parametrize("profile", PROFILES)
def test_two_runs_are_byte_identical_json(profile, capsys):
    a = plan(capsys, profile, as_json=True)
    b = plan(capsys, profile, as_json=True)
    assert a == b and a[1]


# ---------------------------------------------------------------- todo número do markdown existe no JSON
NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")


def _strings(node, acc):
    if isinstance(node, str):
        acc.add(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            acc.add(k)
            _strings(v, acc)
    elif isinstance(node, list):
        for v in node:
            _strings(v, acc)


def _numbers(node, acc):
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, (int, float)):
        acc.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            _numbers(v, acc)
    elif isinstance(node, list):
        for v in node:
            _numbers(v, acc)


BW = re.compile(r"(\d+(?:\.\d+)?) GB/s")


def markdown_numbers_missing_from_json(md: str, doc: dict) -> list[str]:
    strings: set[str] = set()
    _strings(doc, strings)
    # ids, nomes de quant, urls, etc. contêm dígitos que não são números exibidos
    for s in sorted((x for x in strings if len(x) >= 3), key=len, reverse=True):
        md = md.replace(s, " ")
    md = re.sub(r"(?m)^\s*\d+[.)]\s", " ", md)          # numeração de listas
    nums: list = []
    _numbers(doc, nums)
    ints = [n for n in nums if isinstance(n, int) and n >= 0]
    gib = {float(fmt_gb(n).split()[0]) for n in ints}        # memória: GiB (n / 2**30)
    gbps = {round(n / 1e9, 2) for n in ints}                 # banda: GB/s decimal (bytes/s / 1e9)
    missing = []
    for tok in BW.findall(md):                               # banda só vale com rótulo GB/s
        if float(tok) not in gbps:
            missing.append(tok + " GB/s")
    md = BW.sub(" ", md)
    for tok in NUM.findall(md):
        norm = tok.replace(",", ".")
        decimals = len(norm.split(".")[1]) if "." in norm else 0
        value = float(norm)
        if value in gib and decimals == 2:                   # número exibido via fmt_gb (bytes no JSON)
            continue
        if any(round(x, decimals) == value for x in nums):
            continue
        missing.append(tok)
    return missing


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("extra", [(), ("--commercial",), ("--ctx", "2048")])
def test_every_markdown_number_exists_in_json(profile, extra, capsys):
    """Asserção obrigatória: todo número do markdown de plan existe também na saída --json."""
    _, md, _ = plan(capsys, profile, *extra)
    doc, _ = plan_json(capsys, profile, *extra)
    assert markdown_numbers_missing_from_json(md, doc) == []


def test_markdown_checker_detects_a_fabricated_number():
    """Auto-teste do verificador: um número que não está no JSON tem de ser apontado."""
    doc = {"a": 7_650_000_000, "bw": 360_000_000_000, "tokens": 35.29, "ctx": 8192, "id": "test/x-7b"}
    ok = "test/x-7b usa 7.12 GiB, 35.3 tok/s, ctx 8192, banda 360.00 GB/s"
    assert markdown_numbers_missing_from_json(ok, doc) == []
    assert markdown_numbers_missing_from_json(ok + " e 99.99 GiB", doc) == ["99.99"]
    # 7.65 (GB decimal) NÃO vale mais para memória; 335.28 (GiB da banda) não vale para GB/s
    assert markdown_numbers_missing_from_json("usa 7.65 GiB", doc) == ["7.65"]
    assert markdown_numbers_missing_from_json("banda 335.28 GB/s", doc) == ["335.28 GB/s"]


# ---------------------------------------------------------------- semântica na CLI
def test_cli_commercial_has_no_noncommercial_primary(capsys):
    cat = {m["id"]: m for m in load(CATALOG_PATH)["models"]}
    for profile in PROFILES:
        doc, _ = plan_json(capsys, profile, "--commercial")
        for c in doc["candidates"]:
            if cat[c["model"]["id"]]["commercial_ok"] is not True:
                assert c["blocked"] and c["primary_for"] == []
            if c["primary_for"]:
                assert not c["blocked"] and cat[c["model"]["id"]]["commercial_ok"] is True


def test_cli_profile_assertions(capsys):
    d1, _ = plan_json(capsys, "cpu8")
    assert any("embeddings" in c["model"]["tasks"] for c in d1["candidates"])
    assert not any("diffusion" in c["model"]["tasks"] for c in d1["candidates"])
    d2, _ = plan_json(capsys, "rtx3060")
    seven = next(c for c in d2["candidates"] if c["model"]["id"] == "test/dense-7b")
    assert seven["n_gpu_layers"] == 32 and seven["placement"] == "gpu_full"
    d3, _ = plan_json(capsys, "ddr5_64")
    assert any(c["model"]["id"] == "test/moe-30b-a3b" for c in d3["candidates"])


def test_cli_never_lists_incomplete_models(tmp_path, capsys):
    cat = load(CATALOG_PATH)
    for m in cat["models"]:
        if m["id"] == "test/embed-110m":
            m["complete"], m["ctx_max"], m["incomplete_fields"] = False, None, ["ctx_max"]
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(cat))
    argv = ["plan", "--system", str(SYSTEM_PATHS["rtx3060"]), "--catalog", str(p)]
    rc = main(argv)
    md = capsys.readouterr().out
    assert rc == 0 and "test/embed-110m" not in md
    main([*argv, "--json"])
    assert "test/embed-110m" not in capsys.readouterr().out


def test_cli_stdout_only_has_the_report(capsys):
    """--json imprime SÓ o JSON em stdout (pipeável: `scout plan --json > feasibility.json`)."""
    rc, out, _ = plan(capsys, "rtx3060", as_json=True)
    assert rc == 0
    json.loads(out)


def test_cli_missing_system_file_fails_cleanly(tmp_path, capsys):
    argv = ["plan", "--system", str(tmp_path / "nao_existe.json"), "--catalog", str(CATALOG_PATH)]
    try:
        rc = main(argv)
    except SystemExit as e:
        rc = e.code
    cap = capsys.readouterr()
    assert rc not in (0, None)
    assert "Traceback" not in cap.err


def test_cli_does_not_touch_the_network(capsys, monkeypatch):
    import socket
    import urllib.request

    def boom(*a, **k):
        raise AssertionError("acesso à rede durante plan")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    rc, out, _ = plan(capsys, "rtx3060")
    assert rc == 0 and out


@pytest.mark.parametrize("profile", PROFILES)
def test_markdown_units_gib_for_memory_gbps_for_bandwidth(profile, capsys):
    _, md, _ = plan(capsys, profile)
    assert "GiB" in md
    assert not re.search(r"\d GB(?!/s)", md), "memória deve sair em GiB, não GB"
    assert re.search(r"\d GB/s", md)       # banda em GB/s decimal


def test_cli_json_never_has_catalog_stale(capsys):
    for profile in PROFILES:
        doc, _ = plan_json(capsys, profile)
        assert "catalog_stale" not in {w["code"] for w in doc["warnings"]}
