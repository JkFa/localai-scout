"""R8: nenhum arquivo de tests/ referencia o diretório data/ (catálogo, tuning e tabelas reais)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r"(?<![A-Za-z0-9_])da" + r"ta/")


def test_no_test_file_references_data_dir():
    offenders = [
        f"{p.relative_to(HERE)}:{n}"
        for p in sorted(HERE.rglob("*"))
        if p.is_file() and p != Path(__file__).resolve() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if PATTERN.search(line)
    ]
    assert not offenders, offenders
