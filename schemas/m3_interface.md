# Contrato de interface do M3 (Fase 3)

Definido pela sessão orquestradora para `coletor` (HTTP + cache) e `nucleo`
(mapeamento + comandos) trabalharem em paralelo. Fonte: `PROMPT.md` M3,
`schemas/catalog.schema.json`, `DECISIONS.md` (Q4, Q8, campo incompleto).

## `scout/hf.py` — dono: `coletor`

Única parte do projeto que toca a rede. Só `urllib.request`. Nunca importado
por `solver.py`, `estimate.py`, `collect.py` nem `parsers.py`.

    class HFError(Exception):           # .url, .status (int | None), .reason (str)
    BASE = "https://huggingface.co"

    get_model_info(repo: str, *, blobs: bool = False,
                   cache: Path | None = None, offline: bool = False) -> dict
        # GET {BASE}/api/models/{repo}[?blobs=true]
    get_config(repo: str, *, cache: Path | None = None,
               offline: bool = False) -> dict | None
        # GET {BASE}/{repo}/raw/main/config.json ; 404 -> None
    model_card_url(repo: str) -> str   # f"{BASE}/{repo}"

- Timeout 20 s, User-Agent genérico `localai-scout/<versão>` (sem e-mail nem
  dado do usuário). Sem token por padrão; `HF_TOKEN` só se estiver no ambiente.
- `repo` validado por regex `^[A-Za-z0-9][\w.-]*/[\w.-]+$` antes de montar URL.
- Cache em disco: default `~/.cache/localai-scout/hf/`, um JSON por URL (nome =
  sha256 da URL), com `fetched_at`. `offline=True` só lê cache; ausente →
  `HFError(reason="offline_cache_miss")`.
- Erro de rede, 401/403 (gated), 404, 429, JSON inválido → `HFError` com
  `reason` em {network, gated, not_found, rate_limited, bad_json,
  offline_cache_miss}. Nunca devolve dado parcial em silêncio.
- Testes sem rede: `urllib.request.urlopen` substituído por monkeypatch.

## `scout/catalog.py` — dono: `nucleo`

Funções puras de mapeamento (sem rede, testáveis com JSON em
`tests/fixtures/hf/`):

    map_config(config: dict | None) -> dict        # campos de arquitetura + motivos de null
    gguf_quants(blob_info: dict, params_total: int | None) -> list[dict]
                                                    # bpw = file_bytes*8/params_total
    license_fields(info: dict, policy: dict) -> dict  # license, license_source, commercial_ok
    build_entry(repo: str, info: dict, config: dict | None,
                gguf_info: dict | None, policy: dict, today: str) -> dict
                                                    # entrada válida contra catalog.schema.json

Comandos (em `scout/cli.py`):

    scout catalog add <repo> [--gguf-repo <repo>] [--catalog data/catalog.json] [--offline]
    scout catalog check [--catalog data/catalog.json] [--today YYYY-MM-DD]

- `add` faz upsert por `id`, grava JSON ordenado (`sort_keys`, indent 2) e
  ordena modelos por `id`, para diff limpo.
- `check`: valida contra o schema quando `jsonschema` existir, coerência
  `complete`/`incomplete_fields`, `source_url` e `license_source` não nulos
  em modelo completo, e avisa `verified_at` > 60 dias. Sai != 0 em erro.
- `data/license_policy.json`: tabela licença → `commercial_ok`, revisada pelo
  humano antes do commit. Licença fora da tabela → `commercial_ok: null`,
  sem tornar o modelo incompleto (Q4).
- Nada é preenchido de memória. Campo que não veio da API fica `null`, com
  motivo, e o modelo recebe `complete: false`.
- `tasks`: derivado de `pipeline_tag` da API por tabela explícita no código;
  tag desconhecida → `tasks` vazio e modelo incompleto.
