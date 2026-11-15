# Contrato de interface do M1 (Fase 1)

Definido pela sessão orquestradora para que `nucleo` (implementação) e `qa`
(testes) trabalhem em paralelo sem ver o código um do outro. Fórmulas:
`PROMPT.md` M1. Fechamentos: `DECISIONS.md` (respostas Q1–Q8 e A17–A23 valem
como adotados). Formato de dados: `schemas/*.schema.json`.

## Arredondamento e tipos

- Entradas e saídas de memória: `int` em bytes. Contexto e batch: `int`.
- Constantes inteiras: `600_000_000`, `2_000_000_000`, `1_000_000_000`,
  `400_000_000`, `20_000`.
- Comparações de tier em inteiros: `total * 100 <= 80 * budget`, etc.
- KV: `elem_bits` = 16 (f16), 8 (q8_0), 4 (q4_0);
  `kv = 2 * n_layers * n_kv_heads * head_dim * ctx * elem_bits * batch // 8`.
- Pesos por fallback: `math.ceil(params_total * bpw / 8 * k_ovh)`, avaliado
  nessa ordem em float. Testes aceitam tolerância de ±1 byte só aqui.
- `bytes_camada = weights // n_layers`.
- Pool unificado (`unified_memory: true`): `budget = ram_total // 2`, um pool só.
- `n_gpu = floor((floor(0.95 * vram_usable) - kv_gpu - buffer_gpu) / bytes_camada)`,
  limitado a `[0, n_layers]`. No híbrido, KV inteiro e buffer ficam no lado GPU.
- Banda desconhecida (`null`) → `tokens_per_s_est = None`. `ttft_s_est = None`
  (A17). Ordenação: tier, depois tokens/s desc com `None` por último, depois
  `model_id` asc como desempate final.

## `scout/estimate.py` — funções puras, sem I/O

    fmt_gb(n_bytes: int) -> str                     # "7.62 GB", base 10, 2 casas
    vram_display(vram_total: int, vram_free: int) -> int
    vram_usable(vram_total: int, vram_free: int) -> int
    ram_usable(ram_total: int) -> int
    unified_budget(ram_total: int) -> int
    k_ovh(params_total: int) -> float
    weights_bytes(params_total: int, bpw: float, file_bytes: int | None) -> tuple[int, bool]
                                                    # (bytes, usou_fallback)
    kv_bytes(n_layers: int, n_kv_heads: int, head_dim: int, ctx: int,
             kv_quant: str = "f16", batch: int = 1) -> int
    buffer_bytes(ctx: int) -> int
    tier(total: int, budget: int) -> str            # "confortavel"|"limite"|"inviavel"
    min_bpw(params_total: int) -> float             # piso de qualidade
    n_gpu_layers(weights: int, n_layers: int, vram_usable: int,
                 kv_gpu: int, buffer_gpu: int) -> int
    bytes_per_token(weights: int, params_active: int, params_total: int) -> int
                                                    # weights * active // total
    tokens_per_s(bytes_camada: int, n_layers: int, n_gpu: int,
                 bw_gpu: int | None, bw_cpu: int | None,
                 eta_gpu: float, eta_cpu: float) -> float | None
                                                    # bw em bytes/s; MoE: chamador passa
                                                    # bytes_camada já escalado (A20)

## `scout/solver.py`

    @dataclass(frozen=True)
    class Intent:
        tasks: tuple[str, ...]
        ctx: int = 8192
        commercial_use: bool = False
        batch: int = 1
        kv_quant: str = "f16"

    solve(system: dict, catalog: dict, intent: Intent,
          tuning: dict) -> dict   # feasibility.json, válido contra o schema

`solve` não lê arquivo nem rede; `tuning` traz `eta_gpu` e `eta_cpu`.

## `scout/cli.py`

    main(argv: list[str] | None = None) -> int

    scout plan [--system F] [--catalog F] [--task chat,code] [--ctx 8192]
               [--commercial] [--json]

`--task` omitido = todas as tarefas do catálogo. Markdown por padrão; `--json`
imprime o feasibility.json com `sort_keys=True, indent=2`. Sem timestamp em
nenhuma das duas saídas. Todo número do markdown aparece também no `--json`
(bytes no JSON; o teste compara via `fmt_gb`). `data/tuning.json` com
`eta_gpu: 0.75`, `eta_cpu: 0.60` é lido pela CLI, não pelo solver.

## Fixtures (`qa` é dono)

`tests/fixtures/catalog_test.json` (4 modelos fictícios: 1B, 7B denso,
30B MoE com 3B ativos, embeddings; ao menos um com `commercial_ok: false`) e
`sys_cpu8.json`, `sys_rtx3060.json`, `sys_ddr5_64.json`, `sys_apple24.json`,
`sys_rx7800xt.json`. Todos válidos contra `schemas/`, sintéticos, sem nomes de
modelos reais.
