"""Fórmulas do M1 (PROMPT.md), uma função pura por fórmula.

Sem I/O, sem estado, sem rede. Memória em bytes inteiros; contexto e batch em
tokens inteiros. GiB/GB só aparecem em fmt_gb()/fmt_gbps(), na apresentação.

Aritmética inteira conforme DECISIONS.md A21 e schemas/m1_interface.md:
constantes inteiras, comparações de tier multiplicando dos dois lados, KV em
bits com divisão por 8 só no fim. A única conta em float é o fallback de
pesos (params * bpw / 8 * k_ovh), arredondada para cima.

Nota sobre swap: nenhuma função aqui considera swap como memória. Swap evita
o crash por falta de memória mas não resolve lentidão — pesos paginados para
disco derrubam a geração para ~0,1 token/s. Nunca some swap ao orçamento.
"""

from __future__ import annotations

import math

# Reservas fixas (PROMPT.md M1), em bytes.
GPU_RESERVE_BYTES = 600_000_000
RAM_OS_RESERVE_BYTES = 2_000_000_000
RAM_APP_RESERVE_BYTES = 1_000_000_000

# Buffer de computação: base + por token de contexto, em bytes.
BUFFER_BASE_BYTES = 400_000_000
BUFFER_PER_TOKEN_BYTES = 20_000

# Cortes de tier, em porcentagem do orçamento.
TIER_COMFORT_PCT = 80
TIER_LIMIT_PCT = 95

# Bits por elemento do KV cache.
KV_ELEM_BITS: dict[str, int] = {"f16": 16, "q8_0": 8, "q4_0": 4}

TIERS: tuple[str, ...] = ("confortavel", "limite", "inviavel")


def fmt_gb(n_bytes: int) -> str:
    """Bytes -> "7.10 GiB" (base binária, duas casas). Só para apresentação.

    O nome fica fmt_gb por contrato (m1_interface.md, adendo item 1).
    """
    return f"{n_bytes / 2**30:.2f} GiB"


def fmt_gbps(bytes_per_s: int) -> str:
    """Banda em GB/s decimal (convenção de fabricante). Só para apresentação."""
    return f"{bytes_per_s / 1_000_000_000:.2f} GB/s"


def vram_display(vram_total: int, vram_free: int) -> int:
    """VRAM já ocupada no instante da coleta (display, outros processos)."""
    return max(0, vram_total - vram_free)


def vram_usable(vram_total: int, vram_free: int) -> int:
    """vram_total - vram_display - 600e6, truncado em 0."""
    return max(0, vram_total - vram_display(vram_total, vram_free) - GPU_RESERVE_BYTES)


def ram_usable(ram_total: int) -> int:
    """RAM TOTAL menos reservas fixas (nunca RAM livre: não determinística)."""
    return max(0, ram_total - RAM_OS_RESERVE_BYTES - RAM_APP_RESERVE_BYTES)


def unified_budget(ram_total: int) -> int:
    """Pool unificado: 50% da RAM total (DECISIONS.md, Q7)."""
    return ram_total // 2


def k_ovh(params_total: int) -> float:
    """Fator de overhead do fallback de pesos (embeddings/saída não quantizados)."""
    if params_total > 7_000_000_000:
        return 1.05
    if params_total > 3_000_000_000:
        return 1.10
    return 1.15


def weights_bytes(params_total: int, bpw: float, file_bytes: int | None) -> tuple[int, bool]:
    """Bytes dos pesos e se usou o fallback. Prefere sempre o arquivo real."""
    if file_bytes is not None:
        return file_bytes, False
    return math.ceil(params_total * bpw / 8 * k_ovh(params_total)), True


def kv_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    ctx: int,
    kv_quant: str = "f16",
    batch: int = 1,
) -> int:
    """KV cache: 2 * n_layers * n_kv_heads * head_dim * ctx * elem_bytes * batch."""
    elem_bits = KV_ELEM_BITS[kv_quant]
    return 2 * n_layers * n_kv_heads * head_dim * ctx * elem_bits * batch // 8


# Decode previsto (decisão B do humano): resposta de 1024 tokens terminando no
# limite da janela pedida. Não é fator de eficiência; é a definição do número.
DECODE_ANSWER_TOKENS = 1024


def kv_read_bytes_per_token(model: dict, position: int, kv_quant: str = "f16") -> int:
    """Bytes de KV lidos para gerar o token na posição `position`:
    2 * n_layers * n_kv_heads * head_dim * position * elem_bytes.

    Mesma fórmula e mesma tabela de bits de kv_bytes (com ctx = position,
    batch = 1). Só kv_model "standard" (MHA/GQA): MLA não segue esta fórmula
    e é rejeitado pelo solver, aqui levanta ValueError.
    """
    if model.get("kv_model") != "standard":
        raise ValueError(f"kv_model {model.get('kv_model')!r}: leitura de KV só modelada para 'standard'")
    if position < 0:
        raise ValueError("position precisa ser >= 0")
    return kv_bytes(model["n_layers"], model["n_kv_heads"], model["head_dim"], position, kv_quant, 1)


def decode_bytes_per_token(weights_active: int, model: dict, prompt_tokens: int, gen_tokens: int,
                           kv_quant: str = "f16") -> int:
    """Média de bytes lidos por token numa geração de N tokens após um prompt de P:
    weights_active + KV(P + N/2). Inteiro: KV(2P + N) // 2 (o KV é linear na posição)."""
    if prompt_tokens < 0 or gen_tokens < 0:
        raise ValueError("prompt_tokens e gen_tokens precisam ser >= 0")
    return weights_active + kv_read_bytes_per_token(model, 2 * prompt_tokens + gen_tokens, kv_quant) // 2


def decode_start_position(ctx: int) -> int:
    """⌈min(DECODE_ANSWER_TOKENS, ctx)/2⌉: resposta de N = min(1024, ctx) tokens
    começando com o contexto vazio (decode_range_spec.md, D2). Só exibido."""
    if ctx < 1:
        raise ValueError("ctx precisa ser >= 1")
    return (min(DECODE_ANSWER_TOKENS, ctx) + 1) // 2


def decode_position(ctx: int) -> int:
    """Posição média do decode previsto: ctx − N/2 com N = 1024 (resposta que termina
    no limite da janela); ctx < 1024 -> ctx/2, arredondado para cima (mais KV, mais lento)."""
    if ctx < 1:
        raise ValueError("ctx precisa ser >= 1")
    if ctx < DECODE_ANSWER_TOKENS:
        return (ctx + 1) // 2
    return ctx - DECODE_ANSWER_TOKENS // 2


def buffer_bytes(ctx: int) -> int:
    """Buffer de computação: 400e6 + 20e3 * ctx."""
    return BUFFER_BASE_BYTES + BUFFER_PER_TOKEN_BYTES * ctx


def tier(total: int, budget: int) -> str:
    """confortavel se total <= 80% do orçamento, limite se <= 95%, senão inviavel."""
    if total * 100 <= TIER_COMFORT_PCT * budget:
        return "confortavel"
    if total * 100 <= TIER_LIMIT_PCT * budget:
        return "limite"
    return "inviavel"


def limit_cap(budget: int) -> int:
    """floor(0.95 * budget): maior total que ainda é tier limite."""
    return TIER_LIMIT_PCT * budget // 100


def min_bpw(params_total: int) -> float:
    """Piso de qualidade por tamanho. Filtro duro antes de qualquer cálculo."""
    if params_total <= 1_500_000_000:
        return 6.5
    if params_total <= 4_000_000_000:
        return 5.5
    if params_total <= 9_000_000_000:
        return 4.5
    if params_total <= 30_000_000_000:
        return 3.8
    return 2.6


def n_gpu_layers(weights: int, n_layers: int, vram_usable: int, kv_gpu: int, buffer_gpu: int) -> int:
    """Camadas na GPU no offload: floor((floor(0.95*vram_usable) - kv - buffer) / bytes_camada).

    O corte de 0,95 (DECISIONS.md Q6) evita que todo híbrido encha a VRAM a
    100% e caia em inviavel. Limitado a [0, n_layers].
    """
    bytes_camada = weights // n_layers
    if bytes_camada <= 0:
        return n_layers
    free = limit_cap(vram_usable) - kv_gpu - buffer_gpu
    if free <= 0:
        return 0
    return max(0, min(free // bytes_camada, n_layers))


def bytes_per_token(weights: int, params_active: int, params_total: int) -> int:
    """Bytes lidos por token: weights * active / total (MoE lê só os ativos)."""
    return weights * params_active // params_total


def tokens_per_s(
    bytes_camada: int,
    n_layers: int,
    n_gpu: int,
    bw_gpu: int | None,
    bw_cpu: int | None,
    eta_gpu: float,
    eta_cpu: float,
) -> float | None:
    """Decode limitado por banda: 1 / (t_gpu + t_cpu).

    bw em bytes/s. Retorna None se faltar a banda de um trecho que tem
    camadas. MoE: o chamador passa bytes_camada já escalado pela fração ativa.
    """
    n_cpu = n_layers - n_gpu
    t = 0.0
    if n_gpu > 0:
        if bw_gpu is None:
            return None
        t += (bytes_camada * n_gpu) / (bw_gpu * eta_gpu)
    if n_cpu > 0:
        if bw_cpu is None:
            return None
        t += (bytes_camada * n_cpu) / (bw_cpu * eta_cpu)
    if t <= 0:
        return None
    return 1 / t


def segment_seconds(bytes_read: int, bw: int | None, eta: float,
                    cal_bw: int | None = None, cal_overhead_s: float = 0.0) -> float | None:
    """Tempo por token de um trecho (pool) do decode.

    Com calibração do pool (schemas/bench_spec.md): bytes / banda_calibrada +
    overhead_fixo, sem eta. Sem calibração: bytes / (bw * eta), como em
    tokens_per_s. None se faltar a banda.
    """
    if cal_bw is not None:
        return bytes_read / cal_bw + cal_overhead_s
    if bw is None:
        return None
    return bytes_read / (bw * eta)
