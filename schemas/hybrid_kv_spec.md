# KV do híbrido por camada: memória alinhada ao throughput — enunciado

Corrige a contradição apontada pelo revisor (achado 3). Na memória, `PROMPT.md`
§Offload e o A18 dizem: "no híbrido, o KV inteiro e o buffer ficam no lado da
GPU". No throughput, `PROMPT.md` §Throughput diz: "cada camada lê o SEU KV, no
pool onde a camada está". O humano decidiu pelo modelo POR CAMADA. Revê o A18
e a fórmula de `n_gpu` do Q6. Não há mudança de schema (§4).

## Perguntas abertas — respondidas pelo humano em 2026-11-16

**Q1 = (b):** o mesmo buffer de computação nos dois pools do híbrido até haver medição do lado CPU.
**Q2:** o llama.cpp é a referência; conferir o Ollama com `ollama ps` quando houver máquina com GPU.
O único indício citado sobre o Ollama veio de um espelho não oficial do GitHub e é tratado como fraco.


- **Q1. Buffer de computação no lado CPU do híbrido. Bloqueia só uma linha de
  §1.** O llama.cpp reserva um buffer de computação POR BACKEND: o
  `sched_reserve` registra `"<backend> compute buffer size"` para cada um, e o
  CPU usa o buffer *host* da GPU. Portanto o lado CPU do híbrido tem buffer
  próprio, maior que zero. O tamanho dele depende da divisão do grafo e não
  existe medição no projeto. A fórmula `400e6 + 20e3·ctx` descreve um buffer
  só.
  - (a) Buffer só na GPU, como hoje. A RAM fica com buffer zero, o que é
    certamente subestimado: é a direção proibida. Custo: zero mudança.
  - (b) **O mesmo `buffer` nos dois pools (recomendado até haver medição).**
    Erra na direção aceita. No MoE Q4_K_M com rtx3060 e ctx 32768, soma
    1 055 360 000 B à RAM. `candidate.buffer_bytes` passa a ser 2× no híbrido,
    a soma de `by_pool`, coerente com `total_bytes`. Custo: híbridos no limite
    da RAM podem cair de tier ou de quant, sempre do lado seguro.
  - (c) Medir: o bench lê o log de compute buffer por backend do runner numa
    máquina com GPU e grava o resultado em `data/tuning.json`. Hoje não há
    máquina com GPU calibrada, então (c) substitui (b) depois, não agora.
- **Q2. O motor próprio do Ollama segue a mesma regra? Não bloqueia.** No
  llama.cpp está conferido no código: em `src/llama-kv-cache.cpp`, com
  `offload`, `buft = ggml_backend_dev_buffer_type(model.dev_layer(il))`, ou
  seja, o KV fica no dispositivo da camada. O padrão é `offload_kqv` ligado; a
  flag `--no-kv-offload` leva todo o KV para a CPU. No motor Go do Ollama
  (`kvcache`) só há indício (contexto por camada), não conferência. Se ele
  puser todo o KV na GPU, o modelo novo subestima a VRAM.
  - Recomendação: aceitar o llama.cpp como referência e conferir com
    `ollama ps` (divisão CPU/GPU) quando houver máquina com GPU.

## 1. Memória por pool no híbrido (`nucleo`, `scout/estimate.py`)

Notação: L = `n_layers`, `bc = weights // L` (como hoje), `kv` = KV total da
intenção (com batch), `buf = buffer_bytes(ctx)`.

    kv_gpu  = kv * n_gpu // L                 # mesma partição do throughput (seg_bytes)
    kv_cpu  = kv - kv_gpu                     # resto inteiro vai para a CPU
    w_gpu   = bc * n_gpu ;  w_cpu = weights - w_gpu        (como hoje)
    GPU     = w_gpu + kv_gpu + buf
    CPU     = w_cpu + kv_cpu + buf_cpu        # buf_cpu = buf com Q1(b); 0 com Q1(a)

Valem para os dois pools: `by_pool[i].kv_bytes`, `buffer_bytes` e
`total_bytes`. No candidato, `kv_bytes = kv` (não muda), `buffer_bytes` é a
soma de `by_pool` e `total_bytes` é a soma de `by_pool`. O tier continua sendo
o pior dos dois (A18). `gpu_full` e `cpu_only` não mudam.

## 2. Escolha de `n_gpu` (`nucleo`)

Cada camada na GPU custa pesos mais o seu KV. O teto 0,95 continua:

    def n_gpu_layers(weights: int, kv: int, n_layers: int, vram_usable: int, buffer_gpu: int) -> int:
        """floor((limit_cap(vram_usable) - buffer_gpu) / (weights//L + ceil(kv/L))), em [0, L]."""

- O `ceil(kv/L)` garante `w_gpu + kv_gpu + buf <= limit_cap(vram_usable)`,
  porque `kv*n//L <= n*ceil(kv/L)`. A GPU nunca passa de `limite`. Em borda
  pode sobrar uma camada que caberia, o que é aceito por ser o lado seguro.
- A assinatura muda de propósito: `kv_gpu` deixa de existir e `kv` vem antes
  de `n_layers`. Chamador antigo quebra no teste, não em silêncio.
- Consequência: `n_gpu_novo >= n_gpu_antigo`, porque `kv*n/L <= kv`. Com
  contexto longo podem aparecer híbridos novos, onde antes `kv + buf > cap`
  dava `n_gpu = 0`.

## 3. Efeito previsto nos fixtures (sem rodar o solver; o `nucleo` confere)

Conta feita só com funções puras de `estimate.py`, nos perfis com pool
`gpu`, ctx de 512 a 32768 e kv f16/q8_0. Nenhum tier e nenhum placement muda
nos fixtures sem mutação. Só `n_gpu_layers` sobe, e com ele o tokens/s do
híbrido e o `primary_for` quando houver empate de tier. Q1 não muda `n_gpu`.

| caso | antes | depois |
| --- | --- | --- |
| rtx3060, MoE Q4_K_M, ctx 8192 f16 (perfil padrão dos testes e do exemplo) | n_gpu 25 | 26 |
| rtx3060, MoE Q4_K_M, ctx 32768 f16 | 18 | 22 |
| rtx3060, dense-7b Q8_0, ctx 32768 f16 | 24 | 27 |
| rx7800xt, MoE Q6_K, ctx 8192 f16 | 26 | 27 |
| rx7800xt, MoE Q6_K, ctx 32768 f16 | 21 | 24 |

Testes que quebram e precisam de conta nova:
- `test_estimate.py`: os cinco testes de `n_gpu_layers` (assinatura).
  `test_tokens_per_s_default_etas_and_rtx_profile` usa n_gpu 25 explícito e
  continua válido como conta, mas o comentário "perfil 2" passa a ser 26.
- `test_phase7.py::test_f4_unknown_gpu_bw_picks_least_cpu_bytes`: 28 → 29,
  com (8 550 000 000 − 563 840 000) // (239 062 500 + 33 554 432) = 29. Os
  bytes na CPU passam a ser 3 camadas mais o KV lido delas.
- `test_phase7.py::test_fa_limit_is_read_from_tuning`: 21 → 22, com
  10 646 160 000 // 483 554 432 = 22.
- `test_cli.py::test_cli_passes_max_bpw_offload_to_solver` e
  `test_task3.py::test_hybrid_with_cpu_majority_triggers_warning`: conferir.
  Este último é invariante e deve passar.
- `schemas/examples/feasibility_example.json`: o candidato híbrido vai para
  n_gpu 26, com `by_pool` recalculado. O `tokens_per_s_est` passa de 21,6 para
  22,33 e o `tokens_per_s_est_start` (opção C, já implementada) de 29,69 para
  30,69. A inversão de ordem citada em `decode_range_spec.md` continua
  (22,33 < 25,99 e 30,69 > 29,16). O `cpu` de `by_pool` depende de Q1. As
  fixtures de `web/fixtures/` com híbrido continuam válidas contra o schema
  (a suíte as valida desde 1ee1a59); os números delas ficam como estão, porque
  são fixtures de apresentação.

## 4. Schema e eco

Não mudam. `pool_usage` já tem `kv_bytes` e `buffer_bytes` por pool, e o
schema não obriga zero no lado CPU. Só muda o significado: o lado CPU do
híbrido passa a ter `kv_bytes > 0`, e `buffer_bytes > 0` com Q1(b). O web já
lista `by_pool` e não precisa mudar. `schema_version` continua 1.

## 5. Testes com conta à mão (`nucleo`)

1. `n_gpu_layers(18_300_000_000, 805_306_368, 48, 11_800_000_000, 563_840_000) == 26`:
   (11 210 000 000 − 563 840 000) // (381 250 000 + 16 777 216) =
   10 646 160 000 // 398 027 216 = 26.
2. Divisão por pool nesse caso: `kv_gpu = 805_306_368*26//48 = 436_207_616`,
   `kv_cpu = 369_098_752`, GPU = 9 912 500 000 + 436 207 616 + 563 840 000 =
   10 912 547 616 (`limite`), CPU = 8 387 500 000 + 369 098 752 = 8 756 598 752,
   mais 563 840 000 com Q1(b).
3. **A RAM deixa de caber e o tier cai.** rtx3060 com `ram_total` =
   15 100 000 000 (orçamento 12 100 000 000, `limit_cap` 11 495 000 000),
   chat, ctx 32768, f16, MoE:
   - Antes: Q4_K_M hybrid, n_gpu 18, CPU 11 437 500 000 (`limite`).
   - Agora: o Q4_K_M hybrid tem n_gpu 22, kv 3 221 225 472 e
     `kv_cpu = 3_221_225_472 - 3_221_225_472*22//48 = 1_744_830_464`. A CPU
     fica com 9 912 500 000 de pesos + 1 744 830 464 de KV = 11 657 330 464
     (Q1a) ou 12 712 690 464 (Q1b), acima de 11 495 000 000: `inviavel`.
   - O candidato passa a ser Q3_K_M hybrid, n_gpu 26, GPU 10 871 023 780
     (`limite`), CPU 8 305 561 692 (Q1a) ou 9 360 921 692 (Q1b), as duas
     `confortavel`.
   - Assert: quant Q3_K_M e Q4_K_M em `skipped_quants`. O desfecho é o mesmo
     com Q1(a) e Q1(b).
4. Invariantes para todo perfil × ctx × kv: `sum(u.kv_bytes) == kv_bytes`,
   `sum(u.total_bytes) == total_bytes`, GPU do híbrido `<= limit_cap` e
   `by_pool[gpu].kv_bytes == kv*n_gpu//L`. Este último amarra memória e
   throughput (`seg_bytes`) à mesma partição.

## 6. Critério de pronto

    .venv/bin/pytest -q                              # 0 falhas
    .venv/bin/python schemas/validate_examples.py    # exit 0
    F="--system tests/fixtures/sys_rtx3060.json --catalog tests/fixtures/catalog_test.json --tuning tests/fixtures/tuning_test.json --no-calibration"
    .venv/bin/scout plan $F --task chat --ctx 8192 --json | .venv/bin/python -c "import json,sys; c=[c for c in json.load(sys.stdin)['candidates'] if c['placement']=='hybrid'][0]; u={p['pool']:p for p in c['by_pool']}; assert c['n_gpu_layers']==26 and u['cpu']['kv_bytes']==369_098_752 and u['gpu']['kv_bytes']+u['cpu']['kv_bytes']==c['kv_bytes']"
    grep -n "KV inteiro" PROMPT.md schemas/m1_interface.md scout/*.py   # sem saída

No mesmo commit, o `nucleo` atualiza `PROMPT.md` §Offload (fórmula de `n_gpu`
e frase do lado CPU), `schemas/m1_interface.md` (linhas 20–21 e 42) e o
comentário em `scout/solver.py` (`_attempts`), além do exemplo.
