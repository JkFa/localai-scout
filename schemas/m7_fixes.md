# Correções da Fase 7 — enunciado

Decisões do humano em 2026-11-15 sobre o relatório do revisor. Este arquivo é
o enunciado comum para `nucleo`, `coletor`, `web` e `qa`.

## F4 — ordem quando tokens/s é desconhecido (achados 4 e 5)

Quando `tokens_per_s_est` é `null`, o desempate não é mais por `model_id`.

- **Escolha do placement de um modelo (`_choose`)**: entre os placements
  viáveis, se todos têm tokens/s conhecido, vence o maior (como antes). Se
  algum é desconhecido, vence o de MENOR `bytes_cpu_por_token`, onde
  `bytes_cpu_por_token = bytes_camada_lidos * (n_layers - n_gpu)` (MoE: com a
  fração ativa, como em A20). Empate: menor `bytes_por_token` total; depois a
  ordem fixa `gpu_full`, `hybrid`, `cpu_only`.
- **Ordenação dos candidatos e `primary_for`**: chave = (tier,
  tokens/s desc com null por último, `bytes_cpu_por_token` asc,
  `bytes_por_token` asc, `model_id` asc). Isto só muda a ordem entre
  candidatos com tokens/s null.
- Nenhum número novo é exibido como velocidade: os bytes são só critério de
  ordem. O motivo vai em `confidence_reasons` como `ranked_by_bytes`.

## F4b — tabela de banda de GPU (revisa a resposta Q5)

O humano revisou a Q5: a banda da GPU pode vir de uma tabela de referência
quando a sonda local não a fornece.

- `data/gpu_bandwidth.json`: `{"reviewed": false, "entries": [{"match":
  "<substring exata do nome da GPU, case-insensitive>", "bandwidth_bytes_per_s":
  <int>, "source_url": "<página de especificação do fabricante>",
  "verified_at": "YYYY-MM-DD"}]}`.
- Toda entrada tem `source_url` de página oficial do fabricante, conferida
  nesta execução. Sem fonte conferida, a entrada não entra. Nada de memória.
- Na coleta: se a banda não veio da sonda e o nome da GPU casa com uma única
  entrada, `bandwidth_bytes_per_s` = valor da tabela, `source: "table"`,
  `confidence: "medium"`. Mais de uma entrada casando → não usa, motivo
  `ambiguous_table_match`. iGPU que compartilha RAM do sistema não entra na
  tabela.

## Fa — quant acima do limiar não vale offload (nota de especificação)

- `data/tuning.json` ganha `max_bpw_offload` (float, valor inicial `8.5`), lido pela CLI e passado em `tuning` ao solver.
- Para uma quant com `bpw > max_bpw_offload`, o placement `hybrid` não é
  avaliado (pulado com motivo `bpw_above_offload_limit` em
  `skipped_quants`). `gpu_full` e `cpu_only` continuam avaliados.
- `PROMPT.md` M1 é atualizado com essa regra, no MESMO commit do código.

## F2 — CPU híbrida sem sysfs (achado 2, `coletor`)

Sem `/sys/devices/cpu_core` e `cpu_atom`: usar `lscpu -p=CPU,CORE,MAXMHZ`. Se
os núcleos têm frequências máximas distintas em dois grupos, é híbrida: o
grupo de maior MAXMHZ são os de performance, `confidence: "medium"`. Se não
houver MAXMHZ ou o resultado for inconclusivo e o fabricante for Intel,
`cores_performance` e `cores_efficiency` ficam com `confidence: "low"`, motivo
`hybrid_undetermined`. Nunca `high` sem sysfs.

## F3 — canais de RAM no dmidecode (achado 3, `coletor`)

A chave de canal inclui o controlador (`Controller0-ChannelA` ≠
`Controller1-ChannelA`). Quando os canais vêm da contagem de módulos (sem
locator interpretável), `channels` sai `low` e a banda derivada também sai
`low`. A confiança da banda nunca é maior que a menor confiança entre MT/s e
canais.

## F6 — marcas no markdown da CLI (achado 6, `nucleo`)

Número estimado leva prefixo `~`; candidato com confiança low leva `⚠` ao lado
da memória e dos tokens/s. Legenda curta no topo do markdown. O teste
"todo número do markdown existe no JSON" continua valendo (ignora `~`/`⚠`).

## F7 — proveniência nos rejeitados (achado 7, `nucleo` + `web`)

`rejected[].provenance` (opcional no schema), com `missing_bytes`:
`estimated` quando a tentativa mais próxima usou pesos por fallback, senão
`calculated`. `web/render.js` usa esse campo; ausente → `calculated`.

## F8 — tuning de teste (achado 8, `qa`)

`tests/fixtures/tuning_test.json` (eta 0,75 / 0,60, `max_bpw_offload` 8.5).
Nenhum teste lê `data/tuning.json`; a CLI nos testes recebe `--tuning`
(o `nucleo` adiciona a flag ao `plan` e ao `serve`).

---

# Rodada 2 — decisões do humano sobre o fechamento da Fase 7

## R2 — método e confiança da CPU híbrida (`coletor`)

- `source` de `cores_performance`/`cores_efficiency` diz o método:
  `sysfs:cpu_core/cpu_atom`, `lscpu:maxmhz_grouping (separação X%)`, ou
  `undetermined`.
- Confiança por método: sysfs → high; agrupamento por frequência → medium;
  indeterminado → low. Agrupamento nunca sai high.
- Limiar de 15% aprovado.
- DECISIONS.md, limitação conhecida (não implementar): em Intel híbrido os
  E-cores compartilham L2 em grupos de quatro e os P-cores têm L2 próprio;
  sinal estrutural em `/sys/devices/system/cpu/cpu*/cache/`, que sobrevive em
  container com mais frequência que cpufreq.

## R3 — limiar de offload nunca silencioso (`nucleo` + `web`)

- Quando o limiar descarta o `hybrid` de uma quant e o modelo ainda entra por
  outro placement ou por outra quant, o candidato vencedor carrega
  `notes: [{"code": "offload_skipped_above_bpw", "message": "..."}]` (campo
  opcional no schema; lista, ordem determinística). Mensagem do tipo:
  "FP16 híbrido descartado: acima de 8,5 bpw o offload não traz ganho
  perceptível; Q8_0 cabe inteiro na GPU." Os números da mensagem saem do
  tuning e do candidato, nunca fixos no texto.
- NÃO cria entrada em `rejected`.
- `web/render.js` mostra as `notes` do candidato na linha dele. O markdown da
  CLI também.

## R4 — banda de GPU: valor real, fórmula como fallback (`coletor`)

- `data/bandwidth_tables.json` ganha a seção `gpu`, chaveada pelo NOME EXATO
  que `nvidia-smi`/`rocm-smi` devolvem. Por entrada: `bus_width_bits`,
  `data_rate_gbps`, `bandwidth_bytes_s` (já calculado), `source_url` da página
  do fabricante. `data/gpu_bandwidth.json` deixa de existir.
- Resolução: nome na tabela → `bandwidth_bytes_s`, confiança high; fora da
  tabela com barramento conhecido e clock legível → fórmula com o multiplicador
  do fabricante, confiança medium; nada → null, low (o desempate F4 assume).
- Multiplicador por fabricante (nvidia-smi, rocm-smi) validado reproduzindo a
  banda publicada de uma placa conhecida; se a conta não bater com o número
  oficial, não grava.
- Escopo: só as GPUs dos cinco fixtures e a da máquina do humano.
- Nenhum número de memória. Cada entrada vem de busca com URL gravada. A lista
  de placas com URLs é mostrada ao humano ANTES de gravar.

## R5 — PROMPT.md é a regra (`nucleo`)

- Seção "Algoritmo de geração de candidatos" do PROMPT.md passa a conter o
  desempate F4, o limiar Fa e a nota R3. A fórmula de `n_gpu` passa a ter o
  teto 0,95 (Q6, Fase 0), hoje só no código e no DECISIONS.md. Vai no mesmo
  commit do código de R3.

## R8 — testes nunca referenciam data/ (`qa`)

- Teste que varre `tests/` e falha se qualquer arquivo referenciar `data/`
  (exceto o próprio teste). Testes que hoje validam arquivos reais de `data/`
  saem; essa validação vive no carregador em runtime ou num comando `check`.
