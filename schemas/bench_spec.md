# scout bench — enunciado (Tarefa 4)

Decisões do humano em 2026-11-15, depois da validação em hardware real
(Ryzen 5 7520U). Enunciado comum para `coletor` (medição) e `nucleo`
(consumo da calibração no plan). Fonte da regra: M4 do PROMPT.md, alterado
por este documento — o PROMPT.md é atualizado no mesmo commit.

## Princípio

Não calibrar eta. Calibrar o tempo por token medido na carga real:

    t_por_token = bytes_por_token / banda_efetiva + overhead_fixo

`bytes_por_token` é o do solver (pesos × fração ativa; só o trecho do pool).
O eta global de `data/tuning.json` continua só como chute inicial antes da
primeira calibração e NÃO é alterado com base em nenhuma máquina.

## Medição (`coletor`, `scout/bench.py`)

- `scout bench --model <id> [--model <id> ...] [--run] [--ollama-model <nome>]...`
  sem `--run` só imprime o comando e o que falta.
- Ferramenta: `llama-bench` se estiver no PATH e o arquivo GGUF existir em
  disco; senão a API local do Ollama (`127.0.0.1:11434`), com o modelo
  indicado por `--ollama-model` (pareado na ordem com `--model`). Grava qual
  ferramenta foi usada.
- A medição casa com o que o plan recomendou para o candidato: mesma quant,
  mesmo `n_gpu_layers`, mesmo contexto, threads = núcleos de performance.
  Com Ollama, a quant é conferida comparando o tamanho do blob GGUF com o
  `file_bytes` do catálogo (diferença ≤ 1 MiB); se não casar, recusa com
  motivo e não mede.
- Prompt canônico versionado no repositório (arquivo único, sempre o mesmo),
  com contagem de tokens registrada no resultado. Geração de 128 tokens,
  temperatura 0, semente fixa; prompt com prefixo variável por rodada para
  não acertar cache de prompt.
- Estabilização: 1 aquecimento descartado; rodadas até as duas últimas
  ficarem a até 5% uma da outra (decode e, separadamente, prefill), teto de 6.
  Reporta TODAS as rodadas, marcando as que entraram na conta (as duas
  estáveis). Não estabilizou em 6 → grava com `stable: false` e confiança
  rebaixada, e diz isso na saída. Vale igualmente para o prefill.
- Energia lida no momento da medição (`online` de Mains/USB*, perfil em
  `/sys/firmware/acpi/platform_profile` ou `powerprofilesctl get`). Na
  bateria: aviso alto na saída e no `measured.json`.
- `measured.json` (no diretório corrente ou `--out`): por modelo medido, as
  rodadas, decode e prefill estáveis, bytes_por_token, banda efetiva
  implícita (bytes × tok/s), pool (cpu/gpu), quant, n_gpu_layers, ctx,
  threads, ferramenta e versão, arquitetura do modelo, energia, perfil, data.

## Ajuste e gravação da calibração (`coletor`)

> Substituído em parte pela Tarefa 5 (`schemas/calibration_store_spec.md`):
> a calibração não vai mais para `data/tuning.json`, e sim para o arquivo
> local `calibration.json`, com identidade da máquina, versão do motor e uma
> entrada por estado de energia. O plan só a aplica a um system.json que case
> com essa identidade. As regras de energia do "Consumo no plan" passam a ser
> as do §4 daquele enunciado.

- Com ≥ 2 modelos medidos no MESMO pool e tamanhos diferentes: mínimos
  quadrados em `t = bytes/banda + overhead`. Se `overhead` < 5% do tempo por
  token do menor modelo, ou negativo, descarta-o e refaz com um parâmetro
  (`banda = média ponderada de bytes×tok/s`). Registra a forma usada e os
  dois valores.
- Com 1 modelo: só um parâmetro, `overhead = 0`, forma registrada.
- `data/tuning.json` ganha `calibration.<pool>` = {`bandwidth_bytes_per_s`
  (int), `overhead_s_per_token` (float ou 0), `form` ("1p"|"2p"),
  `stable` (bool), `power`: {"on_ac": bool|null, "profile": str|null},
  `tool`, `models` (ids + arquitetura + bytes), `date`, `prefill_tokens_per_s`
  (se estável; senão null com motivo)}. Escreve só com `--run` bem-sucedido;
  nunca altera `eta_gpu`/`eta_cpu`.
- Arredondamento e tipos: bytes inteiros; tempos em float.

## Consumo no plan (`nucleo`, `scout/solver.py` + `cli.py`)

- Se `tuning.calibration.<pool>` existe, o trecho daquele pool usa
  `t = bytes/bandwidth + overhead` em vez de `bytes/(bw × eta)`. Sem
  calibração, comportamento atual.
- Confiança: sem calibração, tokens/s sai no máximo `medium`; com calibração
  estável e mesmo estado de energia, pode sair `high`; `stable: false` → no
  máximo `medium`.
- Estado de energia: a CLI lê o estado ATUAL (mesma regra da coleta) e o
  passa ao solver; se diferir do `power` da calibração, aviso
  `calibration_power_mismatch` e confiança rebaixada um nível.
- Arquitetura: se o modelo do candidato tem arquitetura diferente de todas
  as dos modelos da calibração, o candidato não passa de `medium`
  (motivo `calibration_other_arch`).
- `ttft_s_est`: com `prefill_tokens_per_s` calibrado e estável,
  `ttft_s_est = prompt_tokens / prefill` só se o intent trouxer o tamanho do
  prompt; senão continua null. O aviso `slow_prefill_cpu` passa a citar a
  medição desta máquina quando houver.
- Proveniência: tokens/s calibrado = `measured`-derivado → `calculated` com
  origem `calibration` (não `estimated`).

## Critério de pronto

`scout bench --model Qwen/Qwen3-4B --ollama-model qwen3:4b-instruct --run`
reproduz, na mesma condição de energia, os números medidos à mão (decode
~7,9 tok/s, banda efetiva ~19,8 GB/s); e um `scout plan` posterior usa a
calibração e prevê o mesmo modelo dentro de 30%.
