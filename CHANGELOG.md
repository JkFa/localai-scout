# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

## [0.2.1] - 2026-11-16

### Adicionado

- Tok/s previsto no início de uma conversa (`tokens_per_s_est_start`), ao lado do número principal no fim da janela: mesma conta, mesmo placement e mesma calibração, só com o KV lido na posição do início. Aparece como linha secundária no markdown e no frontend; o número do fim da janela continua o único que ordena e escolhe a recomendação.
- `inputs.decode.start_position_tokens` no `feasibility.json`. O campo é opcional: arquivos da 0.2.0 continuam válidos e abrem no modo ARQUIVO.

## [0.2.0] - 2026-11-16

### Adicionado

- `scout bench`: mede tok/s de decode e de prefill de modelos no Ollama, com estabilização das rodadas e registro das condições (CPU, temperatura, swap) no `measured.json`.
- Geração de 1024 tokens no bench; rodada com menos de 768 tokens gerados é recusada (`short_generation`).
- Calibração local de hardware, guardada fora do git, com identidade de hardware em duas camadas, contexto de medição (`measured_with`: kernel, MT/s e canais) em cada entrada e uma entrada por estado de energia; sem casamento exato de energia, o plan usa a menor medição da máquina e diz a condição por extenso.
- Energia com dois tipos de ausência: desconhecido (atributo existe e não foi lido; limita a confiança a média) e não aplicável (a máquina não tem bateria ou mecanismo de perfil; não limita).
- Versão do motor de inferência gravada na calibração; versão diferente rebaixa a confiança e sugere recalibrar.
- Leitura do KV cache na fórmula de decode, com a suposição de posição (`inputs.decode`) exibida nas recomendações.
- Modelo Qwen3-0.6B no catálogo; com ele, a calibração passa a ter dois parâmetros (banda efetiva e custo fixo por token).
- Frontend: seção de calibração no diagnóstico e linha com a suposição de posição do decode.

### Alterado

- **Incompatível:** calibrações da versão 1 (só pesos) são descartadas com `calibration_schema_outdated`; é preciso medir de novo com `scout bench` (ou recalcular a partir do `measured.json`).
- **Incompatível:** `data/tuning.json` não aceita mais a chave `calibration`.
- **Incompatível:** os tokens/s previstos agora são na posição do fim da janela de contexto, por isso são menores que os da 0.1.0 para o mesmo modelo.
- **Incompatível:** `inputs.calibration` e `inputs.decode` passam a ser obrigatórios no `feasibility.json`.
- Calibração com `calibration_schema_version` 2: o `bytes_por_token` do decode inclui o KV cache lido (pesos ativos + KV médio em f16). Os valores antigos e novos estão registrados em `DECISIONS.md`.
- O bench usa P e N reais das rodadas usadas para calcular a leitura média do KV.
- O prefill gravado na calibração com vários modelos é o menor entre eles, não a média.
- O bench descarrega cada modelo do Ollama ao fim da medição, para não deixar memória ocupada na rodada seguinte.

### Corrigido

- Bateria detectada pelo `type` e pelo `scope` da fonte em `/sys/class/power_supply`, e não pelo nome `BAT*`.
- Leitura do Ollama ignora o proxy do ambiente.
- Versão do motor lida só para as ferramentas das entradas escolhidas, não para pools descartados.
- Fora do casamento exato de energia, o TTFT e o aviso de prefill lento usam o menor prefill elegível.
- Bench encontra modelo criado com `ollama create x`, que o Ollama lista como `x:latest`.
- `measured_with` (kernel e camada 2 de identidade) fica em cada entrada e não é reescrito pelo merge.
- Mensagens de identidade mostram bytes com unidade; num pool descartado, kernel e camada 2 vão com efeito `none` no eco e o frontend não mostra a correspondência de energia.

## [0.1.0] - 2026-11-15

### Adicionado

- Núcleo calculável: estimativa de memória, solver de offload e `scout plan`.
- Teto de 0,95 para offload de camadas registrado nas decisões.
- Coleta real de hardware com `scout collect`.
- Aceitação de saída do `dmidecode` gerada pelo usuário (`--dmidecode`).
- Cliente HTTP do Hugging Face com cache.
- Construtor de catálogo com `catalog add` e `catalog check`.
- Catálogo semente com 7 modelos, obtidos pela API do Hugging Face.
- Backend local `scout serve` com token, CORS e Private Network Access.
- Frontend estático com modos ARQUIVO e LIVE.
- Exibição de memória em GiB.
- Testes e fixtures do núcleo, da coleta, do catálogo, do backend e dos endpoints.

### Corrigido

- Agrupamento de GGUF com arquivo único e com shards.
- Validação de schemas para `jsonschema` 4.18 ou superior.
- Remoção de cópias duplicadas do setup.
- Preferência de qualidade aplicada abaixo de 7e9 parâmetros.
