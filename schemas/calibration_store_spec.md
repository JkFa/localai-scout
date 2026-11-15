# Calibração local da máquina — enunciado (Tarefa 5, rev. 3)

Fecha ONDE a calibração do `scout bench` é gravada e QUANDO o `scout plan` a
aplica. Substitui, em `schemas/bench_spec.md`, o local de gravação e as regras
de energia do consumo no plan. Fórmula, estabilização e regra de arquitetura
ficam como estão. Contrato do arquivo e do eco:
`schemas/calibration.schema.json`. Exemplo validado pela CI:
`schemas/examples/calibration_example.json`.

## Decisões do humano incorporadas (sem perguntas abertas)

- **P1.** A identidade tem papéis separados. A camada 1 DECIDE e usa só campos
  de hardware legíveis sem privilégio. A camada 2 CORROBORA (MT/s e canais):
  conhecidos e diferentes invalida, desconhecido não conta e limita a medium.
  O teto vem acompanhado de orientação, dizendo como liberar `high`.
- **P2.** A calibração não expira por idade; expirar por idade está fora.
  `calibration_schema_version` muda só quando muda o significado dos números,
  e divergência descarta. Motor e versão entram na comparação: divergência
  rebaixa a confiança e sugere recalibrar.
- **PE.** O critério é a DIREÇÃO do erro, não o tamanho. Superestimar faz o
  usuário baixar gigabytes esperando uma velocidade que não vem, e a
  ferramenta perde a credibilidade. Subestimar, no pior caso, faz pular um
  modelo que daria certo. Daí as regras de §4:
  - casamento exato de energia primeiro; sem ele, a MENOR medição disponível,
    com a confiança um nível abaixo e a condição dita por extenso;
  - nenhuma medição desta máquina perde para o eta global, que só entra
    quando o pool não tem medição nenhuma que valha.
- **PE.1.** Há dois tipos de ausência, que deixam de ser o mesmo `null`:
  - **desconhecido**: o atributo existe e não foi lido. Não decide e limita a
    confiança a medium.
  - **não aplicável**: o atributo não existe nesta classe de máquina. Não
    decide e não limita, porque não há evidência faltando. Isso leva o desktop
    a `high` sem exceção escrita à mão. Vale para §2 (camada 2) e §4 (energia).
- **PK.** O kernel sai da identidade. Identidade de HARDWARE decide; a PILHA DE
  SOFTWARE (kernel e motor) só modula a confiança. O kernel é comparado por
  `major.minor`, e divergência rebaixa um nível e sugere recalibrar, sem
  invalidar.
- **Exemplo.** O `calibration_example.json` usa uma máquina fictícia, com
  valores sinteticamente óbvios. As medições reais ficam só no DECISIONS.md,
  onde valem como evidência. O motivo: o exemplo carrega uma impressão digital
  de hardware (mesmo raciocínio do `mem.txt` no .gitignore), e um número real
  num exemplo acaba virando semente ou entrando em teste.

## 1. Separação e local do arquivo (`coletor`, `scout/calstore.py`)

- `data/tuning.json` (versionado) guarda só as constantes do projeto: `eta_gpu`,
  `eta_cpu`, `max_bpw_offload`, `prefill_reference` e `source`. Nenhum código o
  grava.
- A calibração fica no arquivo local `calibration.json`. O caminho é resolvido
  por uma função pura,
  `resolve_calibration_path(flag, env, platform, home) -> (Path | None, origin)`,
  com `platform` vindo de `sys.platform`. A primeira regra que se aplicar vence:
  1. `--calibration PATH` → origem `flag`.
  2. `SCOUT_CALIBRATION` com valor não vazio → origem `env`.
  3. Padrão do SO → origem `default`. No linux/other,
     `$XDG_STATE_HOME/localai-scout/calibration.json` se a variável for um
     caminho absoluto; senão `~/.local/state/...`. No darwin,
     `~/Library/Application Support/localai-scout/calibration.json`. No windows,
     `%LOCALAPPDATA%\localai-scout\calibration.json`; sem a variável,
     `~\AppData\Local\...`, nunca Roaming. Se não houver home, devolve
     `(None, "default")`.
- `--no-calibration` (plan e serve) leva à origem `disabled`: nada é lido e
  nenhum motor é consultado. É exclusivo com `--calibration`. `--tuning` é
  independente das duas opções.
- Arquivo ausente: com origem `flag`, é erro; com `env` ou `default`, status
  `absent`.
- Arquivo malformado ou com `schema_version` ≠ 1: o plan falha alto.
- Migração: `load_tuning` falha se encontrar a chave `calibration`. A
  mensagem mostra o caminho novo e pede para rodar o bench de novo. Não há
  conversão: o bloco antigo não tem identidade, e completá-lo seria inventar
  dado. `solve()` também levanta `ValueError` se receber `tuning["calibration"]`.

## 2. Identidade de hardware (`nucleo`, `scout/calibration.py`, puro)

`machine_identity(system)` copia o `value` de cada campo abaixo do system.json.
`gpu.*` vem da GPU que o solver escolhe para o pool gpu. Não entram hostname,
serial, MAC, usuário, caminho nem hash.

**Camada 1 — DECIDE.** Só campos legíveis sem privilégio. Se o campo diferir,
ou for null em qualquer lado, o efeito é `discard`: a calibração do pool não
vale. Aqui vale a regra "null = diferente".

| campo | escopo | igual quando | por que está aqui |
| --- | --- | --- | --- |
| `cpu.model`, `cpu.cores_physical` | all | exato | definem o controlador de memória e as threads |
| `memory.ram_total_bytes` | all | \|a−b\| ≤ 5% de max | MemTotal oscila menos de 1% com firmware e kernel; uma troca de módulo muda 25% ou mais |
| `platform` | all | exato | outro SO no mesmo hardware troca drivers e runtime inteiros; não é atualização da pilha, é outra pilha |
| `os.arch` | all | exato | x86_64 contra arm64 (Rosetta, emulação) muda tudo |
| `gpu.name` | gpu | exato | escopo gpu porque, numa máquina sem GPU, o campo é null e o pool cpu não pode depender dele |
| `gpu.backend` | gpu | exato | a mesma GPU via cuda, rocm ou vulkan passa por outro caminho de execução |
| `gpu.vram_total_bytes` | gpu | ±5% | a reserva da iGPU na BIOS muda a VRAM |

`memory.unified` não entra: ele só decide se o pool se chama `unified` ou
`cpu`, e a calibração já é guardada por pool. `os.kernel` saiu da camada 1 e
foi para §3 (PK).

**Camada 2 — CORROBORA.** São `memory.speed_mts` e `memory.channels`, que vêm
do dmidecode. Para cada campo:
- conhecido nos dois lados e igual → `none`;
- conhecido nos dois lados e diferente → `discard` (`calibration_machine_mismatch`),
  porque é um sinal real de mudança de configuração;
- **desconhecido** em algum lado → não conta e limita a confiança a medium
  (`cap_medium`, `calibration_memory_unverified`);
- **não aplicável** → não conta e não limita (`none`). Hoje, isso é o pool
  `unified` (SoC): a memória vem do chip, que a camada 1 já identifica, e a
  dimensão MT/s × canais não existe à parte. Em §2 a camada 2 é avaliada só
  para o pool `cpu`.

O que não é "não aplicável": plataforma em que o coletor ainda não lê MT/s
(darwin, windows). O atributo existe; falta a leitura. Fica desconhecido, com
o texto abaixo. "Não aplicável" exige evidência positiva de que a dimensão não
existe; na dúvida, o valor é desconhecido.

**Orientação no lugar do teto.** Aviso `calibration_memory_unverified`
(severidade `info`, no máximo um), com o texto escolhido pelo lado em que falta
o dado e pelo `platform` do system.json:
- falta no system atual, em linux: "MT/s e canais da RAM não foram lidos neste
  system.json; sem essa evidência a calibração do pool cpu fica em confiança
  média. Rode uma vez `sudo scout collect` (ou `scout collect --dmidecode
  <arquivo de sudo dmidecode -t memory>`) e refaça o plan para liberar
  confiança alta."
- falta na calibração, em linux: "A calibração foi gravada sem MT/s e canais
  da RAM; gere o system.json com o dmidecode legível (como acima) e rode
  `scout bench --run` de novo para liberar confiança alta."
- outra plataforma: "O coletor ainda não lê MT/s e canais da RAM nesta
  plataforma; a calibração do pool cpu fica em confiança média." Não promete
  um caminho que não existe.

**Versão do significado.** `calibration_schema_version` é um inteiro, cujo
valor esperado é a constante `CALIBRATION_SCHEMA_VERSION = 1` em
`scout/calibration.py`. O bump só acontece quando muda o SIGNIFICADO dos
números gravados. Se for diferente, todos os pools são descartados
(`calibration_schema_outdated`, aviso warn que manda rodar o bench).
`scout_version` é gravado e nunca comparado.

## 3. Pilha de software: modula, não decide (`coletor` lê; `nucleo` compara)

Divergência na pilha de software nunca descarta a calibração. Com a máquina
igual, a medição desta máquina continua melhor que o eta global (PE). O que
muda é a confiança, e a saída diz o que mudou e sugere recalibrar.

**Kernel.** `os.kernel` fica gravado na identidade, mas não decide. A chave de
comparação é `kernel_key(s)`: o prefixo `^\d+\.\d+` ("7.0.0-43-generic" →
"7.0"); sem esse padrão, a string inteira. Efeitos, na linha `match` de
`os.kernel` (camada `software`):
- igual → `none`;
- diferente → `downgrade` (um nível abaixo), aviso warn
  `calibration_kernel_changed`: "A calibração foi medida com o kernel 6.8;
  agora é 7.0. Rode `scout bench --run` para recalibrar.";
- null em algum lado → `cap_medium`, aviso info `calibration_kernel_unknown`.
Atualização de patch (`7.0.0-43` → `7.0.0-45`) não muda nada. O `major.minor`
muda com o HWE, mais ou menos a cada 6 meses.

**Motor de inferência.**
- Cada entrada tem uma única ferramenta (`tool`) e `tool_version`. O bench
  recusa, antes de medir, um ajuste que misture ferramentas no mesmo pool.
- `engine_version(tool) -> str | None`, em `scout/bench.py`, é a ÚNICA fonte
  da versão. O bench a usa para gravar e a CLI do plan para ler a versão
  atual; assim as duas strings são comparáveis. Para `ollama`:
  `GET http://127.0.0.1:11434/api/version`, campo `version`, timeout de 2 s.
  Para `llama-bench`: `llama-bench --version`, timeout de 3 s, primeira linha
  sem espaços nas pontas. Se o binário não estiver no PATH, não aceitar a
  opção ou não responder, o resultado é None. Isso substitui a versão tirada
  do JSON da rodada. O `coletor` confere o formato com o binário real e
  registra no DECISIONS.md se não houver `--version`.
- Rede: quem lê é `run_plan` (CLI e serve), como já faz com a energia. A
  leitura acontece só se o arquivo de calibração existe e a origem não é
  `disabled`, uma vez por ferramenta das entradas escolhidas, sempre em
  loopback, que o bench já usa. O solver recebe `engines: dict[str, str | None]`
  e só compara: a regra 5 continua valendo.
- Efeitos:
  - `equal`: nenhum.
  - `different`: `downgrade` e aviso warn `calibration_engine_changed`: "A
    calibração do pool X foi medida com ollama 0.17.6; agora está 0.18.0. Rode
    `scout bench --run` para recalibrar."
  - `unknown` (null em algum lado, por exemplo com o Ollama parado):
    `cap_medium` e aviso info `calibration_engine_unknown`: "Versão atual de
    ollama não lida (motor parado ou ausente); inicie-o e refaça o plan." O
    número calibrado continua sendo usado.

## 4. Energia (`coletor` lê; `nucleo` escolhe)

**Estado com três tipos de valor.** `read_power()` devolve
`{on_ac, profile, not_applicable}`. `not_applicable` é a lista dos atributos
que não existem nesta máquina. Um atributo nessa lista tem valor null; um
atributo null FORA da lista é desconhecido. Detecção no linux:
- `on_ac` não se aplica quando não existe `/sys/class/power_supply/BAT*`
  (sem bateria, não há dimensão "na tomada ou na bateria"). Com bateria e
  leitura falhando, é desconhecido.
- `profile` não se aplica quando não existe `/sys/firmware/acpi/platform_profile`
  E `powerprofilesctl` não está no PATH. Com um dos dois presente e a leitura
  falhando, é desconhecido.
- darwin, windows e other: `not_applicable` só com evidência positiva (por
  exemplo, a API do SO dizendo que não há bateria). O que o coletor não lê
  fica desconhecido, e o aviso diz que a leitura não existe nesta plataforma.

O bench grava esse mesmo objeto em `power` de cada entrada.

**Comparação de um atributo** (gravado × atual): os dois não aplicáveis →
igual; os dois conhecidos → igual se os valores coincidem; conhecido de um
lado e não aplicável do outro → diferente; desconhecido em qualquer lado →
desconhecido.

**Escolha da entrada, por pool:**
1. Entrada em que os dois atributos dão igual → `power_match: same`, nenhum
   efeito.
2. Se não houver, e nenhum atributo atual for desconhecido → a entrada de
   MENOR `bandwidth_bytes_per_s` entre todas as do pool. `power_match:
   other_state`, `downgrade` e aviso warn `calibration_power_mismatch`.
3. Se algum atributo atual for desconhecido → a de menor banda entre as
   entradas compatíveis com a parte conhecida (todas, se nenhuma for). Ela é
   usada mesmo que pareça casar, porque o estado não está provado.
   `power_match: unknown`, `cap_medium`, aviso info `calibration_power_unknown`.
Empate de banda: a data mais recente; depois, a ordem do arquivo.

**A condição dita por extenso.** As mensagens nomeiam os dois estados com
`describe_power(power)`, que monta "na tomada, perfil performance", "na
bateria, perfil balanced", "sem bateria", "perfil não lido", e assim por
diante:
- mismatch: "Calibrado na tomada, perfil performance; você está na bateria,
  perfil balanced. Usando a medição mais lenta desta máquina
  (11,0 GB/s efetivos) para não prometer velocidade. Rode `scout bench --run`
  neste estado para uma estimativa exata."
- unknown: "Estado de energia atual não lido (perfil desconhecido); usando a
  medição mais lenta compatível desta máquina."

**O eta só entra sem medição.** O pool usa `bytes/(bw × eta)` só quando não há
calibração que valha: o arquivo está ausente ou desabilitado, o pool não tem
entradas, ou a identidade de §2 descartou o pool. Nenhuma divergência de
energia, kernel ou motor leva ao eta.

`available_power` no eco lista todos os estados gravados no pool.

**Como a confiança é combinada, por candidato de pool aplicado.** O ponto de
partida é `high`. Primeiro entram os tetos, que limitam a medium:
`stable: false`, `power_unknown`, `engine_unknown`, `kernel_unknown`,
`memory_unverified` e `other_arch`. Depois, cada motivo de "um nível abaixo"
desce um nível, com piso `low`: `power_mismatch`, `engine_changed` e
`kernel_changed`. Não aplicável não entra em nenhuma das duas listas.

## 5. Eco no feasibility.json (`nucleo`; schema muda, `schema_version` fica 1)

- `inputs.calibration` passa a ser obrigatório, no formato
  `calibration.schema.json#/$defs/echo`. Traz a origem, o status, a versão
  esperada, o arquivo lido (versões e identidade), uma linha `match` por
  campo (com camada `1`, `2` ou `software`, escopo e efeito) e, por pool, o
  status, os motivos, `power_match`, `available_power`, `engine` e a entrada
  escolhida. Nunca traz o caminho do arquivo.
- `inputs.tuning.calibration` sai. O `$defs/calibration` do feasibility vira
  `$ref` para `calibration.schema.json#/$defs/pool_calibration`.
- `inputs.power` aparece quando o status é `used` ou `discarded`, no formato
  `calibration.schema.json#/$defs/power` (com `not_applicable`).
- Avisos novos: `calibration_machine_mismatch` e `calibration_machine_unknown`
  (warn, com o campo e os dois valores), `calibration_schema_outdated` (warn),
  `calibration_engine_changed` (warn), `calibration_engine_unknown` (info),
  `calibration_kernel_changed` (warn), `calibration_kernel_unknown` (info) e
  `calibration_memory_unverified` (info). `calibration_power_mismatch` (warn) e
  `calibration_power_unknown` (info) ganham o texto de §4.
- Razões de confiança novas: `calibration_memory_unverified`,
  `calibration_engine_changed`, `calibration_engine_unknown`,
  `calibration_kernel_changed` e `calibration_kernel_unknown`.
- O markdown do plan ganha uma linha de calibração, com o estado de energia
  por extenso.
- `schema_version` fica 1, porque o eco antigo nunca foi publicado.
- API: `solve(system, catalog, intent, tuning, power=None, calibration=None,
  engines=None)`, com `calibration: CalibrationInput | None` (dataclass com
  `origin` e `doc`).

## 6. Escrita (`coletor`, `calstore.py` + `bench.py` + `cli.py`)

- **Antes de medir:** o bench recusa se algum campo de camada 1 e escopo all
  for null no `--system` (`identity_incomplete`), ou se `gpu.*` for null num
  alvo do pool gpu. Também recusa se esses mesmos campos não forem `equal` aos
  de uma leitura ao vivo, que não precisa de privilégio
  (`system_not_this_machine`). Kernel, motor e energia nunca impedem a medição.
- **Merge:** `B.write_calibration` deixa de existir. Na gravação:
  - com `calibration_schema_version` diferente, todos os pools antigos caem;
  - entradas antigas caem se algum campo de identidade do escopo do pool
    (camadas 1 e 2) der `different` ou `unknown`; a pilha de software não
    derruba entradas;
  - no pool medido, a entrada com a mesma chave de energia (os dois atributos
    iguais, pela comparação de §4) é substituída; as outras ficam;
  - a identidade gravada passa a ser a do `--system` atual (o kernel
    inclusive).
  O stderr diz o que caiu e por quê.
- **Arquivo:** a escrita é atômica (`mkdir(..., mode=0o700)`, `mkstemp` no
  mesmo diretório, `fsync`, `os.replace`; o temporário é removido se houver
  erro) e determinística (`sort_keys`, `indent=2`, `\n` no fim).
- **Comandos:** `scout bench --run --calibration PATH` grava e imprime o
  caminho. `scout serve` lê o arquivo na inicialização e nunca grava; a energia
  e o motor são lidos a cada `/plan`. A seção "Calibração local" do
  ARCHITECTURE.md já foi escrita.

## 7. Frontend (`web`, `web/render.js`)

O Diagnóstico mostra `inputs.calibration`: status, origem e, por pool, data,
ferramenta e versões, energia (a gravada e a atual, por extenso) e os estados
disponíveis. Mostra também as linhas de `match` com efeito diferente de
`none`. O `CONF_REASON` ganha as razões novas. Os textos vêm do JSON; sem
`inputs.calibration`, nada quebra.

## 8. Testes (`qa`, entregues com o `coletor`)

- Um fixture autouse aponta `SCOUT_CALIBRATION`, `HOME`, `XDG_STATE_HOME`,
  `LOCALAPPDATA` e `USERPROFILE` para `tmp_path`.
- `read_power` e `engine_version` são trocados por versões fixas; a versão de
  `--no-calibration` levanta exceção, para provar que nada é lido.
- A detecção de `not_applicable` é testada com árvores falsas de `/sys` e PATH
  (sem BAT*, sem platform_profile e sem powerprofilesctl; com o daemon
  presente e a leitura falhando).
- `resolve_calibration_path` e `kernel_key` são testadas por parâmetros.
- Os bytes de `tuning_test.json` são iguais antes e depois do bench.
- Duas gravações iguais produzem bytes iguais; uma falha em `os.replace` deixa
  o arquivo antigo intacto.
- Os casos de §2 a §4 são testados um a um, e toda saída é validada no schema.
  As fixtures de calibração são sintéticas, como o exemplo.

## Critério de pronto

    pytest -q && python schemas/validate_examples.py
    T=$(mktemp -d); sha256sum data/tuning.json > $T/sum
    export SCOUT_CALIBRATION=$T/cal.json
    scout bench --model Qwen/Qwen3-4B --ollama-model qwen3:4b-instruct --run
    sha256sum -c $T/sum && git status --porcelain data/       # intacto, vazio
    python3 schemas/check_calibration_done.py "$T"            # escrito pelo qa

O `check_calibration_done.py` (do `qa`, sem rede: não mede nada, só chama
`scout plan` com variações do arquivo do bench e do system.json, todas em `$T`,
com energia e motor injetados) falha a menos que:
1. o plan normal dê `status: used`, pool cpu `applied`, `power_match: same`;
   o Qwen3-4B fique a até 30% do medido e com confiança high;
2. **camada 2 desconhecida:** com `memory.speed_mts.value = null` no system,
   `applied`, o candidato em medium com `calibration_memory_unverified`, e um
   aviso cuja mensagem contenha `dmidecode`;
3. **camada 2 divergente:** com `memory.speed_mts.value = 4800`, `discarded`
   com `calibration_machine_mismatch`, e o tokens/s volta ao eta;
4. **camada 1:** com `sys_rtx3060.json`, `discarded`;
5. **kernel:** com `os.kernel` do system trocado só no patch (mesmo
   `major.minor`), nenhum efeito e confiança high; com outro `major.minor`,
   `applied`, `calibration_kernel_changed`, um nível abaixo e mensagem com
   `scout bench --run`;
6. **motor:** com `tool_version: "0.0.0"` no arquivo,
   `calibration_engine_changed`, um nível abaixo e mensagem com
   `scout bench --run`; com a versão atual null, `calibration_engine_unknown`
   e medium;
7. **energia, outro estado:** com uma segunda entrada sintética
   `{on_ac: false, profile: "balanced"}` de banda = metade da medida e o plan
   na bateria/balanced → `same`, tokens/s ≈ metade; com o plan na
   bateria/performance (nenhuma casa) → `other_state`, entrada escolhida = a de
   menor banda, um nível abaixo, mensagem contendo "na tomada" e "na bateria";
8. **energia não aplicável:** arquivo e plan com
   `{on_ac: null, profile: null, not_applicable: ["on_ac", "profile"]}` →
   `same` e confiança high; com o plan em `profile` null fora de
   `not_applicable` → `unknown` e medium;
9. **schema da calibração:** com `calibration_schema_version: 99`,
   `discarded` e `calibration_schema_outdated`;
10. com `--no-calibration`, `disabled` e sem `inputs.power`.

## Ordem

1. `nucleo` (§2, §3 na parte da comparação, §4 na parte da escolha, §5).
2. `coletor` (§1, §3 na parte da leitura, §4 na parte de `read_power`, §6 e
   testes).
3. `web`.
4. `revisor`.

Um commit por agente.
