Construa `localai-scout`: uma ferramenta de linha de comando em Python que lê
o hardware da máquina e recomenda modelos de IA locais que de fato rodam nela.

## Por que a arquitetura é assim — não negocie este ponto

Ferramentas desse tipo falham sempre do mesmo jeito: pedem a um LLM que
lembre de modelos, calcule memória e escreva o relatório na mesma passada.
O resultado são nomes de modelo inventados e contas de VRAM que não fecham.

Aqui o trabalho é separado em cinco estágios e o LLM aparece em um só:

  1. COLETA     script determinístico           -> system.json
  2. CATÁLOGO   derivado da API do Hugging Face -> catalog.json
  3. SOLVER     código puro, sem rede, testado  -> feasibility.json
  4. CONSULTOR  LLM: só interpreta e escreve    -> relatório
  5. BENCHMARK  medição real                    -> recalibra o estágio 3

Regra que governa o projeto inteiro: nenhum número é produzido por LLM, e
nenhum modelo é citado sem existir no catálogo com URL de origem.

## Stack

- Python 3.11+, pyproject.toml, instalável com pip install -e .
- Entry point: scout = scout.cli:main
- Dependências: apenas pytest, e só para testes. Para HTTP use urllib.request
  da stdlib. Se achar que precisa de outra dependência, pare e me pergunte
- Type hints em tudo, dataclasses para os modelos de dados
- Alvo testado: Linux/Ubuntu. macOS e Windows em best effort, sem testes,
  jamais com crash
- git init, commits pequenos por módulo, sem push

## Convenção de unidades — fonte de bug número um aqui

Internamente TUDO em bytes inteiros e tokens inteiros. Nenhum GB no meio do
cálculo. Converta para GB só na apresentação, por uma função fmt_gb(). O
catálogo guarda params_total como inteiro (7620000000), nunca em bilhões.

## Como lidar com o que você não sabe

Esta regra vale para todas as etapas. Você nunca escreve em arquivo um nome
de modelo, um tamanho, uma contagem de camadas ou uma licença vindos da sua
memória. Ou o dado veio de uma API nesta execução, ou o campo fica null e o
registro é marcado como incompleto. Em dúvida sobre escopo, pergunte em vez
de assumir.

---

# M1 — núcleo calculável

Faça primeiro e pare para eu revisar. É o coração: se estiver errado, todo o
resto mente com confiança.

Entregáveis: scout/estimate.py, scout/solver.py, scout/cli.py (só o comando
plan), testes, e um catálogo de TESTE em tests/fixtures/catalog_test.json com
4 modelos fictícios porém plausíveis (um 1B, um 7B denso, um 30B MoE com 3B
ativos, um de embeddings).

Critério de pronto — isto roda e produz saída correta:

  scout plan --system tests/fixtures/sys_rtx3060.json \
             --catalog tests/fixtures/catalog_test.json --ctx 8192

Os testes NUNCA leem data/catalog.json. Sempre o catálogo de teste, para que
mudança no catálogo real não quebre a suíte.

## Fórmulas — implemente exatamente assim, cada uma função pura

Orçamento, dois pools separados (um só quando unified_memory é true):

  vram_display = vram_total - vram_free        (medido na coleta)
  vram_usable  = vram_total - vram_display - 600e6
  ram_usable   = ram_total - 2e9 - 1e9

Use RAM TOTAL menos reserva fixa, nunca "RAM livre": em Linux o valor livre
inclui page cache recuperável e oscila a cada segundo, o que tornaria a saída
não determinística entre duas execuções seguidas.

Pesos — prefira sempre o tamanho REAL do arquivo:

  se quant.file_bytes existe:  weights = quant.file_bytes
  senão, fallback:             weights = params_total * bpw / 8 * k_ovh
                               k_ovh = 1.05 se params_total > 7e9
                                       1.10 se 3e9 < params_total <= 7e9
                                       1.15 se params_total <= 3e9

O fallback existe só para modelo cujo tamanho não foi obtido, e o candidato
que o usar sai com confidence "low". O fator k_ovh sobe em modelos pequenos
porque os tensores de embedding e saída, que não são quantizados no mesmo
nível, representam fração maior do arquivo.

KV cache:

  kv = 2 * n_layers * n_kv_heads * head_dim * ctx * elem_bytes * batch
  elem_bytes: 2 em f16, 1 em q8_0, 0.5 em q4_0

n_kv_heads e head_dim vêm do catálogo e NUNCA são chutados: com GQA, o número
de cabeças de chave e valor é bem menor que o de atenção, e errar isso erra o
orçamento em várias vezes. Se o modelo usar atenção latente (kv_model ==
"mla"), esta fórmula superestima muito: rejeite com motivo em M1.

Buffer de computação:

  buffer = 400e6 + 20e3 * ctx

Viabilidade, contra o orçamento do pool onde o modelo roda:

  total = weights + kv + buffer
  "confortavel"  se total <= 0.80 * orcamento
  "limite"       se total <= 0.95 * orcamento
  "inviavel"     acima disso

Offload de camadas (só GGUF; EXL2, AWQ e GPTQ exigem residência total em GPU
e devem ser rejeitados quando não couberem inteiros):

  bytes_camada = weights / n_layers
  n_gpu = floor((vram_usable - kv_gpu - buffer_gpu) / bytes_camada)
  n_gpu = max(0, min(n_gpu, n_layers))

Throughput, somando os dois trechos:

  t_tok = (bytes_camada * n_gpu) / (bw_gpu * eta_gpu)
        + (bytes_camada * (n_layers - n_gpu)) / (bw_cpu * eta_cpu)
  tokens_per_s = 1 / t_tok

eta_gpu = 0.75 e eta_cpu = 0.60 são empíricos: coloque-os em data/tuning.json
para que o estágio 5 os recalibre por máquina, nunca fixos no código. Reporte
ttft_s_est separado de tokens_per_s_est: prefill é limitado por computação e
decode por banda de memória, e o usuário sente os dois de formas diferentes.

MoE, onde memória e velocidade divergem:

  memória usa params_total
  bytes_lidos_por_token = weights * (params_active / params_total)

Isto é obrigatório. Um MoE de 30B com 3B ativos roda em CPU com RAM sobrando,
e um solver que use só params_total o rejeita por engano — justamente o caso
que mais interessa a quem tem hardware fraco.

Piso de qualidade por tamanho, filtro duro antes de qualquer cálculo:

  params_total <= 1.5e9  -> bpw minimo 6.5   (Q6_K)
  <= 4e9                 -> bpw minimo 5.5   (Q5_K_M)
  <= 9e9                 -> bpw minimo 4.5   (Q4_K_M)
  <= 30e9                -> bpw minimo 3.8   (Q3_K_M)
  acima disso            -> bpw minimo 2.6   (IQ2_M)

Modelos pequenos têm menos redundância interna e sofrem muito mais com
quantização agressiva. Abaixo de 7e9, quando duas opções couberem, prefira o
modelo MENOR em bpw ALTO ao modelo maior em bpw destrutivo.

## Algoritmo de geração de candidatos

  para cada modelo do catálogo:
      se complete == false                         -> pular, silencioso
      se tasks do modelo não cruza intent.tasks    -> pular, silencioso
      se intent.ctx > ctx_max                      -> rejeitar com motivo
      se kv_model == "mla"                         -> rejeitar com motivo
      blocked = intent.commercial_use and not commercial_ok
      para cada quant, da MAIOR bpw para a menor:
          se bpw < piso(params_total)              -> pular com motivo
          para cada placement em [gpu_full, hybrid, cpu_only]:
              se o formato não permite o placement -> pular
              calcular mem, tier e tokens_per_s
          se houver placement viável:
              escolher o de maior tokens_per_s
              registrar candidato e PARAR as quants deste modelo
  ordenar por (tier, tokens_per_s desc)
  modelo sem nenhum candidato entra em rejected com motivo e bytes faltantes

Parar na primeira quant viável garante ao usuário a melhor qualidade que cabe,
não a menor que cabe. Um candidato blocked continua no relatório, marcado,
mas nunca como recomendação principal.

---

# M2 — coleta real

scout/collect.py. Todo campo de saída tem a forma {"value": ..., "source":
"...", "confidence": "high|medium|low"}. Campo não detectado sai como null
com confidence "low", jamais como zero.

Em Linux:
- SO e kernel: /etc/os-release e platform.release()
- CPU: lscpu, fallback /proc/cpuinfo. Separe núcleos FÍSICOS e de PERFORMANCE
  dos lógicos: em Intel híbrido os E-cores atrapalham inferência e nunca devem
  entrar em n_threads. Flags de ISA: avx2, avx512, amx, neon, sve
- RAM: /proc/meminfo. Tipo, MT/s e canais via dmidecode. dmidecode exige root:
  tente sem sudo, NUNCA peça senha, e degrade para confidence "low"
- Banda de memória: se não detectada, estime MT/s * canais * 8 e marque como
  estimada. Sem MT/s, use a tabela em data/bandwidth_tables.json por geração
- GPU: nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,
  compute_cap. rocm-smi para AMD. /sys/class/drm/card* como último recurso.
  Derive bf16 (cc >= 8.0), fp8 (cc >= 8.9) e flash attention
- Disco: os.statvfs e lsblk -d -o name,rota para separar NVMe, SATA e HDD
- Energia: /sys/class/power_supply, para avisar sobre throttling na bateria
- Já instalado: ollama, llama-cli, llama-server, lm-studio e comfyui no PATH;
  se ollama existir, rode ollama list

Em macOS use sysctl e system_profiler e marque unified_memory true. Em Windows
use powershell. Qualquer detecção que falhe registra o motivo e segue.

Critério de pronto: scout collect roda sem crash num ambiente onde nvidia-smi,
lscpu e dmidecode não existem, e devolve confidence "low" nos campos afetados.
Teste isso de verdade, com PATH vazio.

---

# M3 — catálogo real

scout/catalog.py mais um construtor. NÃO escreva o catálogo à mão e não
pesquise fichas de modelo na web: derive tudo de APIs, por este comando.

  scout catalog add <repo_hf> [--gguf-repo <repo>]

Busque, via urllib:
  GET https://huggingface.co/api/models/{repo}
  GET https://huggingface.co/{repo}/raw/main/config.json
  GET https://huggingface.co/api/models/{gguf_repo}?blobs=true

Mapeie do config.json, sem adivinhar nada:
  num_hidden_layers        -> n_layers
  num_key_value_heads      -> n_kv_heads  (ausente: num_attention_heads, MHA)
  num_attention_heads      -> n_heads
  head_dim                 -> head_dim
       (ausente: hidden_size / num_attention_heads)
  max_position_embeddings  -> ctx_max
  num_experts ou n_routed_experts presentes -> é MoE; calcule params_active
  kv_lora_rank presente    -> kv_model = "mla"

bpw efetivo é CALCULADO, não tabelado:

  bpw = file_bytes * 8 / params_total

data/quant_bpw.json existe só como fallback quando não há file_bytes.

Licença: leia cardData.license da API e grave license_source com a URL do
cartão do modelo. Se a API não devolver licença, grave null e marque o modelo
incompleto. NUNCA preencha licença de memória nem de fonte secundária: há
páginas na web que contradizem os cartões oficiais, e licença errada aqui vira
problema jurídico do usuário.

Campo que não vier da API fica null e o modelo recebe "complete": false.
Modelo incompleto nunca entra em recomendação.

scout catalog check avisa se verified_at tem mais de 60 dias.

Semente: 6 a 8 modelos apenas, todos adicionados por esse comando, cobrindo
chat, código, embeddings, VLM, STT e um MoE. Me mostre a lista de repos que
você pretende adicionar ANTES de rodar. Se a rede falhar, entregue o catálogo
vazio e documente o comando no README — não preencha à mão.

---

# M4 — relatório e benchmark

report.py monta o pacote de contexto (system.json, feasibility.json e o
recorte do catálogo com os modelos citados) e preenche prompts/consultant.md.
Por padrão apenas imprime o prompt montado. Com --call e ANTHROPIC_API_KEY no
ambiente, faz POST via urllib. Falhe alto se feasibility.json não existir.

bench.py gera o comando por candidato:

  llama-bench -m <arquivo> -p 512 -n 128 -ngl <n_gpu_layers> -t <cores_perf>

Com --run, execute SOMENTE se o binário estiver no PATH e o arquivo do modelo
existir em disco; caso contrário imprima o comando e diga o que falta. Parseie
a saída, grave measured.json e atualize data/tuning.json:

  eta_novo = eta_atual * (tokens_medidos / tokens_estimados)
  limitado ao intervalo [0.30, 1.00]

---

# M5 — difusão, por último e assumidamente heurístico

  vram_pico = pesos_dit + pesos_text_encoders + vae_pico(res)
            + ativacoes(res, batch)
  tempo = passos * tempo_por_passo(res, backend)

Os text encoders costumam dominar e são o erro clássico: um T5-XXL em fp16
pesa cerca de 9 GB, mais que muitos DiTs. Modele as alavancas como flags:
offload sequencial de CPU, VAE em tiles, pesos em fp8 ou nf4, resolução,
passos.

Toda saída deste módulo sai com confidence "low" e aviso de que só a medição
confirma. Se você não conseguir dados confiáveis por API, entregue o módulo
retornando "não suportado" em vez de números inventados. Isso é aceitável e
preferido.

---

# CLI

  scout collect [--out system.json]
  scout plan [--system F] [--catalog F] [--task chat,code] [--ctx 8192]
             [--commercial] [--offline] [--json]
  scout bench [--model ID] [--run]
  scout report [--call]
  scout catalog add <repo> | check

plan imprime markdown legível por padrão e JSON com --json.

# Regras negativas

- NUNCA recomende aumentar swap para rodar modelo maior. Swap evita o crash
  por falta de memória e não resolve lentidão: pesos paginados para disco
  derrubam a geração para a casa de 0,1 token/s. Deixe isso como comentário
  no código e como aviso no relatório. As alavancas corretas, nesta ordem:
  quantizar o KV para q8_0, reduzir contexto, reduzir bpw dentro do piso,
  trocar por modelo menor
- Nunca recomende EXL2, AWQ ou GPTQ sem residência total na VRAM
- Nunca derive n_threads dos núcleos lógicos
- Nunca invente valor de hardware não detectado nem campo de catálogo
- Nunca deixe o solver tocar a rede

# Testes

Fixtures em tests/fixtures, uma por perfil, todas contra o catálogo de teste:

  1. notebook CPU-only, 8 GB DDR4 single-channel, HDD
  2. desktop 16 GB DDR4 dual-channel + RTX 3060 12 GB
  3. desktop 64 GB DDR5 sem GPU (o caso MoE)
  4. Apple Silicon 24 GB unified
  5. AMD RX 7800 XT 16 GB via ROCm

Asserções obrigatórias:
- nenhum candidato com total acima do orçamento do seu pool
- nenhum modelo <= 3e9 recomendado abaixo de bpw 5.5
- perfil 1: nenhuma recomendação de difusão, e ao menos um de embeddings
- perfil 3: o MoE aparece como viável
- perfil 2: n_gpu_layers == n_layers para o 7B na melhor quant que couber
- com --commercial, nenhum modelo não-comercial como principal
- duas execuções seguidas de plan sobre a mesma fixture dão saída idêntica
- todo número do markdown de plan existe também na saída --json

O teste de determinismo é o mais revelador: se divergir, há leitura de "RAM
livre" ou aritmética escapando para fora das funções puras.

Comece por M1. Não avance para M2 sem minha revisão.
