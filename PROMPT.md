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
cálculo. Converta só na apresentação: memória em GiB (base 2**30, duas casas,
"7.10 GiB") por fmt_gb(); banda em GB/s decimal (bytes/s / 1e9, convenção de
fabricante) por fmt_gbps() (DECISIONS.md, "memória apresentada em GiB"). O
catálogo guarda params_total como inteiro (7620000000), nunca em bilhões.

Aritmética do M1 em inteiros: constantes inteiras (600_000_000), cortes de
tier comparados multiplicando (total * 100 <= 80 * orcamento), KV em bits com
a divisão por 8 só no fim, bytes_camada = weights // n_layers. A única conta
em float é o fallback de pesos, arredondado para cima (math.ceil)
(DECISIONS.md A21; schemas/m1_interface.md).

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

vram_display e vram_usable são truncados em 0; vram_usable truncado gera o
aviso vram_usable_clamped (schemas/feasibility.schema.json, pool.usable_bytes).

Pool unificado (unified_memory true): um pool só, "unified", com orçamento
ram_total // 2 (DECISIONS.md Q7) e aviso unified_memory_single_pool. Nele só
existe o placement gpu_full, com a banda da RAM e eta_gpu (código da Fase 1;
sem entrada própria em DECISIONS.md). EXL2, AWQ e GPTQ não contam como
residentes na GPU em memória unificada: cada quant desses formatos é pulada
com no_gpu_for_format (DECISIONS.md, "formatos GPU-only rejeitados em memória
unificada").

Várias GPUs: o solver usa uma só, a de maior vram_total (desempate: menor
índice), com aviso multi_gpu_ignored (DECISIONS.md, "Representação de
multi-GPU").

Campo de hardware null (DECISIONS.md A23): sem ram_total o plano falha com
erro; GPU sem vram_total ou sem vram_free não forma pool (aviso
hardware_field_missing); gpus null (detecção falhou) = plano só com CPU, com o
mesmo aviso.

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
  n_gpu = floor((floor(0.95 * vram_usable) - kv_gpu - buffer_gpu) / bytes_camada)
  n_gpu = max(0, min(n_gpu, n_layers))

O teto de 0.95 é o mesmo corte do tier "limite": sem ele o offload encheria a
VRAM a 100% e todo híbrido cairia em "inviavel" (DECISIONS.md, Q6). No
híbrido, o KV inteiro e o buffer ficam no lado da GPU; o lado CPU leva as
camadas restantes (weights - n_gpu * bytes_camada). O tier do híbrido é
calculado por pool, contra o orçamento de cada um, e vale o PIOR dos dois
(DECISIONS.md A18). O placement hybrid só existe com 1 <= n_gpu < n_layers
(schemas/feasibility.schema.json, candidate.n_gpu_layers).

Quant acima de max_bpw_offload não é avaliada em hybrid: ver o limiar de
offload no algoritmo de geração de candidatos, abaixo.

Throughput, somando os dois trechos:

  t_tok = (bytes_camada * n_gpu) / (bw_gpu * eta_gpu)
        + (bytes_camada * (n_layers - n_gpu)) / (bw_cpu * eta_cpu)
  tokens_per_s = 1 / t_tok

eta_gpu = 0.75 e eta_cpu = 0.60 são empíricos: coloque-os em data/tuning.json
para que o estágio 5 os recalibre por máquina, nunca fixos no código. Reporte
ttft_s_est separado de tokens_per_s_est: prefill é limitado por computação e
decode por banda de memória, e o usuário sente os dois de formas diferentes.

Enquanto não houver fórmula de prefill nem medição, ttft_s_est = null com o
motivo ttft_unspecified (DECISIONS.md A17). Se falta a banda de um trecho que
tem camadas, tokens_per_s_est = null (DECISIONS.md Q5). No JSON,
tokens_per_s_est sai arredondado em 2 casas; se o arredondamento der 0, sai o
valor cru (código da Fase 1; sem entrada própria em DECISIONS.md).

MoE, onde memória e velocidade divergem:

  memória usa params_total
  bytes_lidos_por_token = weights * (params_active / params_total)
  no throughput, bytes_camada lidos = bytes_lidos_por_token // n_layers,
  nos DOIS trechos (GPU e CPU) do híbrido (DECISIONS.md A20)

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
modelo MENOR em bpw ALTO ao modelo maior em bpw destrutivo. Regra concreta
(DECISIONS.md, "preferência de qualidade abaixo de 7e9"): se o principal de
uma tarefa tem params_total < 7e9 e bpw < 5.5, e outro candidato não
bloqueado da mesma tarefa tem params_total menor, bpw >= 6.5 e tier não pior,
o principal passa a ser esse (o primeiro na ordem final). A ordem dos
candidatos não muda; o preterido gera o aviso quality_preference.

## Algoritmo de geração de candidatos

  para cada modelo do catálogo:
      se complete == false                         -> pular, silencioso
      se tasks do modelo não cruza intent.tasks    -> pular, silencioso
      ctx = min(intent.ctx, ctx_max) se a única tarefa avaliada
            (tasks do modelo ∩ intent.tasks) é embeddings; senão intent.ctx
      se ctx > ctx_max                             -> rejeitar com motivo
      se kv_model == "mla"                         -> rejeitar com motivo
      blocked = intent.commercial_use and commercial_ok != true
      para cada quant, da MAIOR bpw para a menor:
          se bpw é null                            -> pular com motivo
          se bpw < piso(params_total)              -> pular com motivo
          para cada placement em [gpu_full, hybrid, cpu_only]:
              se o formato não permite o placement -> pular
              se placement == hybrid e bpw > max_bpw_offload
                                                   -> não avaliar (limiar)
              calcular mem, tier e tokens_per_s
          se houver placement viável:
              escolher o de maior tokens_per_s     (desempate abaixo)
              registrar candidato e PARAR as quants deste modelo
  ordenar por (tier, tokens_per_s desc, bytes_cpu_por_token asc,
               bytes_por_token asc, model_id asc)
  modelo sem nenhum candidato entra em rejected com motivo e bytes faltantes

Desempate quando tokens_per_s é desconhecido (banda null). Na escolha do
placement, se algum placement viável tem tokens_per_s null, vence o de menor
bytes_cpu_por_token = bytes_camada_lidos * (n_layers - n_gpu) (MoE: com a
fração ativa); empate pelo menor bytes_por_token total; depois a ordem fixa
gpu_full, hybrid, cpu_only. Na ordenação dos candidatos e na escolha do
principal por tarefa vale a chave acima, que só muda a ordem entre candidatos
com tokens_per_s null. Esses bytes são só critério de ordem, nunca exibidos
como velocidade; o candidato recebe o motivo ranked_by_bytes em
confidence_reasons.

Limiar de offload. Se bpw > max_bpw_offload (em data/tuning.json, valor
inicial 8.5, nunca fixo no código), o placement hybrid não é avaliado para
essa quant; gpu_full e cpu_only continuam. A banda da CPU derrubaria a
velocidade; vale mais uma quant menor inteira na GPU. O limiar nunca é
silencioso: se o hybrid descartado teria sido viável e
  - a quant não coube em gpu_full nem em cpu_only: ela sai em skipped_quants
    com motivo bpw_above_offload_limit;
  - o modelo ainda entra (por outro placement da mesma quant ou por quant
    menor): o candidato vencedor leva em notes
    {"code": "offload_skipped_above_bpw", "message": ...}, uma nota por quant
    descartada, na ordem de avaliação. A mensagem diz a quant descartada, o
    limiar (vindo do tuning) e onde o vencedor roda.
  Nenhum dos dois casos cria entrada em rejected. Se o hybrid descartado não
  teria sido viável, o limiar não mudou nada e não há nota.

Parar na primeira quant viável garante ao usuário a melhor qualidade que cabe,
não a menor que cabe. Um candidato blocked continua no relatório, marcado,
mas nunca como recomendação principal.

Contexto efetivo de embeddings: o candidato registra o ctx usado no KV e no
buffer em ctx_effective. Modelo avaliado também para outra tarefa continua
rejeitado por ctx_exceeds_max (DECISIONS.md, "embeddings usam ctx efetivo").

Bloqueio comercial: commercial_ok false -> blocked_reason non_commercial;
commercial_ok null (licença fora da tabela) -> license_unknown, também
bloqueado sob --commercial, sem tornar o modelo incompleto (DECISIONS.md Q4).

Principal por tarefa (primary_for): para cada tarefa da intenção, o primeiro
candidato não bloqueado que tem a tarefa, na ordem final, salvo a preferência
de qualidade abaixo de 7e9 (DECISIONS.md, "principal por tarefa").

Rejeição. Se alguma tentativa (quant, placement) foi calculada, closest é a de
menor déficit (empate: a primeira avaliada) e missing_bytes = max sobre os
pools de (total_bytes - floor(0.95 * budget_bytes)); o motivo é
format_requires_full_gpu se closest é EXL2/AWQ/GPTQ, senão does_not_fit. Sem
nenhuma tentativa calculada: below_quality_floor se todas as quants caíram no
piso; no_gpu_for_format se só houve piso e formato sem GPU (com ao menos um
desses); no_usable_quant no resto. closest e missing_bytes ficam null
(schemas/feasibility.schema.json, rejected; código da Fase 1, sem entrada
própria em DECISIONS.md). rejected[].provenance.missing_bytes é estimated se
closest usou pesos por fallback, senão calculated (DECISIONS.md F7).

Confiança do candidato: low se usou pesos por fallback ou se algum pool usado
tem dado de hardware com confidence low; senão medium se eta é default, se a
banda de um trecho com camadas é estimada ou se tokens/s é null; senão high.
Motivos em confidence_reasons, em ordem alfabética (código da Fase 1; sem
entrada própria em DECISIONS.md). Proveniência de cada número: a pior entre
as entradas, em provenance (DECISIONS.md, "Proveniência por número").

Avisos fixos: swap_not_a_fix sai sempre (regra negativa). O solver não emite
catalog_stale, porque não lê relógio; a idade do catálogo é checada por
scout catalog check (DECISIONS.md, "catalog_stale fora do solver").

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
- Banda de memória: se não detectada, estime MT/s * canais * data_width_bits / 8
  (Data Width do dmidecode, nunca Total Width, que inclui ECC) e marque como
  estimada. Data Width ausente ou fora de {32, 64}: assuma 64 e rebaixe para
  confidence "low". O tipo de memória não entra na fórmula; se divergir do que
  MT/s e largura indicam, registre a divergência em source/note do tipo sem
  trocar o valor. Sem MT/s, use a tabela em data/bandwidth_tables.json por geração
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
       (ausente ou null: hidden_size / num_attention_heads, só se a divisão
        for exata; senão null)
  max_position_embeddings  -> ctx_max
  num_experts, n_routed_experts ou num_local_experts > 1 -> é MoE; calcule
                              params_active (fórmula abaixo)
  kv_lora_rank presente    -> kv_model = "mla"

VLM: se config.json tem text_config, TODOS os campos acima vêm só de
text_config; campo ausente lá fica null, nunca o do nível de cima (que é do
encoder de visão). params_total = safetensors.total da API do modelo
(ausente: null e incompleto). (Mapeamento da Fase 3; sem entrada própria em
DECISIONS.md.)

params_active de MoE (DECISIONS.md, "fórmula de params_active do MoE
aprovada"):

  params_active = params_total - (n_experts - num_experts_per_tok)
                  * camadas_moe * 3 * hidden_size * ffn_expert
  ffn_expert   = moe_intermediate_size (estilo Mixtral, sem ele:
                 intermediate_size quando a chave é num_local_experts)
  camadas_moe  = n_layers - first_k_dense_replace - len(mlp_only_layers)

Exige hidden_act com porta (silu, swish, swiglu). Dado faltando,
decoder_sparse_step ou moe_layer_freq != 1, ou resultado fora de
[1, params_total) -> params_active null e modelo incompleto. Modelo denso:
params_active = params_total.

Tarefas: derivadas de pipeline_tag por tabela no código (text-generation ->
chat; image-text-to-text -> vlm; automatic-speech-recognition -> stt;
feature-extraction e sentence-similarity -> embeddings; text-to-image ->
diffusion); tag fora da tabela -> tasks null e modelo incompleto
(catalog.schema.json exige ao menos uma tarefa quando não é null). "code" não tem pipeline_tag: vem de
data/task_overrides.json (reviewed), que por repo só ACRESCENTA tarefas, nunca
remove nem altera outro campo, e não se aplica se tasks veio null
(DECISIONS.md, "tarefa code por override revisado").

Quants GGUF (?blobs=true): arquivos .gguf agrupados pelo nome sem o sufixo
-0000K-of-0000N; tamanho de lfs.size (senão size). Por quant: se há arquivo
único, vale ele (file_bytes e file_url dele); senão, o único conjunto
completo de shards, somado, com file_url do primeiro shard. Conjunto
incompleto é descartado; arquivo único e shards divergindo mais de 1% gera
aviso e fica o único. mmproj e arquivos sem nome de quant são ignorados.
(Correção da Fase 3; sem entrada própria em DECISIONS.md.)

bpw efetivo é CALCULADO, não tabelado:

  bpw = file_bytes * 8 / params_total

bpw fora de (0, 40] fica null, com aviso no catalog add (DECISIONS.md, "teto
de bpw no catálogo sobe para 40").

data/quant_bpw.json existe só como fallback quando não há file_bytes.

Licença: leia cardData.license da API e grave license_source com a URL do
cartão do modelo. Se a API não devolver licença, grave null e marque o modelo
incompleto. commercial_ok vem de data/license_policy.json (reviewed): licença
fora da tabela -> commercial_ok null, sem tornar o modelo incompleto
(DECISIONS.md Q4 e "license_policy revisada"). NUNCA preencha licença de memória nem de fonte secundária: há
páginas na web que contradizem os cartões oficiais, e licença errada aqui vira
problema jurídico do usuário.

Campo que não vier da API fica null e o modelo recebe "complete": false.
Modelo incompleto nunca entra em recomendação.

scout catalog check avisa se verified_at tem mais de 60 dias e falha (saída
!= 0) se o catálogo viola o schema ou a coerência complete/incomplete_fields.

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

  scout collect [--out system.json] [--dmidecode ARQUIVO]
  scout plan [--system F] [--catalog F] [--task chat,code] [--ctx 8192]
             [--commercial] [--offline] [--json] [--tuning F]
  scout serve [--port 8765] [--catalog F] [--tuning F]
  scout bench [--model ID] [--run]
  scout report [--call]
  scout catalog add <repo> [--gguf-repo R] [--catalog F] [--offline]
  scout catalog remove <id> [--catalog F]
  scout catalog check [--catalog F] [--today YYYY-MM-DD]

--tuning (padrão data/tuning.json) existe para os testes usarem
tests/fixtures/tuning_test.json e nunca lerem data/ (DECISIONS.md, "flag
--tuning em plan e serve"). catalog remove sai != 0 se o id não existe
(DECISIONS.md, "scout catalog remove"). catalog add grava JSON com
sort_keys, indent 2 e modelos ordenados por id (schemas/m3_interface.md).

plan imprime markdown legível por padrão e JSON com --json. No markdown,
número com proveniência estimated leva o prefixo "~" (tokens/s sempre) e a
memória e os tokens/s de candidato com confiança low levam "⚠", com legenda
no topo; o JSON não muda (DECISIONS.md, "marcas de estimativa no markdown").
As notes do candidato aparecem na coluna de nota (DECISIONS.md R3).

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
