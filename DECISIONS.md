# Decisões

Formato: `## <data> — <decisão>`, depois contexto, alternativas descartadas e
consequência aceita. Decisões marcadas **PROPOSTA** aguardam aprovação humana.

---

## 2026-11-15 — Token de pareamento do backend local

**Contexto.** `127.0.0.1:8765` é alcançável por qualquer página aberta no
navegador. O token é a única barreira entre um site qualquer e o perfil da
máquina.

**Decisão.**
- Geração: `secrets.token_urlsafe(32)` a cada `scout serve`. Nunca persistido
  em disco, nunca aceito por variável de ambiente ou flag, nunca escrito em log.
  Rotação = reiniciar o servidor; o token anterior morre com o processo.
- Transporte inicial: no **fragment** (`#token=...&port=8765`), nunca na query.
  O fragment não é enviado ao GitHub Pages nem vai no `Referer`.
- O frontend lê o fragment no carregamento e chama imediatamente
  `history.replaceState` para removê-lo da URL atual.
- Armazenamento: **só em memória** (variável do módulo `api.js`). Não usa
  `sessionStorage` nem `localStorage`. Recarregar a página exige novo
  pareamento (clicar de novo no link do terminal ou colar o token num campo).
- Envio: `Authorization: Bearer <token>` em toda rota exceto `/health`.
- Comparação: `hmac.compare_digest` sobre bytes, em tempo constante. Token
  ausente, malformado ou errado dão a mesma resposta 401 `unauthorized`.

**Alternativas descartadas.** Query string (vai para logs do Pages e para o
`Referer`). `sessionStorage` (sobrevive a reload, mas é legível por XSS do
mesmo jeito e pode ser persistido pela restauração de sessão do navegador; o
ganho de UX não paga a superfície extra). Token fixo em arquivo de config
(vira segredo de longa duração em disco). Cookie (exigiria `SameSite=None`
cross-site e abriria CSRF).

**Consequência.** Reload perde o pareamento — aceito. O link com o token pode
ficar no histórico global do navegador antes do `replaceState`; o risco é
limitado porque o token só vale enquanto aquele processo `scout serve` vive.

## 2026-11-15 — Frontend sem build

**Contexto.** Frontend estático no GitHub Pages, subcaminho `/localai-scout/`,
zero requisição externa.

**Decisão.** ES modules nativos, sem framework, sem bundler, sem transpilação,
sem `package.json`. Caminhos sempre relativos. Fonte do sistema.
Acrescentar ao `index.html` uma CSP em `<meta>`:
`default-src 'self'; connect-src 'self' http://127.0.0.1:* http://localhost:*; img-src 'self' data:; object-src 'none'; base-uri 'none'`
— transforma "zero requisição externa" em regra imposta pelo navegador.

**Alternativas descartadas.** Vite/esbuild (dependência de toolchain Node,
artefato a publicar diferente do fonte, CI mais longo). Framework (peso e
cadeia de dependências desproporcionais a 5 telas).

**Consequência.** Sem TypeScript nem JSX; validação do contrato no frontend é
manual (checagem de `schema_version` e `kind`). Qualquer pedido de build passa
por pergunta ao humano e entrada neste arquivo.

## 2026-11-15 — Política de campo incompleto

**Decisão.** Campo não obtido = `null`, nunca `0`, nunca string vazia.
- `system.json`: todo campo é `{value, source, confidence}`; `value: null`
  força `confidence: "low"` (imposto pelo schema).
- `catalog.json`: todo campo do modelo é chave obrigatória, nullable. O schema
  impõe `complete:true` ⇒ nenhum campo estrutural null, ao menos uma quant e
  `incomplete_fields` vazio; `complete:false` ⇒ `incomplete_fields` não vazio.
  A coerência inversa (a lista bate exatamente com os campos null) fica em
  código: `scout catalog check` e um teste do `qa`.
- Modelo com `complete:false` nunca é recomendado; é pulado em silêncio pelo
  solver e só aparece como contagem em `feasibility.excluded.incomplete`.
- Exceções deliberadas, que NÃO tornam o modelo incompleto: `quant.file_bytes`
  null (há fallback, com `confidence: low`), `quant.bpw` null (só aquela quant
  é ignorada, `bpw_unknown`), `commercial_ok` null (ver pergunta Q4).

**Alternativas descartadas.** Toda a coerência em código (falha tarde).
Toda no schema (a coerência inversa exigiria um `if` por campo e duplicaria a
lista; mais partes móveis sem ganho de segurança — o sentido perigoso,
`complete:true` com buraco, já é barrado no schema).

**Consequência.** JSON Schema aceita `16.0` como inteiro (spec 2020-12); a
garantia "bytes como int" depende de o produtor emitir `int` do Python. O
schema barra `16.5` e `"16 GB"`.

## 2026-11-15 — Agentes rodando como general-purpose

**Contexto.** `.claude/agents/` estava aninhado em `localai-scout-setup/` e
não foi carregado como tipos de agente. A sessão principal copiou o diretório
para a raiz; nesta sessão os agentes rodam como general-purpose instruídos a
ler sua própria definição.

**Decisão.** Modelo e esforço continuam os da tabela do `CLAUDE.md`
(`arquiteto` = opus/high). Não é escalonamento.

**Consequência.** Na próxima sessão os tipos nativos devem estar disponíveis;
`localai-scout-setup/` e o `.zip` são cópias duplicadas e podem ser removidos
(decisão do humano).

## 2026-11-15 — Critério de pronto da Fase 0 reinterpretado

**Contexto.** `ORCHESTRATION.md` pede que os schemas "validem os exemplos de
`PROMPT.md`", mas `PROMPT.md` não contém nenhum exemplo JSON.

**Decisão.** Exemplos construídos em `schemas/examples/` a partir do perfil 2
de `PROMPT.md` e de modelos explicitamente fictícios (`test/...`, URLs em
`example.invalid`, bloco `fixture.synthetic: true`). Números de hardware são
sintéticos e rotulados como tal; números de memória do plano foram calculados
pelas fórmulas de M1. Critério executável: `python schemas/validate_examples.py`
sai 0, incluindo 11 casos negativos que precisam ser rejeitados.

**Consequência.** Os exemplos não são golden files do solver: dependem das
propostas A18–A20 abaixo. O `nucleo` deve recalcular e o `qa` derivar fixtures
próprias.

## 2026-11-15 — jsonschema como dependência — PROPOSTA

**Contexto.** `PROMPT.md`: "dependências: apenas pytest".

**Proposta.** `jsonschema` NUNCA em runtime (nem CLI nem `scout serve`).
Usado só por `schemas/validate_examples.py` e, opcionalmente, por um teste
que valida as saídas de `scout plan --json` e `scout collect` contra os
schemas, com `pytest.importorskip("jsonschema")` — a suíte passa sem ele.
O backend valida o body de `/plan` com checagem manual mínima (tipos e chaves
de `intent`, `kind`/`schema_version` do `system`).

**Em aberto.** Pergunta Q1.

## 2026-11-15 — Representação de multi-GPU

**Decisão.** `system.gpus` é um Field cujo `value` é a lista de todas as GPUs
(`null` = detecção falhou; `[]` = não há GPU). O solver v1 usa UMA GPU: a de
maior `vram_total_bytes`, desempate pelo menor `index`, e emite o aviso
`multi_gpu_ignored`. Split de tensores entre GPUs fica fora de escopo.

**Consequência.** Máquinas multi-GPU recebem plano conservador. Suportar
split depois exige novo `pool_id` e bump de `schema_version`.

## 2026-11-15 — Versionamento e identificação dos arquivos

**Decisão.** Todo arquivo tem `schema_version` (inteiro, hoje `1`) e `kind`
(`system` | `catalog` | `feasibility`). O frontend identifica o arquivo
solto pelo `kind` e recusa `schema_version` desconhecido com mensagem clara.
Mudança incompatível = incremento. `feasibility.json` não tem timestamp
(determinismo).

## 2026-11-15 — Proveniência por número

**Decisão.** Números ficam planos (inteiros/números), e cada objeto que exibe
números tem um mapa `provenance: {campo: catalog|measured|calculated|estimated}`.
`estimated` = depende de constante empírica (eta, k_ovh, tabela) ou de
entrada `confidence: low`; a proveniência de um derivado é a pior entre as
entradas. `tokens_per_s_est` é sempre `estimated`.

**Alternativa descartada.** Cada número como `{value, provenance}`: quebra a
asserção "todo número do markdown existe no --json" com comparações simples e
incha o arquivo.

---

# Em aberto / subespecificado

Lacunas encontradas em `ARCHITECTURE.md` e `PROMPT.md`, com fechamento
PROPOSTO. Nada aqui está decidido até o humano aprovar.

### A1. Onde roda o solver no modo ARQUIVO — crítico
`CLAUDE.md` diz que ARQUIVO sobe o `system.json`, mas o navegador não tem
solver. Reimplementar as fórmulas em JS duplica o núcleo fora da suíte pytest
(viola a regra 1). **Proposta:** o frontend nunca calcula. `feasibility.json`
é autocontido (embute `system`, `intent`, tuning e o recorte do catálogo de
cada modelo citado). Modo ARQUIVO aceita:
- `feasibility.json` (de `scout plan --json > feasibility.json`): todas as telas;
- `system.json` sozinho: só a tela Diagnóstico, com instrução para rodar
  `scout plan --json`.
Filtro por tarefa e toggle comercial no frontend são só filtros de exibição
sobre a ordem do solver; mudar `ctx` exige rodar a CLI de novo (ou modo LIVE).
Para que o arquivo sirva ao filtro, `scout plan --json` usa por padrão todas
as tarefas. Pergunta Q2.

### A2. Origin do GitHub Pages fora da allowlist
Default `http://localhost:5173` faz o LIVE falhar justamente no Pages.
**Proposta:** allowlist default = `http://localhost:5173`,
`http://127.0.0.1:5173` e a origem do Pages (`https://<usuario>.github.io`,
constante do pacote). `SCOUT_ALLOWED_ORIGIN` aceita lista separada por vírgula
e SUBSTITUI o default. Comparação exata de string, sem curinga. Nota: a origem
`https://<usuario>.github.io` é compartilhada por todos os repositórios Pages
do mesmo usuário — o token continua sendo a barreira real. Pergunta Q3.

### A3. Requisição sem header `Origin` (curl, scripts)
Navegadores sempre enviam `Origin` em fetch cross-origin e em POST.
**Proposta:** sem `Origin` é aceito se o token for válido (uso via curl).
Com `Origin` fora da lista: 403 `forbidden_origin` antes de qualquer trabalho,
inclusive no preflight.

### A4. DNS rebinding — não coberto
Página em `evil.example` re-resolvida para 127.0.0.1 faz requisições
same-origin, que podem vir sem `Origin`. **Proposta:** validar `Host` ∈
{`127.0.0.1:<porta>`, `localhost:<porta>`}; senão 403 `forbidden_host`.

### A5. `/health` sem token
Permite a qualquer página detectar que o scout está rodando (fingerprinting).
**Proposta:** aceitar; responde só `{"ok": true, "version": ...}`, com CORS da
allowlist. Documentar o risco.

### A6. Catálogo no `POST /plan` e em `GET /catalog`
**Proposta:** o servidor usa o próprio `data/catalog.json` instalado; o
cliente nunca envia catálogo. Body `{intent, system?}`: sem `system`, usa a
coleta ao vivo (cacheada desde o start, `GET /system?refresh=1` recoleta).
`GET /catalog` resumido serve só para listar/filtrar, nunca para cálculo — a
resposta de `/plan` já traz o recorte de cada modelo citado.
Shape proposto: `{schema_version, kind: "catalog_summary", models: [{id, source_url, tasks, license, commercial_ok, complete}]}` (schema a fechar na fase backend).

### A7. Limites do body e Content-Type
**Proposta:** body máximo 1 MiB (413 `payload_too_large`, checado por
`Content-Length` antes de ler; sem `Content-Length` → 411 com code `bad_request`). POST exige
`Content-Type: application/json` (415) — isso também garante preflight.

### A8. `POST /bench` executa processo
Riscos: execução arbitrária, caminho de arquivo vindo do cliente, CPU/GPU
saturados por qualquer página com o token, concorrência. Onde fica o arquivo
do modelo não está especificado. **Proposta v1:** `/bench` nunca executa;
devolve só `{command, ran: false, missing: [...]}`. Execução só pelo terminal
(`scout bench --run`). Se um dia executar: argv sem shell, `model_id` só do
catálogo, caminho resolvido pelo servidor, um processo por vez, timeout, e
flag explícita `scout serve --allow-bench`. Até lá, `bench_disabled` (501)
para pedidos de execução.

### A9. Token no histórico do navegador
Ver decisão do token. Mitigação: `replaceState` imediato + rotação por start.
O terminal deve imprimir também o token puro, para colar.

### A10. Porta
`--host` proibido, mas a porta 8765 pode estar ocupada. **Proposta:** `--port`
permitido; a porta vai no fragment (`#token=...&port=...`); falha alta se
ocupada, sem tentar outra porta silenciosamente.

### A11. Private Network Access / Local Network Access
`ARCHITECTURE.md` assume o preflight PNA do Chrome. Navegadores vêm mudando
esse mecanismo (permissão do usuário para acesso à rede local). **Proposta:**
manter os headers PNA e validar empiricamente na fase web/backend em Chrome,
Firefox e Safari; o modo ARQUIVO é o fallback documentado.

### A12. Logs
"Logs opcionais" sem regra. **Proposta:** só stderr, nunca arquivo; nunca
registrar `Authorization`, token, nem corpo de request/response.

### A13. Fixture de demo em `web/fixtures/`
**Proposta:** `web/fixtures/demo-feasibility.json`, derivado de
`schemas/examples/`, com `fixture.synthetic: true`; o frontend mostra faixa
"DADOS SINTÉTICOS" sempre que esse bloco existe. Nunca nomes de modelos reais.

### A14. CI → deploy
"O deploy depende do CI verde" com dois workflows separados não é automático.
**Proposta:** um workflow com jobs `test` → `deploy` (`needs: test`), ou
`workflow_run`. Decisão do `ops`.

### A15. Privacidade do `system.json`
**Decisão no schema:** sem hostname, usuário, serial, MAC ou caminhos de
binário (só presença no PATH). `additionalProperties: false` impede que
apareçam por acidente.

### Lacunas de `PROMPT.md` que afetam o contrato (para o `nucleo`)

- **A16. Origem de `bw_gpu`.** M2 não diz como obter a banda da GPU
  (`nvidia-smi --query-gpu` não a fornece diretamente). Sem ela,
  `tokens_per_s_est` sai null. Pergunta Q5.
- **A17. `ttft_s_est` sem fórmula.** Prefill é limitado por computação e não
  há FLOPS coletados. **Proposta:** null + `confidence_reasons:
  ["ttft_unspecified"]` até haver fórmula ou medição por bench.
- **A18. Tier do híbrido.** "Contra o orçamento do pool onde roda" é ambíguo
  com dois pools. **Proposta:** `by_pool` por pool; tier = pior dos dois.
  Lado GPU = camadas na GPU + KV inteiro + buffer; lado CPU = camadas restantes.
- **A19. `n_gpu` enche a VRAM a 100%.** A fórmula usa `vram_usable` inteiro,
  então todo híbrido cai em `inviavel` (>0,95). **Proposta:** usar
  `floor(0.95 * vram_usable)` no lugar de `vram_usable` (Q6).
- **A20. MoE + offload.** Não está dito como a fração ativa entra em `t_tok`
  com dois trechos. **Proposta:** aplicar `params_active/params_total` a
  `bytes_camada` nos dois trechos.
- **A21. Aritmética inteira.** `600e6`, `0.80 * orcamento`, `elem_bytes =
  0.5` são floats. **Proposta:** constantes inteiras (`600_000_000`),
  comparações `total * 100 <= 80 * orcamento`, KV em bits/8 no fim,
  `bytes_camada = weights // n_layers`.
- **A22. Pool unificado.** Não há fórmula de orçamento para `unified_memory`
  (quanto da RAM a GPU pode usar). Pergunta Q7.
- **A23. Campo de hardware null no solver.** Ex.: `vram_free` null → `vram_display`
  indefinido. **Proposta:** sem `vram_total`/`vram_free` a GPU não forma pool
  (aviso `hardware_field_missing`); sem `ram_total` o plano falha alto.
- **A24. Tarefas e arquiteturas não-decoder.** O mapeamento de M3 não diz como
  obter `tasks` (pipeline_tag?) nem cobre configs encoder-decoder (STT) ou
  VLM (projetor de visão, memória extra). Esses modelos tenderão a
  `complete:false`. Pergunta Q8.

---

# Respostas do humano — 2026-11-15

1. **Q1 — aprovado.** `jsonschema` só como ferramenta de dev (script de fase e
   teste com `importorskip`), nunca em runtime.
2. **Q2 — aprovado.** O modo ARQUIVO aceita `feasibility.json` (renderização
   pura); `system.json` sozinho mostra só o Diagnóstico. CLAUDE.md e
   ARCHITECTURE.md atualizados.
3. **Q3 — usuário GitHub `JkFa`.** Origem do Pages: `https://jkfa.github.io`
   (hostname de origem é minúsculo); app em `https://jkfa.github.io/localai-scout/`.
   Allowlist default do backend: `https://jkfa.github.io` e
   `http://localhost:5173`.
4. **Q4 — aprovado, provisório.** `commercial_ok` derivado de `license` por
   tabela revisada por humano; licença fora da tabela → `commercial_ok: null`
   → bloqueado sob `--commercial`, sem tornar o modelo incompleto. Revisitar.
5. **Q5 — da máquina local.** Sem tabela curada por nome de GPU. A banda da GPU
   vem da sonda local (clock × largura de barramento, quando o driver
   informar) ou da medição de `scout bench`; sem isso fica `null` e
   `tokens_per_s_est` sai null.
6. **Q6 — (a) 0,95.** `n_gpu = floor((0,95 * vram_usable - kv_gpu - buffer_gpu)
   / bytes_camada)`, limitado a [0, n_layers]. Mesmo corte de "limite" do
   resto da classificação; o offload resultante sai como "limite", não
   "inviavel". Registrado à parte: preferência por modelos de licença livre na
   semente do catálogo (Fase 3).
7. **Q7 — 50%.** No pool unificado o orçamento é 50% da RAM total.
8. **Q8 — aprovado.** STT/VLM/embeddings podem ficar `complete:false` até haver
   mapeamento próprio em M3.

## 2026-11-15 — formatos GPU-only rejeitados em memória unificada

**Contexto.** PROMPT.md exige residência total na VRAM para EXL2/AWQ/GPTQ, mas
não diz o que fazer no pool unificado (Apple Silicon), onde esses formatos não
têm backend.

**Decisão (humano).** No pool `unified` esses formatos NÃO contam como
residentes na GPU. Cada quant EXL2/AWQ/GPTQ é pulada com
`skipped_quants[].reason_code = "no_gpu_for_format"`; se o modelo ficar sem
candidato só por isso, o `rejected` usa `reason_code: "no_gpu_for_format"`,
`closest: null`, `missing_bytes: null`. Código já existente no enum do schema;
nenhuma mudança em `schemas/`.

**Consequência.** Modelo só com quants GPU-only nunca é recomendado em Mac,
mesmo que coubesse na memória unificada.

## 2026-11-15 — A17–A23 aprovadas

As propostas A17 a A23 (ttft null, tier do híbrido pelo pior pool, fração
ativa do MoE nos dois trechos, aritmética inteira, campo de hardware null)
foram aprovadas pelo humano como estão e valem para o M1.

## 2026-11-15 — memória apresentada em GiB, banda em GB/s decimal

**Decisão (humano, adendo de `schemas/m1_interface.md` item 1).** `fmt_gb`
passa a usar base binária: `f"{n_bytes / 2**30:.2f} GiB"`; o nome da função
fica. Banda continua decimal (`bytes/s / 1e9`, "GB/s"), convenção de
fabricante, por `fmt_gbps()` em `scout/estimate.py`.

**Consequência.** Internamente nada muda (bytes inteiros). O teste "todo
número do markdown existe no --json" precisa converter bytes por `fmt_gb` e
banda por `fmt_gbps`.

## 2026-11-15 — principal por tarefa

**Decisão (humano, adendo item 2).** `primary_for` de uma tarefa = o primeiro
candidato não bloqueado que tem a tarefa, na ordem final do solver (tier,
tokens/s desc com null por último, `model_id`), salvo a preferência de
qualidade abaixo.

## 2026-11-15 — preferência de qualidade abaixo de 7e9

**Decisão (humano, adendo item 3; PROMPT.md M1).** Se o principal escolhido
pela regra anterior tem `params_total < 7e9` e bpw < 5.5, e existe outro
candidato não bloqueado da mesma tarefa com `params_total` menor, bpw >= 6.5
e tier não pior, o principal passa a ser esse (o primeiro na ordem final). A
ordem da lista de candidatos não muda. O solver emite o aviso
`quality_preference` (novo código no enum de `warning.code` em
`schemas/feasibility.schema.json`) com `field: "candidates[i]"` apontando o
preterido; o markdown marca o preterido na coluna de nota.

## 2026-11-15 — catalog_stale fora do solver

**Decisão (humano, adendo item 4).** O solver não emite `catalog_stale`: ele
não lê relógio (determinismo). A checagem de `verified_at` > 60 dias fica
para `scout catalog check` (Fase 3). O código continua no enum do schema.

## 2026-11-15 — Fase 2: coleta (scout collect)

**Decisão.** Parsers puros em `scout/parsers.py`; execução, timeouts e montagem do
`system.json` em `scout/collect.py`. dmidecode nunca com sudo: sem root, tipo/MT/s/canais
ficam `null`/low. Banda de RAM: MT/s*canais*8 (medium) quando ambos detectados; senão
`data/bandwidth_tables.json` (DDR3..LPDDR5X, 2 canais de 64 bits, velocidade típica da
geração) com `confidence: low` e `source` começando em `estimate:table`. É dado de
REFERÊNCIA, não detectado; sem tipo de memória, fica `null`. Banda de GPU: sem tabela por
nome (Q5); nenhuma sonda local atual dá clock x largura do barramento, então fica `null`.
Híbrido Intel: P/E via `/sys/devices/cpu_core|cpu_atom/cpus` + `lscpu -p=CPU,CORE`;
`cores_performance` é a base de n_threads. Canais pelo dmidecode: locator de canal, ou nº de
módulos populados (low). Flash attention derivado de cc >= 8.0 (medium). `vram_free` é
medido na coleta (schema exige) e oscila entre execuções.

## 2026-11-15 — dmidecode por escolha do usuário

**Decisão.** O scout nunca invoca sudo. Tipo/MT/s/canais da RAM exigem root; o usuário escolhe:
`sudo scout collect` (processo já root) ou `sudo dmidecode -t memory > mem.txt` seguido de
`scout collect --dmidecode mem.txt`. O arquivo passa pelo mesmo parser; `source` = `dmidecode_file`.
Quando negado, a dica com os dois caminhos vai no probe `dmidecode` do system.json e no stderr.

## 2026-11-15 — embeddings usam ctx efetivo em vez de rejeitar

**Decisão (humano).** Quando a única tarefa avaliada de um modelo (interseção
de `model.tasks` com `intent.tasks`) é `embeddings`, o solver usa
`ctx = min(intent.ctx, ctx_max)` no KV e no buffer, em vez de rejeitar com
`ctx_exceeds_max` (texto longo é fatiado em ctx_max). O candidato leva o campo
opcional `ctx_effective` (novo em `schemas/feasibility.schema.json`). Modelo
avaliado também para outra tarefa, ou qualquer outra tarefa: comportamento
anterior (rejeita).

**Consequência.** Testes antigos que esperavam `ctx_exceeds_max` ou KV com
`intent.ctx` para o modelo de embeddings precisam do ctx efetivo.

## 2026-11-15 — tarefa code por override revisado

**Decisão (humano).** A API não expressa "code" (sem pipeline_tag próprio).
`data/task_overrides.json` (`reviewed: true`) lista, por repo, `add_tasks`,
`reason` e `source` (URL do cartão). `scout catalog add` aplica o override
depois de derivar `tasks` da API. Override só ACRESCENTA tarefas: nunca
remove, nunca toca outro campo, e não se aplica se `tasks` veio null (não
preenche campo incompleto). A origem fica registrada só aqui e no stderr do
`add`: `catalog.schema.json` não tem campo de proveniência de tarefas e não
foi alterado. Entrada única: `Qwen/Qwen2.5-Coder-7B-Instruct` → `code`.

## 2026-11-15 — fórmula de params_active do MoE aprovada

**Decisão (humano).** `params_active = safetensors.total − (n_experts −
num_experts_per_tok) × camadas_MoE × 3 × hidden_size × ffn_expert`, com
`ffn_expert = moe_intermediate_size` (ou `intermediate_size` no estilo
Mixtral), camadas_MoE descontando `first_k_dense_replace` e
`mlp_only_layers`, exigindo ativação com porta (silu/swish/swiglu). Qualquer
dado faltando, `decoder_sparse_step`/`moe_layer_freq` ≠ 1 ou resultado fora de
[1, total) → null e modelo incompleto. Implementada em
`scout/catalog.py:moe_inactive_params`.

## 2026-11-15 — license_policy revisada

**Decisão (humano).** `data/license_policy.json` aprovado e marcado
`reviewed: true`: apache-2.0, mit, bsd-2-clause, bsd-3-clause, cc-by-4.0 →
true; cc-by-nc-4.0, cc-by-nc-sa-4.0, cc-by-nc-nd-4.0 → false; resto → null.

## 2026-11-15 — scout catalog remove; SmolLM2 fora do catálogo

**Decisão (humano).** Novo comando `scout catalog remove <id> [--catalog F]`,
sai != 0 se o id não existir. Usado para remover
`HuggingFaceTB/SmolLM2-1.7B-Instruct` de `data/catalog.json`.

## 2026-11-15 — teto de bpw no catálogo sobe para 40

**Contexto.** F32 com metadados passa de 32 bpw efetivo (nomic-embed F32 =
32,04) e era anulado pelo teto do schema.

**Decisão (humano).** `catalog.schema.json`: `quant.bpw.maximum` 32 → 40.
`gguf_quants` só anula bpw fora de (0, 40], com aviso no `catalog add`.

## 2026-11-15 — Fase 4: detalhes do backend não cobertos por A1–A15

**Decisão (agente `backend`).**
- `scout serve --port` e `--catalog` (padrão `data/catalog.json`); sem `--host`. Porta ocupada falha com
  erro, sem tentar outra. `ScoutServer(port, token, catalog_path, origins, collect_fn)` aceita porta 0 e
  coleta injetável só para testes; bind sempre 127.0.0.1.
- Ordem de validação: Host (403) → Origin (403) → OPTIONS responde → token (401, inclusive para rota
  inexistente, para não permitir sondar rotas) → 404/405 → Content-Type (415) → Content-Length (411/413) → JSON.
- Requisição sem Origin passa (A3); respostas de erro também levam o CORS exato quando o Origin é permitido,
  para o navegador conseguir ler o erro. OPTIONS dispensa token (preflight não carrega credencial).
- Coleta ao vivo é preguiçosa: roda na primeira chamada a /system ou /plan sem `system`, fica em cache;
  `/system?refresh=1` recoleta. (A6 dizia "desde o start"; evita travar a abertura do servidor.)
- `/plan`: `intent` exigido, só campos tasks/ctx/commercial_use/batch/kv_quant (desconhecido → 400);
  `tasks` ausente = todas as do catálogo (como a CLI); `system` do cliente precisa ser kind=system,
  schema_version=1, senão 400 `schema_invalid`. Se o solver falhar com KeyError/TypeError num `system` do
  cliente, vira 400; na coleta ao vivo vira 500 `internal`.
- `/bench`: `{model_id}` precisa estar no catálogo (404 senão); resposta
  `{command: "scout bench --run <id>", ran: false, missing: ["model_file_path","runtime"]}`; `run: true` → 501.
- Sem Content-Length em POST → 411 com code `bad_request` (como A7). Helpers `load_tuning` e `run_plan`
  extraídos de `scout/cli.py` (plan inalterado).
- Log: stderr, uma linha `METHOD path -> status`, sem query, sem cabeçalhos, sem corpo.

## 2026-11-15 — Fase 5: decisões do frontend não cobertas pelas fontes

**Decisão (agente `web`).**
- CSP final no `<meta>`: `default-src 'self'; connect-src 'self' http://127.0.0.1:* http://localhost:*; img-src 'self' data:;
  style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'` (sem inline script/style) e
  `referrer: no-referrer`.
- Três fixtures geradas pela CLI (não escritas à mão): `demo-feasibility.json` (ctx 8192), `demo-feasibility-rejeitados.json`
  (mesmos insumos, `--ctx 40000`, para exercitar a tela Rejeitados) e `demo-system.json` (cópia de `sys_rtx3060.json`).
  A fixture de rejeitados só cobre `ctx_exceeds_max`; o caminho `does_not_fit`/`missing_bytes` não tem dado real de demo.
- Tabela de recomendações agrupa por tarefa de `inputs.intent.tasks`; um modelo com duas tarefas aparece nas duas, com chip
  "principal" onde `primary_for` o inclui. Toggle comercial esconde `blocked` ou `commercial_ok !== true`, só lendo o JSON.
- Proveniência de `quant.file_bytes` e da licença não está nos mapas `provenance`; a UI as rotula "catálogo" (origem real).
- `scout bench` não existe na CLI desta versão; o "Próximo passo" mostra `scout bench --run <id>` (contrato do `/bench`) e
  avisa na tela que ainda não executa.
- LIVE: `POST /plan` sem `tasks` (todas), `commercial_use: false` (o toggle é filtro de exibição), ctx editável; fluxo
  /health → /system → /plan; 401 limpa o token em memória. A porta do fragment tem precedência sobre o campo.
- Um 403 de origem não permitida chega ao navegador sem CORS e é indistinguível de "servidor fora do ar"; a mensagem de erro
  lista as três causas (servidor parado, PNA/Safari, origem fora da allowlist).

## 2026-11-15 — desempate por bytes da RAM quando tokens/s é null (F4)

**Contexto.** Na Fase 1, `_choose` caía para a ordem fixa gpu_full > hybrid >
cpu_only quando algum tokens/s era null. Era um desvio do PROMPT.md ("escolher
o de maior tokens_per_s") que só foi relatado no resumo do agente, nunca
registrado aqui. Na ordenação, candidatos com tokens/s null eram desempatados
por `model_id`, critério sem relação com velocidade (achados 4 e 5 da
revisão).

**Decisão (humano, `schemas/m7_fixes.md` F4).** Substitui aquele desvio:
- `_choose`: com algum tokens/s null, vence o placement de menor
  `bytes_cpu_por_token = bytes_camada_lidos × (n_layers − n_gpu)` (MoE com a
  fração ativa, A20); empate pelo menor `bytes_por_token` total; depois
  gpu_full, hybrid, cpu_only.
- Ordem dos candidatos e `primary_for`: (tier, tokens/s desc com null por
  último, bytes_cpu_por_token asc, bytes_por_token asc, model_id asc).
- Os bytes são só critério de ordem, nunca exibidos. Candidato com tokens/s
  null recebe `ranked_by_bytes` em `confidence_reasons` (novo no enum).

## 2026-11-15 — limiar de bpw para offload (Fa)

**Decisão (humano, m7_fixes.md Fa).** `data/tuning.json` ganha
`max_bpw_offload` (inicial 8.5), lido pela CLI e passado em `tuning`. Quant
GGUF com bpw > limiar não tem o placement hybrid avaliado (só quando haveria
GPU dedicada e CPU); gpu_full e cpu_only continuam. Se a quant não couber em
nenhum dos dois, sai em `skipped_quants` com `bpw_above_offload_limit` (novo
no enum) em vez de `does_not_fit`. Se ela couber em gpu_full ou cpu_only, vira
candidata normalmente e nada é registrado. Sem a chave em `tuning`, não há
limite. O solver ecoa o valor em `inputs.tuning.max_bpw_offload` (opcional no
schema). PROMPT.md M1 atualizado no mesmo commit.

## 2026-11-15 — marcas de estimativa no markdown (F6)

**Decisão (humano).** No markdown de `scout plan`, número com proveniência
`estimated` leva prefixo `~` (tokens/s sempre); memória e tokens/s de
candidato com confiança low levam `⚠`. Legenda no topo, sem números. O JSON
não muda.

## 2026-11-15 — proveniência nos rejeitados (F7)

**Decisão (humano).** `rejected[].provenance` opcional no schema. O solver o
emite quando há `closest`: `missing_bytes` = `estimated` se a tentativa mais
próxima usou pesos por fallback (sem `file_bytes`), senão `calculated`.
Ausente (rejeição antes do cálculo) = `calculated` para o frontend.

## 2026-11-15 — flag --tuning em plan e serve

**Decisão (humano, m7_fixes.md F8).** `scout plan` e `scout serve` aceitam
`--tuning F` (padrão `data/tuning.json`), para que os testes usem
`tests/fixtures/tuning_test.json` e nunca leiam `data/`.
