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

## 2026-11-15 — Fase 7 (coletor): F2, F3, F4b

- **F2.** Sem `/sys/devices/cpu_core|cpu_atom`: `lscpu -p=CPU,CORE,MAXMHZ`; dois grupos de MAXMHZ separados por >15% =
  híbrida (maior = performance, medium). Frequência uniforme = não híbrida (medium). Sem MAXMHZ/inconclusivo e Intel:
  `cores_performance` = físicos com low (`hybrid_undetermined`) e `cores_efficiency` null. Nunca high sem sysfs.
- **F3.** Chave de canal = (controlador, canal). Canais por contagem de módulos = low, e a banda derivada nunca excede
  a menor confiança entre MT/s e canais (teto medium).
- **F4b (revisa a resposta Q5).** Banda de GPU pode vir de `data/gpu_bandwidth.json` (`reviewed: false`), só com casamento
  único por substring e `source_url` em nvidia.com/amd.com/intel.com (o código ignora entradas fora disso). Resultado:
  `source: "table"`, medium; duplo casamento = `ambiguous_table_match`, null. A tabela foi entregue VAZIA: as páginas
  oficiais conferidas hoje (nvidia.com: RTX 3060/3060 Ti, 4060/4060 Ti, 4070, 4090; amd.com: RX 7800 XT, ir.amd.com e
  rocm.docs.amd.com) trazem largura de barramento, mas nenhuma traz GB/s (NVIDIA) ou não carregou/não traz o número (AMD).
  Entradas ficam para o humano com fonte oficial que traga o número.

## 2026-11-15 — Fase 7 rodada 2 (coletor): R2

`source` de `cores_performance`/`cores_efficiency` diz o método: `sysfs:cpu_core/cpu_atom` (high),
`lscpu:maxmhz_grouping (separação X%)` (medium; X = 1 - f_lento/f_rápido, medido; limiar 15% aprovado) ou
`undetermined` (low). **Limitação conhecida, não implementada:** em Intel híbrido os E-cores compartilham L2 em
grupos de quatro e os P-cores têm L2 próprio; esse sinal estrutural está em
`/sys/devices/system/cpu/cpu*/cache/` e sobrevive em container com mais frequência que cpufreq. Candidato a
terceiro método de detecção.

## 2026-11-15 — limiar de offload nunca silencioso (R3) e PROMPT.md como regra (R5)

**Contexto.** Na rodada 1, o limiar `max_bpw_offload` só deixava rastro quando
a quant não cabia em nenhum placement; se o modelo entrava por outro placement
ou outra quant, o descarte do hybrid sumia. O humano recusou essa
interpretação.

**Decisão (humano, m7_fixes.md R3/R5).**
- O hybrid acima do limiar continua sendo CALCULADO, só para saber se seria
  viável; nunca entra na escolha. Se seria viável e o modelo entra mesmo
  assim, o candidato vencedor leva `notes: [{"code":
  "offload_skipped_above_bpw", "message": ...}]` (opcional no schema), uma por
  quant descartada, na ordem de avaliação. A mensagem diz a quant
  descartada, o limiar (do tuning, com ponto decimal como o resto do
  relatório) e onde o vencedor roda. Nenhuma entrada em `rejected`.
- Se a quant não coube em gpu_full nem cpu_only e o hybrid seria viável:
  `skipped_quants` com `bpw_above_offload_limit` (e a nota vai para o
  vencedor). Se o hybrid não seria viável, o limiar não mudou nada:
  `does_not_fit`, sem nota.
- O markdown da CLI mostra as notes na coluna de nota do candidato.
- PROMPT.md M1 agora contém, no algoritmo de candidatos, o desempate F4, o
  limiar Fa e a nota R3, e a fórmula de `n_gpu` com o teto 0,95 (Q6).

## 2026-11-15 — R4 etapa 2 (coletor): banda de GPU

- `data/gpu_bandwidth.json` removido (e o código que o lia). A seção `gpu` de `data/bandwidth_tables.json` é chaveada por
  `nome exato|VRAM total em bytes` (a RTX 3060 de 8 GB, 128-bit, não casa com a de 12 GB). Casou: `source: "table"`, high.
- Única entrada: RTX 3060 12 GB, 192 bit, 15 Gbps, 360 GB/s. Fontes: nvidia.com/en-us/geforce/news/game-on-you-asked-we-answered-qa/
  ("192-bit and GDDR6 (at 15 Gbps), delivering 360GB/s") e nvidianews.nvidia.com (192-bit, 12GB GDDR6), verificadas em 2026-11-15.
  **Pendência:** `smi_name_verified: false`. "NVIDIA GeForce RTX 3060" é o nome candidato e a VRAM 12288 MiB (12884901888 bytes) é
  suposta; a entrada pode NÃO casar na coleta real até alguém confirmar com `nvidia-smi --query-gpu=name,memory.total`.
- RX 7800 XT fora: amd.com não carregou (timeouts); a página oficial carregada (ir.amd.com) traz 256-bit e 16 GB, sem
  Gbps nem GB/s. Os 624 GB/s só vieram de resumo de busca, não de página citável.
- Fallback por fórmula DESLIGADO: nenhum multiplicador gravado; sem casamento, banda null/low e o desempate por bytes (F4) assume.
  **Pendência:** validar `clocks.max.memory` do nvidia-smi contra os 360 GB/s da 3060 e o equivalente do rocm-smi.
- iGPU: `bandwidth_bytes_per_s` = banda da RAM do sistema, mesma confiança, `source: "igpu_shares_ram"`; RAM null => null.
  Detecção: AMD com um único cartão em /sys/class/drm sem `mem_info_vram_vendor` (amdgpu só o expõe em GPU discreta; na
  Mendocino desta máquina o arquivo não existe). Heurística: com mais de um cartão AMD não classifica (trata como discreta).

## 2026-11-15 — Banda de RAM com Data Width e energia (coletor)

- **Banda de RAM.** `MT/s * canais * data_width_bits / 8`. Usa "Data Width" do dmidecode (nunca "Total Width", que inclui
  ECC). Largura ausente, fora de {32, 64} ou diferente entre módulos: assume 64 e a confiança cai a low. Canais por contagem
  de módulos continuam low. Tipo não entra na fórmula; divergência vai em `source`/`note` do campo `type`, valor mantido, e só existe para
  canal de 32 bits (Data Width 32) com tipo DDR3/DDR4 (observação dos dados coletados, sem números). Sem limites de MT/s por
  tipo: seria constante de memória sem fonte e daria falso positivo em DDR4 com XMP. Caso real: 2 canais de 32 bits a 5500 MT/s = 44 GB/s
  (a máquina do humano reporta "DDR4", padrão de LPDDR5). PROMPT.md M2 atualizado. `mem.txt` da raiz não vai a tests/ (tem seriais).
- **Energia.** Causa do defeito: `parse_power` nunca lia `online` dos supplies Mains/USB; concluía "na tomada" do status da
  bateria ("Full", "Not charging", "Charging"), que firmware/limite de carga reportam mesmo na bateria. Agora: qualquer
  Mains/USB* `online=1` => AC; todos `online=0` => bateria (contradição com bateria `Charging` => null); sem Mains legível,
  só `Discharging` prova bateria, o resto é null. Baterias com `scope=Device` (periféricos) são ignoradas. Lido a cada coleta,
  sem cache. Não reproduzi o defeito original (esta máquina estava na tomada); a causa vem da análise do código.

## 2026-11-15 — embeddings sem decode e aviso de prefill lento

**Contexto.** Validação em hardware real: tokens/s por banda (decode) foi
exibido para modelos de embeddings, que não geram texto; e o prefill em CPU,
que o solver não estima, domina a espera do usuário (medido 11,7 tok/s de
prefill num Ryzen 5 7520U com Qwen3-4B Q4_K_M).

**Decisão (humano).**
- Candidato cuja única tarefa avaliada é embeddings: `tokens_per_s_est`
  null, nota `decode_not_applicable` (métrica aplicável = throughput de
  prefill, não estimado) e `confidence_reasons` `decode_not_applicable` no
  lugar de `bandwidth_unknown`/`ranked_by_bytes`/`eta_default`/
  `bandwidth_estimated` (nenhum deles afeta número desse candidato; com isso
  a confiança fica high salvo fallback de pesos ou hardware low). Placement e
  ordem pelo desempate por bytes (F4). Markdown mostra "—".
- Aviso `slow_prefill_cpu` (warn) quando algum candidato não-embeddings roda
  em cpu_only ou em hybrid com maioria das camadas na CPU (escolha do
  `nucleo`: maioria = n_layers − n_gpu > n_gpu). Candidatos só de embeddings
  não disparam: a referência é de um LLM e mediria outra coisa.
- A ordem de grandeza vem de `data/tuning.json` → `prefill_reference`
  (medição, com máquina, ferramenta, data e energia), ecoada em
  `inputs.tuning`. A mensagem deriva "~N tok/s" e "prompt de M tokens ≈ um
  minuto" (M = tok/s × 60, 2 algarismos significativos) do valor; sem o
  bloco, sai sem número. `ttft_s_est` continua null.

## 2026-11-15 — scout bench: banda efetiva + overhead em vez de eta (coletor)

**Decisão.** A calibração deixou de ser `eta_novo = eta * medido/estimado` e passou a ser `t = bytes/banda_efetiva + overhead`
por pool, gravada em `data/tuning.json` (`calibration.<pool>`). `eta_gpu`/`eta_cpu` nunca são alterados pelo bench.
**Evidência (mesma máquina, Ryzen 5 7520U, Qwen3-4B Q4_K_M, banda teórica 44 GB/s):** na bateria 4,46 tok/s = 11,1 GB/s
efetivos; na tomada com perfil performance 7,94 tok/s = 19,8 GB/s. Um eta global teria de valer 0,25 numa condição e 0,45
noutra: a eficiência não é propriedade da máquina, depende do estado de energia e do modelo. Banda efetiva medida, rotulada
com o estado de energia, é um fato; eta calibrado em cima de um chute é um chute com cara de dado. O overhead fixo separa o
que escala com os bytes do que não escala (2 modelos de tamanhos diferentes); se for < 5% do tempo do menor modelo ou
negativo, volta a 1 parâmetro e a forma usada e os dois valores ficam registrados.
**Estabilização.** A 1ª rodada após o carregamento subestima (frequência e caches); 1 aquecimento descartado e rodadas
até as duas últimas ficarem a até 5% (decode e prefill separados), teto de 6; sem par estável, `stable: false`,
confiança rebaixada e prefill null. Todas as rodadas ficam no measured.json.
**Escolhas de implementação (não estavam no enunciado).** `--gguf` pareia com `--model` para o llama-bench (sem caminho de
arquivo no catálogo). Banda 1p ponderada pelos bytes. `hybrid` é recusado (tempo não atribuível a um pool só) e embeddings
também (sem decode). A arquitetura vem de `general.architecture` do Ollama; com llama-bench fica null (o catálogo não tem o
campo). A única rede do bench é a API local do Ollama em 127.0.0.1; `hf.py` continua o único módulo que fala com a internet.

## 2026-11-15 — consumo da calibração no plan (Tarefa 4)

**Decisão (humano, `schemas/bench_spec.md`, parte do `nucleo`).**
- Trecho de pool com `tuning.calibration.<pool>` usa `bytes/banda +
  overhead` em vez de `bytes/(bw × eta)`; sem calibração, nada muda. eta
  nunca é alterado. Candidato ganha `tokens_per_s_basis`
  (calibration/eta/mixed); só `calibration` dá proveniência `calculated`.
- Confiança: trecho por eta → no máximo medium (`eta_default` passa a sair
  sempre que um trecho usa eta, não só com `source: default`). Trecho
  calibrado: `stable: false` → medium; energia atual desconhecida → medium
  (`calibration_power_unknown`); energia diferente → aviso
  `calibration_power_mismatch` e um nível abaixo; modelo fora dos medidos e de
  arquitetura que não casa → medium (`calibration_other_arch`).
- Escolhas do `nucleo` dentro do enunciado: (a) híbrido com os dois pools
  calibrados soma o overhead inteiro de cada trecho (leitura literal; overhead
  proporcional às camadas seria mais físico, fica para o humano); (b) o
  próprio modelo medido conta como mesma arquitetura mesmo com
  `architecture` null; (c) `ttft_s_est` só com `--prompt-tokens`, placement
  de um pool, `prefill_tokens_per_s` presente e calibração medida com UM
  modelo que é o candidato — o prefill de outro modelo não se transfere;
  (d) energia atual lida pela CLI com `scout.bench.read_power` só quando há
  calibração, para que a saída sem calibração não dependa do estado da
  máquina; (e) calibração malformada (banda não inteira > 0, overhead < 0)
  faz o plan falhar alto.

## 2026-11-15 — arquitetura no catálogo e comparação normalizada

**Decisão (humano, adendo da Tarefa 4).** Catálogo ganha `architecture`
(opcional, nullable) = `model_type` do config.json (VLM:
`text_config.model_type`). Dado de API que não torna o modelo incompleto.
O bench grava `general.architecture` do GGUF/Ollama ("qwen3moe") e o HF usa
`model_type` ("qwen3_moe"): a comparação é feita em minúsculas, sem "_" e
"-". Arquitetura null em qualquer lado = diferente (não sobe para high).

## 2026-11-15 — calibração local com identidade da máquina, fora do git (Tarefa 5)

**Contexto.** `scout bench --run` gravava `calibration.<pool>` em `data/tuning.json`, que é versionado. O bloco não
tinha identidade, e o solver o aplicava a qualquer system.json. Commitado, ele daria a todo usuário a banda do Ryzen 5
7520U do autor (21,3 GB/s) com `high`. Regravar também reserializava o arquivo inteiro.
**Decisão (arquiteto em duas rodadas; respostas do humano a P1, P2, PE, PE.1 e PK; rev. 3 fechada pela sessão
principal; `schemas/calibration_store_spec.md`, `calibration.schema.json`).**
- `data/tuning.json` guarda só constantes e nunca é gravado. A calibração vai para `calibration.json` local, no
  diretório de estado do usuário: `$XDG_STATE_HOME` ou `~/.local/state`, `~/Library/Application Support`,
  `%LOCALAPPDATA%`. Precedência: `--calibration` > `SCOUT_CALIBRATION` > padrão. `--no-calibration` desliga a leitura.
  STATE e Local, e não CONFIG nem Roaming, porque o dado não é portável e esses diretórios são sincronizados.
- Identidade de HARDWARE decide; PILHA DE SOFTWARE só modula a confiança.
  - Camada 1 (decide; diferiu ou null → descarta): cpu.model, cpu.cores_physical, memory.ram_total_bytes (±5%),
    platform e os.arch, mais gpu.name, gpu.backend e gpu.vram_total_bytes (±5%), estes só para o pool gpu.
  - Camada 2 (corrobora; só pool cpu): MT/s e canais. Conhecidos e diferentes → descarta. Desconhecidos = sem
    evidência → medium, com aviso que explica como liberar `high` (dmidecode legível).
  - Pilha de software: kernel por `major.minor` e versão do motor (lida pela mesma `engine_version()` no bench e no
    plan; a CLI lê em loopback, o solver compara). Diferente → um nível abaixo, com sugestão de recalibrar. Não lido
    → medium. Nunca descarta: invalidar por kernel ou motor força um bench que a maioria não roda, e o usuário cai
    no eta.
  - `memory.unified` não entra: é redundante com o id do pool.
- `calibration_schema_version` (significado dos números): diferente → descarta. `scout_version` nunca é comparado.
- Energia (PE): uma entrada por estado em cada pool. Casamento exato primeiro; sem ele, a MENOR medição do pool, um
  nível abaixo, com a condição por extenso ("calibrado na tomada, você está na bateria"). O critério é a direção do
  erro, não o tamanho: superestimar custa o download e a credibilidade; subestimar custa, no pior caso, pular um
  modelo que daria certo. Nenhuma medição desta máquina perde para o eta global, que só entra quando o pool não tem
  medição que valha.
- Dois tipos de ausência (PE.1): desconhecido (existe e não foi lido → não decide, limita a medium) e não aplicável
  (não existe nesta classe de máquina → não decide e não limita). `power` ganha `not_applicable`. No linux, sem
  `BAT*` → `on_ac` não se aplica; sem `platform_profile` e sem `powerprofilesctl` → `profile` não se aplica. O
  desktop chega a `high` sem regra especial. Plataforma em que o coletor ainda não lê o atributo é desconhecido, não
  "não aplicável".
- O feasibility.json ganha `inputs.calibration` obrigatório (origem, status, casamento campo a campo, motor, energia,
  entrada usada), sem caminho. Sai `inputs.tuning.calibration`; `schema_version` fica 1 (o eco antigo nunca foi
  publicado).
- O bench recusa antes de medir se a identidade de hardware estiver incompleta, se o system.json não for desta
  máquina ou se o ajuste misturar ferramentas. A escrita é atômica e determinística. Tuning com `calibration` → o
  plan falha alto.
- O exemplo do schema e as fixtures de calibração usam máquina fictícia e números redondos. Medição real fica só
  aqui, como evidência: exemplo carrega impressão digital de hardware e número real em exemplo vira semente.
**Evidência (mesma máquina, Qwen3-4B Q4_K_M, banda teórica 44 GB/s, eta daria 26,4 GB/s).** Bateria: 11,1 GB/s.
Tomada/performance: 19,8 GB/s à mão e 21,3 GB/s pelo bench (8,52 tok/s). Com swap ativo e 316 MB disponíveis, o
bench mediu 15,0 GB/s e marcou `stable`: pressão de memória ainda não é conferida (tarefa futura do `coletor`).
**Alternativas descartadas.**
1. Pôr `data/tuning.json` no .gitignore: perde as constantes versionadas.
2. Arquivo dentro do checkout: não existe numa instalação via pip.
3. Hash opaco: não explica por que a calibração caiu.
4. Migrar preenchendo a identidade com o sistema atual: inventa dado (regra 3).
5. Toda divergência só rebaixar: banda de outra RAM é um número errado, não um número incerto.
6. Comparar `scout_version`: obrigaria recalibrar a cada release, sem que os números tenham mudado de significado.
7. Expirar por idade: exige relógio no solver e quebra a reprodutibilidade (descartada pelo humano).
8. Kernel exato na camada 1: derruba a calibração a cada atualização do Ubuntu, quase semanal.
9. Sem medição no estado de energia atual, voltar ao eta: erra +138% na bateria desta máquina, e para cima.
10. Regra especial "profile null dos dois lados = igual": resolvida pela distinção desconhecido × não aplicável.
**Consequência aceita.**
- Trocar CPU, RAM, GPU ou reserva da iGPU invalida a calibração até um novo bench.
- Sem dmidecode legível, o pool cpu não passa de medium, e a saída diz como resolver.
- Com o motor parado no momento do plan, a calibração não passa de medium, e a saída do plan passa a depender do
  motor estar de pé.
- Fora do estado de energia medido, o plan subestima de propósito (na tomada desta máquina, −44% se só houver a
  medição da bateria).
- O pool gpu não confere a configuração de RAM, nem numa iGPU.
- Um bench rodado durante o `serve` só vale depois de reiniciá-lo.

## Tarefa 5, parte do `coletor`: armazenamento, energia e motor (sonnet/medium)
- `llama-bench --version`: NÃO verificado. `llama-bench` não está no PATH desta máquina (`which` vazio), então o
  formato real e a existência da opção são desconhecidos. `engine_version("llama-bench")` aceita a primeira linha não
  vazia de stdout ou, se vazio, de stderr (llama.cpp costuma imprimir versão em stderr), e devolve None se a opção não
  existir (código ≠ 0). Conferir com o binário real antes de confiar na comparação de versão.
- Layer 2 desconhecido (MT/s ou canais null) NÃO derruba entradas antigas no merge de §6, ao contrário da letra de §6
  ("different ou unknown"). Motivo: o plan aplica esse caso em medium; derrubar faria uma máquina sem dmidecode
  legível perder o estado de energia anterior a cada bench, sem nunca acumular entradas. Camada 1 null ou diferente e
  camada 2 diferente continuam derrubando. Revisar se o humano preferir a letra.
- Chave de energia no merge: igual pela comparação de §4 OU estados normalizados idênticos (desconhecido nos dois
  lados também substitui), para a rerodada não duplicar entradas quando a leitura de energia falha igual.
- `read_power` no linux: `/sys/class/power_supply` ilegível é desconhecido, não "sem bateria". Darwin só marca
  `on_ac` não aplicável quando o pmset diz que não há bateria; `profile` fora do linux é desconhecido.
- `build_tool` do bench: `tool_version` vem de `engine_version` (uma chamada após medir), e não mais do JSON do
  Ollama ou do llama-bench. `ollama_check_quant` devolve só a arquitetura.
- Bench lê o arquivo existente antes de medir (malformado = erro antes de gastar tempo) e usa a origem `default` na
  leitura, para `--calibration` apontar para um arquivo novo sem erro.
- `conditions` no measured.json (só diagnóstico): três medições do mesmo estado declarado (tomada/performance) deram
  7,94, 8,52 e 9,95 tok/s, com o prefill subindo junto. Gravamos governor/EPP, frequência, temperatura, loadavg, memória
  e o delta de swap por rodada (before/after de cada modelo) para testar as hipóteses de frequência, calor e carga de
  fundo. Nada disso entra no calibration.json, na comparação, na estabilidade nem na confiança.

## 2026-11-15 — referência do critério de pronto da calibração passa a ser a medição reproduzível

**Decisão (humano).** O critério de pronto da calibração (bench_spec.md e calibration_store_spec.md) passa a comparar
com 9,9 tok/s de decode / 24,8 GB/s efetivos (Qwen3-4B Q4_K_M, Ryzen 5 7520U, tomada, perfil performance, governor
performance, EPP performance, Ollama 0.17.6). A medição à mão (7,94 tok/s, 19,8 GB/s) fica como histórico.
**Evidência.** Três benches no mesmo estado, com condições gravadas no measured.json: 9,95 (18:20), 9,87 e 9,85
tok/s, prefill 20,2–20,4. As remedições chegaram a 96 °C e à mesma velocidade; troca de swap de dezenas de MB nas
rodadas também não mudou o resultado. Medições anteriores mais lentas (7,94 à mão, 8,52 às 14:25) não registravam
condições, e a causa não pode ser recuperada; a suspeita mais forte é pressão de memória (às 14:25 a máquina saía de
7,5 GB em swap). Pressão pesada derruba o decode: 5,99 tok/s com 316 MB livres e swap ativo.
**Consequência.** Qualquer divergência futura deste tamanho fica diagnosticável pelo bloco `conditions`.

## 2026-11-15 — correções da revisão da Tarefa 5 (rev. 4 do enunciado)

**Decisão (humano, sobre o relatório do `revisor`).**
- Achado 1 (grave): kernel e camada 2 saem de `machine` e vão para `measured_with` em cada entrada, nunca reescrito
  por benches posteriores. O plan compara camada 2 e kernel contra a entrada escolhida. Antes, um segundo bench
  reescrevia o contexto das entradas antigas e uma medição de outro kernel, com RAM nunca verificada, saía `high`.
  Alternativa descartada: apagar entradas antigas quando kernel ou MT/s mudam (joga fora medição boa).
- Achado 2 (grave): bateria do sistema decidida por `type == Battery` e `scope != Device`, não pelo nome `BAT*`
  (Asahi e ARM usam `macsmc-battery`; antes viravam "sem bateria" e davam `high` na bateria com a banda da tomada).
- Achado 3 (médio): a leitura do Ollama ignora proxy do ambiente (o urllib mandava 127.0.0.1 ao `http_proxy`).
- Achado 4: versão do motor lida só para ferramentas das entradas escolhidas (`chosen_tools`).
- Achado 5: bytes com unidade na tabela de divergências do frontend. Achado 6: kernel real trocado por sintético.
- Achado 8: fora do casamento exato de energia, o TTFT usa o menor prefill entre as entradas elegíveis.
- Achado 7 NÃO aceito, registrado como limite conhecido: num desktop sem `platform_profile` e sem
  `powerprofilesctl`, mas com TLP, tuned ou governor trocado à mão, o perfil é "não aplicável" dos dois lados e a
  troca não é detectada. Tratar cpufreq/EPP como evidência de perfil tornaria o perfil desconhecido em quase toda
  máquina Linux e prenderia todo desktop em `medium`, desfazendo a decisão PE.1.

## 2026-11-15 — achados menores da segunda revisão da Tarefa 5 (corrigidos pela sessão principal)

**Decisão (humano: opção 1, corrigir inline em vez de abrir terceira rodada com `nucleo`/`coletor`).**
- (a) Bytes de campos de identidade aparecem via `fmt_gb` nas mensagens do solver e no stderr do merge
  (`calibration.fmt_field_value`), conforme a regra 4. A linha de calibração do bench usa `fmt_gbps`.
- (b) Num pool não aplicado, as linhas `match` do pool vão com efeito `none` no eco, e o frontend não mostra a
  correspondência de energia: kernel e camada 2 não modulam um pool que caiu no eta.
- C. O aviso `slow_prefill_cpu` usa o menor prefill elegível fora do casamento exato de energia, como o TTFT.
- D. O bench grava em `measured_with.os.kernel` o kernel lido ao vivo, não o do `--system`, e avisa se diferem.

## 2026-11-15 — ajuste de dois parâmetros adiado: falta modelo pequeno no catálogo

**Decisão (humano).** A calibração desta máquina sai na forma de um parâmetro (banda efetiva, sem overhead), medida
só com o Qwen3-4B Q4_K_M. O segundo caso de sempre, Qwen3.5 0.8B Q8_0 (`qwen3.5:0.8b` no Ollama), não está em
`data/catalog.json`, e o bench só aceita ids do catálogo (regra 2). O bloqueio é uma entrada de catálogo, não uma
limitação de desenho: com dois tamanhos, o ajuste `t = bytes/banda + overhead` separa o custo fixo por token e a
calibração melhora de verdade.
**Acompanhamento (aberto).** Incluir um modelo pequeno no catálogo com `scout catalog add` (consulta o Hugging Face,
altera `data/catalog.json`; commit próprio, fora da Tarefa 5) e refazer o bench com os dois modelos.

## 2026-11-15 — o tier "limite" previu a recusa de carga do Ollama

**Achado.** O plan classificou o Qwen3-4B Q4_K_M (ctx 8192) como tier `limite` nesta máquina (7,0 GiB de RAM). Às
14:23, com o swap desligado e 2,4 GiB livres, o Ollama recusou a carga ("model requires more system memory (3.8 GiB)
than is available (2.4 GiB)"); às 19:31, com 4,0 GiB livres, carregou mas empurrou ~700 MB de outros processos para
o swap. O rótulo previu, na prática, que o modelo cabe só com folga apertada e falha quando as condições apertam.
**Por que registrar.** É evidência de que a faixa de tier significa alguma coisa fora da aritmética, e nenhum teste
cobre isso: os testes provam que o solver calcula o tier, não que o tier corresponde ao comportamento real.

## 2026-11-15 — o bench mede resposta de tamanho real (1024 tokens), não o começo da geração

**Contexto.** A verificação cruzada contra a medição manual (7,94 tok/s com `ollama run --verbose`, contexto padrão)
deu +25% no bench (9,95). Não era bug na aritmética: mesma máquina, mesmo Ollama, swap parado, tomada/performance.
**Evidência (mesmo prompt, modelo já carregado com ctx 4096).** `ollama run` com resposta longa: 8,34 e 8,19 tok/s
(1477 e 1551 tokens gerados), a 3–5% da medição manual. Pela API, gerando 128 tokens: 10,96 e 10,88 tok/s. O bench
(prompt de 465 tokens + 128 gerados, ctx 8192): 9,95. A cada token o modelo relê o KV cache, que cresce; o decode cai
ao longo da resposta e a média dos primeiros 128 tokens superestima a de uma resposta real em ~20–30%.
**Decisão (humano, opção 1).** `GEN_TOKENS` = 1024 e o prompt canônico pede um guia longo, para o modelo não parar
cedo; rodada com menos de 768 tokens gerados (75%) é recusada (`short_generation`), porque a média viria de outra
posição da resposta. Superestimar é a direção de erro que o projeto não aceita: prometer 9,9 tok/s e entregar ~8.
**Custo aceito.** Cada rodada leva ~2–3 min; um bench completo, de 8 a 20 min. A calibração continua presa a um
tamanho de resposta (~1000 tokens), e respostas mais longas ainda ficam um pouco abaixo do previsto.
**Tarefa futura (opção 2, registrada, não iniciada).** Pôr a leitura do KV cache na fórmula: bytes por token =
pesos + KV lido até a posição atual, com o bench medindo em duas posições para separar os dois termos. É mais físico
e vale para qualquer tamanho de resposta; exige enunciado novo (`arquiteto`) e mudança no `nucleo`. Quando existir,
o tamanho fixo de 1024 deixa de ser necessário.

## 2026-11-15 — calibração de dois parâmetros instalada (Qwen3-4B + Qwen3-0.6B)

**Resultado (tomada, perfil performance, Ollama 0.17.6, ctx 8192, 1024 tokens gerados).** Qwen3-4B Q4_K_M: 7,95
tok/s (19,9 GB/s efetivos), rodadas 7,95 e 7,95. Qwen3-0.6B Q8_0 (GGUF oficial importado no Ollama como
`qwen3-0.6b-hf-q8_0`): 25,08 tok/s, rodadas 25,09 e 25,07. Ajuste `t = bytes/banda + overhead`: banda 21,63 GB/s,
overhead 10,3 ms/token (26% do tempo por token do 0.6B, acima do piso de 5%, então a forma 2p vale). Com só o 4B
(forma 1p), a banda era 19,1 GB/s. As 11 checagens do critério de pronto passaram contra o arquivo instalado.
**Critério de swap (humano, opção 2).** Swap na carga do modelo é aceito desde que as rodadas usadas tenham troca
desprezível com o swap; o que invalida a medição é o modelo ser paginado durante a medição, não processos ociosos
irem para o swap quando ele carrega. A rodada 1 do 4B teve 5540 páginas (~22 MB) saindo e foi aceita pelo humano:
o decode dela é idêntico ao da rodada 2, limpa (7,95 nas duas).
**Prefill com vários modelos (humano).** `build_calibration` grava o MENOR prefill entre os modelos, não a média: a
média de 19,9 e 146,8 (83,4) não descreve nenhum dos dois e faria o aviso `slow_prefill_cpu` citar um prefill quatro
vezes otimista para o 4B.
**Bug corrigido.** O bench não achava modelo criado com `ollama create x`, que o Ollama lista como `x:latest`.

## 2026-11-16 — leitura do KV cache na fórmula de decode (calibration_schema_version 2)

**Decisão (humano).** `bytes_por_token = pesos_ativos + 2 × n_layers × n_kv_heads × head_dim × (P + N/2) ×
elem_bytes` (média da leitura do KV numa geração de N tokens a partir de P). O bench grava esse valor com P e N reais
das rodadas usadas e KV suposto f16, o padrão do Ollama (`BENCH_KV_QUANT`, registrado em cada `measured.json` como
`kv_quant_assumed`; se o Ollama rodar com `OLLAMA_KV_CACHE_TYPE` diferente, a banda calibrada sai deslocada). O plan
prevê na posição p̄ = ctx − 512, ou seja, uma resposta de 1024 tokens terminando no limite da janela pedida (opção B,
conservadora pela direção do erro; ctx < 1024 → ctx/2). A suposição aparece em `inputs.decode` e numa linha do
markdown. Evolução registrada, não iniciada: mostrar também o tok/s no início da janela (opção C; muda schema e web).
**Por que agora.** Parcela do KV no tráfego por token (KV f16, pesos ativos), calculada sobre `data/catalog.json`:
nas condições do bench (P≈520, N=1024) é 1–6% nos modelos de 4B para cima e 15,6% no Qwen3-0.6B; perto do
`ctx_max` é 28–88% (Qwen3-4B: 70%). A correção quase não mexe na calibração medida e muda muito a previsão de quem
usa contexto longo.
**Valores antigos e novos lado a lado (mesmas medições de 2026-11-15, sem medir de novo).** A banda subiu porque o
numerador cresceu: é redefinição, não regressão.

| | v1 (só pesos) | v2 (pesos + KV médio) |
| --- | --- | --- |
| bytes/token, Qwen3-4B Q4_K_M (P=524, N=1024) | 2 497 280 256 | 2 650 044 672 (+6,1%) |
| bytes/token, Qwen3-0.6B Q8_0 (P=516, N=1024) | 639 446 668 | 757 345 932 (+18,4%) |
| banda só do 4B (forma 1p) — a validada à mão | 19,86 GB/s (manual: 19,8) | 21,07 GB/s |
| ajuste 1p com os dois modelos | 19,08 GB/s | 20,61 GB/s |
| ajuste 2p: banda | 21,63 GB/s | 22,04 GB/s |
| ajuste 2p: overhead | 10,3 ms/token | 5,5 ms/token |

Quase metade do overhead da v1 era KV: o 0.6B lê proporcionalmente mais cache, e o ajuste só com pesos atribuía essa
leitura a um custo fixo. Os 19,8 GB/s da medição manual correspondem a 21,07 GB/s na definição v2.
**Consequência.** Calibrações v1 são descartadas (`calibration_schema_outdated`) e o bench precisa gravar de novo,
ou a v1 ser recalculada a partir do `measured.json` como foi feito aqui.
