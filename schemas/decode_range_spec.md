# Tokens/s no início e no fim da janela — enunciado (opção C)

Fecha a evolução registrada em DECISIONS.md, 2026-11-16, "leitura do KV cache
na fórmula de decode": além do tokens/s previsto no FIM da janela (opção B,
já implementada), cada candidato passa a trazer o tokens/s de uma conversa
que COMEÇA com o contexto vazio. Contrato: `schemas/feasibility.schema.json`
(`inputs.decode.start_position_tokens`, `candidate.tokens_per_s_est_start`).
Exemplo validado pela CI: `schemas/examples/feasibility_example.json`.

## Perguntas abertas (nenhuma bloqueia; o enunciado vale com a recomendação)

- **Q1. `--prompt-tokens` desloca o início?** Hoje o início supõe prompt
  P = 0. Com `intent.prompt_tokens`, o início poderia ser P + 512 (limitado à
  posição do fim).
  - (a) **Não, P = 0 sempre (recomendado).** O número do início fica uma
    definição fixa, comparável entre execuções, com um campo só no eco. A flag
    existe para o TTFT calibrado e quase ninguém a usa.
  - (b) Sim. É mais fiel para quem informou o prompt, mas custa mais um campo
    no eco (`start_prompt_tokens`), um texto a mais no markdown e no web, e um
    caso de borda (P + 1024 > ctx).
  Mudar de (a) para (b) depois é aditivo e não quebra arquivo.

## 0. Decisões

- **D1. O principal não muda.** `tokens_per_s_est` continua sendo o número do
  fim da janela (p̄ = ctx − 512; ctx < 1024 → ⌈ctx/2⌉). É o ÚNICO que entra em
  `_choose` (placement), `_sort_key`, `primary_for`, `quality_preference` e
  qualquer aviso. O critério é a direção do erro: nunca superestimar.
- **D2. Definição do início.** É uma resposta de N = min(1024, ctx) tokens
  começando com o contexto vazio (P = 0), com o KV lido na posição média
  `start_position_tokens = ⌈N/2⌉`: 512 para ctx ≥ 1024 e igual à posição do
  fim para ctx < 1024. Não existe constante de "prompt típico". Ela seria um
  número sem medição nem fonte: o prompt de 465 tokens do bench é do bench, e
  nada diz que é típico de usuário. O template de chat soma algumas dezenas de
  tokens, e o efeito disso em p̄ = 512 é desprezível.
- **D3. Mesmo cálculo, só a posição muda.** Valem o mesmo candidato, o mesmo
  placement, o mesmo `n_gpu_layers`, os mesmos trechos e a mesma calibração,
  eta e overhead por trecho. O início nunca escolhe placement: no híbrido ele
  descreve o placement escolhido pelo fim.
- **D4. Proveniência.** `provenance.tokens_per_s_est_start` é sempre igual a
  `provenance.tokens_per_s_est`, porque as entradas são as mesmas e só muda a
  posição, que é uma definição. Mesma `confidence`, mesmos
  `confidence_reasons` e mesmo `tokens_per_s_basis`. Não há motivo novo.
- **D5. Schema compatível, sem bump.** O campo é opcional POR ARQUIVO e
  obrigatório POR CANDIDATO quando o arquivo declara a suposição (§3).
  `schema_version` continua 1, e arquivo de scout ≤ 0.2.0 continua válido.
- **D6. Apresentação hierárquica, nunca faixa.** O principal fica onde está.
  O início aparece como uma linha secundária, rotulada "no início de uma
  conversa". Não há faixa "X–Y", porque o leitor ancora no extremo alto e lê
  "até Y". Não há segunda coluna, porque peso visual igual vira duas
  promessas. Não há só tooltip, porque some no toque e no leitor de tela.
- **D7. Quando o segundo número some.** Ele não aparece quando é null ou
  quando o seu TEXTO FORMATADO é igual ao do principal no mesmo formatador
  (CLI: 2 casas; web: `fmtTps`, 1 casa). Não há limiar percentual, porque
  seria uma constante inventada.

## 1. Posição (`nucleo`, `scout/estimate.py`)

    def decode_start_position(ctx: int) -> int:
        """⌈min(DECODE_ANSWER_TOKENS, ctx)/2⌉. ctx < 1 -> ValueError."""

Invariante: `1 <= decode_start_position(ctx) <= decode_position(ctx)` para
todo ctx >= 1, com igualdade exata se e só se ctx <= 1024. Pontos fixos:
1 → 1, 600 → 300, 1023 → 512, 1024 → 512 (fim 512), 1025 → 512 (fim 513),
8192 → 512.

## 2. Solver (`nucleo`, `scout/solver.py`)

- `kv_read_start = E.kv_read_bytes_per_token(m, E.decode_start_position(ctx), intent.kv_quant)`,
  ao lado do `kv_read` atual, com o mesmo `ctx`.
- O Attempt ganha `tps_start`, calculado pela mesma `tps(n_gpu)` com
  `kv_read_start` no `seg_bytes`, sem duplicar a fórmula. `_choose`,
  `_sort_key`, `_rank` e a preferência de qualidade não leem `tps_start`.
- Quando `embed_only`, `tps_start = None` junto com `tps`.
- O candidato recebe `tokens_per_s_est_start = _round_tps(a.tps_start)` e
  `provenance["tokens_per_s_est_start"] = provenance["tokens_per_s_est"]`,
  SEMPRE, inclusive com valor null.
- `inputs.decode` ganha `start_position_tokens = E.decode_start_position(intent.ctx)`.
- Invariante após arredondar: `start >= end`, porque `_round_tps` é monótona.
  Os dois são null juntos.

## 3. Schema (`arquiteto`, feito)

- `inputs.decode.start_position_tokens`: inteiro >= 1, opcional.
- `candidate.tokens_per_s_est_start`: number|null, > 0, opcional.
- Na raiz, a regra é tudo ou nada. Se `inputs.decode` tem
  `start_position_tokens`, todo candidato tem o campo. Se não tem, nenhum tem,
  porque número sem a sua suposição é número sem proveniência.
- No candidato, o campo é null se e só se `tokens_per_s_est` é null. Se o
  campo existe, `provenance` tem a entrada dele.
- `start >= end` e `start_position_tokens <= position_tokens` não são
  expressáveis em JSON Schema e ficam com o pytest (§6).
- `validate_examples.py` tem um caso positivo (forma 0.2.0) e quatro
  negativos.

## 4. Markdown do plan (`nucleo`, `scout/cli.py`)

- A frase acima da tabela continua quando `start_position_tokens` está
  ausente. Quando está presente, troca "Conversas mais curtas geram mais
  rápido." por: "Esse é o número que ordena e escolhe a recomendação. Na nota,
  "no início de uma conversa" é o de uma resposta de {N} tokens começando com
  o contexto vazio (posição média {start_position_tokens}); conversas mais
  curtas geram mais rápido, mas não conte com ele numa conversa longa."
- Coluna nota, antes do TTFT: `no início de uma conversa: {_fmt_tps(start, prov, "tokens_per_s_est_start")} tokens/s`,
  com `_warn`. Fica fora quando `decode_not_applicable` ou quando D7 a
  suprime. `_fmt_tps` ganha o parâmetro `field` (padrão `"tokens_per_s_est"`).
- A coluna tokens/s continua com um número só.

## 5. Frontend (`web`, `web/render.js`, `web/app.js`)

- `tpsCell`: o principal fica como está. Quando D7 permite, entra abaixo um
  `div.muted.small` com "no início de uma conversa: " e
  `num(fmtTps(start), provenance.tokens_per_s_est_start || "estimated")`.
  O cabeçalho da coluna não muda.
- `decodeNote`: sem `start_position_tokens` (ou sem `inputs.decode`), o texto
  é idêntico ao de hoje. Com ele, a última frase vira: "É esse número que
  ordena e escolhe a recomendação. O número menor, embaixo, é o de uma
  resposta de {N} tokens começando com o contexto vazio (posição média
  {start}): conversas mais curtas geram mais rápido, mas não conte com ele
  numa conversa longa."
- Em `validate`, `tokens_per_s_est_start`, quando presente, tem de ser número
  ou null. Senão é erro "Candidato #i: tokens_per_s_est_start deveria ser
  número ou null.". Ausente não é erro.
- Nenhuma aritmética: o web só compara os dois textos formatados (D7).

## 6. Bordas

- **ctx <= 1024:** as posições são iguais, os valores são iguais, o campo é
  emitido e a apresentação mostra um número só.
- **ctx pouco acima de 1024:** se os valores formatados coincidirem, aparece
  um número só (D7).
- **Embeddings e banda desconhecida:** os dois são null, a linha mostra "—"
  ou "não se aplica"/"desconhecido" como hoje e não há linha de início.
- **Híbrido:** mesmo `n_gpu_layers`. O KV de cada trecho entra pela fração
  n/n_layers das suas camadas, como no fim. No exemplo, o MoE híbrido fica
  abaixo do denso no fim (21,6 contra 25,99) e acima no início (29,69 contra
  29,16), e a ordem segue o fim.
- **TTFT:** não muda. É prefill, independe da posição do decode e não ganha
  variante.

## 7. Testes (`nucleo`; o `qa` roda)

Os valores esperados vêm de aritmética comentada no teste, como os de
`test_larger_context_predicts_lower_decode`.

1. `test_estimate.py`: os pontos fixos de §1, `ValueError` para ctx 0 e a
   invariante para ctx em `range(1, 40000, 7)`.
2. `test_solver.py`: com `ddr5_64`, chat e ctx 8192, `inputs.decode` igual a
   `{8192, 1024, 7680, start 512}`. Para o `test/dense-7b` cpu_only,
   (7.65e9 + 67_108_864) / (89.6e9 × 0.6) dá `start ≈ 6.9663`, e o
   `end ≈ 6.2103` não muda.
3. Para todo perfil de `SYSTEM_PATHS`, ctx em (512, 1024, 2048, 8192, 32768)
   e kv em (f16, q8_0), todo candidato tem a chave. Vale
   `(start is None) == (end is None)`, `start >= end`, proveniência igual e
   ctx <= 1024 → `start == end`. A saída valida contra o schema.
4. Inversão: dois modelos sintéticos, um com pesos pequenos e KV grande
   (n_kv_heads alto), outro com pesos maiores e KV pequeno, numa ordem que
   inverte entre o início e o fim. `candidates` e `primary_for` seguem o fim.
5. Híbrido: o candidato hybrid da fixture `rtx3060` (ou equivalente) tem
   `start > end`, com valor calculado à mão por trecho no comentário.
6. Embeddings: `start is None` e `"tokens_per_s_est_start" in provenance`.
7. Markdown: com ctx 8192 há uma linha com "no início de uma conversa: ~6.97
   tokens/s" e a frase nova. Com ctx 600 não aparece "no início". A linha de
   embeddings não tem o texto.
8. Compatibilidade: `render_markdown` de um feasibility sem
   `start_position_tokens` e sem o campo (forma 0.2.0) produz a frase antiga e
   nenhuma nota de início.

## 8. Critério de pronto (por comando)

    .venv/bin/pytest -q                                    # 0 falhas
    .venv/bin/python schemas/validate_examples.py          # exit 0
    F="--system tests/fixtures/sys_rtx3060.json --catalog tests/fixtures/catalog_test.json --tuning tests/fixtures/tuning_test.json --no-calibration"
    .venv/bin/scout plan $F --ctx 8192 --json | .venv/bin/python -c "import json,sys; f=json.load(sys.stdin); assert f['inputs']['decode']['start_position_tokens']==512; assert all('tokens_per_s_est_start' in c for c in f['candidates'])"
    .venv/bin/scout plan $F --ctx 8192 | grep -c "no início de uma conversa"  # >= 1
    .venv/bin/scout plan $F --ctx 600  | grep -c "no início de uma conversa"  # imprime 0
    node --check web/render.js && node --check web/app.js
    git diff --exit-code web/fixtures/demo-feasibility.json   # forma pré-0.2.0 (sem inputs.decode) intacta

Os passos manuais ficam com o `web` e precisam de evidência no resumo. Em
`python -m http.server -d web 5173`, carregar `web/fixtures/demo-feasibility.json`
deve mostrar a tela idêntica à anterior. É a forma anterior à 0.2.0, sem inputs.decode. `demo-feasibility-calibracao.json` está na forma 0.2.0, com decode e sem início, e deve mostrar a frase antiga sem linha secundária. O `web` pode criar `web/fixtures/demo-feasibility-inicio.json` gerado pela CLI nova, mas não edita as fixtures existentes. Carregar a saída de
`scout plan $F --ctx 8192 --json` deve mostrar a linha secundária e a frase nova.
Com `--ctx 600`, nenhuma linha secundária.

## 9. PROMPT.md, seção Throughput (`nucleo`, mesmo commit do solver)

Acrescentar depois do parágrafo da decisão B: "Junto dele vai
tokens_per_s_est_start (opção C), com o mesmo candidato, placement, trechos e
calibração, e o KV lido em p̄ = ⌈min(1024, ctx)/2⌉, que é uma resposta de 1024
tokens começando com o contexto vazio. Ele só é exibido, nunca ordena nem
decide, e é ecoado em inputs.decode.start_position_tokens
(schemas/decode_range_spec.md)."
