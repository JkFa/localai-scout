# Plano de execução — LocalAI Scout ponta a ponta

Este arquivo é o roteiro da sessão principal. Leia `CLAUDE.md` para as regras
de roteamento de agentes e `ARCHITECTURE.md` para o contrato frontend/backend.

## Prompt de abertura

Cole isto numa sessão nova do Claude Code, na raiz do repositório:

    Leia CLAUDE.md, ARCHITECTURE.md, PROMPT.md e ORCHESTRATION.md.
    Você é a sessão orquestradora: delega, revisa e integra; não implementa
    o que cabe inteiro num agente. Siga a tabela de roteamento de CLAUDE.md
    para escolher modelo e esforço de cada agente.
    Comece pela Fase 0 e pare ao fim dela para minha revisão.

## Como cada fase funciona

Toda fase tem objetivo, agente dono, entregável e um critério de pronto
verificável por comando. A sessão principal abre a fase, delega, recebe o
resumo, roda o critério de pronto ela mesma e só então para para revisão.

Nenhuma fase começa antes de a anterior ser aprovada por mim. Se um critério
de pronto falhar, a fase não acabou — não siga adiante registrando dívida.

---

## Fase 0 — contratos

**Agente:** `arquiteto` (opus, high). **Entrega:** schemas JSON de
`system.json`, `catalog.json` e `feasibility.json` em `schemas/`, mais
`DECISIONS.md` com as decisões já tomadas e as que ficaram em aberto.

Revise `ARCHITECTURE.md` criticamente antes de aceitar: aponte o que está
subespecificado e proponha o fechamento. Não escreva código nesta fase.

**Pronto quando:** os três schemas validam os exemplos de `PROMPT.md` com
`jsonschema`, e `DECISIONS.md` lista ao menos as decisões sobre token de
pareamento, ausência de build no frontend e política de campo incompleto.

---

## Fase 1 — núcleo calculável

**Agentes:** `nucleo` (opus, medium) implementa; `qa` (sonnet, high) escreve
os testes em paralelo, a partir do mesmo enunciado, sem ver a implementação.

Escopo exato: M1 de `PROMPT.md`. `scout/estimate.py`, `scout/solver.py`,
`scout/cli.py` com o comando `plan`, catálogo de teste e cinco fixtures de
perfil de máquina.

Testes escritos por quem não escreveu o código é deliberado: é o que pega
fórmula implementada ao contrário da especificação.

**Pronto quando:**

    pytest -q
    scout plan --system tests/fixtures/sys_rtx3060.json \
               --catalog tests/fixtures/catalog_test.json --ctx 8192

roda verde, e duas execuções seguidas do `plan` sobre a mesma fixture
produzem saída byte a byte idêntica.

---

## Fase 2 — coleta real

**Agente:** `coletor` (sonnet, medium). Escopo: M2 de `PROMPT.md`.

**Pronto quando:** `scout collect` roda na máquina real e produz
`system.json` coerente; e roda também com `env -i PATH= python -m scout
collect` sem crash, devolvendo `confidence: "low"` nos campos afetados.

---

## Fase 3 — catálogo real

**Agentes:** `nucleo` (opus, medium) para o mapeamento `config.json` →
campos do catálogo, que é a parte onde errar é caro; `coletor`
(sonnet, medium) para o cliente HTTP e o cache.

Escopo: M3 de `PROMPT.md`. Antes de rodar o `catalog add`, o agente
apresenta a lista de repositórios que pretende adicionar e espera aprovação.

**Pronto quando:** `data/catalog.json` tem de 6 a 8 modelos, todos com
`source_url` e `license_source` preenchidos por API, nenhum campo inventado,
e `scout catalog check` passa.

---

## Fase 4 — backend local

**Agente:** `backend` (sonnet, medium). Escopo: `scout/server.py` e o comando
`scout serve`, conforme a seção de backend de `ARCHITECTURE.md`.

**Pronto quando:**

    scout serve &
    curl -s localhost:8765/health                      # 200
    curl -s localhost:8765/system                      # 401 sem token
    curl -s -H "Authorization: Bearer $TOKEN" localhost:8765/system   # 200

e o preflight `OPTIONS` devolve `Access-Control-Allow-Private-Network: true`.

---

## Fase 5 — frontend

**Agente:** `web` (sonnet, medium). Construa o modo ARQUIVO inteiro e só
depois o modo LIVE. Se o LIVE atrasar, o produto ainda está entregue.

**Pronto quando:** servindo `web/` localmente, arrastar um `system.json` de
fixture renderiza diagnóstico, recomendações e rejeitados; e a aba de rede do
navegador mostra zero requisição para fora da máquina.

---

## Fase 6 — publicação

**Agente:** `ops` (haiku, low). Workflows de CI e Pages, README com captura
de tela e instruções de uso, `CHANGELOG.md`.

**Pronto quando:** a URL do Pages abre, carrega o `system.json` de exemplo
embutido e mostra a tela de recomendações. O CI está verde na `main`.

---

## Fase 7 — revisão independente

**Agente:** `revisor` (opus, high), sem ter visto nenhuma das fases
anteriores. Entrada: o diff completo contra o commit inicial.

Procure especificamente: fórmula divergente de `PROMPT.md`, unidade trocada
entre bytes e GB, leitura de RAM livre em vez de total, caminho em que o
frontend vaza dado para fora, endpoint sem checagem de token, número exibido
sem proveniência, e teste que lê o catálogo real.

**Pronto quando:** todo achado está corrigido ou registrado em
`DECISIONS.md` com justificativa aceita por mim.

---

## Orçamento de atenção

Se uma fase passar de duas rodadas de ida e volta com o mesmo agente, pare e
me chame. Fase que não fecha em duas rodadas quase sempre tem enunciado
ruim, não modelo fraco — e insistir com modelo maior só queima tokens.
