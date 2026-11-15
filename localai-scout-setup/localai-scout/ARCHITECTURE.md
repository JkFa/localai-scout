# Arquitetura ponta a ponta

## O problema que define o desenho

Um navegador não lê hardware. Qualquer coisa que informe VRAM, canais de
memória ou modelo de CPU precisa rodar como processo nativo na máquina do
usuário. Logo o backend é LOCAL, não hospedado, e o frontend publicado no
GitHub Pages é um cliente que fala com ele — ou que trabalha offline a
partir de um arquivo.

    GitHub Pages (HTTPS, estático, grátis)
      web/  SPA sem build
        modo ARQUIVO  -> usuário sobe system.json          [sempre funciona]
        modo LIVE     -> fetch http://127.0.0.1:8765       [opcional]
                               |
                               v
    Máquina do usuário
      scout serve  -> http.server stdlib, bind 127.0.0.1
                      token obrigatório, CORS restrito

O modo ARQUIVO é o primário. O modo LIVE é conveniência. Se o LIVE quebrar
numa atualização de navegador, o produto continua inteiro.

## Backend local

### Endpoints

    GET  /health            -> {"ok": true, "version": "0.1.0"}
    GET  /system            -> system.json, coleta ao vivo
    POST /plan              -> body {system, intent} -> feasibility.json
    GET  /catalog           -> catálogo resumido (id, tasks, licença)
    POST /bench             -> {model_id} -> {command, ran, measured?}

Todos respondem JSON. Erro é `{"error": {"code": "...", "message": "..."}}`
com status HTTP coerente. `/plan` é puro: mesma entrada, mesma saída.

### Segurança — não é opcional

O endereço `127.0.0.1:8765` é alcançável por QUALQUER página aberta no
navegador do usuário. Sem proteção, um site qualquer faz fetch e lê o perfil
completo da máquina. Portanto:

- `scout serve` gera um token aleatório (`secrets.token_urlsafe(32)`) a cada
  inicialização e o imprime no terminal, junto com uma URL de pareamento:
  `https://<usuario>.github.io/localai-scout/#token=<token>`
- Todo endpoint, exceto `/health`, exige `Authorization: Bearer <token>`
- Bind SOMENTE em 127.0.0.1. Nunca 0.0.0.0. Nunca `--host` configurável
- Origin allowlist por variável `SCOUT_ALLOWED_ORIGIN`, default
  `http://localhost:5173`. Requisição de origin fora da lista é recusada
  antes de qualquer trabalho
- Sem persistência: o servidor não grava nada em disco além de logs opcionais

### CORS e Private Network Access

Dois pontos que vão custar uma tarde se não estiverem no código desde o início:

1. Uma página HTTPS chamando `http://127.0.0.1` NÃO é bloqueada como conteúdo
   misto: loopback é tratado como origem potencialmente confiável pela
   especificação de contextos seguros. Chrome, Edge e Firefox permitem.
   Safari é mais restrito — documente o modo ARQUIVO como alternativa.
2. O Chrome aplica Private Network Access: o preflight chega com o cabeçalho
   `Access-Control-Request-Private-Network: true` e o servidor PRECISA
   responder `Access-Control-Allow-Private-Network: true`, além dos
   cabeçalhos CORS normais. Sem isso o fetch falha com erro de rede opaco,
   que é fácil de confundir com servidor fora do ar.

O `OPTIONS` deve responder: `Access-Control-Allow-Origin` (o origin exato,
nunca `*` quando há credencial), `Access-Control-Allow-Headers:
Authorization, Content-Type`, `Access-Control-Allow-Methods`,
`Access-Control-Allow-Private-Network: true`, `Vary: Origin`.

## Frontend

Sem framework, sem build, ES modules nativos. Isso mantém o deploy trivial
e o repositório pequeno. Se um agente achar que precisa de bundler, ele
pergunta antes e registra a justificativa em `DECISIONS.md`.

    web/
      index.html
      app.js          roteamento de estado e render
      api.js          cliente do backend, detecção de modo
      render.js       tabelas, badges de tier, avisos
      style.css
      fixtures/       um system.json de exemplo para demo

Telas, em ordem de importância:

1. **Entrada.** Arrastar `system.json`, ou colar, ou conectar ao backend.
   Mostra qual modo está ativo e o que cada um exige.
2. **Diagnóstico.** Pools de memória, gargalos, avisos (bateria, HDD, swap).
3. **Recomendações.** Tabela por tarefa com modelo, quant, colocação, memória,
   tokens/s, download e licença. Filtro por tarefa e toggle de uso comercial.
4. **Rejeitados.** O que não coube e quanto faltou. Esta tela é o que dá
   credibilidade ao resto.
5. **Próximo passo.** O comando de benchmark, com botão de copiar.

Regras de interface:

- Todo número exibido carrega sua proveniência: catálogo, calculado ou
  estimado. Estimado aparece com marca visual distinta.
- Confiança `low` nunca é renderizada como número cheio sem ressalva.
- Zero requisição externa: sem Google Fonts, sem CDN, sem analytics. Fonte
  do sistema. Isso é requisito de privacidade, não preferência estética.
- Funciona sem JavaScript habilitado? Não — mas deve falhar com mensagem
  clara, não com tela branca.
- Acessível: contraste AA, navegação por teclado, tabelas com cabeçalho real.

## Deploy

`.github/workflows/pages.yml` — publica `web/` no GitHub Pages a cada push
na `main`, com `actions/configure-pages`, `actions/upload-pages-artifact`
(path `web/`) e `actions/deploy-pages`. Permissões `pages: write` e
`id-token: write`.

`.github/workflows/ci.yml` — roda `pytest -q` em push e pull request, em
Python 3.11 e 3.12. O deploy depende do CI verde.

O frontend precisa saber que está sob subcaminho (`/localai-scout/`): use
caminhos relativos em tudo, nunca absolutos começando em `/`.
