# LocalAI Scout — instruções do projeto

Ferramenta que lê o hardware da máquina e recomenda modelos de IA locais que
de fato rodam nela. CLI em Python, backend local, frontend estático no
GitHub Pages.

Especificação do núcleo: `PROMPT.md`.
Contrato frontend/backend e deploy: `ARCHITECTURE.md`.
Plano de execução por fases: `ORCHESTRATION.md`.
Decisões tomadas e por quê: `DECISIONS.md`.

## Arquitetura em uma tela

- `scout/` — CLI e biblioteca Python, stdlib-first
- `scout/server.py` — backend LOCAL, `scout serve`, bind só em 127.0.0.1:8765
- `web/` — frontend estático, sem build, publicado no GitHub Pages
- O frontend tem dois modos. ARQUIVO: o usuário sobe o `system.json` gerado
  pela CLI; funciona sempre, sem backend, offline. LIVE: fala com o backend
  local por HTTP. ARQUIVO é o caminho primário e nunca pode quebrar.
- Nenhum dado de hardware sai da máquina. Sem telemetria, sem analytics,
  sem CDN de terceiros no frontend.

## Regras invioláveis

1. Nenhum número vem de LLM. Toda aritmética em função pura, testada.
2. Nenhum modelo é citado sem existir no catálogo com `source_url`.
3. Campo não obtido fica `null` e o registro recebe `complete: false`.
   Nunca invente valor de hardware, tamanho, licença ou arquitetura.
4. Unidades: bytes inteiros e tokens inteiros internamente. GB só na
   apresentação, por `fmt_gb()`.
5. O solver nunca toca a rede. A coleta nunca toca a rede.
6. Nunca recomende aumentar swap para rodar modelo maior.
7. Testes nunca leem `data/catalog.json`. Usam `tests/fixtures/catalog_test.json`.
8. Nenhum segredo no repositório. `ANTHROPIC_API_KEY` só via ambiente.
9. Toda fase termina com parada para revisão humana. Não encadeie fases.

## Roteamento de agentes — modelo e esforço por tarefa

A sessão principal roda em Opus 5.5 e ORQUESTRA: decide, delega, revisa e
integra. Ela não implementa o que cabe inteiro num agente.

| Tarefa | Agente | Modelo | Esforço |
| --- | --- | --- | --- |
| Contrato, schema, decisão de arquitetura | `arquiteto` | opus | high |
| Fórmulas, solver, geração de candidatos | `nucleo` | opus | medium |
| Detecção de hardware, parsing de SO | `coletor` | sonnet | medium |
| Endpoints HTTP, CORS, token, serialização | `backend` | sonnet | medium |
| Interface, estado, render, acessibilidade | `web` | sonnet | medium |
| Testes, fixtures, rodar suíte | `qa` | sonnet | high |
| Revisão independente antes do merge | `revisor` | opus | high |
| CI, deploy, README, changelog, commits | `ops` | haiku | low |
| Busca ampla no repositório | `Explore` | — | low |

Ao invocar um agente, passe o esforço desta tabela explicitamente. Se a
tarefa não estiver na tabela, classifique-a pela mais próxima e registre a
escolha em `DECISIONS.md`.

### Quando NÃO delegar

- Tarefa de menos de três chamadas de ferramenta: faça inline. O spawn custa
  mais que o trabalho.
- Pergunta que você já consegue responder com o que está em contexto.
- Edição pontual num arquivo que você acabou de escrever.

### Economia de tokens

- Comece no tier mais barato que resolve. Escale só depois de falha real,
  nunca por precaução.
- Um agente por FASE, não por arquivo. Cada spawn novo re-deriva contexto
  do zero e esse é o maior desperdício do projeto.
- Para continuar um agente que já tem contexto, use `SendMessage` com o id
  dele. Não abra um agente novo para a mesma linha de trabalho.
- Agentes devolvem resumo e diff, no máximo 30 linhas. Nunca o conteúdo
  inteiro dos arquivos que escreveram.
- `qa` roda `pytest -q` e devolve só as falhas, nunca a saída completa.
- Busca ampla vai para `Explore`, não para a sessão principal nem para um
  agente caro.
- Nunca peça a um agente que leia o repositório inteiro. Dê os caminhos.

### Escada de escalonamento

1. O agente `sonnet` falhou duas vezes na mesma tarefa → reenvie para
   `opus` com esforço `medium` E o log das duas falhas.
2. `opus`/`medium` insuficiente → revise o ENUNCIADO antes de subir para
   `high`. Na maioria dos casos o problema é a especificação, não o modelo.
3. Toda escalada vira uma linha em `DECISIONS.md` com o motivo.

## Comandos

    pip install -e .            # instala a CLI
    pytest -q                   # suíte
    scout collect               # gera system.json
    scout plan --ctx 8192       # recomendações
    scout serve                 # backend local, imprime o token
    python -m http.server -d web 5173   # frontend em dev

## Convenções

- Python 3.11+, type hints em tudo, dataclasses para dados.
- Frontend em ES modules puros, sem framework e sem build. Se achar que
  precisa de build, pergunte antes.
- Commits pequenos, em português, no imperativo: `adiciona solver de offload`.
- Um commit nunca mistura fases diferentes.
