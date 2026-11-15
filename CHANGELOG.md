# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

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
