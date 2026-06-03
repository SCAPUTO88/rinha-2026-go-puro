# Rinha de Backend 2026 - Go (Busca Vetorial)

Esta é a minha submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), focada em detecção de fraude usando busca vetorial com restrições severas de infraestrutura (1 CPU e 350MB de RAM).

## 🚀 Arquitetura e Estratégia

A API foi construída em **Go puro**, sem nenhuma dependência externa (nem frameworks web, nem libs matemáticas). A estratégia principal foi atingir **latência sub-milissegundo (p99 < 1ms)** com **100% de exatidão** (0 falsos positivos ou falsos negativos comparado ao *brute force* exato). Para atingir o máximo do score avaliado, aplicamos as seguintes otimizações agressivas:

* **Vantage Point Tree (VP-Tree):** Em vez de aproximações (ANNs) que resultariam em perdas de pontos, ou de *brute force* que destruiria a latência, implementamos uma VP-Tree manual. A busca aplica poda severa baseada na Desigualdade Triangular, derrubando a complexidade para O(log N).
* **Mmap + Zero-Copy:** Os 3 milhões de vetores pré-processados (`~230MB`) são carregados na API através de `mmap` (com mapeamento compartilhado `MAP_SHARED`). O arquivo nem sequer sobe para a RAM alocada (Heap) do Go, e através do `unsafe.Slice` converte ponteiros binários direto para arrays legíveis. E melhor: o SO compartilha as páginas de memória entre as duas instâncias da API de forma transparente.
* **SIMD Padding:** As 14 dimensões da Rinha foram estendidas com padding para um array estático de `[16]float32`. Esse alinhamento torna a rotina de cálculo de distância Euclidiana perfeitamente amigável à cache L1 e ao auto-vetorizador do compilador do Go.
* **Zero-Allocation HTTP Handlers:** 
  * O Garbage Collector está **desativado** (`GOGC=off`). 
  * Não há chamadas para `json.Marshal`. As seis possíveis respostas finais já ficam pre-compiladas estaticamente em arrays de bytes.
  * O hot path não possui alocações não-gerenciadas: `sync.Pool` reaproveita massivamente os buffers de request do socket, os Structs JSON, e até mesmo as árvores de Heap locais da busca KNN.

**Métrica no Hot Pipeline (HTTP Socket -> JSON Unmarshal -> Vectorize -> VP-Tree -> Static Response):** média de `~6 µs/op`.

## 🛠 Como executar

Para rodar localmente respeitando os limites originais do `docker-compose` da Rinha (0.10c pro Nginx, 0.45c para as APIs):

```bash
docker compose pull
docker compose up -d
```

E para realizar um teste de carga apontando para `http://localhost:9999`:
```bash
k6 run .\rinha-de-backend-2026\test\smoke.js
```

## 📂 Estrutura de Destaque

- `cmd/api`: Entrypoint do servidor HTTP.
- `cmd/preprocess`: CLI injetado no multi-stage build que varre o `.json.gz` e constrói a árvore balanceada num binário limpo.
- `internal/vptree.go`: Implementação O(log N) de KNN para os vetores.
- `internal/mmap_unix.go`: Truque principal da contenção de consumo de RAM do app.

## Autor
- Sandro Caputo ([SCAPUTO88](https://github.com/SCAPUTO88))
