# ============================================================
# Stage 1: Build Go binaries
# ============================================================
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Cache dependencies
COPY go.mod ./
RUN go mod download

# Copy source code
COPY internal/ ./internal/
COPY cmd/ ./cmd/

# Build both binaries (static, no CGO)
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /bin/api ./cmd/api

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /bin/preprocess ./cmd/preprocess

# ============================================================
# Stage 2: Preprocess — build VP-Tree binary from dataset
# ============================================================
FROM builder AS preprocessor

# Copy the dataset from the challenge repo
COPY rinha-de-backend-2026/resources/references.json.gz /data/references.json.gz

# Build the VP-Tree binary (this is the expensive step, cached by Docker)
RUN /bin/preprocess /data/references.json.gz /data/vptree.bin

# ============================================================
# Stage 3: Runtime — minimal image
# ============================================================
FROM scratch

# Copy the API binary
COPY --from=builder /bin/api /api

# Copy the pre-built VP-Tree binary data
COPY --from=preprocessor /data/vptree.bin /data/vptree.bin

# Default environment
ENV PORT=8080
ENV VPTREE_BIN=/data/vptree.bin

# Performance tuning:
# GOGC=200: GC roda quando heap dobra, reduz frequência sem acumular memória indefinidamente.
# GOMAXPROCS não é definido: o runtime usa o número de CPUs disponíveis automaticamente.
# GOMEMLIMIT limita o heap do Go, mas a mmap do vptree.bin não conta nesse limite.
ENV GOGC=200
ENV GOMEMLIMIT=120MiB

EXPOSE 8080

ENTRYPOINT ["/api"]
