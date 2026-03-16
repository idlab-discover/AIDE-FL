# Build application
FROM golang:1.24 AS builder

WORKDIR /app

# mod and sum separate allows to cache dependencies between builds
COPY go.mod go.sum ./
RUN go mod download
COPY . .
COPY .env.example /app/.env

RUN CGO_ENABLED=0 GOOS=linux go build -o /orchestrator ./main.go

# Create minimal container
FROM scratch
WORKDIR /app
COPY --from=builder /orchestrator /app/orchestrator
COPY --from=builder /app/.env /app/.env
CMD ["/app/orchestrator"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
