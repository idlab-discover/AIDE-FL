# Build application
FROM golang:1.24 AS builder

WORKDIR /app

# mod and sum separate allows to cache dependencies between builds
COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o /orchestrator ./main.go

# Create minimal container
FROM scratch
COPY --from=builder /orchestrator /app/.env .
CMD ["/orchestrator"]

LABEL org.opencontainers.image.source=https://github.com/MartienskieNum1/AIDE-FL