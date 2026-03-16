# Build application
FROM golang:1.24 AS builder

WORKDIR /app

# mod and sum separate allows to cache dependencies between builds
COPY go.mod go.sum ./
RUN go mod download
COPY . .
COPY .env.example /app/.env

RUN CGO_ENABLED=0 GOOS=linux go build -o /datastore-logic ./main.go

# Create minimal container
FROM scratch
WORKDIR /app
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /datastore-logic /app/datastore-logic
COPY --from=builder /app/.env /app/.env
CMD ["/app/datastore-logic"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
