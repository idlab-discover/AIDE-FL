# Build application
FROM golang:1.24 AS builder

WORKDIR /app

# mod and sum separate allows to cache dependencies between builds
COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o /preprocessor ./main.go

# Create minimal container
FROM scratch
COPY --from=builder /preprocessor /app/.env .
CMD ["/preprocessor"]

LABEL org.opencontainers.image.source=https://github.com/MartienskieNum1/AIDE-FL