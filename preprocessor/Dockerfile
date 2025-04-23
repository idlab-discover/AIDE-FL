FROM golang:1.24

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o /preprocessor ./main.go

EXPOSE 8080

CMD ["/preprocessor"]

LABEL org.opencontainers.image.source=https://github.com/MartienskieNum1/AIDE-FL