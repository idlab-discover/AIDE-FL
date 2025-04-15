FROM golang:1.24

WORKDIR /app

COPY go.mod ./
#COPY go.sum ./
#RUN go mod download

COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /orchestrator

EXPOSE 8080

CMD ["/orchestrator"]

LABEL org.opencontainers.image.source https://github.com/MartienskieNum1/AIDE-FL