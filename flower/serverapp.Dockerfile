ARG BASE_IMAGE=ghcr.io/str-gen/flower-serverapp-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app
ENV FLWR_HOME=/app/.flwr

COPY --chown=49999:49999 main.py .
COPY --chown=49999:49999 config.toml /app/.flwr/config.toml
COPY --chown=49999:49999 pyproject.toml .
COPY --chown=49999:49999 flower ./flower

ENTRYPOINT ["python", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
