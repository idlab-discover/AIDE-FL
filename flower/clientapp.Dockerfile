ARG BASE_IMAGE=ghcr.io/idlab-discover/aide-fl/flower-clientapp-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --chown=49999:49999 pyproject.toml .
COPY --chown=49999:49999 flower ./flower

ENTRYPOINT ["flwr-clientapp"]

LABEL org.opencontainers.image.source=https://github.com/idlab-discover/AIDE-FL
