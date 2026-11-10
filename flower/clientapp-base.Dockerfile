FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 FLWR_HOME=/app/.flwr
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY clientapp-requirements.txt .
RUN python -m pip install -r clientapp-requirements.txt
RUN groupadd --gid 49999 flower && useradd --uid 49999 --gid 49999 --home-dir /app flower && chown -R 49999:49999 /app
USER 49999:49999
LABEL org.opencontainers.image.source=https://github.com/idlab-discover/AIDE-FL
