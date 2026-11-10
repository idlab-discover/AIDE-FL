FROM flwr/serverapp:1.27.0

WORKDIR /app
ENV FLWR_HOME=/app/.flwr

COPY serverapp-requirements.txt .
RUN python -m pip install -U --no-cache-dir -r serverapp-requirements.txt

LABEL org.opencontainers.image.source=https://github.com/idlab-discover/AIDE-FL
