FROM flwr/serverapp:latest

WORKDIR /app
ENV FLWR_HOME=/app/.flwr

COPY serverapp-requirements.txt .
RUN python -m pip install -U --no-cache-dir -r serverapp-requirements.txt

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
