FROM flwr/clientapp:1.27.0

WORKDIR /app

COPY clientapp-requirements.txt .
RUN python -m pip install -U --no-cache-dir -r clientapp-requirements.txt

LABEL org.opencontainers.image.source=https://github.com/idlab-discover/AIDE-FL
