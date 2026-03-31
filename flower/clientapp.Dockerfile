FROM flwr/clientapp:latest

WORKDIR /app

COPY clientapp-requirements.txt .
RUN python -m pip install -U --no-cache-dir -r clientapp-requirements.txt

COPY pyproject.toml .
COPY flower ./flower

ENTRYPOINT ["flwr-clientapp"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
