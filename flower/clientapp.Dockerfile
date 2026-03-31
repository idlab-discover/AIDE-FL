FROM flwr/clientapp:latest

WORKDIR /app

COPY clientapp-requirements.txt .
RUN python -m pip install -U --no-cache-dir -r clientapp-requirements.txt

COPY --chown=49999:49999 pyproject.toml .
COPY --chown=49999:49999 flower ./flower

ENTRYPOINT ["flwr-clientapp"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
