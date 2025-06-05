FROM flwr/serverapp:1.18.0

WORKDIR /app

COPY pyproject.toml .
RUN python -m pip install -U  --no-cache-dir .

COPY main.py .env .
COPY flower ./flower

ENTRYPOINT ["python", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/MartienskieNum1/AIDE-FL