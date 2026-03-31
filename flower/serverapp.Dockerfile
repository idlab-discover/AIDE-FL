FROM flwr/serverapp:latest

WORKDIR /app

COPY serverapp-requirements.txt .
RUN python -m pip install -U --no-cache-dir -r serverapp-requirements.txt

COPY main.py .
COPY pyproject.toml .
COPY flower ./flower

ENTRYPOINT ["python", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
