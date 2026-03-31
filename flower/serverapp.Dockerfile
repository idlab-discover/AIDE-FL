FROM flwr/serverapp:latest

WORKDIR /app

COPY serverapp-requirements.txt .
RUN python -m pip install -U --no-cache-dir -r serverapp-requirements.txt

COPY --chown=49999:49999 main.py .
COPY --chown=49999:49999 pyproject.toml .
COPY --chown=49999:49999 flower ./flower

ENTRYPOINT ["python", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
