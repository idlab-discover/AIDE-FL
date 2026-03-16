FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY .env.example .env

CMD ["python", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/Str-Gen/AIDE-FL
