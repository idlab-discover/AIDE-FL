FROM python:3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/MartienskieNum1/AIDE-FL