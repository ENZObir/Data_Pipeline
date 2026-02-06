FROM python:3.10-slim AS builder

WORKDIR /deps

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/deps -r /tmp/requirements.txt

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /deps /usr/local
COPY . /app

CMD ["python", "models/train.py"]
