FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend
COPY models ./models
COPY docs ./docs

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "rail-eta serve --host 0.0.0.0 --port ${PORT:-8000}"]
