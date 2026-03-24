FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-chatbot.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-chatbot.txt

COPY run_chatbot.py ./
COPY src ./src
COPY static ./static
COPY templates ./templates
COPY data/chatbot ./data/chatbot
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "run_chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
