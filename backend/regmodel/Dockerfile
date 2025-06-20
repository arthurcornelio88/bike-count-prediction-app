FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# Nettoyage immédiat pour pas gonfler l'image intermédiaire
RUN apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Final image
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.11/site-packages/
ENV PATH="/usr/local/lib/python3.11/site-packages/bin:$PATH"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
