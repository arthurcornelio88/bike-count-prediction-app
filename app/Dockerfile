# ---- Base image avec Python ----
FROM python:3.11-slim

# ---- Variables d’environnement ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Dépendances système minimales ----
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Dossier de travail ----
WORKDIR /app

# ---- Copie des fichiers ----
COPY . /app

# ---- Dépendances Python ----
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---- Port exposé ----
EXPOSE 8000

# ---- Commande de démarrage ----
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
