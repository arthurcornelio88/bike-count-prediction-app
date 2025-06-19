# Utiliser l'image officielle de Python
FROM python:3.11-slim

# Définir un répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur pour installer les dépendances
COPY requirements.txt /app/

# Créer et activer un environnement virtuel
RUN python3 -m venv /env

# Activer l'environnement virtuel et installer les dépendances dans l'environnement virtuel
RUN /env/bin/pip install --no-cache-dir -r requirements.txt

# Copier tout le code source dans le conteneur
COPY . /app/

# Exposer le port 8080
EXPOSE 8080

# Utiliser l'environnement virtuel pour démarrer l'application
CMD ["/env/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
