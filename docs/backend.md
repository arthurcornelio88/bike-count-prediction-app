# 📘 Documentation – Backend `ds_traffic_cyclist1`

## 🔧 Environnement de développement (DEV)

### 📂 `.env` requis pour chaque service

```env
ENV=DEV
GOOGLE_APPLICATION_CREDENTIALS=gcp.json
```

### ▶️ Lancer les deux API en local

```bash
docker compose up --build
```

Cela instancie deux services :

* `cloudrun-classmodel-backend` (port 8080)
* `cloudrun-regmodel-backend` (port 8000)

---

### 🔁 Endpoints locaux

#### ClassModel API

```bash
curl -X POST 'http://localhost:8080/predict' \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE", "date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00", "coordonnées_géographiques": "48.8672, 2.3501", "mois_annee_comptage": "mai 2025"}]}'
```

#### RegModel API

```bash
curl -X POST 'http://localhost:8000/predict' \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE", "date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00", "coordonnées_géographiques": "48.8672, 2.3501", "mois_annee_comptage": "mai 2025"}], "model_type": "nn", "metric": "r2"}'
```

---

## ☁️ Déploiement en production (GCP Cloud Run)

### 🧱 Configuration initiale

Créer le repository Docker :

```bash
gcloud artifacts repositories create cloud-run-images \
  --project=datascientest-460618 \
  --location=europe-west1 \
  --repository-format=docker \
  --description="Docker repository for Cloud Run images"
```

Créer le secret GCP :

```bash
gcloud secrets create gcp-service-account \
  --data-file=gcp.json
```

Donner accès au service Cloud Run pour lire les secrets :

```bash
gcloud projects add-iam-policy-binding datascientest-460618 \
  --member="serviceAccount:467498471756-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

Configurer Docker pour GCP :

```bash
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

---

## 🧩 Déploiement des services

### 🔹 ClassModel API

```bash
cd backend/classmodel

docker build -t europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/classmodel-api:latest .

docker push europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/classmodel-api:latest

gcloud run deploy classmodel-api \
  --image europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/classmodel-api:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory=4Gi \
  --set-env-vars=ENV=PROD \
  --update-secrets=GCP_JSON_CONTENT=gcp-service-account:latest
```

---

### 🔹 RegModel API

```bash
cd backend/regmodel

docker build -t europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/regmodel-api:latest .

docker push europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/regmodel-api:latest

gcloud run deploy regmodel-api \
  --image europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/regmodel-api:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory=4Gi \
  --set-env-vars=ENV=PROD \
  --update-secrets=GCP_JSON_CONTENT=gcp-service-account:latest
```

---

## 🌐 Endpoints en production

### ✅ RegModel API

```bash
curl -X POST "https://regmodel-api-467498471756.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE", "date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00", "coordonnées_géographiques": "48.8672, 2.3501", "mois_annee_comptage": "mai 2025"}], "model_type": "nn", "metric": "r2"}'
```

### ✅ ClassModel API

```bash
curl -X POST "https://classmodel-api-467498471756.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE", "date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00", "coordonnées_géographiques": "48.8672, 2.3501", "mois_annee_comptage": "mai 2025"}]}'
```

---

## 📊 Intégration Streamlit Cloud

Utiliser les URL suivantes dans le front Streamlit :

* [https://classmodel-api-467498471756.europe-west1.run.app/predict](https://classmodel-api-467498471756.europe-west1.run.app/predict)
* [https://regmodel-api-467498471756.europe-west1.run.app/predict](https://regmodel-api-467498471756.europe-west1.run.app/predict)

---

## 🧹 Maintenance

* Nettoyer régulièrement les artefacts dans : [https://console.cloud.google.com/artifacts](https://console.cloud.google.com/artifacts)
* Le **repository `cloud-run-images`** doit être utilisé pour tous les déploiements (éviter `gcf_artifacts` limité à 2GB).

---

## ✅ Conclusion – Retours d'expérience (REX)

### 🧠 Leçons apprises

#### 1. **Optimisation de la taille des images et des modèles**

* Pour rester dans les quotas du **free tier** (Render, Cloud Run, Cloud Functions), il faut **absolument compresser les modèles** et limiter les dépendances.
* La taille des containers est cruciale : `Render` limite à **512MB**, `Cloud Functions` à **2GB**.
* La meilleure architecture trouvée ici est un **déploiement via Cloud Run avec 4Gi de mémoire**, réparti entre **deux backends distincts** :

  * `regmodel-api`
  * `classmodel-api`
* Un seul backend pour tout ? Trop lourd. Deux microservices = déploiements plus efficaces.

#### 2. **Architecture de chargement des modèles**

* Initialement, les modèles étaient chargés directement depuis **Streamlit Cloud**. Résultat : trop lent, trop fragile, ou sinon, crash.
* Ensuite, les modèles étaient **chargés à chaque prédiction via API**. Trop coûteux, notamment si plusieurs modèles doivent être disponibles.
* Solution optimale :

  * **Chargement des "best models" dès le démarrage du container**
  * Possibilité de les **actualiser dynamiquement** via un endpoint `/refresh_model`
  * Les modèles sont stockés dans un **bucket GCS centralisé** et lus à chaque démarrage du backend (prod ou dev)

#### 3. **Développement et DevOps**

* L'aller-retour **dev/prod** est essentiel. Pouvoir reproduire un comportement localement avec `docker compose`, `curl` et `FastAPI` est une immense aide.
* Les temps de build/déploiement sur Cloud Run ou Render sont **très longs**. Il faut tester **un maximum de logique en local** pour éviter les frustrations.
* Les requêtes `curl` avec des JSON de test sont **indispensables** pour itérer vite et valider l'API sans dépendre du front.

#### 4. **Gestion des environnements (DEV / PROD)**

* Sur Render, la gestion d'environnements est **compliquée et peu flexible** (versions Python, dépendances, secrets...).
* En construisant ses propres images Docker, **tout est sous contrôle** :

  * Les bugs apparaissent **en local d’abord**, donc plus facile à debugger.
  * L’environnement est **isolé, reproductible et stable** : ce qui marche en local marchera en prod.
  * On maîtrise la version de Python, les libs, les chemins, les variables — **pas de mauvaise surprise au déploiement**.

---