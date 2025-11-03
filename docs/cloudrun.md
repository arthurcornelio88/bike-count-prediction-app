# ENV

- FOr DEV
In .env
GOOGLE_APPLICATION_CREDENTIALS=gcp.json

---

in dev

```bash
docker compose up --build
```

> on aura les deux containers : `cloudrun-classmodel-backend` et `cloudrun-regmodels-backend`

---

ClassModel API

```bash
curl -X POST 'http://localhost:8080/predict' \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}]}'
```

RegModel API

```bash
curl -X POST 'http://localhost:8000/predict' \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}], "model_type": "nn","metric": "r2"}'
```

in prod

- For PROD

gcloud artifacts repositories create cloud-run-images \
  --project=datascientest-460618 \
  --location=europe-west1 \
  --repository-format=docker \
  --description="Docker repository for Cloud Run images"

gcloud secrets create gcp-service-account \
  --data-file=gcp.json

gcloud projects add-iam-policy-binding datascientest-460618 \
  --member="serviceAccount:<467498471756-compute@developer.gserviceaccount.com>" \
  --role="roles/secretmanager.secretAccessor"

gcloud run services update predict-api-v3 \
  --region=europe-west1 \
  --update-secrets="GOOGLE_APPLICATION_CREDENTIALS=gcp-service-account:latest"

gcloud auth configure-docker europe-west1-docker.pkg.dev

# Mounting ClassModel API

```bash
cd backend/classmodel
```

```bash
docker build -t europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/classmodel-api:latest .
```

docker push europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/classmodel-api:latest

gcloud run deploy classmodel-api \
  --image europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/classmodel-api:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory=4Gi \
  --set-env-vars=ENV=PROD \
  --update-secrets=GCP_JSON_CONTENT=gcp-service-account:latest

```text

# Mouting RegModel

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
```text

# Test endpoint

Pour RegModel API

```bash
curl -X POST "https://regmodel-api-467498471756.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}],
  "model_type": "nn","metric": "r2"}'
```text

Pour ClassModel API

```bash
curl -X POST "https://classmodel-api-467498471756.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}],
  "model_type": "rf_ class","metric": "f1_score"}'
```text

# Secrets in Streamlit Cloud

Copier coller les url from CloudRun + predict

<https://classmodel-api-467498471756.europe-west1.run.app/predict>
Important pour gestion

-> nettoyer de temps en temps les artifacts à chaque run deploy
<https://console.cloud.google.com/artifacts>
->  le nom du repo dans Artifact Registry détermine le contexte d'exécution final.
  cloud-run-images, et pas gcf_artifacts (limité à 2gb)
