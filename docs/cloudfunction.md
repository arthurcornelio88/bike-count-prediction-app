# ENV 

- FOr DEV
In .env 
GOOGLE_APPLICATION_CREDENTIALS=gcp.json

---

in dev 

docker build -t my_model_api .
docker run -p 8080:8080 my_model_api
docker run -p 8080:8080 --env-file .env my_model_api

---

```bash
curl -X POST 'http://localhost:8080/predict' \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}]}'
```
in prod

- For PROD
in env.yaml
GCP_JSON_CONTENT: |
  {
    "type": "service_account",
    "project_id": "datascientest-460618",
    etc...

cd cloud_run

gcloud artifacts repositories create gcf-artifacts \
  --project=datascientest-460618 \
  --location=europe-west1 \
  --repository-format=docker \
  --description="Docker repository for Cloud Run images"

gcloud secrets create gcp-service-account \
  --data-file=gcp.json

gcloud projects add-iam-policy-binding datascientest-460618 \
  --member="serviceAccount:467498471756-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud run services update predict-api-v3 \
  --region=europe-west1 \
  --update-secrets="GOOGLE_APPLICATION_CREDENTIALS=gcp-service-account:latest"

gcloud auth configure-docker europe-west1-docker.pkg.dev

docker build -t europe-west1-docker.pkg.dev/datascientest-460618/gcf-artifacts/predict-api-v3:latest .

docker push europe-west1-docker.pkg.dev/datascientest-460618/gcf-artifacts/predict-api-v3:latest

# Nom de l'image avec le bon repo
docker build -t europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/predict-api-v3:latest .

docker push europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/predict-api-v3:latest

gcloud run deploy predict-api-v3 \
  --image europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/predict-api-v3:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory=4Gi \
  --set-env-vars=ENV=PROD \
  --update-secrets=GCP_JSON_CONTENT=gcp-service-account:latest

# Test endpoint

```bash
curl -X POST 'https://europe-west1-datascientest-460618.cloudfunctions.net/predict_api_v3/predict' \
  -H "Content-Type: application/json" \
  -d '{"records": [{"nom_du_compteur": "35 boulevard de Ménilmontant NO-SE","date_et_heure_de_comptage": "2025-05-17 18:00:00+02:00","coordonnées_géographiques": "48.8672, 2.3501","mois_annee_comptage": "mai 2025"}]}'
```
Important pour gestion

-> nettoyer de temps en temps les artifacts à chaque run deploy
https://console.cloud.google.com/artifacts
->  le nom du repo dans Artifact Registry détermine le contexte d'exécution final. 
  cloud-run-images, et pas gcf_artifacts (limité à 2gb)


optionnel

gcloud projects add-iam-policy-binding datascientest-460618 \
  --member="user:arthur.cornelio@gmail.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding datascientest-460618 \
  --member="user:arthur.cornelio@gmail.com" \
  --role="roles/artifactregistry.writer"



  datascientest-460618



