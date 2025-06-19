cd cloud_function

gcloud beta functions deploy predict_api_v3 \
  --runtime python312 \
  --entry-point handler \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2048MB \
  --region=europe-west1 \
  --source=. \
  --env-vars-file env.yaml

  datascientest-460618

gcloud functions describe predict_api_v3 --region=europe-west1

docker build -t my_model_api .
docker run -p 8080:8080 my_model_api
